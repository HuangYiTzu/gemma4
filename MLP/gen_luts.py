#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gen_luts.py -- Offline table generator + bit-accurate model check for the
Gemma-4-E2B single-layer MLP HLS kernel (decode stage, INT2 QAT weights).

Target model : google/gemma-4-E2B
Quantization : google/gemma-4-E2B-it-qat-mobile-transformers
                 (INT2 weights, INT8 activations, per-tensor activation scales)
Target layer : layer 15 (INT2, intermediate_size 12288) -- the scales are read
               from the bin_packing_mlp.py export in  mlp_bin_output/layer_15/
               (activation_scales.json + the weight-scale .bin files).

The kernel stops at down_proj: post_feedforward_layernorm and the residual add
live in a separate downstream block.  The MLP has FOUR quantization points,
all of which exist in the QAT checkpoint itself (no extra calibration is
introduced by the hardware):

  1) gate_proj output : S_G = gate_proj.output_activation_scale  -> requant q_g
  2) up_proj   output : S_U = up_proj.output_activation_scale    -> requant q_u
  3) down_proj input  : S_H = down_proj.input_activation_scale   -> folded in G'
  4) down_proj output : S_Y = down_proj.output_activation_scale  -> c_down

(the fifth scale S_IN = gate_proj.input_activation_scale = up_proj.input_
activation_scale is the grid of the INT8 input x_q; it only enters the
per-channel SRQ multipliers r_g / r_u that the host derives from the weight
scales, see below.)

G'[q_g] folds  "dequant(gate) -> GELU(tanh) -> rescale to the down_proj input
scale"  into ONE table lookup:

      G'[q_g] = round_half_even( GELU(q_g * S_G) * S_U / S_H * 2^GLUT_FRAC )

q_g only has 256 possible values, so the table is EXACT, not an approximation.
GLUT_FRAC is chosen AUTOMATICALLY as the largest number of fraction bits for
which every entry still fits int16 (layer 15: S_U/S_H = 1.52, peak 5.90 ->
Q4.12, i.e. GLUT_FRAC = 12).  It is a compile-time constant of the kernel:
regenerate the header when the layer changes.

The elementwise stage of the kernel is then pure integer arithmetic:

      q_g  = clamp( rne( r_g[i] * acc_gate, 2^-20 ), -128, 127 )   # SRQ
      q_u  = clamp( rne( r_u[i] * acc_up,   2^-20 ), -128, 127 )   # SRQ
      h[i] = clamp( rne( G'[q_g] * q_u , >>GLUT_FRAC ), -128, 127 )
      (h is already at the down_proj input scale S_H -> no further quant)

      r_g[i] = round( ws_gate[i] * S_IN / S_G * 2^20 )             # host, INT32
      r_u[i] = round( ws_up[i]   * S_IN / S_U * 2^20 )             # host, INT32

The down_proj output is REQUANTIZED (not dequantized) to INT16 codes:

      p[j]      = sum_i W_down[j,i] * h[i]                       # INT32
      y_q[j]    = clamp( rne( p[j] * c_down[j], 2^-24 ), -32768, 32767 )
      c_down[j] = round( ws_down[j] * S_H / S_Y16 * 2^24 )       # host, INT32
      S_Y16     = S_Y / 256

S_Y16 is 256x finer than the checkpoint's INT8 output grid, so the downstream
RMSNorm block gets the down_proj result with 8 extra bits of resolution and
8 bits of headroom; its real value is y_q * S_Y16.

Usage:
    python gen_luts.py                       # layer_15 -> gelu_luts.h + checks
    python gen_luts.py --layer-dir DIR       # another export directory
    python gen_luts.py --out FILE            # custom output header path
    python gen_luts.py --no-check            # generate only
"""

import argparse
import json
import math
import os
import random
import struct
import sys
from fractions import Fraction

# --------------------------------------------------------------------------
# Kernel geometry (must match kernel.h) and fixed-point formats
# --------------------------------------------------------------------------
K_HID   = 1536          # hidden_size
F_INT   = 12288         # intermediate_size of the INT2 layers
NPC     = 32            # HBM pseudo-channels
DN      = 512           # macro tile width
CPP     = DN // NPC     # 16 fused columns per PC per tile
TILES   = 2 * F_INT // DN   # 48 fused gate/up tiles

RQ_SHIFT  = 20      # SRQ requant multipliers r_g / r_u are raw * 2^-20
CD_SHIFT  = 24      # c_down requant multipliers are raw * 2^-24

Y16_MIN, Y16_MAX = -32768, 32767
PS_BOUND  = F_INT * 2 * 128      # |p|   <= F * max|w| * max|h|
ACC_BOUND = K_HID * 2 * 127      # |acc| <= K * max|w| * max|x|

# GELU tanh approximation (gelu_pytorch_tanh, as used by Gemma)
K1 = 0.7978845608028654       # sqrt(2/pi)
K3 = 0.044715


def gelu_tanh(x):
    return 0.5 * x * (1.0 + math.tanh(K1 * (x + K3 * x * x * x)))


# --------------------------------------------------------------------------
# Scales of the target layer, read from the bin_packing_mlp.py export.
#
#   activation_scales.json : 6 per-tensor activation scales (float32 values)
#   gate_up_scale_pcNN.bin : 768 x float32 LE per PC, fused-column order
#                            (tile t, column j) -> fused column 512t + 16p + j,
#                            even = gate_proj.weight_scale, odd = up_proj
#   down_scale_slr0.bin    : 1536 x float32 LE, down_proj.weight_scale[j]
# (the full format description lives at the top of mlp_model.h)
# --------------------------------------------------------------------------
class LayerScales(object):
    def __init__(self, layer_dir):
        self.dir = layer_dir
        with open(os.path.join(layer_dir, "activation_scales.json")) as f:
            a = json.load(f)
        self.S_IN = a["gate_proj_input"]
        if a["up_proj_input"] != a["gate_proj_input"]:
            raise SystemExit("gate_proj/up_proj input scales differ: the "
                             "kernel shares one x_q between them")
        self.S_G = a["gate_proj_output"]
        self.S_U = a["up_proj_output"]
        self.S_H = a["down_proj_input"]
        self.S_Y = a["down_proj_output"]
        self.S_Y16 = self.S_Y / 256.0

        # the same six scalars as float32 LE, in JSON key order
        p = os.path.join(layer_dir, "activation_scales.bin")
        if os.path.exists(p):
            with open(p, "rb") as f:
                b = struct.unpack("<6f", f.read(24))
            for x, y in zip(b, a.values()):
                if x != y:
                    raise SystemExit("activation_scales.bin disagrees with .json")

        fused = [0.0] * (2 * F_INT)
        for pc in range(NPC):
            with open(os.path.join(layer_dir, "gate_up_scale_pc%02d.bin" % pc),
                      "rb") as f:
                v = struct.unpack("<%df" % (TILES * CPP), f.read(TILES * CPP * 4))
            for t in range(TILES):
                for j in range(CPP):
                    fused[t * DN + pc * CPP + j] = v[t * CPP + j]
        self.ws_gate = fused[0::2]
        self.ws_up   = fused[1::2]
        with open(os.path.join(layer_dir, "down_scale_slr0.bin"), "rb") as f:
            self.ws_down = list(struct.unpack("<%df" % K_HID, f.read(K_HID * 4)))

    def layer_index(self):
        m = os.path.basename(os.path.normpath(self.dir))
        return int(m.split("_")[-1]) if "_" in m else -1


# --------------------------------------------------------------------------
# Bit-accurate integer helpers (mirror MLP.cpp / mlp_model.h exactly)
# --------------------------------------------------------------------------
def rne(v, sh):
    """round-half-to-even of v * 2^-sh on integers (matches HLS datapath)."""
    q = v >> sh                     # floor
    rem = v - (q << sh)             # in [0, 2^sh)
    half = 1 << (sh - 1)
    if rem > half or (rem == half and (q & 1)):
        q += 1
    return q


def rne_real(x):
    """round-half-to-even of a real number (Python round(), numpy.round)."""
    f = math.floor(x)
    d = x - f
    if d > 0.5:
        return f + 1
    if d < 0.5:
        return f
    return f if (f % 2 == 0) else f + 1


def clamp8(v):
    return max(-128, min(127, v))


def clamp16(v):
    return max(Y16_MIN, min(Y16_MAX, v))


def srq(acc, r):
    """SRQ: INT32 accumulator -> INT8 at the proj output activation scale."""
    return clamp8(rne(acc * r, RQ_SHIFT))


def h_hw(qg, qu, glut, frac):
    """elementwise GELU*up via the exact G' table."""
    return clamp8(rne(glut[qg + 128] * qu, frac))


def cd_from_ws(ws, sc):
    """host side: c_down = round(ws_down * S_H / S_Y16 * 2^24), must fit INT32."""
    c = rne_real(ws * sc.S_H / sc.S_Y16 * (1 << CD_SHIFT))
    assert 0 <= c <= 0x7FFFFFFF, "c_down does not fit INT32 (ws_down=%g)" % ws
    return c


def rq_from_ws(ws, s_in, s_out):
    """host side: r = round(ws * S_IN / S_out * 2^20), must fit INT32."""
    r = rne_real(ws * s_in / s_out * (1 << RQ_SHIFT))
    assert 0 <= r <= 0x7FFFFFFF, "requant multiplier does not fit INT32"
    return r


def y_hw(p, c):
    """down_proj requant: INT32 partial sum -> INT16 code at S_Y16."""
    return clamp16(rne(p * c, CD_SHIFT))


# --------------------------------------------------------------------------
# G' table construction (exact by definition: 256 entries, one per q_g code)
# GLUT_FRAC = the most fraction bits for which every entry fits int16.
# --------------------------------------------------------------------------
def glut_real(sc, i):
    return gelu_tanh((i - 128) * sc.S_G) * sc.S_U / sc.S_H


def pick_glut_frac(sc):
    peak = max(abs(glut_real(sc, i)) for i in range(256))
    for frac in range(15, -1, -1):
        if all(-32768 <= rne_real(glut_real(sc, i) * (1 << frac)) <= 32767
               for i in range(256)):
            return frac, peak
    raise SystemExit("G' peak %g does not fit int16 at all" % peak)


def build_glut(sc, frac):
    g = []
    for i in range(256):
        v = rne_real(glut_real(sc, i) * (1 << frac))
        assert -32768 <= v <= 32767, "G' entry does not fit int16"
        g.append(v)
    return g


# --------------------------------------------------------------------------
# Verification: full elementwise pipeline vs. double-precision model
# --------------------------------------------------------------------------
def check(sc, glut, frac):
    ok = True

    # 1) table sanity: exactness (recompute) + endpoints
    for i in range(256):
        ref = rne_real(glut_real(sc, i) * (1 << frac))
        assert glut[i] == ref
    print("G' table  : 256 entries, exact by construction, Q%d.%d; "
          "G'[-128]=%d  G'[0]=%d  G'[+127]=%d"
          % (16 - frac, frac, glut[0], glut[128], glut[255]))

    # 2) SRQ -> G' -> multiply -> rne pipeline, requant multipliers drawn from
    #    the REAL per-channel range, against two double-precision models:
    #      a) same INT8 codes q_g / q_u -> isolates the G' table rounding:
    #         |err| <= 0.5 (final rne) + 128 * 2^-(frac+1) (table entry)
    #      b) unrounded g / u (the SRQ rounding is left in) -> the inherent
    #         error of INT8 activations at S_G / S_U propagated through
    #         GELU(g) * u / S_H, bounded analytically from the scales:
    #         0.5 + 0.5 * max|dh/dq_g| + 0.5 * max|dh/dq_u|
    rg_lo, rg_hi = min(RG), max(RG)
    ru_lo, ru_hi = min(RU), max(RU)
    dg = 1e-4
    slope = max(abs(gelu_tanh(x + dg) - gelu_tanh(x - dg)) / (2 * dg)
                for x in [(-128 + n / 8.0) * sc.S_G for n in range(256 * 8 + 1)])
    peak = max(abs(glut_real(sc, i)) for i in range(256))
    bound_a = 0.5 + 128.0 * 2.0 ** -(frac + 1)
    bound_b = 0.5 + 0.5 * slope * sc.S_G * 128 * sc.S_U / sc.S_H + 0.5 * peak
    rnd = random.Random(20260825)
    err_a = err_b = 0.0
    for _ in range(200000):
        accg = rnd.randint(-ACC_BOUND, ACC_BOUND)
        accu = rnd.randint(-ACC_BOUND, ACC_BOUND)
        rg = rnd.randint(rg_lo, rg_hi)
        ru = rnd.randint(ru_lo, ru_hi)
        qg, qu = srq(accg, rg), srq(accu, ru)
        h = h_hw(qg, qu, glut, frac)

        ha = max(-128.0, min(127.0, glut_real(sc, qg + 128) * qu))
        err_a = max(err_a, abs(h - ha))

        gd = max(-128.0, min(127.0, accg * rg / 2.0 ** RQ_SHIFT)) * sc.S_G
        ud = max(-128.0, min(127.0, accu * ru / 2.0 ** RQ_SHIFT)) * sc.S_U
        hb = max(-128.0, min(127.0, gelu_tanh(gd) * ud / sc.S_H))
        err_b = max(err_b, abs(h - hb))
    print("h pipeline: r_g in [%d, %d], r_u in [%d, %d], 200000 vectors"
          % (rg_lo, rg_hi, ru_lo, ru_hi))
    print("h pipeline: a) same q_g/q_u codes : max |err| = %.3f LSB "
          "(bound %.3f: rne 0.5 + table 128*2^-%d)" % (err_a, bound_a, frac + 1))
    print("h pipeline: b) unrounded g / u    : max |err| = %.3f LSB "
          "(bound %.3f: INT8 q_g/q_u rounding through GELU*up, max slope %.3f)"
          % (err_b, bound_b, slope))
    if err_a > bound_a or err_b > bound_b:
        ok = False
    return ok


# --------------------------------------------------------------------------
# Verification: down_proj requant to INT16 codes
#
#   1) rne(p * c_down, 24) is checked EXACTLY against Fraction rounding
#      (Python's round() on a Fraction is round-half-to-even), with exact .5
#      ties forced on a quarter of the vectors;
#   2) the unclamped code is checked against the real value p * ws * S_H in
#      S_Y16 units.  Two errors add up: the output rounding (<= 0.5 LSB) and
#      the host rounding of c_down to an integer (<= |p| * 2^-25 LSB).
# --------------------------------------------------------------------------
def check_down_requant(sc):
    ok = True
    gain = sc.S_H / sc.S_Y16
    ws_max = 0x7FFFFFFF / (gain * (1 << CD_SHIFT))
    print("down rq  : c_down = round(ws_down * S_H / S_Y16 * 2^%d), "
          "S_H / S_Y16 = %g" % (CD_SHIFT, gain))
    print("down rq  : c_down fits INT32 for ws_down < %.6g "
          "(layer max ws_down = %.6g)" % (ws_max, max(sc.ws_down)))
    print("down rq  : |p * c_down| < 2^%d  -> ap_int<64> product, "
          "INT32 before the clamp" % (PS_BOUND * 0x7FFFFFFF).bit_length())

    rnd = random.Random(20260914)
    worst = 0.0
    nsat = nties = nbad = 0
    for n in range(200000):
        p = rnd.randint(-PS_BOUND, PS_BOUND)
        if n % 4 == 0:
            # c_down = 2^23 (gain 0.5) and an odd p -> an exact .5 tie
            ws = 0.5 * sc.S_Y16 / sc.S_H
            p |= 1
        elif n % 4 == 1:
            ws = rnd.choice(sc.ws_down)          # a real layer-15 channel
        else:
            ws = ws_max * 10.0 ** rnd.uniform(-7.0, -1e-3)
        c = cd_from_ws(ws, sc)

        raw = rne(p * c, CD_SHIFT)
        if raw != round(Fraction(p * c, 1 << CD_SHIFT)):
            nbad += 1
        if (p * c) & ((1 << CD_SHIFT) - 1) == 1 << (CD_SHIFT - 1):
            nties += 1
        y = y_hw(p, c)
        if y != raw:
            nsat += 1
            continue

        err = abs(y - p * ws * sc.S_H / sc.S_Y16)
        bound = 0.5 + abs(p) * 2.0 ** -(CD_SHIFT + 1) + 1e-6
        worst = max(worst, err)
        if err > bound:
            ok = False

    print("down rq  : 200000 vectors, rne vs exact Fraction: %d mismatches, "
          "%d exact ties" % (nbad, nties))
    print("down rq  : max |y_q - p*ws*S_H/S_Y16| = %.3f LSB unclamped "
          "(bound 0.5 + |p|*2^-25 <= %.3f), %d INT16 clamps"
          % (worst, 0.5 + PS_BOUND * 2.0 ** -(CD_SHIFT + 1), nsat))
    return ok and nbad == 0 and nties > 0 and nsat > 0


# --------------------------------------------------------------------------
# Verification: the real per-channel multipliers of the layer
# --------------------------------------------------------------------------
def check_layer_multipliers(sc):
    print("layer     : ws_gate [%.6g, %.6g]  ws_up [%.6g, %.6g]  "
          "ws_down [%.6g, %.6g]"
          % (min(sc.ws_gate), max(sc.ws_gate), min(sc.ws_up), max(sc.ws_up),
             min(sc.ws_down), max(sc.ws_down)))
    print("layer     : r_g [%d, %d]  r_u [%d, %d]  c_down [%d, %d]  (all INT32)"
          % (min(RG), max(RG), min(RU), max(RU), min(CD), max(CD)))
    # SRQ product and rounded code must fit the kernel's 64 / 32-bit paths
    top = ACC_BOUND * max(max(RG), max(RU))
    assert top < (1 << 63)
    assert rne(top, RQ_SHIFT) < (1 << 31)
    print("layer     : |acc * r| < 2^%d, rounded SRQ code < 2^%d before the "
          "INT8 clamp" % (top.bit_length(), rne(top, RQ_SHIFT).bit_length()))
    return True


# --------------------------------------------------------------------------
# Header emission (plain C types: header is shared by HLS, tb AND host code)
# --------------------------------------------------------------------------
def fmt_array(name, ctype, vals, per_line=8):
    lines = ["static const %s %s[256] = {" % (ctype, name)]
    for i in range(0, 256, per_line):
        row = ", ".join("%6d" % v for v in vals[i:i + per_line])
        lines.append("    " + row + ("," if i + per_line < 256 else ""))
    lines.append("};")
    return "\n".join(lines)


def emit_header(path, sc, glut, frac):
    t = []
    t.append("// gelu_luts.h -- AUTO-GENERATED by gen_luts.py.  DO NOT EDIT BY HAND.")
    t.append("//   regenerate with:  python gen_luts.py --layer-dir %s"
             % sc.dir.replace("\\", "/"))
    t.append("// Gemma-4-E2B MLP, layer %d: folded GELU table G' (Q%d.%d, exact) and"
             % (sc.layer_index(), 16 - frac, frac))
    t.append("// the activation scales of the QAT checkpoint (per-tensor, float32).")
    t.append("// Plain C types on purpose: shared by MLP.cpp, tb_MLP.cpp and host.cpp.")
    t.append("#ifndef GELU_LUTS_H")
    t.append("#define GELU_LUTS_H")
    t.append("")
    t.append("#define MLP_LAYER_IDX %d   // checkpoint layer the tables below belong to"
             % sc.layer_index())
    t.append("")
    t.append("// quantization points, read from activation_scales.json of the export")
    t.append("static const double MLP_S_IN  = %.17g;  // gate_proj/up_proj.input_activation_scale (x_q grid)" % sc.S_IN)
    t.append("static const double MLP_S_G   = %.17g;  // gate_proj.output_activation_scale" % sc.S_G)
    t.append("static const double MLP_S_U   = %.17g;  // up_proj.output_activation_scale" % sc.S_U)
    t.append("static const double MLP_S_H   = %.17g;  // down_proj.input_activation_scale" % sc.S_H)
    t.append("static const double MLP_S_Y   = %.17g;  // down_proj.output_activation_scale" % sc.S_Y)
    t.append("static const double MLP_S_Y16 = %.17g;  // S_Y / 256: grid of the INT16 output y_q" % sc.S_Y16)
    t.append("")
    t.append("#define MLP_GLUT_FRAC %d   // G' is Q%d.%d (auto: peak |G'| = %.4f)"
             % (frac, 16 - frac, frac, max(abs(glut_real(sc, i)) for i in range(256))))
    t.append("#define MLP_RQ_SHIFT  %d   // SRQ multipliers r_g/r_u:  raw * 2^-%d" % (RQ_SHIFT, RQ_SHIFT))
    t.append("#define MLP_CD_SHIFT  %d   // c_down requant multiplier: raw * 2^-%d" % (CD_SHIFT, CD_SHIFT))
    t.append("")
    t.append("// G'[q_g + 128] = rne( GELU_tanh(q_g*S_G) * S_U / S_H * 2^MLP_GLUT_FRAC ).")
    t.append("// Folds dequant -> GELU -> rescale-to-S_H into one EXACT 256-entry ROM.")
    t.append(fmt_array("GELU_LUT_I16", "short", glut))
    t.append("")
    t.append("#endif // GELU_LUTS_H")
    t.append("")
    with open(path, "w", newline="\n") as f:
        f.write("\n".join(t))
    print("wrote %s" % path)


def main():
    global RG, RU, CD
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer-dir", default=os.path.join("mlp_bin_output", "layer_15"))
    ap.add_argument("--out", default="gelu_luts.h")
    ap.add_argument("--no-check", action="store_true")
    args = ap.parse_args()

    sc = LayerScales(args.layer_dir)
    frac, peak = pick_glut_frac(sc)
    glut = build_glut(sc, frac)
    RG = [rq_from_ws(w, sc.S_IN, sc.S_G) for w in sc.ws_gate]
    RU = [rq_from_ws(w, sc.S_IN, sc.S_U) for w in sc.ws_up]
    CD = [cd_from_ws(w, sc) for w in sc.ws_down]

    print("layer %d: S_IN = %g  S_G = %g  S_U = %g  S_H = %g  S_Y = %g  S_Y16 = %g"
          % (sc.layer_index(), sc.S_IN, sc.S_G, sc.S_U, sc.S_H, sc.S_Y, sc.S_Y16))
    print("G' peak |GELU(q_g*S_G)*S_U/S_H| = %.4f  ->  GLUT_FRAC = %d (Q%d.%d)"
          % (peak, frac, 16 - frac, frac))
    emit_header(args.out, sc, glut, frac)

    if not args.no_check:
        print("\n--- bit-accurate model verification ---")
        ok = check_layer_multipliers(sc)
        ok = check(sc, glut, frac) and ok
        print("")
        ok = check_down_requant(sc) and ok
        print("RESULT: %s" % ("PASS" if ok else "FAIL"))
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
