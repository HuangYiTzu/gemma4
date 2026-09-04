#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gen_luts.py -- Offline table generator + bit-accurate model check for the
Gemma-4-E2B single-layer MLP HLS kernel (decode stage, INT2 QAT weights).

Target model : google/gemma-4-E2B
Quantization : google/gemma-4-E2B-it-qat-mobile-transformers
                 (INT2 weights, INT8 activations, per-tensor activation scales)

The MLP has exactly THREE quantization points, all of which exist in the QAT
checkpoint itself (no extra calibration is introduced by the hardware):

  1) gate_proj output : S_G = gate_proj.output_activation_scale  -> requant q_g
  2) up_proj   output : S_U = up_proj.output_activation_scale    -> requant q_u
  3) down_proj input  : S_H = down_proj.input_activation_scale   -> folded in G'

G'[q_g] folds  "dequant(gate) -> GELU(tanh) -> rescale to the down_proj input
scale"  into ONE table lookup:

      G'[q_g] = round_half_even( GELU(q_g * S_G) * S_U / S_H * 2^13 )   (Q3.13)

q_g only has 256 possible values, so the table is EXACT, not an approximation.
The elementwise stage of the kernel is then pure integer arithmetic:

      q_g  = clamp( rne( r_g[i] * acc_gate, 2^-20 ), -128, 127 )   # SRQ
      q_u  = clamp( rne( r_u[i] * acc_up,   2^-20 ), -128, 127 )   # SRQ
      h[i] = clamp( rne( G'[q_g] * q_u , >>13 ),     -128, 127 )
      (h is already at the down_proj input scale S_H -> no further quant)

The header also carries the fast-inverse magic seeds (paper ISCAS'25 fig.2);
the 1/sqrt(x) mode is used by the post_feedforward_layernorm (RMSNorm) on the
kernel's final stage.

NOTE: the real QAT scale export is not ready yet, therefore S_G / S_U / S_H
below are PLACEHOLDER values picked for functional verification only.
Regenerating this header with the real checkpoint scales is the only change
needed when they become available.

Usage:
    python gen_luts.py                 # generate gelu_luts.h + run model check
    python gen_luts.py --out FILE      # custom output header path
    python gen_luts.py --no-check      # generate only
"""

import argparse
import math
import random
import sys

# --------------------------------------------------------------------------
# Quantization points -- MUST match kernel.h / MLP.cpp
# (placeholder values, see file header)
# --------------------------------------------------------------------------
S_G = 0.0625        # gate_proj.output_activation_scale : g = q_g * S_G
S_U = 0.0625        # up_proj.output_activation_scale   : u = q_u * S_U
S_H = 0.25          # down_proj.input_activation_scale  : h = q_h * S_H
                    # (S_H is deliberately tight enough that the strongest
                    #  channels saturate, so the INT8 clamp on h is covered by
                    #  the functional test instead of being dead code)

GLUT_FRAC = 13      # G' is Q3.13
RQ_SHIFT  = 20      # SRQ requant multipliers r_g / r_u are raw * 2^-20
CD_SHIFT  = 24      # c_down dequant scales are raw * 2^-24
LN_FRAC   = 13      # layernorm (1+gamma) weights are raw * 2^-13 (Q3.13)
HID_FRAC  = 8       # hidden-state / residual / output fixed point is Q7.8
LN_EPS    = 2.0 ** -20

# GELU tanh approximation (gelu_pytorch_tanh, as used by Gemma)
K1 = 0.7978845608028654       # sqrt(2/pi)
K3 = 0.044715

# fast-inverse magic seeds: I_y = MAGIC - (I_x >> s)
#   s=0 for 1/x       (a2 = 2(B-sigma)L)
#   s=1 for 1/sqrt(x) (a1 = 3/2(B-sigma)L)   <- used by the RMSNorm stage
SIGMA = 0.0450465
FP_B, FP_L = 127, 1 << 23
MAGIC_RECIP = int(round(2.0 * (FP_B - SIGMA) * FP_L))
MAGIC_RSQRT = int(round(1.5 * (FP_B - SIGMA) * FP_L))


def gelu_tanh(x):
    return 0.5 * x * (1.0 + math.tanh(K1 * (x + K3 * x * x * x)))


# --------------------------------------------------------------------------
# Bit-accurate integer helpers (mirror MLP.cpp / kernel.h exactly)
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


def srq(acc, r):
    """SRQ: INT32 accumulator -> INT8 at the proj output activation scale."""
    return clamp8(rne(acc * r, RQ_SHIFT))


def h_hw(qg, qu, glut):
    """elementwise GELU*up via the exact G' table."""
    return clamp8(rne(glut[qg + 128] * qu, GLUT_FRAC))


# --------------------------------------------------------------------------
# G' table construction (exact by definition: 256 entries, one per q_g code)
# --------------------------------------------------------------------------
def build_glut():
    g = []
    for i in range(256):
        qg = i - 128
        val = gelu_tanh(qg * S_G) * S_U / S_H * (1 << GLUT_FRAC)
        v = rne_real(val)
        assert -32768 <= v <= 32767, "G' entry does not fit int16"
        g.append(v)
    return g


# --------------------------------------------------------------------------
# Verification: full elementwise pipeline vs. double-precision model
# --------------------------------------------------------------------------
def check(glut):
    ok = True

    # 1) table sanity: exactness (recompute) + endpoints
    for i in range(256):
        qg = i - 128
        ref = rne_real(gelu_tanh(qg * S_G) * S_U / S_H * (1 << GLUT_FRAC))
        assert glut[i] == ref
    print("G' table  : 256 entries, exact by construction; "
          "G'[-128]=%d  G'[0]=%d  G'[+127]=%d" % (glut[0], glut[128], glut[255]))

    # 2) SRQ -> G' -> multiply -> rne pipeline against the double model.
    #    The double model applies the same INT8 clamp domain so the comparison
    #    isolates the LUT/rounding error (not saturation policy).
    rnd = random.Random(20260825)
    max_err = 0.0
    for _ in range(200000):
        accg = rnd.randint(-390144, 390144)   # |acc| <= K*2*127
        accu = rnd.randint(-390144, 390144)
        rg = rnd.randint(8192, 24576)
        ru = rnd.randint(8192, 24576)
        h = h_hw(srq(accg, rg), srq(accu, ru), glut)

        gd = max(-128.0, min(127.0, accg * rg / 2.0 ** RQ_SHIFT)) * S_G
        ud = max(-128.0, min(127.0, accu * ru / 2.0 ** RQ_SHIFT)) * S_U
        hd = max(-128.0, min(127.0, gelu_tanh(gd) * ud / S_H))
        max_err = max(max_err, abs(h - hd))
    print("h pipeline: max |err| vs double model = %.3f LSB "
          "(SRQ 0.5 + GELU-prop ~0.6 + RNE 0.5)" % max_err)
    if max_err > 2.0:
        ok = False
    return ok


# --------------------------------------------------------------------------
# Bit-accurate model of MLP.cpp's fast_inv_unit<ms_t, rsq_t>(x, false)
#
# The elementwise check above hammers the SRQ/LUT path with 200k vectors, but
# the RMSNorm inverse-square-root unit is called ONCE per kernel invocation --
# so an end-to-end simulation, however many tokens it runs, only ever tests it
# at a handful of points.  Sweeping it here costs nothing and is the only way
# to see the whole input range.
#
# Fixed-point formats (kernel.h / MLP.cpp):
#   x   ms_t   ap_ufixed<56,32,AP_RND>   -> 24 fractional bits
#   y   rsq_t  ap_ufixed<32,12>          -> 20 fractional bits, AP_TRN
#   yb         ap_ufixed<56,25>          -> 31 fractional bits
#   xy  xy_t   ap_ufixed<32,18,AP_RND>   -> 14 fractional bits
#   t   t_t    ap_ufixed<26,2,AP_RND>    -> 24 fractional bits
# --------------------------------------------------------------------------
MS_F, RSQ_F, RSQ_W, YB_F, XY_F, XY_W, T_F, T_W = 24, 20, 32, 31, 14, 32, 24, 26


def _rnd(n, sh):
    """AP_RND: round to nearest, ties toward +inf, on n * 2^-sh."""
    return (n + (1 << (sh - 1))) >> sh if sh > 0 else n << (-sh)


def fast_inv_rsqrt(ms):
    """Returns (y, branch_pos, branch_ey) exactly as the HLS unit computes it."""
    n_x = min(_rnd(int(ms * (1 << (MS_F + 8))), 8), (1 << 56) - 1)   # ms_t
    if n_x == 0:
        return 0.0, None, None

    pos = n_x.bit_length() - 1                    # leading-one detector
    mb = n_x & ~(1 << pos)
    if pos >= 23:
        M_x = (mb >> (pos - 23)) & 0x7FFFFF
        br_pos = "ge23"
    else:
        M_x = (mb << (23 - pos)) & 0x7FFFFF
        br_pos = "lt23"
    e_x = pos - MS_F

    I_x = ((((e_x + 127) & 0x1FF) << 23) | M_x) & 0xFFFFFFFF
    I_y = (MAGIC_RSQRT - (I_x >> 1)) & 0xFFFFFFFF

    e_y = (I_y >> 23) - 127
    M_y = I_y & 0x7FFFFF
    n_yb = (1 << YB_F) | (M_y << (YB_F - 23))     # yb, 31 fractional bits

    if e_y >= 0:
        n_sh, br_ey = n_yb << e_y, "ge0"
    else:
        n_sh, br_ey = n_yb >> (-e_y), "lt0"
    n_y = (n_sh >> (YB_F - RSQ_F)) & ((1 << RSQ_W) - 1)               # -> rsq_t

    for _ in range(2):                            # two Newton-Raphson steps
        n_xy = _rnd(n_x * n_y, MS_F + RSQ_F - XY_F) & ((1 << XY_W) - 1)
        n_half = (n_xy * n_y) >> 1                # F = XY_F + RSQ_F
        n_b1 = 3 << (XY_F + RSQ_F - 1)            # 1.5 at F = XY_F + RSQ_F
        n_t = _rnd((n_b1 - n_half) & ((1 << 64) - 1),
                   XY_F + RSQ_F - T_F) & ((1 << T_W) - 1)
        n_y = ((n_y * n_t) >> T_F) & ((1 << RSQ_W) - 1)

    return n_y / float(1 << RSQ_F), br_pos, br_ey


def check_fastinv():
    """Sweep the RMSNorm rsqrt unit and report where it is actually usable."""
    print("fast_inv : sweeping the RMSNorm 1/sqrt unit (called ONCE per token,")
    print("           so end-to-end simulation can never sweep it)")
    print("           %-12s %-12s %-12s %-9s %s"
          % ("mean-square", "rsqrt(hw)", "rsqrt(ref)", "rel.err", "branches"))

    seen = set()
    worst_ok, first_bad = 0.0, None
    ms = 2.0 ** -20
    rows = []
    while ms < 2.0 ** 34:
        y, bp, be = fast_inv_rsqrt(ms)
        ref = 1.0 / math.sqrt(ms)
        rel = abs(y - ref) / ref if ref else 0.0
        if bp:
            seen.add((bp, be))
        rows.append((ms, y, ref, rel, bp, be))
        # 1e-3 relative on rs is ~0.03 LSB of output error at |y_norm| ~ 8
        if rel <= 1e-3:
            worst_ok = max(worst_ok, ms)
        elif first_bad is None and ms > 1.0:
            first_bad = ms
        ms *= 16.0

    for (ms, y, ref, rel, bp, be) in rows:
        print("           %-12.3e %-12.5e %-12.5e %-9.2e %s/%s"
              % (ms, y, ref, rel, bp, be))

    print("fast_inv : branch pairs reached by this sweep : %s"
          % ", ".join(sorted("%s/%s" % b for b in seen)))
    print("fast_inv :   (only two of the four combinations are reachable at all:")
    print("fast_inv :    pos < 23 means ms < 0.5, which forces rs > 1.41 and e_y >= 0)")
    print("fast_inv : rel.err <= 1e-3 holds up to mean-square ~ %.3e" % worst_ok)
    print("fast_inv : NOTE -- rsq_t is ap_ufixed<32,12>, i.e. 20 fractional bits, so its")
    print("fast_inv :         ABSOLUTE step is 2^-20 and the RELATIVE error grows as rs")
    print("fast_inv :         shrinks.  The comment on rsq_t in kernel.h claims the unit")
    print("fast_inv :         covers rs in [2^-16, 2^10], but at rs = 2^-16 the relative")
    print("fast_inv :         error is 6.25% -- that end of the claimed range is not")
    print("fast_inv :         usable at this width.  Either keep the RMSNorm operating")
    print("fast_inv :         point inside the window above (MLP_CD_DECADE in mlp_model.h")
    print("fast_inv :         is sized for it) or widen rsq_t.")
    # both reachable branch pairs must be exercised
    return len(seen) >= 2


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


def emit_header(path, glut):
    t = []
    t.append("// gelu_luts.h -- AUTO-GENERATED by gen_luts.py.  DO NOT EDIT BY HAND.")
    t.append("//   regenerate with:  python gen_luts.py")
    t.append("// Gemma-4-E2B MLP: folded GELU table G' (Q3.13, exact) + fast-inverse")
    t.append("// magic seeds (rsqrt mode drives the post_feedforward RMSNorm).")
    t.append("// Plain C types on purpose: shared by MLP.cpp, tb_MLP.cpp and host.cpp.")
    t.append("#ifndef GELU_LUTS_H")
    t.append("#define GELU_LUTS_H")
    t.append("")
    t.append("// quantization points (PLACEHOLDER values until the real QAT scale")
    t.append("// export is ready -- see gen_luts.py header)")
    t.append("static const double MLP_S_G = %.17g;  // gate_proj.output_activation_scale" % S_G)
    t.append("static const double MLP_S_U = %.17g;  // up_proj.output_activation_scale" % S_U)
    t.append("static const double MLP_S_H = %.17g;  // down_proj.input_activation_scale" % S_H)
    t.append("")
    t.append("#define MLP_GLUT_FRAC %d   // G' is Q3.13" % GLUT_FRAC)
    t.append("#define MLP_RQ_SHIFT  %d   // SRQ multipliers r_g/r_u: raw * 2^-20" % RQ_SHIFT)
    t.append("#define MLP_CD_SHIFT  %d   // c_down dequant scales:   raw * 2^-24" % CD_SHIFT)
    t.append("#define MLP_LN_FRAC   %d   // layernorm (1+gamma):     raw * 2^-13" % LN_FRAC)
    t.append("#define MLP_HID_FRAC  %d    // hidden / residual / output: Q7.8" % HID_FRAC)
    t.append("#define MLP_LN_EPS    %.17g  // 2^-20, RMSNorm epsilon" % LN_EPS)
    t.append("")
    t.append("// I_y = MAGIC - (I_x >> s):  s=0 for 1/x, s=1 for 1/sqrt(x) (RMSNorm)")
    t.append("static const unsigned int FASTINV_MAGIC_RECIP = 0x%08XU; // %d" % (MAGIC_RECIP, MAGIC_RECIP))
    t.append("static const unsigned int FASTINV_MAGIC_RSQRT = 0x%08XU; // %d" % (MAGIC_RSQRT, MAGIC_RSQRT))
    t.append("")
    t.append("// G'[q_g + 128] = rne( GELU_tanh(q_g*S_G) * S_U / S_H * 2^13 ), Q3.13.")
    t.append("// Folds dequant -> GELU -> rescale-to-S_H into one EXACT 256-entry ROM.")
    t.append(fmt_array("GELU_LUT_Q313", "short", glut))
    t.append("")
    t.append("#endif // GELU_LUTS_H")
    t.append("")
    with open(path, "w", newline="\n") as f:
        f.write("\n".join(t))
    print("wrote %s" % path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="gelu_luts.h")
    ap.add_argument("--no-check", action="store_true")
    args = ap.parse_args()

    glut = build_glut()
    print("S_G = %g  S_U = %g  S_H = %g" % (S_G, S_U, S_H))
    print("MAGIC_RECIP = 0x%08X   MAGIC_RSQRT = 0x%08X" % (MAGIC_RECIP, MAGIC_RSQRT))
    emit_header(args.out, glut)

    if not args.no_check:
        print("\n--- bit-accurate model verification ---")
        ok = check(glut)
        print("")
        ok = check_fastinv() and ok
        print("RESULT: %s" % ("PASS" if ok else "FAIL"))
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
