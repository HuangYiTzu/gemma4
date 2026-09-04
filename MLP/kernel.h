/*******************************************************************************
** kernel.h -- shared configuration for the Gemma-4-E2B MLP HLS accelerator
**             and for the host / testbench code.
**
** Model  : google/gemma-4-E2B                (hidden 1536, intermediate 12288)
** Weights: google/gemma-4-E2B-it-qat-mobile-transformers  (INT2 QAT)
** Board  : Xilinx Alveo U280 (3 SLRs, 32 HBM pseudo-channels)
** Stage  : decode (batch = 1 token)  ->  the MLP is three GEMV operations
**
**     gate = W_gate . x     (12288 x 1536)
**     up   = W_up   . x     (12288 x 1536)
**     h    = GELU(gate) * up                         -> INT8 @ S_H
**     y    = W_down . h     (1536 x 12288)
**     out  = post_feedforward_layernorm(y) + residual
**
** Only THREE quantization points exist, all of them already in the QAT
** checkpoint (see gen_luts.py):
**     gate_proj.output_activation_scale  -> r_g[i] requant  -> q_g  (INT8)
**     up_proj.output_activation_scale    -> r_u[i] requant  -> q_u  (INT8)
**     down_proj.input_activation_scale   -> folded into the G' LUT
** h therefore needs NO extra quantization step: it is produced directly at
** the down_proj input scale.
**
** ---------------------------------------------------------------------------
** Spatial mapping (512 x 512 macro tiling, 3 SLRs share all tiles evenly)
** ---------------------------------------------------------------------------
**   W_gate / W_up : the two matrices are interleaved column-wise into ONE
**                   fused matrix of N = 2*F = 24576 columns (even = gate,
**                   odd = up), split into 48 macro tiles of DN = 512 columns.
**                   SLR(t) = t / 16   -> 16 tiles per SLR
**                   SLR s therefore owns fused columns [8192s, 8192s+8192)
**                   and produces h[4096s : 4096s+4096].
**
**   W_down        : split along K.  SLR(n_tile, k_tile) = k_tile / 8, i.e.
**                   SLR s owns W_down[:, 4096s : 4096s+4096] (1.5 MB), which
**                   is exactly aligned with the h slice that SLR s produced.
**                   Each SLR computes a full 1536-wide INT32 partial sum; the
**                   three partial sums are added across the SLRs at the end.
**
**   HBM           : one data mover in SLR0 owns all 32 AXI ports (local, zero
**                   SLL), and forwards macro tiles to the owning SLR as wide
**                   streams:
**                     SLR0/SLR1 boundary : 512 B/cy (SLR1) + 512 B/cy (SLR2)
**                                        = 1024 B/cy = 8192 bit  (~8500 SLL)
**                     SLR1/SLR2 boundary : 512 B/cy (~4300 SLL)
**                   both well inside the 17280 SLL/boundary budget of the U280.
**
** ---------------------------------------------------------------------------
** Compute array (per SLR): 128 N-lanes x 32 K-lanes = 4096 MAC/cycle
** ---------------------------------------------------------------------------
** INT2 x INT8 products are packed two-per-DSP48E2: the 27-bit A port carries
** two INT2 weights of two DIFFERENT output columns at bit offsets 0 and
** MLP_PACK_SHIFT, the 18-bit B port carries the shared INT8 activation, and
** the 48-bit P accumulator holds both partial sums in separate fields.  So
** 4096 MAC/cycle costs 2048 DSPs per SLR.  The field pitch of 18 bits also
** covers INT4 x INT8 (16 accumulations of |w*q| <= 8*128 need 15 bits), so
** the very same array is reused when other layers move to INT4.
*******************************************************************************/

#ifndef MLP_KERNEL_H
#define MLP_KERNEL_H

// ---------------------------------------------------------------------------
// Model geometry
// ---------------------------------------------------------------------------
#define MLP_K            1536        // hidden_size
#define MLP_F            12288       // intermediate_size
#define MLP_N_FUSED      (2 * MLP_F) // 24576 fused gate/up columns
#define MLP_WBITS        2           // INT2 weights
#define MLP_WPB          (8 / MLP_WBITS)          // 4 weights per byte
#define MLP_BPC          (MLP_K / MLP_WPB)        // 384 B per fused column
#define MLP_BPC_FULL     (MLP_F / MLP_WPB)        // 3072 B per down column

// ---------------------------------------------------------------------------
// Macro tiling / SLR allocation
// ---------------------------------------------------------------------------
#define MLP_NSLR         3
#define MLP_DN           512         // macro tile width (columns)
#define MLP_NPC          32          // HBM pseudo-channels owned by the mover
#define MLP_CPP          (MLP_DN / MLP_NPC)       // 16 gate/up columns per PC
#define MLP_CPP_D        4                        // 4 down rows per PC

#define MLP_TILES_GU     (MLP_N_FUSED / MLP_DN)   // 48 fused tiles
#define MLP_TILES_SLR    (MLP_TILES_GU / MLP_NSLR)// 16 tiles per SLR
#define MLP_F_SLR        (MLP_F / MLP_NSLR)       // 4096 h values per SLR
#define MLP_N_DOWN       MLP_K                    // down_proj output dim 1536
#define MLP_TILES_DN     (MLP_N_DOWN / MLP_DN)    // 3 down n_tiles

// ---------------------------------------------------------------------------
// Compute array shape
// ---------------------------------------------------------------------------
#define MLP_NLANE        128         // output columns computed in parallel
#define MLP_KLANE        32          // K terms computed in parallel
#define MLP_NPASS        (MLP_DN / MLP_NLANE)     // 4 passes per macro tile
#define MLP_KSTEP        (MLP_DN / MLP_KLANE)     // 16 steps per K macro tile
#define MLP_KTILE_GU     (MLP_K / MLP_DN)         // 3   (K = 1536)
#define MLP_KTILE_DN     ((MLP_F / MLP_NSLR) / MLP_DN)  // 8 (K slice = 4096)

// DSP48E2 packing: two output columns share one DSP, fields 18 bits apart
#define MLP_PACK_SHIFT   18
#define MLP_PACK_WAYS    2

// ---------------------------------------------------------------------------
// HBM byte layout produced by the offline repacker (see tb / host)
//   channel p :  [ gate/up stream        ]  48 tiles x 6144 B = 288 KB
//                [ down stream, SLR 0..2 ]   3 x 49152 B      = 144 KB
// ---------------------------------------------------------------------------
#define MLP_GU_TILE_BYTES (MLP_CPP * MLP_BPC)             // 6144
#define MLP_GU_CH_BYTES   (MLP_TILES_GU * MLP_GU_TILE_BYTES)   // 294912
#define MLP_DN_BLK_BYTES  (MLP_CPP_D * (MLP_F / MLP_NSLR / MLP_WPB)) // 4*1024
#define MLP_DN_SLR_BYTES  (MLP_TILES_DN * MLP_NPASS * MLP_DN_BLK_BYTES) // 49152
#define MLP_DN_CH_BYTES   (MLP_NSLR * MLP_DN_SLR_BYTES)   // 147456
#define MLP_CH_BYTES      (MLP_GU_CH_BYTES + MLP_DN_CH_BYTES)  // 442368

#define MLP_AXI_BYTES     32                              // 256-bit HBM port
#define MLP_CH_WORDS      (MLP_CH_BYTES / MLP_AXI_BYTES)  // 13824 beats

// One data-mover beat = 16 pseudo-channels = 512 B, so a macro tile crosses to
// its SLR at 512 B/cycle -- exactly the per-SLR crossing width assumed in the
// SLL budget above, and within the 4096-bit ceiling Vitis HLS puts on an
// aggregated stream element.  The mover reads all 32 channels per address and
// emits the low and the high half back to back.
#define MLP_BEAT_PC       16
#define MLP_BEAT_HALF     (MLP_NPC / MLP_BEAT_PC)         // 2 beats per address

// offline bias-to-two's-complement mask applied by the repacker
#define MLP_XOR_MASK      0xAA

// ---------------------------------------------------------------------------
// Scale/table fixed-point formats live in gelu_luts.h (auto-generated):
//   MLP_RQ_SHIFT (20), MLP_GLUT_FRAC (13), MLP_CD_SHIFT (24),
//   MLP_LN_FRAC (13), MLP_HID_FRAC (8), MLP_LN_EPS, GELU_LUT_Q313[256]
// ---------------------------------------------------------------------------
#include "gelu_luts.h"

#define MLP_HID_SCALE  (1 << MLP_HID_FRAC)        // 256.0 : Q7.8 hidden states

// ---------------------------------------------------------------------------
// Consistency checks on the geometry above.  These are the invariants the HBM
// layout and the address arithmetic in MLP.cpp rely on; if a dimension is ever
// retuned, a broken combination fails to compile instead of silently producing
// wrong addresses.  (C++98-compatible: no static_assert in the host build.)
// ---------------------------------------------------------------------------
typedef char mlp_geom_assert[
	(MLP_N_FUSED == MLP_TILES_GU * MLP_DN) &&
	(MLP_TILES_GU == MLP_NSLR * MLP_TILES_SLR) &&
	(MLP_DN == MLP_NPC * MLP_CPP) &&
	(MLP_NLANE == MLP_NPC * MLP_CPP_D) &&
	(MLP_F == MLP_NSLR * MLP_F_SLR) &&
	(MLP_F_SLR == MLP_TILES_SLR * (MLP_DN / 2)) &&
	(MLP_NLANE % MLP_PACK_WAYS == 0) &&
	(MLP_BPC % MLP_AXI_BYTES == 0) &&
	(MLP_CH_BYTES % MLP_AXI_BYTES == 0) &&
	(MLP_NPC == MLP_BEAT_PC * MLP_BEAT_HALF) ? 1 : -1];

#ifndef MLP_HOST_ONLY
// ===========================================================================
// HLS-only section
// ===========================================================================
#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_stream.h>

typedef ap_uint<8 * MLP_AXI_BYTES>  hbm_word_t;   // one 256-bit HBM beat
typedef ap_int<MLP_WBITS>           w_t;          // INT2 weight
typedef ap_int<8>                   act_t;        // INT8 activation
typedef ap_int<32>                  acc_t;        // INT32 accumulator
typedef ap_int<48>                  dsp_t;        // DSP48E2 P register
typedef ap_int<27>                  dsp_a_t;      // DSP48E2 A port
typedef ap_int<18>                  dsp_b_t;      // DSP48E2 B port
typedef ap_int<32>                  raw32_t;      // scalars on the AXI bus

// ---- epilogue fixed-point formats -----------------------------------------
// y_pre = psum * c_down  (the single dequantization of the down_proj output)
typedef ap_fixed<32, 16, AP_RND, AP_SAT>  norm_t;
// sum of squares for the RMSNorm mean
typedef ap_ufixed<64, 42>                 ssq_t;
// mean square (+ eps) fed to the fast inverse-square-root unit
typedef ap_ufixed<56, 32, AP_RND>         ms_t;
// 1/sqrt(mean square) in [2^-16, 2^10]
typedef ap_ufixed<32, 12>                 rsq_t;
// (1 + gamma) post_feedforward_layernorm weight, Q3.13
typedef ap_fixed<16, 3>                   lnw_t;
// hidden state / residual / output, Q7.8
typedef ap_fixed<16, 8, AP_RND, AP_SAT>   hid_t;

// One data-mover beat carries MLP_BEAT_PC pseudo-channels (see above).
typedef struct { hbm_word_t w[MLP_BEAT_PC]; } mover_beat_t;

// ---------------------------------------------------------------------------
// Top-level kernel.  w_hbm_0 .. w_hbm_31 are the 32 HBM pseudo-channels owned
// by the SLR0 data mover; every small vector shares one narrow AXI bundle.
// ---------------------------------------------------------------------------
// the low / high halves are separate so the data mover can build one 512 B
// beat from each without a runtime-variable channel index
#define MLP_HBM_PORTS_LO(D) \
	D( 0) D( 1) D( 2) D( 3) D( 4) D( 5) D( 6) D( 7) \
	D( 8) D( 9) D(10) D(11) D(12) D(13) D(14) D(15)
#define MLP_HBM_PORTS_HI(D) \
	D(16) D(17) D(18) D(19) D(20) D(21) D(22) D(23) \
	D(24) D(25) D(26) D(27) D(28) D(29) D(30) D(31)
#define MLP_HBM_PORTS(D)  MLP_HBM_PORTS_LO(D) MLP_HBM_PORTS_HI(D)

#define MLP_HBM_ARG(i) const hbm_word_t *w_hbm_##i,

extern "C" void mlp(
	MLP_HBM_PORTS(MLP_HBM_ARG)
	const raw32_t *x_q,       // [1536] INT8 MLP input (gate/up activation)
	const raw32_t *r_g,       // [12288] gate requant multiplier, raw * 2^-20
	const raw32_t *r_u,       // [12288] up   requant multiplier, raw * 2^-20
	const raw32_t *c_down,    // [1536] down dequant scale,       raw * 2^-24
	const raw32_t *ln_gamma,  // [1536] (1+gamma) layernorm weight, raw * 2^-13
	const raw32_t *resid,     // [1536] residual hidden state, Q7.8
	raw32_t *y_out);          // [1536] MLP output hidden state, Q7.8

#endif // !MLP_HOST_ONLY

#endif // MLP_KERNEL_H
