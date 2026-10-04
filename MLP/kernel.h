/*******************************************************************************
** kernel.h -- shared configuration for the Gemma-4-E2B MLP HLS accelerator
**             and for the host / testbench code.
**
** Model  : google/gemma-4-E2B                (hidden 1536, intermediate 12288)
** Weights: google/gemma-4-E2B-it-qat-mobile-transformers  (INT2 QAT), layer 15,
**          read from the bin_packing_mlp.py export (mlp_bin_output/layer_15,
**          format documented at the top of mlp_model.h)
** Board  : Xilinx Alveo U280 (3 SLRs, 32 HBM pseudo-channels)
** Stage  : decode (batch = 1 token)  ->  the MLP is three GEMV operations
**
**     gate = W_gate . x     (12288 x 1536)
**     up   = W_up   . x     (12288 x 1536)
**     h    = GELU(gate) * up                         -> INT8  @ S_H
**     y    = W_down . h     (1536 x 12288)           -> INT16 @ S_Y16
**
** The kernel ends at down_proj.  post_feedforward_layernorm and the residual
** add are NOT in this kernel: the downstream RMSNorm block receives y_q and
** recovers the real value as y_q * S_Y16.
**
** FOUR quantization points, all of them already in the QAT checkpoint (see
** gen_luts.py):
**     gate_proj.output_activation_scale  -> r_g[i] requant    -> q_g (INT8)
**     up_proj.output_activation_scale    -> r_u[i] requant    -> q_u (INT8)
**     down_proj.input_activation_scale   -> folded into the G' LUT
**     down_proj.output_activation_scale  -> c_down[j] requant -> y_q (INT16)
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
**   HBM           : FOUR data movers in SLR0 share the 32 AXI ports, 8 each
**                   (local, zero SLL), and forward macro tiles to the owning
**                   SLR as wide streams.  SLR0 and SLR2 are NOT adjacent, so
**                   everything bound for SLR2 ALSO crosses the SLR0/SLR1
**                   boundary:
**                     SLR0/SLR1 :  8 lanes (engine 2 + engine 3) + 2 ctrl + 2 psum
**                     SLR1/SLR2 :  4 lanes (engine 3)            + 1 ctrl + 1 psum
**                   and every wire of a crossing lane costs one SLL.  A lane
**                   is 512 bit of payload but 514 PHYSICAL WIRES (+ TVALID,
**                   + TREADY), so the busy boundary needs
**                     10 x 514 + 2 x 34 = 5208 SLL
**                   against 23040 SLL per boundary on the U280.  (With the
**                   8 lanes per engine of builds 1..8 it was 18 x 514 +
**                   2 x 34 = 9320 -- see FOUR LANES PER ENGINE below.)
**
**                   WHY FOUR MOVERS (MLP_NMV).  The boundary TOTAL was never
**                   the problem: the failing build sat at 83 % / 87 % of the
**                   SLL budget.  What killed it was the DISTRIBUTION.  SLLs
**                   are handed out in 16 independent COLUMNS of 1440, and a
**                   crossing net can only use the column above where its two
**                   endpoints were placed.  With ONE mover every crossing net
**                   started from the same block, so route_design measured
**                     SLR[1-2]  ... 139% 155% 207% 202% 198%   (columns 11..15)
**                   against 14..71 % on columns 0..10, and gave up with
**                     ERROR: [VPL 35-3] Design is not routable as its global
**                                       congestion level is 7
**                   Four CUs, each anchored to its own contiguous block of 8
**                   HBM banks, start their crossings from four points spread
**                   across the die width, so the demand lands on four groups
**                   of columns instead of one.  mlp_link.cfg pins that down
**                   with a pblock per mover (PLACE_DESIGN.TCL.PRE).
**
**                   The split itself cost NOTHING in throughput: with 8 lanes
**                   a mover still issued 3 HBM reads per port (one address
**                   per SLR) and MLP_WSUB = 2 writes per lane, so mv_beat kept
**                   II=3.  (The lane count is a separate decision, see FOUR
**                   LANES PER ENGINE below.)  It also needs no change to the
**                   HBM byte layout -- only the lane-to-channel index formula
**                   in mlp_common.h, which is a pure relabeling (see send_addr
**                   there).
**
**                   NOTE: the wire count used to read "1024 B/cy = 8192 bit
**                   (~8500 SLL)", which counted TDATA ONLY.  With ap_axiu the
**                   lanes also carried TKEEP (64 wires), TSTRB (64) and TLAST
**                   (1) -- 643 wires each -- and route_design refused the
**                   design with
**                     ERROR: [VPL 35-5] Design is not routable as its vertical
**                                       wire utilization is 82.49 %
**                   The lanes are bare AXI4-Stream now (see wstrm_t below).
**
**                   FOUR LANES PER ENGINE (MLP_WSTREAMS 8 -> 4, after the
**                   eighth Hardware build).  Builds 7 and 8 failed route_design
**                   by a hair (router iteration 1 got down to 15-17 overlaps,
**                   then iteration 2 blew up; 1,045 left).  The reports put
**                   most of the demand on the weight lanes: they were ~90 %
**                   of the kernel's SLR0 -> SLR1 crossing, SLL column 7 of
**                   that boundary was at 162 %, and all 24 lanes (12,288 bit)
**                   run vertically through SLR0 from the movers, where the
**                   router's initial estimate had "South Long" congestion at
**                   level 7 on 33 % of the tiles.  Halving the lanes gives
**                     SLR0/SLR1  9320 -> 5208 SLL      SLR1/SLR2  4660 -> 2604
**                     lane wires running up through SLR0  12,288 -> 6,144
**                   and halves the lane input of every engine's load loop.
**                   Cost: 256 B/cycle per engine instead of 512, so a mover
**                   needs MLP_WSUB = 4 writes per lane per address and mv_beat
**                   runs at II = 4 instead of 3 (the HBM ports idle 1/4 of the
**                   time), and the engine's load loop at II = 4 instead of 2.
**                   Load and compute do not overlap inside the engine, so a
**                   token takes 4,608 addresses x (1 .. 2) cycles = +4.6 k ..
**                   +9.2 k cycles more.  The HBM layout, the host and the
**                   weight files are untouched.  To go back, set MLP_WSTREAMS
**                   to 8 and restore the port lists named at
**                   mlp_lane_arity_assert below.
**
** ---------------------------------------------------------------------------
** FOUR KERNELS, nine compute units -- this is what puts one MAC array in
** every SLR (a single kernel is one placement unit: v++ can only pin it to ONE
** SLR, which is why the previous monolithic "mlp" kernel had all three engines
** in SLR0).  The split is along the SLR boundary itself:
**
**   mlp_ctrl     1 CU in SLR0.  Owns the x_q / r_g / r_u pointers and feeds
**                each engine one control lane (x_q, then that engine's r_g
**                and r_u slice).  It is its OWN kernel so that the four mover
**                CUs below can be four instances of ONE identical kernel --
**                if the control path lived in the mover, CU 1 would need
**                three extra AXIS ports the other three do not have, and v++
**                has no way to express that.
**
**   mlp_mover    MLP_NMV = 4 CUs in SLR0, IDENTICAL hardware.  CU j owns HBM
**                pseudo-channels 8j .. 8j+7 (mlp_link.cfg does the mapping;
**                every CU just calls its ports w_hbm_0 .. w_hbm_7) and feeds
**                each engine MLP_WSTREAMS_MV = 1 AXI4-Stream lane of
**                MLP_AXIS_BITS = 512 bit.  The four CUs together give an
**                engine all MLP_WSTREAMS = 4 lanes = 256 B/cycle, exactly the
**                per-SLR crossing width above.  Nothing in the kernel knows
**                which j it is: the channel block and the lane numbers are
**                decided entirely by sp= / stream_connect=.
**
**   mlp_engine   3 CUs, one per SLR, IDENTICAL hardware (no SLR parameter: the
**                movers pick the addresses, so an engine only ever sees its
**                own tiles, its own h slice and its own requant tables).
**                Free-running (ap_ctrl_none): stream ports only, started by
**                data, never enqueued by the host.
**
**   mlp_collect  1 CU in SLR0.  Reads the three INT32 partial-sum streams in
**                lockstep, adds them, applies c_down and writes y_q.
**
** The CU-to-CU links are made by v++ (stream_connect in mlp_link.cfg); Vitis
** inserts the SLR-crossing register slices, so the SLL count is the stream
** width, not the AXI fabric of a 32-port mover replicated per SLR.
**
** The four movers run free of each other -- there is no common beat counter.
** They do not need one: an engine's recv_addr reads ALL FOUR lanes for every
** HBM address, so a mover that runs ahead simply fills its lane FIFO and is
** back-pressured.  The system self-aligns.  The weight links are declared 512
** deep in mlp_link.cfg.  With 8 lanes that held one WHOLE job, so the movers
** prefetched the next job while an engine computed and the load then ran
** faster than the movers (II 2 against 3).  With 4 lanes the movers and the
** load loop both move one address per 4 cycles, so a prefetched job would not
** load any faster; the FIFOs only absorb the skew between the four movers.
**
** ---------------------------------------------------------------------------
** Compute array (per SLR): 128 N-lanes x 24 K-lanes = 3072 MAC/cycle
** ---------------------------------------------------------------------------
** INT2 x INT8 products are packed two-per-DSP48E2: the 27-bit A port carries
** two INT2 weights of two DIFFERENT output columns at bit offsets 0 and
** MLP_PACK_SHIFT, the 18-bit B port carries the shared INT8 activation, and
** the 48-bit P register holds both partial sums in separate fields.  So
** 3072 MAC/cycle costs 64 x 24 = 1536 DSPs per SLR.
**
** Every DSP accumulates its own K lane over a K tile IN ITS OWN P REGISTER --
** an RTL blackbox, mlp_dsp_mac.v, because HLS C++ always routes a pipelined
** accumulator out to the fabric and back (32 LUT + 32 FF + 64 wires per DSP;
** see mlp_common.h).  At the end of the tile the 24 lanes of a column are
** pre-summed in two groups of 12 and split (mlp_common.h, job_drain).  Summing them on the DSP cascade wires instead
** was tried: written in HLS C++ it does not become a cascade (measured, see
** job_drain in mlp_engine.cpp).
**
** The array was 32 K-lanes / 2048 DSPs until the Hardware builds failed with
** routing congestion level 7 in every SLR; report_design_analysis put this
** MAC pipeline at the top of every congested window.  See MLP_KLANE_HW below.
**
** gate/up and down_proj SHARE that one array and one working set per SLR
** (MLP_JOBS below): both are the same INT2 x INT8 product and they run in
** different phases of the token, so a second copy would idle through half of
** every token.
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
#define MLP_NPASS        (MLP_DN / MLP_NLANE)     // 4 passes per macro tile
#define MLP_KTILE_GU     (MLP_K / MLP_DN)         // 3   (K = 1536)
#define MLP_KTILE_DN     ((MLP_F / MLP_NSLR) / MLP_DN)  // 8 (K slice = 4096)

// K direction.  TWO different widths, on purpose:
//   MLP_KLANE     32 INT2 weights per 64-bit working-set word.  This is the
//                 DATA FORMAT (HBM image -> jbuf) and cannot change without
//                 re-exporting the weights.
//   MLP_KLANE_HW  24 K terms the MAC array actually multiplies per cycle.
//                 The array was 32 wide (2048 DSPs); the failed Hardware
//                 builds showed it was the main source of routing congestion
//                 in every SLR, so it now takes 3/4 of a word per cycle and
//                 costs 64 x 24 = 1536 DSPs.  A small per-lane gearbox in
//                 compute_job cuts 24-weight slices out of consecutive
//                 32-weight words (4 phases; see mlp_engine.cpp).
// A 512-deep K tile therefore takes MLP_KSTEP_HW = 22 steps: 21 full ones and
// a last one with 8 live K terms (the other 16 are zeroed).
#define MLP_KLANE        32          // INT2 weights per 64-bit word (format)
#define MLP_KSTEP        (MLP_DN / MLP_KLANE)     // 16 words per K macro tile
#define MLP_KLANE_HW     24          // K terms multiplied per cycle
#define MLP_KSTEP_HW     ((MLP_DN + MLP_KLANE_HW - 1) / MLP_KLANE_HW)  // 22 steps

// DSP48E2 packing: two output columns share one DSP, fields 18 bits apart
#define MLP_PACK_SHIFT   18
#define MLP_PACK_WAYS    2

// ---------------------------------------------------------------------------
// Job list of the SHARED engine (one MAC array + one working set per SLR).
// gate/up and down_proj are the same 128 x 24 INT2 x INT8 product and run in
// different phases of the token, so one engine serves both; it walks
//   job  0 .. 15 : the SLR's 16 fused gate/up macro tiles  -> produces h
//   job 16 .. 27 : the SLR's 12 down_proj (n_tile, n_pass) blocks -> psum
// in this order, so h is complete before the first down job reads it.
// ---------------------------------------------------------------------------
#define MLP_JOBS_GU      MLP_TILES_SLR                 // 16 gate/up jobs
#define MLP_JOBS_DN      (MLP_TILES_DN * MLP_NPASS)    // 12 down jobs
#define MLP_JOBS         (MLP_JOBS_GU + MLP_JOBS_DN)   // 28 jobs per SLR

// ---------------------------------------------------------------------------
// HBM byte layout, produced offline by bin_packing_mlp.py (one file per
// region and pseudo-channel, see mlp_model.h):
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

// ---------------------------------------------------------------------------
// Kernel-to-kernel streaming (mover -> engines -> collector)
//
// One HBM address yields MLP_NPC x 32 B = 1024 B.  That is carried to the
// owning engine on MLP_WSTREAMS parallel 512-bit AXI4-Stream lanes, MLP_WSUB
// words per lane, so a lane group sustains 4 x 64 B = 256 B/cycle -- the
// per-SLR crossing width the SLL budget above is built on.
//
// LANE-TO-CHANNEL MAPPING.  Lane i, sub-word u carries the pseudo-channel pair
// of word m = i * MLP_WSUB + u, i.e. channels 2m and 2m + 1, so
//
//     lane i owns pseudo-channels 8i .. 8i + 7   (CONTIGUOUS)
//
// The old formula was m = u * MLP_WSTREAMS + i, which (with 8 lanes) gave
// lane i the SCATTERED set {2i, 2i+1, 2i+16, 2i+17}.  Both formulas produce
// every m in 0 .. MLP_ADDR_WORDS-1 exactly once and in the same per-lane
// order, so this is a pure relabeling of the LINK -- the HBM byte layout,
// bin_packing_mlp.py and the exported weight files are untouched.  What the
// contiguous version buys is the four-mover split: engine lane j is exactly
// HBM[8j .. 8j+7], so mover CU j can own that block and nothing else.
// ---------------------------------------------------------------------------
#define MLP_AXIS_BITS     512
#define MLP_AXIS_BYTES    (MLP_AXIS_BITS / 8)             // 64 B per AXIS word
#define MLP_WSTREAMS      4                               // lanes an ENGINE sees
#define MLP_ADDR_WORDS    (MLP_NPC * MLP_AXI_BYTES / MLP_AXIS_BYTES)  // 16
#define MLP_WSUB          (MLP_ADDR_WORDS / MLP_WSTREAMS) // 4 words per lane
#define MLP_PC_PER_WORD   (MLP_AXIS_BYTES / MLP_AXI_BYTES)// 2 channels per word

// ---------------------------------------------------------------------------
// The mover is MLP_NMV identical CUs, split by HBM channel (see the CU map).
// CU j owns pseudo-channels [j*MLP_NPC_MV, (j+1)*MLP_NPC_MV) and drives engine
// lanes j*MLP_WSTREAMS_MV .. +MLP_WSTREAMS_MV-1 of every engine.  Inside the
// kernel the channels are numbered 0 .. MLP_NPC_MV-1 and the lanes w<s>_0 ..
// w<s>_(MLP_WSTREAMS_MV-1), so all four CUs are the same netlist.
//
// With 4 lanes a mover drives ONE lane per engine and puts all of its
// MLP_ADDR_WORDS_MV = 4 words of an address on it: MLP_WSUB = 4 writes per
// lane per address.  That, not the HBM side, now sets the pace:
//     mv_beat   II = max(3 HBM reads per port, MLP_WSUB writes per lane) = 4
//     gu_load / dn_load (engine)  II = MLP_WSUB reads per lane         = 4
// (With 8 lanes they were 3 and 2.)  HLS pragmas cannot take a #define, so
// the II= of those loops is a literal; MLP_MV_II / MLP_LD_II say what it must
// be, and an assert next to each pragma fails the build if they drift apart.
// ---------------------------------------------------------------------------
#define MLP_NMV           4                               // data-mover CUs
#define MLP_NPC_MV        (MLP_NPC / MLP_NMV)             // 8 HBM channels each
#define MLP_WSTREAMS_MV   (MLP_WSTREAMS / MLP_NMV)        // 1 lane per engine
#define MLP_ADDR_WORDS_MV (MLP_NPC_MV * MLP_AXI_BYTES / MLP_AXIS_BYTES)  // 4
#define MLP_MV_II         ((MLP_NSLR > MLP_WSUB) ? MLP_NSLR : MLP_WSUB)  // 4
#define MLP_LD_II         MLP_WSUB                        // 4

// control lane: x_q, then this engine's r_g and r_u slice, 16 INT32 per word
#define MLP_VPW           (MLP_AXIS_BYTES / 4)            // 16 values per word
#define MLP_CTRL_VALS     (MLP_K + 2 * MLP_F_SLR)         // 9728 per engine
#define MLP_CTRL_WORDS    (MLP_CTRL_VALS / MLP_VPW)       // 608 AXIS words

// The control vectors in global memory have EXACTLY the AXIS word format (16
// contiguous INT32 = one 512-bit beat), so the mover reads them as 512-bit
// words and forwards them untouched.  Reading them one INT32 at a time cost
// 16x twice over: the loop ran once per VALUE instead of once per WORD, and
// HLS could not widen the accesses, so every 32-bit element burned a full
// 64 B AXI beat (1.58 MB moved per token where 102 KB was needed).
#define MLP_XQ_WORDS      (MLP_K / MLP_VPW)               // 96  words
#define MLP_RQ_WORDS_SLR  (MLP_F_SLR / MLP_VPW)           // 256 words per engine
#define MLP_RQ_WORDS      (MLP_F / MLP_VPW)               // 768 words total

// offset-binary -> two's-complement mask.  Already applied by the exporter
// (the "magic" 170 in manifest.json): the .bin files hold INT2 two's
// complement, so neither the host nor the kernel XORs anything.
#define MLP_XOR_MASK      0xAA

// ---------------------------------------------------------------------------
// Scale/table fixed-point formats live in gelu_luts.h (auto-generated from
// the layer's activation_scales.json by gen_luts.py):
//   MLP_RQ_SHIFT (20), MLP_GLUT_FRAC (12 for layer 15, picked so that the
//   G' peak fits int16), MLP_CD_SHIFT (24), MLP_LAYER_IDX,
//   MLP_S_IN / MLP_S_G / MLP_S_U / MLP_S_H / MLP_S_Y / MLP_S_Y16,
//   GELU_LUT_I16[256]
// ---------------------------------------------------------------------------
#include "gelu_luts.h"

// ---------------------------------------------------------------------------
// Kernel output: the down_proj result as INT16 codes at S_Y16
//
//     p[j]      = ps0[j] + ps1[j] + ps2[j]                  (cross-SLR, INT32)
//     y_q[j]    = clamp_int16( rne( p[j] * c_down[j], 2^-MLP_CD_SHIFT ) )
//     c_down[j] = round( ws_down[j] * S_H / S_Y16 * 2^MLP_CD_SHIFT )   (host)
//     S_Y16     = down_proj.output_activation_scale / 256
//
// ws_down[j] is the per-channel INT2 weight scale of down_proj.  c_down must
// fit INT32, i.e. ws_down < 2^7 * S_Y16 / S_H.  |p| <= F*2*128 < 2^22, so the
// product needs 53 bits and the rounded code 30 bits before the clamp.
// ---------------------------------------------------------------------------
#define MLP_Y16_MIN    (-32768)
#define MLP_Y16_MAX    32767

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
	(MLP_ADDR_WORDS == MLP_WSTREAMS * MLP_WSUB) &&
	// the four-mover split has to divide evenly everywhere, or CU j would
	// own a channel block that does not line up with its lane(s)
	(MLP_NPC == MLP_NMV * MLP_NPC_MV) &&
	(MLP_WSTREAMS == MLP_NMV * MLP_WSTREAMS_MV) &&
	(MLP_ADDR_WORDS_MV == MLP_WSTREAMS_MV * MLP_WSUB) &&
	(MLP_NPC_MV == MLP_ADDR_WORDS_MV * MLP_PC_PER_WORD) &&
	(MLP_CTRL_VALS == MLP_CTRL_WORDS * MLP_VPW) &&
	(MLP_K == MLP_XQ_WORDS * MLP_VPW) &&
	(MLP_F == MLP_RQ_WORDS * MLP_VPW) &&
	(MLP_RQ_WORDS == MLP_NSLR * MLP_RQ_WORDS_SLR) &&
	(MLP_CTRL_WORDS == MLP_XQ_WORDS + 2 * MLP_RQ_WORDS_SLR) ? 1 : -1];

#ifndef MLP_HOST_ONLY
// ===========================================================================
// HLS-only section
// ===========================================================================
#include <ap_axi_sdata.h>
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
typedef ap_int<16>                  y16_t;        // INT16 down_proj output code

// ---------------------------------------------------------------------------
// CU-to-CU links: 512-bit weight / control lanes, 32-bit partial-sum lane.
//
// BARE streams: an hls::stream<ap_uint<N>> with "#pragma HLS INTERFACE axis"
// becomes TDATA + TVALID + TREADY and nothing else.  ap_axiu<512,0,0,0> would
// additionally carry TKEEP (one wire per BYTE -> 64), TSTRB (64) and TLAST (1),
// i.e. 643 wires per lane instead of 515.
//
// Those 128 extra wires were pure waste here: the kernels always sent every
// byte of every beat (t.keep = t.strb = -1) and never used TLAST for framing.
// On an SSI device they were not free, though -- 18 of the lanes crossed an
// SLR boundary back then, so they cost 18 x 128 = 2304 SLLs and pushed
// route_design over its vertical-wire limit (see the header comment above).
//
// Plain ap_uint also keeps the code portable between Vitis HLS 2022.1 (the
// local csim) and 2023.2 (the server): hls::axis grew extra template
// parameters between those releases, a bare ap_uint did not.
// ---------------------------------------------------------------------------
typedef ap_uint<MLP_AXIS_BITS>           axis_word_t;
typedef axis_word_t                      wstrm_t;   // weight / control lane
typedef ap_uint<32>                      pstrm_t;   // partial-sum lane

// ---------------------------------------------------------------------------
// Port-list macros.
//
//   MLP_HBM_PORTS  the MLP_NPC_MV = 8 pseudo-channels ONE mover CU owns.  The
//                  numbering is LOCAL: every CU has w_hbm_0 .. w_hbm_7 and
//                  mlp_link.cfg maps CU j's ports onto HBM[8j .. 8j+7].
//   MLP_ELANES     the MLP_WSTREAMS = 4 lanes an engine sees (w_<lane>).
//
// There is deliberately NO macro for the mover's own lanes: it drives only
// MLP_WSTREAMS_MV = 1 per engine, and three ports spelled out read better than
// a macro whose trailing comma then needs a dummy argument after it.
// ---------------------------------------------------------------------------
#define MLP_HBM_PORTS(D)  D(0) D(1) D(2) D(3) D(4) D(5) D(6) D(7)

#define MLP_HBM_ARG(i)  const hbm_word_t *w_hbm_##i,
#define MLP_HBM_PASS(i) w_hbm_##i,

#define MLP_ELANES(D)    D(0) D(1) D(2) D(3)
#define MLP_EARG(i)      hls::stream<wstrm_t> &w_##i,
#define MLP_EPASS(i)     w_##i,

// The lane count is spelled out in more places than MLP_WSTREAMS: MLP_ELANES
// above, the mover prototype below and its INTERFACE pragmas (mlp_mover.cpp),
// the engine's INTERFACE pragmas (mlp_engine.cpp), the lane lists of
// send_addr / recv_addr (mlp_common.h), run_case in tb_MLP.cpp, and the
// stream_connect lines of mlp_link.cfg.  They are all written for exactly 4
// lanes per engine (1 per mover); a retune has to update every one of them,
// and this makes sure it cannot be forgotten silently.
typedef char mlp_lane_arity_assert[
	(MLP_WSTREAMS == 4) && (MLP_WSTREAMS_MV == 1) && (MLP_WSUB == 4) &&
	(MLP_MV_II == 4) && (MLP_LD_II == 4) ? 1 : -1];

// ---------------------------------------------------------------------------
// The three kernels (see the CU map at the top of this file)
// ---------------------------------------------------------------------------

// SLR0: the control plane.  1 CU.  Reads x_q / r_g / r_u and hands every
// engine its own control block.  Split out of mlp_mover so that the four mover
// CUs stay one identical kernel (see the CU map above).
//
// x_q / r_g / r_u are the SAME host buffers as before (plain INT32 arrays);
// only the kernel's view of them is 512-bit words, which is a pure
// reinterpretation: value i lives at bits [32*(i%16)+31 : 32*(i%16)] of word
// i/16 on a little-endian bus, exactly where the AXIS control lane wants it.
// The host must keep them 64 B aligned and a whole number of words long --
// 1536 and 12288 INT32 are 96 and 768 words, so both hold exactly.
//
// Pointers first, streams last: AXI4-Stream ports consume kernel argument
// indices in the XRT ABI and must be left unset, so this keeps the host's
// clSetKernelArg indices at 0..2.
extern "C" void mlp_ctrl(
	const axis_word_t *x_q,       // arg 0: [96]  = 1536 INT8 MLP input
	const axis_word_t *r_g,       // arg 1: [768] = 12288 gate requant, 2^-20
	const axis_word_t *r_u,       // arg 2: [768] = 12288 up   requant, 2^-20
	hls::stream<wstrm_t> &ctrl0,  // x_q + the SLR's r_g / r_u slice
	hls::stream<wstrm_t> &ctrl1,
	hls::stream<wstrm_t> &ctrl2);

// SLR0: the weight path.  MLP_NMV = 4 IDENTICAL CUs, one per block of
// MLP_NPC_MV = 8 HBM pseudo-channels.  A CU reads one address per SLR out of
// its own 8 channels and pushes them onto MLP_WSTREAMS_MV = 1 lane per
// engine.  Which HBM banks and which engine lane that is is decided ONLY by
// sp= / stream_connect= in mlp_link.cfg -- the kernel has no CU index.
//
// The 8 pointers come first: they are the only arguments the host binds, so
// they keep the argument indices 0..7 the OpenCL host uses.
extern "C" void mlp_mover(
	MLP_HBM_PORTS(MLP_HBM_ARG)                              // arg 0..7
	hls::stream<wstrm_t> &w0_0,                             // -> engine 1
	hls::stream<wstrm_t> &w1_0,                             // -> engine 2
	hls::stream<wstrm_t> &w2_0);                            // -> engine 3

// one CU per SLR, free-running, identical hardware
extern "C" void mlp_engine(
	hls::stream<wstrm_t> &ctrl,
	MLP_ELANES(MLP_EARG)          // w_0 .. w_3
	hls::stream<pstrm_t> &ps);    // 1536 INT32 partial sums per token

// SLR0: cross-SLR add + c_down requant -> y_q.
// Pointers first, streams last -- AXI4-Stream ports DO consume kernel argument
// indices in the XRT ABI (and must be left unset), so putting them at the end
// keeps the host's clSetKernelArg indices at 0..1.
extern "C" void mlp_collect(
	const raw32_t *c_down,        // arg 0: [1536] down requant, raw * 2^-24
	raw32_t *y_q,                 // arg 1: [1536] output, INT16 code @ S_Y16
	hls::stream<pstrm_t> &ps0,
	hls::stream<pstrm_t> &ps1,
	hls::stream<pstrm_t> &ps2);

#endif // !MLP_HOST_ONLY

#endif // MLP_KERNEL_H
