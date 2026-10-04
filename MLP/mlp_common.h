/*******************************************************************************
** mlp_common.h -- everything the three MLP kernels share: the rounding /
**                 saturation primitives, the DSP48E2 INT2 x INT8 packing, the
**                 geometry of the per-engine working set and the AXI4-Stream
**                 pack / unpack helpers.
**
** Why a header and not a .cpp: every kernel must live in its OWN source file.
** v++ puts a copy of the kernel's source into each .xo for software emulation,
** so two kernels in one .cpp would be defined twice at link time
**   ("multiple definition of `mlp_engine'", v++ 17-1309).
** Everything here is static / inline / a typedef, so each translation unit
** gets a private copy and nothing collides.  (The one exception is the
** DECLARATION of the mlp_dsp_mac RTL blackbox, which has no definition in the
** kernel sources at all -- see "ONE DSP48E2 multiply-accumulate" below.)
**
** See kernel.h for the geometry, the CU map and the kernel prototypes.
*******************************************************************************/

#ifndef MLP_COMMON_H
#define MLP_COMMON_H

#include "kernel.h"

// ===========================================================================
// Rounding / saturation primitives (round-half-to-even, matches gen_luts.py)
// ===========================================================================
template <int SH, int W>
static ap_int<W> rne_sh(ap_int<W> v)
{
#pragma HLS INLINE
	ap_int<W>       q = v >> SH;              // arithmetic shift == floor
	ap_uint<SH>     r = (ap_uint<SH>)v;       // v - (q << SH), always >= 0
	const ap_uint<SH> half = (ap_uint<SH>)1 << (SH - 1);
	if (r > half || (r == half && q[0])) q += 1;
	return q;
}

static inline act_t clamp_i8(ap_int<32> v)
{
#pragma HLS INLINE
	if (v >  127) return (act_t) 127;
	if (v < -128) return (act_t)-128;
	return (act_t)v;
}

static inline y16_t clamp_i16(ap_int<64> v)
{
#pragma HLS INLINE
	if (v > MLP_Y16_MAX) return (y16_t)MLP_Y16_MAX;
	if (v < MLP_Y16_MIN) return (y16_t)MLP_Y16_MIN;
	return (y16_t)v;
}

// ===========================================================================
// DSP48E2 INT2 x INT8 packing
//
//   A(27b) = w_hi * 2^18 + w_lo        two output columns, one shared act.
//   B(18b) = q                         INT8 activation
//   P(48b) = sum(A*B) = 2^18 * S_hi + S_lo
//
// A field pitch of 18 bits leaves room for 16 accumulations of |w*q| <= 2*128
// (INT2, 13 bits) and of |w*q| <= 8*128 (INT4, 15 bits), so the identical
// array is reused when other layers switch to INT4 weights.
// ===========================================================================
static inline dsp_a_t dsp_pack_a(w_t w_lo, w_t w_hi)
{
#pragma HLS INLINE
	return ((dsp_a_t)w_hi << MLP_PACK_SHIFT) + (dsp_a_t)w_lo;
}

// Split a packed accumulator into its two signed fields.  The low field is
// two's complement inside its MLP_PACK_SHIFT bits, so when it is negative it
// has BORROWED one unit from the high field and both halves need correcting
// (decimal analogue at a pitch of 100: hi=3, lo=-5 is stored as 295, and a
// naive split reads 2 / 95 instead of 3 / -5).
//
// The packed format is ADDITIVE: p1 + p2 = (hi1+hi2) * 2^SHIFT + (lo1+lo2),
// so several P registers may be summed as plain 48-bit integers and split
// ONCE, as long as the accumulated low field still fits the signed
// MLP_PACK_SHIFT-bit window.  compute_job uses that to run one split per
// MLP_SPLIT_SPAN K-lanes instead of one per lane; the budget is asserted
// below (MLP_SPLIT_LO_MAX).
static inline void dsp_split(dsp_t p, acc_t &lo, acc_t &hi)
{
#pragma HLS INLINE
	ap_uint<MLP_PACK_SHIFT> f = (ap_uint<MLP_PACK_SHIFT>)p;
	bool borrow = f[MLP_PACK_SHIFT - 1];      // low field is negative
	ap_int<MLP_PACK_SHIFT + 1> fs = (ap_int<MLP_PACK_SHIFT + 1>)f;
	if (borrow) fs -= ((ap_int<MLP_PACK_SHIFT + 1>)1 << MLP_PACK_SHIFT);
	lo = (acc_t)fs;
	hi = (acc_t)((p >> MLP_PACK_SHIFT) + (borrow ? 1 : 0));
} 

// ===========================================================================
// ONE DSP48E2 multiply-accumulate that keeps its sum INSIDE the DSP
//
//   P <= (first ? 0 : P) + (w_hi * 2^18 + w_lo) * q        returns P[31:0]
//
// In hardware this is the RTL blackbox mlp_dsp_mac.v (registered with HLS by
// mlp_blackbox.tcl).  It exists because HLS cannot build an accumulator that
// stays in the DSP: written as "P += a * q" in a pipelined loop, HLS copies P
// out to a fabric register every cycle and feeds it back through a 2:1
// "pipeline bypass" mux and a "first step ? 0" mux into the DSP's C port --
// 32 LUT + 32 FF + 64 wires per DSP, 1536 DSPs per engine, all switched by
// one 49,152-load control net.  That was the main cause of the routing
// conflicts in the third Hardware build (route_status / high_fanout.rpt).
//
// The blackbox is stateful (the sum lives in its P register), so it cannot be
// modelled by a function of its arguments.  mlp_engine therefore calls it
// through MLP_MAC():
//   __SYNTHESIS__ defined  (csynth, hw, hw_emu RTL) -> mlp_dsp_mac(), the RTL
//   otherwise (csim, sw_emu, the g++ testbench)    -> mlp_dsp_mac_model(),
//                          which carries the accumulator in an explicit
//                          per-DSP state variable owned by the caller.
// Both return the same P[31:0]; the state argument disappears from the
// synthesised call, so HLS never sees it.
//
// CALLER'S RULE (see mlp_dsp_mac.v): the calls of one K tile must come on
// consecutive clocks -- a pipelined loop at II = 1 with no stall.
//
// 32 bits are enough: one K tile adds MLP_KSTEP_HW products of
// |w * q| <= MLP_WABS_MAX * MLP_QABS_MAX into each packed field, so
// |P| <= MLP_P_ABS_MAX < 2^31 (asserted with the job geometry below).
// ===========================================================================
typedef ap_int<32> dsp_out_t;                // P[31:0] of one DSP

#define MLP_P_ABS_MAX ((long long)MLP_KSTEP_HW * MLP_WABS_MAX * MLP_QABS_MAX * \
                       ((1LL << MLP_PACK_SHIFT) + 1))

// the RTL blackbox (mlp_dsp_mac.v); its C placeholder is mlp_dsp_mac.cpp
ap_int<32> mlp_dsp_mac(ap_int<2> w_lo, ap_int<2> w_hi, ap_int<8> q,
                       ap_uint<1> first);

// bit-exact software model of the same DSP; st is that DSP's P register
static inline dsp_out_t mlp_dsp_mac_model(dsp_t &st, w_t w_lo, w_t w_hi,
                                          act_t q, bool first)
{
	const ap_int<27> a = ((ap_int<27>)w_hi << MLP_PACK_SHIFT) + (ap_int<27>)w_lo;
	const dsp_t      m = (dsp_t)(a * (dsp_b_t)q);
	st = first ? m : (dsp_t)(st + m);
	return (dsp_out_t)st;
}

#ifdef __SYNTHESIS__
#define MLP_MAC(st, w_lo, w_hi, q, first) \
	mlp_dsp_mac((w_lo), (w_hi), (q), (ap_uint<1>)(first))
#else
#define MLP_MAC(st, w_lo, w_hi, q, first) \
	mlp_dsp_mac_model((st), (w_lo), (w_hi), (q), (first))
#endif

// INT2 weight kl of a 64-bit k-major word (32 weights, k increasing)
static inline w_t w_at(ap_uint<64> word, int kl)
{
#pragma HLS INLINE
	return (w_t)(ap_uint<2>)word.range(2 * kl + 1, 2 * kl);
}

// ===========================================================================
// Geometry of the shared working set (one per engine)
//
// A job always fills the same buffer: 128 lanes x MLP_JOB_WORDS 64-bit words.
// gate/up uses all 192 slots (4 passes x 48 words of a 1536-deep K), down uses
// the first 128 (one pass over the 4096-deep K slice).
// ===========================================================================
#define MLP_GU_WORDS  (MLP_BPC / 8)                 // 48 x 64-bit words per column
#define MLP_WPB_AXI   (MLP_AXI_BYTES / 8)           // 4 x 64-bit words per HBM beat
#define MLP_GU_BPCOL  (MLP_BPC / MLP_AXI_BYTES)     // 12 HBM beats per gate/up column
#define MLP_DN_WORDS  ((MLP_F / MLP_NSLR) / 32)     // 128 x 64-bit words per down row
#define MLP_DN_BPROW  (MLP_DN_WORDS / MLP_WPB_AXI)  // 32 HBM beats per down row
#define MLP_PAIRS     (MLP_NLANE / MLP_PACK_WAYS)   // 64 (gate, up) pairs
#define MLP_DSPS      (MLP_NLANE / MLP_PACK_WAYS)   // 64 DSP columns x MLP_KLANE_HW

// ---------------------------------------------------------------------------
// Lane relabeling of the gate/up load (gu_store_pc in mlp_engine.cpp)
//
// One HBM address delivers MLP_NPC = 32 pseudo-channels IN THE SAME CYCLE, and
// they all have to land in 32 DIFFERENT lane memories or the write port of a
// lane serializes them.  The natural mapping lane = (p*16 + j) % 128 fails
// that: p, p+8, p+16 and p+24 share a lane, so a beat wrote 4 x 4 = 16 words
// into one memory (II = 8 measured).  The fix is to change which columns make
// up a "pass": instead of a contiguous 128-column half of the macro tile, pass
// np takes MLP_GU_GRP consecutive columns out of EVERY pseudo-channel, so for
// a fixed column index j the lane depends on p alone and the 32 writes go to
// 32 memories (4 words each -> 2 cycles on a true dual-port RAM, II = 2;
// since the link went to 4 lanes the load loop runs at II = 4 anyway, but a
// shared lane memory would still serialize it to II = 8).
//
//     np = j / MLP_GU_GRP                lane = p * MLP_GU_GRP + j % MLP_GU_GRP
//
// This is a relabeling of the BUFFER only; the HBM layout is untouched.  Lane
// nl of pass np therefore carries fused column
//     lc = (nl / MLP_GU_GRP) * MLP_CPP + MLP_GU_GRP * np + nl % MLP_GU_GRP
// of the tile, which keeps lanes (2d, 2d+1) an (even, odd) = (gate, up) pair
// for the DSP packing; compute_job undoes the relabeling in one index
// expression built from MLP_GU_HG (see the epilogue there).
// ---------------------------------------------------------------------------
#define MLP_GU_GRP    (MLP_CPP / MLP_NPASS)         // 4 columns per pass per PC
#define MLP_GU_HG     (MLP_GU_GRP / MLP_PACK_WAYS)  // 2 DSP pairs per group

// ---------------------------------------------------------------------------
// Half-grouping of the field split (job_drain in mlp_engine.cpp)
//
// Splitting every one of the MLP_DSPS x MLP_KLANE_HW = 1536 accumulators would
// cost one borrow correction each, and that logic -- not the DSPs -- is what
// made the MAC module big.  Because the packed format is additive, the K-lanes
// can be pre-summed as raw 48-bit values and split afterwards, which divides
// the number of corrections by MLP_SPLIT_SPAN.
//
// The only limit is the low field.  One P holds MLP_KSTEP_HW products of
// |w * q| <= MLP_WABS_MAX * MLP_QABS_MAX, so pre-summing MLP_SPLIT_SPAN of
// them reaches MLP_SPLIT_LO_MAX, which must stay inside the SIGNED
// MLP_PACK_SHIFT-bit window (2^17 = 131072 here):
//
//   span 24 (all lanes) : 24*22*2*128 = 135168  -- too big
//   span 12 (half)      : 12*22*2*128 =  67584  -- about half the window <= used
//
// The assert below is the real guard: switching the layer to INT4 weights
// (MLP_WBITS 4 -> MLP_WABS_MAX 8) makes a span of 12 overflow and FAILS THE
// BUILD instead of silently corrupting the high field.
// ---------------------------------------------------------------------------
#define MLP_WABS_MAX     (1 << (MLP_WBITS - 1))      // 2: INT2 spans -2 .. +1
#define MLP_QABS_MAX     128                         // INT8 activation
#define MLP_SPLIT_SPAN   (MLP_KLANE_HW / 2)          // 12 K-lanes per split
#define MLP_SPLIT_GROUPS (MLP_KLANE_HW / MLP_SPLIT_SPAN) // 2 splits per DSP column
#define MLP_SPLIT_LO_MAX ((long)MLP_SPLIT_SPAN * MLP_KSTEP_HW *                        MLP_WABS_MAX * MLP_QABS_MAX)

// ---------------------------------------------------------------------------
// The 24-of-32 gearbox (compute_job in mlp_engine.cpp)
//
// Step s of a K tile multiplies K terms [24s, 24s + 24).  In 32-weight words
// that start moves by 24 = -8 (mod 32) per step, so it cycles through four
// phases and every step's slice lies inside two consecutive words:
//
//   s % 4 = 0 : word 3m     [ 0..23]
//   s % 4 = 1 : word 3m     [24..31] + word 3m+1 [ 0..15]
//   s % 4 = 2 : word 3m+1   [16..31] + word 3m+2 [ 0.. 7]
//   s % 4 = 3 : word 3m+2   [ 8..31]
//
// Each lane therefore reads ONE word per step (the last word the step
// touches, MLP_GEAR_WORD) and keeps the previous step's word in a register;
// the slice is cut out of {current, previous} with a 4-way choice.  The
// activations use the same four phases: the 24 operands start at bank
// (24s mod 32) of the 32-way banked xq / h, i.e. at bank 0, 24, 16 or 8.
// ---------------------------------------------------------------------------
#define MLP_QBANKS       32                          // xq / h cyclic banking
#define MLP_GEAR_PHASES  4                           // 32 / gcd(24, 32)
#define MLP_XQ_ROWS      (MLP_K / MLP_QBANKS)        // 48  rows of 32 in xq
#define MLP_H_ROWS       (MLP_F_SLR / MLP_QBANKS)    // 128 rows of 32 in h
// the word (0 .. MLP_KSTEP-1) holding the LAST K term step s touches
#define MLP_GEAR_WORD(s) (((s) * MLP_KLANE_HW + MLP_KLANE_HW - 1) / MLP_KLANE)

#define MLP_GU_BEATS  (MLP_GU_TILE_BYTES / MLP_AXI_BYTES)   // 192 addresses / tile
#define MLP_DN_BEATS  (MLP_DN_BLK_BYTES  / MLP_AXI_BYTES)   // 128 addresses / block

#define MLP_JOB_WORDS (MLP_NPASS * MLP_GU_WORDS)    // 192  (down needs 128)

typedef ap_uint<64> wbuf_t;

// the ONE working set of an engine: 128 lanes x 192 words = 192 KB.
// Bound to a TRUE dual-port RAM (RAM_T2P: both ports read OR write) in
// mlp_engine.cpp, which is what turns the 4 word writes a beat leaves in a
// lane into 2 cycles instead of 4.  (With the 4-lane link the load loop is at
// II = 4 regardless, so the RAM is no longer what limits it.)
typedef wbuf_t job_buf_t[MLP_NLANE][MLP_JOB_WORDS];

typedef char mlp_job_assert[
	(MLP_JOB_WORDS >= MLP_DN_WORDS) &&
	(MLP_NPC * MLP_GU_GRP == MLP_NLANE) &&
	(MLP_CPP == MLP_NPASS * MLP_GU_GRP) &&
	(MLP_GU_GRP % MLP_PACK_WAYS == 0) &&
	(MLP_KTILE_GU * MLP_KSTEP == MLP_GU_WORDS) &&
	(MLP_KTILE_DN * MLP_KSTEP == MLP_DN_WORDS) &&
	(MLP_JOBS == MLP_JOBS_GU + MLP_JOBS_DN) &&
	(MLP_GU_BEATS == MLP_CPP * MLP_GU_BPCOL) &&
	(MLP_DN_BEATS == MLP_CPP_D * MLP_DN_BPROW) &&
	(MLP_KLANE_HW == MLP_SPLIT_GROUPS * MLP_SPLIT_SPAN) &&
	(MLP_SPLIT_LO_MAX < (1L << (MLP_PACK_SHIFT - 1))) &&
	// one K tile's P must fit the 32 bits mlp_dsp_mac hands out
	(MLP_P_ABS_MAX < (1LL << 31)) &&
	// the gearbox in compute_job is written for exactly 24 of 32: four
	// phases, slices aligned to 8 weights, never more than two words
	(4 * MLP_KLANE_HW == 3 * MLP_KLANE) &&
	(MLP_KLANE == MLP_QBANKS) &&
	(MLP_KSTEP_HW * MLP_KLANE_HW >= MLP_DN) &&
	((MLP_KSTEP_HW - 1) * MLP_KLANE_HW < MLP_DN) &&
	(MLP_DN % MLP_QBANKS == 0) ? 1 : -1];

// ===========================================================================
// AXIS helpers
//
// The lanes are BARE AXI4-Stream now (TDATA/TVALID/TREADY only, see kernel.h),
// so a put is just a write -- there is no TKEEP / TSTRB / TLAST left to fill
// in.  Nothing is lost: every beat always carried all 64 bytes and TLAST was
// never read anywhere.  What is gained is 128 fewer physical wires per lane.
// ===========================================================================
static inline void axis_put(hls::stream<wstrm_t> &s, axis_word_t d)
{
#pragma HLS INLINE
	s.write(d);
}

static inline void ps_put(hls::stream<pstrm_t> &s, acc_t v)
{
#pragma HLS INLINE
	s.write((pstrm_t)v);
}

// ---------------------------------------------------------------------------
// The link, both ends.  A word index m carries pseudo-channels 2m and 2m + 1,
// and BOTH sides place word m on lane i, sub-word u with the SAME formula
//
//     m = i * MLP_WSUB + u
//
// so lane i owns the contiguous channels 8i .. 8i + 7 (kernel.h explains why
// that matters for the four-mover split).  The loop nest stays "for u { for
// lane }" on both sides, so a lane still sees its MLP_WSUB = 4 words in the
// order m = 4i, 4i + 1, 4i + 2, 4i + 3, and the reconstructed ch[] is
// bit-identical to what the single 32-channel mover used to produce -- with 8
// lanes or with 4.
//
// send_addr runs inside ONE mover CU, so it only ever sees that CU's
// MLP_NPC_MV channels and MLP_WSTREAMS_MV lanes, numbered from zero.  Because
// the formula is the same shape at both scales, CU j's local word m maps onto
// global word MLP_ADDR_WORDS_MV * j + m, i.e. onto exactly the channels the
// link assigns to engine lanes MLP_WSTREAMS_MV * j + i.  recv_addr in the
// engine gathers all four CUs' lanes and needs no notion of which CU sent
// what.
// ---------------------------------------------------------------------------

// one mover CU's slice of an HBM address (MLP_NPC_MV x 256 bit)
//   -> MLP_WSTREAMS_MV = 1 lane x MLP_WSUB = 4 words
// (the lane list below is written for exactly one lane per engine --
// mlp_lane_arity_assert in kernel.h guards it)
static inline void send_addr(const hbm_word_t ch[MLP_NPC_MV],
                             hls::stream<wstrm_t> &l0)
{
#pragma HLS INLINE
	hls::stream<wstrm_t> *lane[MLP_WSTREAMS_MV] = { &l0 };
	for (int u = 0; u < MLP_WSUB; ++u) {
#pragma HLS UNROLL
		for (int i = 0; i < MLP_WSTREAMS_MV; ++i) {
#pragma HLS UNROLL
			const int m = i * MLP_WSUB + u;   // word 0..MLP_ADDR_WORDS_MV-1
			axis_word_t d;
			d.range(8 * MLP_AXI_BYTES - 1, 0)                  = ch[2 * m];
			d.range(MLP_AXIS_BITS - 1, 8 * MLP_AXI_BYTES)      = ch[2 * m + 1];
			axis_put(*lane[i], d);
		}
	}
}

// ... and back: the engine's MLP_WSTREAMS lanes -> one whole HBM address
static inline void recv_addr(MLP_ELANES(MLP_EARG) hbm_word_t ch[MLP_NPC])
{
#pragma HLS INLINE
	hls::stream<wstrm_t> *lane[MLP_WSTREAMS] = { &w_0, &w_1, &w_2, &w_3 };
	for (int u = 0; u < MLP_WSUB; ++u) {
#pragma HLS UNROLL
		for (int i = 0; i < MLP_WSTREAMS; ++i) {
#pragma HLS UNROLL
			const int m = i * MLP_WSUB + u;            // word 0..15
			axis_word_t d = lane[i]->read();
			ch[2 * m]     = d.range(8 * MLP_AXI_BYTES - 1, 0);
			ch[2 * m + 1] = d.range(MLP_AXIS_BITS - 1, 8 * MLP_AXI_BYTES);
		}
	}
}

#endif // MLP_COMMON_H
