/*******************************************************************************
** mlp_engine.cpp -- KERNEL 2: the compute engine, ONE COMPUTE UNIT PER SLR.
**
** 3 CUs of this identical kernel (SLR0 / SLR1 / SLR2), free-running
** (ap_ctrl_none, stream ports only: the host never enqueues them, they block
** on the control lane until a token starts).  No SLR parameter is needed --
** the mover already selected this engine's tiles, so the local h index is a
** pure function of (job, pass, DSP column):
**     local = job*256 + (d/2)*8 + 2*np + (d&1)
** which is what the SLR-parameterised version reduced to.  The (d/2)*8 shape
** comes from the lane relabeling that makes the 32 pseudo-channels of one HBM
** address hit 32 different lane memories (see mlp_common.h).
**
** ONE MAC array (1536 DSP48E2) and ONE working set serve both projections:
**   job  0..15 : fused gate/up macro tile -> h[0..4095] of this SLR
**                (SRQ -> G' LUT -> INT8)
**   job 16..27 : down_proj block over the SLR's K slice, which is exactly the
**                h it just produced -> 128 INT32 partial sums to the collector
** Jobs retire in order, so h is complete before job 16 reads it.
**
** ROUTING-DRIVEN SHAPE.  Two Hardware builds failed in route_design with
** global congestion level 7, and report_design_analysis put THIS kernel at the
** top of every congested window in every SLR: the MAC pipeline (34-65 % of the
** cells) and the jbuf ping-pong (13-35 %).  Three changes answer that:
**   o) NO DATAFLOW.  One working set (128 URAM instead of 256) and no 2:1
**      ping-pong select on its 8192-bit read path.  load_job and compute_job
**      now take turns; the overlap moved into the 512-deep weight FIFOs in
**      front of the engine (mlp_link.cfg).  With 8 lanes those held a whole
**      job, so the movers prefetched the next one while this one computed;
**      with 4 lanes the movers and load_job both need 4 cycles per address,
**      so there is nothing left to gain from prefetching.
**   o) 24 K-LANES, NOT 32: 1536 DSPs instead of 2048, fed through a 4-phase
**      gearbox (mlp_common.h) because a 64-bit word holds 32 weights.
**   o) THE ACCUMULATORS STAY IN THE DSPs.  The third Hardware build routed
**      every net but left 9,024 conflicting nodes, mostly at the input pins
**      of the MAC DSPs: HLS had built each "P += a * q" as DSP -> fabric
**      copy + two muxes -> back into the DSP (49,152 LUT/FF per engine).  The
**      MAC is now the RTL blackbox mlp_dsp_mac.v (see compute_job), so this
**      kernel must be synthesised with mlp_blackbox.tcl registered
**      (mlp_engine_compile.cfg for v++, run_hls.tcl locally).
**   (A DSP cascade for the K reduction was tried and measured; written in
**    HLS C++ it does not become a cascade -- see job_drain.)
** Throughput cost: the job's load (II=4 out of the FIFOs since the link went
** to 4 lanes; II=2 before) and its compute no longer overlap, and a K tile
** takes 22 steps instead of 16.  Measure the token latency in Emulation-HW
** before and after.
**
** One kernel per source file: v++ copies the source into every .xo for
** software emulation, so kernels sharing a .cpp collide at link time.
*******************************************************************************/

#include "mlp_common.h"

// ###########################################################################
// KERNEL 2: mlp_engine  (3 CUs, one per SLR, free-running)
// ###########################################################################

// ---- control block -> the engine's local tables --------------------------
static void recv_ctrl(hls::stream<wstrm_t> &ctrl, act_t xq[MLP_K],
                      acc_t rq_g[MLP_F_SLR], acc_t rq_u[MLP_F_SLR])
{
rc_x:
	for (int i = 0; i < MLP_K; i += MLP_VPW) {
#pragma HLS PIPELINE II=1
		axis_word_t d = ctrl.read();
		for (int v = 0; v < MLP_VPW; ++v) {
#pragma HLS UNROLL
			xq[i + v] = (act_t)(ap_int<8>)(ap_int<32>)
			            d.range(32 * v + 31, 32 * v);
		}
	}
rc_rg:
	for (int i = 0; i < MLP_F_SLR; i += MLP_VPW) {
#pragma HLS PIPELINE II=1
		axis_word_t d = ctrl.read();
		for (int v = 0; v < MLP_VPW; ++v) {
#pragma HLS UNROLL
			rq_g[i + v] = (acc_t)(ap_int<32>)d.range(32 * v + 31, 32 * v);
		}
	}
rc_ru:
	for (int i = 0; i < MLP_F_SLR; i += MLP_VPW) {
#pragma HLS PIPELINE II=1 
		axis_word_t d = ctrl.read();
		for (int v = 0; v < MLP_VPW; ++v) {
#pragma HLS UNROLL
			rq_u[i + v] = (acc_t)(ap_int<32>)d.range(32 * v + 31, 32 * v);
		}
	}
}

// ---- scatter one HBM address into the working set ------------------------
// The 32 pseudo-channels of one address arrive together, so both mappings put
// them in 32 DIFFERENT lane memories -- otherwise the lanes they share
// serialize the writes and set the II of the load loop.
//   gate/up : pseudo-channel p holds columns p*16 + j of the macro tile.  j is
//             fixed within a beat, so the lane must come from p: pass
//             np = j / 4 and lane = p * 4 + j % 4 (the relabeling documented
//             in mlp_common.h).  4 writes per lane per beat -> 2 cycles on
//             the dual-port RAM, below the II = 4 the 4-lane link sets.
//   down    : pseudo-channel p holds rows p*4 + r of the block, r is fixed
//             within a beat, so lane = p*4 + r is already distinct per p and
//             the words start at slot 0.
static inline void gu_store_pc(job_buf_t buf, int p, int j, int wo, hbm_word_t w)
{
#pragma HLS INLINE
	const int np = j / MLP_GU_GRP;                          // 0..3   pass slot
	const int nl = p * MLP_GU_GRP + (j % MLP_GU_GRP);       // 0..127 lane
	const int wb = np * MLP_GU_WORDS + wo;                  // pass slot + word
	for (int u = 0; u < MLP_WPB_AXI; ++u)
		buf[nl][wb + u] = (wbuf_t)w.range(64 * u + 63, 64 * u);
}

static inline void dn_store_pc(job_buf_t buf, int p, int r, int wo, hbm_word_t w)
{
#pragma HLS INLINE
	const int nl = p * MLP_CPP_D + r;                       // 0..127 in the block
	for (int u = 0; u < MLP_WPB_AXI; ++u)
		buf[nl][wo + u] = (wbuf_t)w.range(64 * u + 63, 64 * u);
}

// The two job kinds scatter an address differently, so they keep SEPARATE
// loops: the address arithmetic stays compile-time constant inside each, and
// both retire a beat in II=4.  That is the floor set by the link itself: 16
// words per address on 4 AXIS lanes = MLP_WSUB = 4 reads from every lane, one
// per cycle -- exactly the rate the movers deliver (mv_beat, II=4).  The
// working set alone would allow II=2 (32 conflict-free lanes x 4 words over
// the 2 ports of the true dual-port RAM), which is what the loops ran at with
// 8 lanes.  Both loops write the SAME 128 memories -- only the write address
// logic is duplicated.
//
// the literal II=4 of gu_load / dn_load (HLS pragmas cannot take a #define)
typedef char mlp_ld_ii_assert[(MLP_LD_II == 4) ? 1 : -1];

static void load_job(MLP_ELANES(MLP_EARG) job_buf_t buf, int job)
{
	if (job < MLP_JOBS_GU) {
	gu_load:
		for (int b = 0; b < MLP_GU_BEATS; ++b) {            // 192 addresses
#pragma HLS PIPELINE II=4
			hbm_word_t ch[MLP_NPC];
#pragma HLS ARRAY_PARTITION variable=ch complete dim=1
			recv_addr(MLP_ELANES(MLP_EPASS) ch);

			const int j  = b / MLP_GU_BPCOL;   // column inside the PC block
			const int wo = (b % MLP_GU_BPCOL) * MLP_WPB_AXI;
		gu_load_pc:
			for (int p = 0; p < MLP_NPC; ++p) {
#pragma HLS UNROLL
				gu_store_pc(buf, p, j, wo, ch[p]);
			}
		}
	} else {
	dn_load:
		for (int b = 0; b < MLP_DN_BEATS; ++b) {            // 128 addresses
#pragma HLS PIPELINE II=4
			hbm_word_t ch[MLP_NPC];
#pragma HLS ARRAY_PARTITION variable=ch complete dim=1
			recv_addr(MLP_ELANES(MLP_EPASS) ch);

			const int r  = b / MLP_DN_BPROW;   // row inside the PC block
			const int wo = (b % MLP_DN_BPROW) * MLP_WPB_AXI;
		dn_load_pc:
			for (int p = 0; p < MLP_NPC; ++p) {
#pragma HLS UNROLL
				dn_store_pc(buf, p, r, wo, ch[p]);
			}
		}
	}
}

// ---- the 24-of-32 gearbox: one step's slice of a lane ---------------------
// cur = the word this step reads, prev = the word the previous step read.
// Phase s % 4 picks where the 24 weights (48 bits) start inside {cur, prev};
// the table is in mlp_common.h.  Four constant bit ranges -> a 4:1 select per
// output bit, one LUT each, sitting right at the URAM output.
static inline ap_uint<64> gear_slice(wbuf_t cur, wbuf_t prev, int ph)
{
#pragma HLS INLINE
	const ap_uint<128> win = ((ap_uint<128>)cur << 64) | (ap_uint<128>)prev;
	ap_uint<64> seg = 0;
	switch (ph) {
	case 0:  seg.range(47, 0) = win.range(64 + 47, 64); break; // cur [ 0..23]
	case 1:  seg.range(47, 0) = win.range(48 + 47, 48); break; // prev[24..31] cur[0..15]
	case 2:  seg.range(47, 0) = win.range(32 + 47, 32); break; // prev[16..31] cur[0.. 7]
	default: seg.range(47, 0) = win.range(16 + 47, 16); break; // prev[ 8..31]
	}
	return seg;
}

// ---- 3072 MAC/cycle over one job, then the job's epilogue ----------------
//
// This is the ONLY multiply loop of the engine.  It appears once in the source
// and is called from one place, so HLS builds exactly one 1536-DSP array and
// both projections take turns on it.
//
//   gate/up (job < 16): 4 passes x 3 K tiles, operands from x_q,
//                       epilogue = SRQ -> G' LUT -> INT8 h
//   down    (job >= 16): 1 pass  x 8 K tiles, operands from h,
//                       epilogue = push the INT32 partial sums
//
// A K tile (512 K terms = 16 words per lane) takes MLP_KSTEP_HW = 22 steps of
// MLP_KLANE_HW = 24 K terms; the last step has 8 live terms and zeroes the
// other 16 activations, so the garbage weights it sees multiply by 0.
static void compute_job(const job_buf_t buf, int job,
                        const act_t xq[MLP_K],
                        const acc_t rq_g[MLP_F_SLR],
                        const acc_t rq_u[MLP_F_SLR],
                        act_t h[MLP_F_SLR],
                        hls::stream<pstrm_t> &ps)
{
	const bool is_gu  = (job < MLP_JOBS_GU);
	const int  npass  = is_gu ? MLP_NPASS    : 1;              // 4 : 1
	const int  nktile = is_gu ? MLP_KTILE_GU : MLP_KTILE_DN;   // 3 : 8

	// gearbox history: the word each lane read on the previous step.  Only
	// phases 1..3 look at it, and phase 0 is the first step of every K tile,
	// so what it holds across a tile or pass boundary never matters.
	wbuf_t prev[MLP_NLANE];
#pragma HLS ARRAY_PARTITION variable=prev complete dim=1
	for (int i = 0; i < MLP_NLANE; ++i) {
#pragma HLS UNROLL
		prev[i] = 0;
	}

#ifndef __SYNTHESIS__
	// Software model only: the P registers of the 1536 DSPs (MLP_MAC in
	// mlp_common.h).  In hardware they are inside the DSP48E2s themselves,
	// so this array does not exist in the synthesised design at all.
	dsp_t mac_st[MLP_DSPS][MLP_KLANE_HW];
#endif

job_pass:
	for (int np = 0; np < npass; ++np) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=4
		acc_t acc[MLP_NLANE];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=1
		for (int i = 0; i < MLP_NLANE; ++i) {
#pragma HLS UNROLL
			acc[i] = 0;
		}

		// gate/up: pass np lives in slot np*48 of every lane; down: slot 0
		const int wbase = is_gu ? np * MLP_GU_WORDS : 0;

	job_ktile:
		for (int kt = 0; kt < nktile; ++kt) {
#pragma HLS LOOP_TRIPCOUNT min=3 max=8
			// ---- 1536 DSP48E2, one per (column pair, K lane) ----
			// Each one accumulates the K tile in its OWN P register
			// (mlp_dsp_mac, an RTL blackbox).  There is no P array here any
			// more and no "P = 0" before the tile: step s == 0 tells the DSPs
			// to start over, and the tile's sums come out of the DSPs on the
			// last step, where they are drained straight away (below).
			//
			// WHY NOT "P += a * q" IN C++.  That is what this loop used to
			// say, and HLS built every accumulator as DSP P -> fabric copy
			// register + pipeline-bypass mux (ap_sig_allocacmp) + "first
			// step ? 0" mux -> back into the DSP's C port.  Per engine that is
			// 1536 x 32 = 49,152 LUTs and FFs and ~100 k wires, all switched
			// by ap_enable_reg_pp0_iter8 -- the 49,152-load net of the third
			// Hardware build's high_fanout.rpt -- and the input-pin (IMUX)
			// conflicts that failed its route_design sat right at those DSPs.
			// Three C++ codings were synthesised on a small copy of this loop
			// (reset per tile, never reset + difference, one flat loop with
			// "s == 0 ? m : P + m"): every one kept P in the fabric, because
			// HLS inserts that bypass mux for any loop-carried variable of a
			// pipelined loop.  Only RTL keeps it in the DSP.
			//
			// THE RULE THE BLACKBOX NEEDS: the 22 calls of a K tile arrive on
			// consecutive clocks.  This pipeline is II=1 and has no blocking
			// I/O (HLS 200-1553 said so when style=frp was tried), so it can
			// neither stall nor bubble.  Keep it that way: after any change to
			// this loop, check "Final II = 1" for it in the csynth log, and
			// never read a stream inside it.
		job_kstep:
			for (int s = 0; s < MLP_KSTEP_HW; ++s) {
#pragma HLS PIPELINE II=1
				const int ph  = s & (MLP_GEAR_PHASES - 1);        // gearbox phase
				// the word this step reads: the last one it touches, clamped
				// so the half-empty last step never reads past its K tile
				const int wg  = MLP_GEAR_WORD(s);
				const int wi  = wbase + kt * MLP_KSTEP
				              + (wg < MLP_KSTEP ? wg : MLP_KSTEP - 1);

				// ---- activations: 24 consecutive K terms from 32 banks ----
				// K term kb + kl lives in bank (kb + kl) % 32, row (kb + kl) / 32.
				// kb % 32 is 0 / 24 / 16 / 8, so bank b holds the operand from
				// row r0 when b >= kb % 32 and from row r0 + 1 otherwise.  Every
				// bank is read exactly ONCE per cycle at a computed row, which is
				// what keeps the 32 reads conflict-free (a plain xq[kb + kl]
				// with a run-time kb % 32 is not provably conflict-free and
				// would cost II).  Rows past the end only feed zeroed lanes, so
				// they are clamped to stay in range.
				const int kb  = kt * MLP_DN + s * MLP_KLANE_HW;   // first K term
				const int r0  = kb / MLP_QBANKS;
				const int kph = (kb / 8) & (MLP_GEAR_PHASES - 1); // kb % 32 / 8

				act_t bx[MLP_QBANKS], bh[MLP_QBANKS];
#pragma HLS ARRAY_PARTITION variable=bx complete dim=1
#pragma HLS ARRAY_PARTITION variable=bh complete dim=1
				for (int b = 0; b < MLP_QBANKS; ++b) {
#pragma HLS UNROLL
					const int r  = (b >= kph * 8) ? r0 : r0 + 1;
					// Operand mux: x_q for gate/up, h for down.  is_gu is a
					// RUN-TIME value, so HLS schedules BOTH reads every cycle;
					// the unused one must still be in range and hit bank b.
					const int rx = is_gu ? (r < MLP_XQ_ROWS ? r : MLP_XQ_ROWS - 1) : 0;
					const int rh = (r < MLP_H_ROWS ? r : MLP_H_ROWS - 1);
					bx[b] = xq[rx * MLP_QBANKS + b];
					bh[b] = h [rh * MLP_QBANKS + b];
				}

				act_t q[MLP_KLANE_HW];
#pragma HLS ARRAY_PARTITION variable=q complete dim=1
				for (int kl = 0; kl < MLP_KLANE_HW; ++kl) {
#pragma HLS UNROLL
					// (kph*8 + kl) & 31: the low 3 bits are the constant kl & 7,
					// so this is a 4:1 select per operand, not a 32:1
					const int  b     = (kph * 8 + kl) & (MLP_QBANKS - 1);
					const bool live  = (s * MLP_KLANE_HW + kl) < MLP_DN;
					q[kl] = live ? (is_gu ? bx[b] : bh[b]) : (act_t)0;
				}

				// ---- weights: one word per lane, 24 weights out of the gearbox
				ap_uint<64> seg[MLP_NLANE];
#pragma HLS ARRAY_PARTITION variable=seg complete dim=1
				for (int nl = 0; nl < MLP_NLANE; ++nl) {
#pragma HLS UNROLL
					const wbuf_t cur = buf[nl][wi];
					seg[nl]  = gear_slice(cur, prev[nl], ph);
					prev[nl] = cur;
				}

				// ---- 1536 DSP48E2: each accumulates in its own P register ----
				// w_lo / w_hi are the two packed columns; the (w_hi << 18) +
				// w_lo packing happens in the DSP pre-adder (mlp_dsp_mac.v).
				// r[d][kl] is that DSP's P[31:0] after this step; only the
				// last step's value is used.
				const bool first = (s == 0);
				dsp_out_t r[MLP_DSPS][MLP_KLANE_HW];
#pragma HLS ARRAY_PARTITION variable=r complete dim=0
			job_dsp:
				for (int d = 0; d < MLP_DSPS; ++d) {
#pragma HLS UNROLL
					for (int kl = 0; kl < MLP_KLANE_HW; ++kl) {
#pragma HLS UNROLL
						r[d][kl] = MLP_MAC(mac_st[d][kl],
						                   w_at(seg[2 * d    ], kl),
						                   w_at(seg[2 * d + 1], kl),
						                   q[kl], first);
					}
				}

				// ---- last step: drain -> PRE-ADD -> split fields -> acc ----
				// The packed format is additive (mlp_common.h), so the K-lanes
				// are summed as raw accumulators FIRST and the borrow
				// correction runs once per MLP_SPLIT_SPAN = 12 lanes: 64 x 2 =
				// 128 corrections.  MLP_SPLIT_SPAN x MLP_KSTEP_HW products stay
				// inside the low field's window (asserted in mlp_common.h).
				// The 32-bit DSP outputs are widened to 48 bits before the sum
				// (12 x |P| can exceed 2^31).
				//
				// The drain sits INSIDE the step loop, on its last iteration,
				// so it reads the DSP outputs the cycle they appear instead of
				// parking 1536 x 32 bits in fabric registers until the loop
				// exits.
				//
				// NOT A DSP CASCADE.  Summing the 12 lanes on the DSPs' own
				// PCOUT -> PCIN wires instead of this adder tree was tried and
				// measured on one DSP column (Vivado 2022.1 synthesis, xcu280):
				//   tree  (this code)  : 24 DSP,  745 LUT, 1136 FF, 47.6 k pins
				//   chain (HLS C++)    : 24 DSP,  837 LUT, 1342 FF, 46.7 k pins,
				//                        0 PCIN driven by a PCOUT
				// HLS keeps every partial sum in its own fabric register
				// between two DSPs, which is exactly what stops Vivado from
				// inferring the cascade.  A real cascade would have to be part
				// of mlp_dsp_mac.v.
				if (s == MLP_KSTEP_HW - 1) {
				job_drain:
					for (int d = 0; d < MLP_DSPS; ++d) {
#pragma HLS UNROLL
						acc_t sl = 0, sh = 0;
					job_split:
						for (int g = 0; g < MLP_SPLIT_GROUPS; ++g) {
#pragma HLS UNROLL
							dsp_t sum = 0;
							for (int k = 0; k < MLP_SPLIT_SPAN; ++k) {
#pragma HLS UNROLL
								sum += (dsp_t)r[d][g * MLP_SPLIT_SPAN + k];
							}
							acc_t lo, hi;
							dsp_split(sum, lo, hi);
							sl += lo;
							sh += hi;
						}
						acc[2 * d    ] += sl;
						acc[2 * d + 1] += sh;
					}
				}
			}
		}

		// ---- epilogue: the only other place the two jobs differ ----
		if (is_gu) {
			// 128 fused columns done = 64 (gate, up) pairs.  The engine owns
			// exactly the h slice its 16 tiles produce, so the local index
			// needs no SLR term -- but it does have to undo the lane
			// relabeling of gu_store_pc (mlp_common.h):
			//     lane 2d of pass np = fused column
			//         lc = (2d / 4) * 16 + 4 * np + (2d % 4)
			//     intermediate channel i = (job * 512 + lc) / 2
			//                            = job * 256 + (d / 2) * 8 + 2 * np + (d & 1)
			// d = 0..63 and np = 0..3 cover 0..255 exactly once, so the whole
			// 256-channel slice of a tile is still written exactly once.
		gu_act:
			for (int d = 0; d < MLP_PAIRS; ++d) {
#pragma HLS PIPELINE II=1 
				const int local = job * (MLP_DN / 2)
				                + (d / MLP_GU_HG) * (MLP_CPP / 2)
				                + MLP_GU_HG * np
				                + (d % MLP_GU_HG);

				// SRQ at the gate_proj / up_proj output scales
				ap_int<64> pg = (ap_int<64>)acc[2 * d    ]
				* (ap_int<64>)rq_g[local];
				ap_int<64> pu = (ap_int<64>)acc[2 * d + 1]
				* (ap_int<64>)rq_u[local];
				act_t q_g = clamp_i8((ap_int<32>)
				rne_sh<MLP_RQ_SHIFT>(pg));
				act_t q_u = clamp_i8((ap_int<32>)
				rne_sh<MLP_RQ_SHIFT>(pu));

				// G' folds dequant -> GELU -> rescale to S_H (exact), int16
				// with MLP_GLUT_FRAC fraction bits; |G' * q_u| < 2^23 fits int32
				ap_int<32> gp = (ap_int<32>)
				(ap_int<16>)GELU_LUT_I16[(int)q_g + 128];
				ap_int<32> hv = gp * (ap_int<32>)q_u;
				h[local] = clamp_i8(rne_sh<MLP_GLUT_FRAC>(hv));
			}
		} else {
			// the collector adds the three engines lane by lane, so the block
			// order (job) and the lane order are the whole protocol
		dn_emit:
			for (int nl = 0; nl < MLP_NLANE; ++nl) {
#pragma HLS PIPELINE II=1
				ps_put(ps, acc[nl]);
			}
		}
	}
}

extern "C" void mlp_engine(hls::stream<wstrm_t> &ctrl,
                           MLP_ELANES(MLP_EARG)
                           hls::stream<pstrm_t> &ps)
{
#pragma HLS INTERFACE axis port=ctrl
#pragma HLS INTERFACE axis port=w_0
#pragma HLS INTERFACE axis port=w_1
#pragma HLS INTERFACE axis port=w_2
#pragma HLS INTERFACE axis port=w_3
#pragma HLS INTERFACE axis port=ps
	// free-running: no control registers, no host enqueue, started by data
#pragma HLS INTERFACE ap_ctrl_none port=return

	// NOT static: the three CUs are three instances of this same kernel, and in
	// software emulation they are three threads of ONE process -- function-scope
	// statics would be SHARED storage and the engines would overwrite each
	// other's requant tables and h slice (plausible-looking but wrong results,
	// only the all-zero test case still passing).  As automatic arrays each CU
	// gets its own, which is also the physical truth in hardware: HLS turns
	// them into this module's own on-chip memories either way.
	act_t xq[MLP_K];
	acc_t rq_g[MLP_F_SLR], rq_u[MLP_F_SLR];
	act_t h[MLP_F_SLR];
	// The banking factor follows the WIDEST access of each array:
	//   xq / h    : compute_job reads 24 CONSECUTIVE operands per cycle from a
	//               start that moves by 24 per step; with 32 banks each bank is
	//               read once per cycle at a computed row (see compute_job), so
	//               the reads never collide and the loop keeps II=1.  Both
	//               arrays are INT8 and tiny (1.5 KB / 4 KB), so the banks cost
	//               LUTRAM/FF, not BRAM or URAM.
	//   rq_g/rq_u : written 16 per control word, read ONE per cycle in gu_act
	//               -- 16 is already the widest access, more banks would only
	//               split the 32 BRAMs they sit in.
#pragma HLS ARRAY_PARTITION variable=xq   cyclic factor=32 dim=1
#pragma HLS ARRAY_PARTITION variable=rq_g cyclic factor=16 dim=1
#pragma HLS ARRAY_PARTITION variable=rq_u cyclic factor=16 dim=1
#pragma HLS ARRAY_PARTITION variable=h    cyclic factor=32 dim=1

	// ONE working set, 192 KB, for the whole run.  There used to be two: the
	// job loop was a DATAFLOW region, so HLS double-buffered jbuf (256 URAM,
	// 80 % of an SLR) and put a 2:1 select on every one of the 8192 bits the
	// MAC array reads per cycle -- 13-35 % of the cells in the congested
	// windows of the failed builds.  Load and compute now take turns on one
	// buffer; the overlap they used to get lived in the 512-deep weight FIFOs
	// in front of the engine (mlp_link.cfg), which held a whole job while the
	// link had 8 lanes (with 4 lanes load_job runs at the movers' own rate).
	//
	// TRUE dual port: both ports can read OR write, so the 4 words a beat
	// leaves in a lane retire in 2 cycles.  (load_job is at II=4 now, set by
	// the 4-lane link; it was II=2, set by this RAM, with 8 lanes.)  RAM_2P
	// has one write port only, and HLS used to buy the second one by silently
	// partitioning dim 2 cyclic-2 (HLS 214-270) -- which DOUBLED the memory
	// count.  Asking for T2P directly is the same throughput without that.
	//   impl=URAM : 1 URAM per lane -> 128 of the 320 per SLR
	//   impl=BRAM : 4 BRAM18 per lane -> 512 of the 1344 per SLR
	job_buf_t jbuf;
#pragma HLS ARRAY_PARTITION variable=jbuf complete dim=1
//#pragma HLS BIND_STORAGE variable=jbuf type=RAM_T2P impl=BRAM
#pragma HLS BIND_STORAGE variable=jbuf type=RAM_T2P impl=URAM

token_loop:
	for (;;) {
		// blocks here until the mover starts the next token
		recv_ctrl(ctrl, xq, rq_g, rq_u);

	job_loop:
		for (int job = 0; job < MLP_JOBS; ++job) {
			load_job(MLP_ELANES(MLP_EPASS) jbuf, job);
			compute_job(jbuf, job, xq, rq_g, rq_u, h, ps);
		}

#ifndef __SYNTHESIS__
		break;            // C simulation: exactly one token per call
#endif
	}
}
