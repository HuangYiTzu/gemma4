/*******************************************************************************
** MLP.cpp -- Vitis HLS kernel: one Gemma-4-E2B MLP layer (decode stage)
**            INT2 QAT weights, INT8 activations, Xilinx Alveo U280.
**
**   gate = W_gate . x ; up = W_up . x ; h = GELU(gate) * up ; y = W_down . h
**   out  = post_feedforward_layernorm(y) + residual
**
** Architecture (see kernel.h for the full map):
**
**   mover_gu / mover_dn   one data mover in SLR0 owns all 32 HBM pseudo-
**                         channels (local AXI, zero SLL) and round-robins
**                         512x512 macro tiles to the owning SLR as 512 B
**                         beats (16 PCs each), i.e. the 512 B/cycle per-SLR
**                         crossing width of the SLL budget in kernel.h.
**
**   gu_engine<SLR>        16 fused gate/up macro tiles per SLR.  Working set
**                         512 columns x 1536 K = 192 KB, double buffered so
**                         the load of tile t+1 overlaps the compute of tile t.
**                         128 N-lanes x 32 K-lanes = 4096 MAC/cycle on 2048
**                         DSP48E2 (two output columns packed per DSP).
**                         Produces h[4096s : 4096s+4096] (INT8 @ S_H).
**
**   dn_engine<SLR>        W_down is split along K: SLR s owns the K slice
**                         [4096s, 4096s+4096), exactly the h slice it just
**                         produced -> no h movement between SLRs.  Emits a
**                         full 1536-wide INT32 partial sum.
**
**   epilogue              cross-SLR integer add -> x c_down (the one and only
**                         dequantization) -> RMSNorm (fast inverse square
**                         root, ISCAS'25 shared unit) -> + residual -> Q7.8.
**
** Quantization points: exactly three, all from the QAT checkpoint --
**   r_g[i] (gate out), r_u[i] (up out), and S_H which is folded into the
**   exact 256-entry G' table.  h needs no further quantization.
**
** NO floating point anywhere in the hardware path.
*******************************************************************************/

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

// ===========================================================================
// fast_inv_unit -- shared reciprocal / inverse-square-root unit, ISCAS'25
// "An Accurate and Compact Design Integrating Seven Common Nonlinear
//  Functions in Deep Learning", Fig. 2:
//   1) leading-one detector builds an FP32-style word I_x
//   2) magic seed   I_y = a2 - I_x         (1/x)
//                   I_y = a1 - (I_x >> 1)  (1/sqrt(x))
//   3) FP-to-INT back to fixed point
//   4) two Newton-Raphson iterations
// The MLP epilogue uses the 1/sqrt(x) mode for post_feedforward_layernorm;
// the 1/x mode is kept so the same unit serves softmax/sigmoid in the other
// layers, and is pruned by HLS when mode_recip is a constant.
// ===========================================================================
template <typename DT, typename RT>
static RT fast_inv_unit(DT x, bool mode_recip)
{
#pragma HLS INLINE
	const int W = DT::width;
	const int F = DT::width - DT::iwidth;

	// ---- 1) leading-one detector (priority encoder) ----
	ap_uint<W> bits = x.range(W - 1, 0);
	ap_uint<7> pos = 0;
lod:
	for (int i = 0; i < W; ++i) {
#pragma HLS UNROLL
		if (bits[i]) pos = i;
	}
	ap_int<12> e_x = (ap_int<12>)pos - F;

	ap_uint<W> mb = bits;
	mb[pos] = 0;                              // strip the implicit leading 1
	ap_uint<23> M_x;
	if (pos >= 23) M_x = (ap_uint<23>)(mb >> (int)(pos - 23));
	else           M_x = (ap_uint<23>)(((ap_uint<W + 24>)mb) << (int)(23 - pos));

	// ---- 2) magic-constant seed in the INT32 domain ----
	ap_uint<32> I_x = (((ap_uint<32>)(ap_uint<9>)(e_x + 127)) << 23) | M_x;
	ap_uint<32> I_y = mode_recip
	                ? (ap_uint<32>)(FASTINV_MAGIC_RECIP - I_x)
	                : (ap_uint<32>)(FASTINV_MAGIC_RSQRT - (I_x >> 1));

	// ---- 3) FP-to-INT: rebuild y0 = 1.M * 2^e_y as fixed point ----
	ap_int<12>  e_y = (ap_int<12>)(I_y >> 23) - 127;
	ap_uint<23> M_y = I_y.range(22, 0);
	ap_ufixed<56, 25> yb = 0;                 // 2^24 .. 2^-31
	yb[31] = 1;                               // implicit leading 1 (2^0)
	yb.range(30, 8) = M_y;
	RT y = (e_y >= 0) ? (RT)(yb << (int)e_y) : (RT)(yb >> (int)(-e_y));

	// ---- 4) two Newton-Raphson refinement steps ----
	// the intermediates are cut back to narrow types on purpose: x*y is ~1
	// (reciprocal) or ~sqrt(x) (rsqrt), so a full-width product would cost a
	// huge multiplier for bits that are structurally zero.
	const ap_ufixed<2, 1> B1(1.5);
	typedef ap_ufixed<32, 18, AP_RND> xy_t;   // x*y
	typedef ap_ufixed<26, 2,  AP_RND> t_t;
nr:
	for (int it = 0; it < 2; ++it) {
#pragma HLS UNROLL
		xy_t xy = (xy_t)(x * y);
		t_t  t;
		if (mode_recip) t = (t_t)(2  - xy);                 // y*(2 - x*y)
		else            t = (t_t)(B1 - (xy * y >> 1));      // y*(1.5-0.5xy^2)
		y = (RT)(y * t);
	}
	return y;
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

// split P into the two signed fields (borrow correction on the low field)
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

// INT2 weight kl of a 64-bit k-major word (32 weights, k increasing)
static inline w_t w_at(ap_uint<64> word, int kl)
{
#pragma HLS INLINE
	return (w_t)(ap_uint<2>)word.range(2 * kl + 1, 2 * kl);
}

// ===========================================================================
// Data movers (SLR0): own all 32 HBM pseudo-channels, round-robin macro tiles
// to the three SLRs so all engines stay fed.  Each address yields two 512 B
// beats: channels 0..15 then 16..31.
// ===========================================================================
// MLP_HBM_ARG (kernel.h) expands the 32 pseudo-channel parameters

static void mover_gu(MLP_HBM_PORTS(MLP_HBM_ARG)
                     hls::stream<mover_beat_t> &o0,
                     hls::stream<mover_beat_t> &o1,
                     hls::stream<mover_beat_t> &o2)
{
	const int BEATS = MLP_GU_TILE_BYTES / MLP_AXI_BYTES;   // 192 beats/tile
mv_gu_tile:
	for (int nt = 0; nt < MLP_TILES_SLR; ++nt) {
	mv_gu_slr:
		for (int s = 0; s < MLP_NSLR; ++s) {
			const int t = MLP_TILES_SLR * s + nt;          // SLR(t) = t/16
			const int base = t * BEATS;
		mv_gu_beat:
			for (int b = 0; b < BEATS; ++b) {
#pragma HLS PIPELINE II=2
				mover_beat_t lo, hi;
#define MLP_RD_LO(i) lo.w[i] = w_hbm_##i[base + b];
#define MLP_RD_HI(i) hi.w[(i) - MLP_BEAT_PC] = w_hbm_##i[base + b];
				MLP_HBM_PORTS_LO(MLP_RD_LO)
				MLP_HBM_PORTS_HI(MLP_RD_HI)
#undef MLP_RD_LO
#undef MLP_RD_HI
				if      (s == 0) { o0.write(lo); o0.write(hi); }
				else if (s == 1) { o1.write(lo); o1.write(hi); }
				else             { o2.write(lo); o2.write(hi); }
			}
		}
	}
}

static void mover_dn(MLP_HBM_PORTS(MLP_HBM_ARG)
                     hls::stream<mover_beat_t> &o0,
                     hls::stream<mover_beat_t> &o1,
                     hls::stream<mover_beat_t> &o2)
{
	const int BEATS  = MLP_DN_BLK_BYTES / MLP_AXI_BYTES;   // 128 beats/block
	const int BLOCKS = MLP_TILES_DN * MLP_NPASS;           // 12 blocks
	const int CH0    = MLP_GU_CH_BYTES / MLP_AXI_BYTES;    // down region base
mv_dn_blk:
	for (int blk = 0; blk < BLOCKS; ++blk) {
	mv_dn_slr:
		for (int s = 0; s < MLP_NSLR; ++s) {
			const int base = CH0 + s * (MLP_DN_SLR_BYTES / MLP_AXI_BYTES)
			                     + blk * BEATS;
		mv_dn_beat:
			for (int b = 0; b < BEATS; ++b) {
#pragma HLS PIPELINE II=2
				mover_beat_t lo, hi;
#define MLP_RD_LO(i) lo.w[i] = w_hbm_##i[base + b];
#define MLP_RD_HI(i) hi.w[(i) - MLP_BEAT_PC] = w_hbm_##i[base + b];
				MLP_HBM_PORTS_LO(MLP_RD_LO)
				MLP_HBM_PORTS_HI(MLP_RD_HI)
#undef MLP_RD_LO
#undef MLP_RD_HI
				if      (s == 0) { o0.write(lo); o0.write(hi); }
				else if (s == 1) { o1.write(lo); o1.write(hi); }
				else             { o2.write(lo); o2.write(hi); }
			}
		}
	}
}

// ===========================================================================
// gate/up engine -- one instance per SLR (SLR = template parameter)
// ===========================================================================
#define MLP_GU_WORDS  (MLP_BPC / 8)           // 48 x 64-bit words per column
#define MLP_WPB_AXI   (MLP_AXI_BYTES / 8)     // 4 x 64-bit words per AXI beat
#define MLP_GU_BPCOL  (MLP_BPC / MLP_AXI_BYTES)   // 12 AXI beats per column
#define MLP_PAIRS     (MLP_NLANE / MLP_PACK_WAYS) // 64 (gate, up) pairs

// working set: 512 columns x 384 B, addressed as [n_pass][n_lane][word]
typedef ap_uint<64> wbuf_t;

// working set of one macro tile: 512 columns x 384 B = 192 KB
typedef wbuf_t gu_tile_buf_t[MLP_NPASS][MLP_NLANE][MLP_GU_WORDS];

// scatter one pseudo-channel's 32 B into the gate/up working set.  p / j / wo
// are compile-time constants at every (unrolled) call site.
static inline void gu_store_pc(gu_tile_buf_t buf, int p, int j, int wo,
                               hbm_word_t w)
{
#pragma HLS INLINE
	const int lc = p * MLP_CPP + j;            // 0..511 inside the macro tile
	const int np = lc / MLP_NLANE;
	const int nl = lc % MLP_NLANE;
	for (int u = 0; u < 4; ++u)
		buf[np][nl][wo + u] = (wbuf_t)w.range(64 * u + 63, 64 * u);
}

// ---- load one 512-column macro tile out of the mover stream --------------
static void gu_load_tile(hls::stream<mover_beat_t> &in, gu_tile_buf_t buf)
{
	const int BEATS = MLP_GU_TILE_BYTES / MLP_AXI_BYTES;   // 192 addresses
gu_load:
	for (int b = 0; b < BEATS; ++b) {
#pragma HLS PIPELINE II=2
		mover_beat_t lo = in.read();           // channels 0..15
		mover_beat_t hi = in.read();           // channels 16..31
		const int j  = b / MLP_GU_BPCOL;       // column inside the PC block
		const int wo = (b % MLP_GU_BPCOL) * MLP_WPB_AXI;   // 64-bit word offset
	gu_load_pc:
		for (int pp = 0; pp < MLP_BEAT_PC; ++pp) {
#pragma HLS UNROLL
			gu_store_pc(buf, pp,                 j, wo, lo.w[pp]);
			gu_store_pc(buf, pp + MLP_BEAT_PC,   j, wo, hi.w[pp]);
		}
	}
}

// ---- 4096 MAC/cycle over one macro tile, then the GELU LUT elementwise ----
template <int SLR>
static void gu_compute_tile(const gu_tile_buf_t buf, int nt,
                            const act_t xq[MLP_K],
                            const acc_t rq_g[MLP_F_SLR],
                            const acc_t rq_u[MLP_F_SLR],
                            act_t h_out[MLP_F_SLR])
{
	const int t = MLP_TILES_SLR * SLR + nt;               // global tile index

gu_pass:
	for (int np = 0; np < MLP_NPASS; ++np) {
		acc_t acc[MLP_NLANE];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=1
		for (int i = 0; i < MLP_NLANE; ++i) {
#pragma HLS UNROLL
			acc[i] = 0;
		}

	gu_ktile:
		for (int kt = 0; kt < MLP_KTILE_GU; ++kt) {
			// ---- 2048 DSP48E2, one per (column pair, K lane) ----
			dsp_t P[MLP_NLANE / MLP_PACK_WAYS][MLP_KLANE];
#pragma HLS ARRAY_PARTITION variable=P complete dim=0
			for (int d = 0; d < MLP_NLANE / MLP_PACK_WAYS; ++d) {
#pragma HLS UNROLL
				for (int kl = 0; kl < MLP_KLANE; ++kl) {
#pragma HLS UNROLL
					P[d][kl] = 0;
				}
			}

		gu_kstep:
			for (int ks = 0; ks < MLP_KSTEP; ++ks) {
#pragma HLS PIPELINE II=1
				const int wi = kt * MLP_KSTEP + ks;
				const int k0 = kt * MLP_DN + ks * MLP_KLANE;

				act_t q[MLP_KLANE];
#pragma HLS ARRAY_PARTITION variable=q complete dim=1
				for (int kl = 0; kl < MLP_KLANE; ++kl) {
#pragma HLS UNROLL
					q[kl] = xq[k0 + kl];
				}

			gu_dsp:
				for (int d = 0; d < MLP_NLANE / MLP_PACK_WAYS; ++d) {
#pragma HLS UNROLL
					wbuf_t wl = buf[np][2 * d    ][wi];
					wbuf_t wh = buf[np][2 * d + 1][wi];
					for (int kl = 0; kl < MLP_KLANE; ++kl) {
#pragma HLS UNROLL
						dsp_a_t a = dsp_pack_a(w_at(wl, kl),
						w_at(wh, kl));
						P[d][kl] += (dsp_t)(a * (dsp_b_t)q[kl]);
					}
				}
			}

			// ---- drain -> split fields -> adder tree -> acc ----
		gu_drain:
			for (int d = 0; d < MLP_NLANE / MLP_PACK_WAYS; ++d) {
#pragma HLS UNROLL
				acc_t sl = 0, sh = 0;
				for (int kl = 0; kl < MLP_KLANE; ++kl) {
#pragma HLS UNROLL
					acc_t lo, hi;
					dsp_split(P[d][kl], lo, hi);
					sl += lo;
					sh += hi;
				}
				acc[2 * d    ] += sl;
				acc[2 * d + 1] += sh;
			}
		}

		// ---- 128 fused columns done = 64 (gate, up) pairs --------
		// i is the intermediate-dimension index; SLR s covers
		// [4096s, 4096s+4096) so the local index is i - 4096s.
	gu_act:
		for (int d = 0; d < MLP_NLANE / MLP_PACK_WAYS; ++d) {
#pragma HLS PIPELINE II=1
			const int i     = t * (MLP_DN / 2) + np * MLP_PAIRS + d;
			const int local = i - SLR * MLP_F_SLR;

			// SRQ at the gate_proj / up_proj output scales
			ap_int<64> pg = (ap_int<64>)acc[2 * d    ]
			* (ap_int<64>)rq_g[local];
			ap_int<64> pu = (ap_int<64>)acc[2 * d + 1]
			* (ap_int<64>)rq_u[local];
			act_t q_g = clamp_i8((ap_int<32>)
			rne_sh<MLP_RQ_SHIFT>(pg));
			act_t q_u = clamp_i8((ap_int<32>)
			rne_sh<MLP_RQ_SHIFT>(pu));

			// G' folds dequant -> GELU -> rescale to S_H (exact)
			ap_int<32> gp = (ap_int<32>)
			(ap_int<16>)GELU_LUT_Q313[(int)q_g + 128];
			ap_int<32> hv = gp * (ap_int<32>)q_u;
			h_out[local] = clamp_i8(rne_sh<MLP_GLUT_FRAC>(hv));
		}
	}
	}

// ---- the engine: 16 macro tiles, tile load overlapped with the previous
// tile's compute.  DATAFLOW on the tile loop turns the two calls into two
// concurrent processes and gives the working set an automatic ping-pong.
template <int SLR>
static void gu_engine(hls::stream<mover_beat_t> &in,
                      const act_t  xq[MLP_K],
                      const acc_t  rq_g[MLP_F_SLR],
                      const acc_t  rq_u[MLP_F_SLR],
                      act_t h_out[MLP_F_SLR])
{
gu_tile:
	for (int nt = 0; nt < MLP_TILES_SLR; ++nt) {
#pragma HLS DATAFLOW
		gu_tile_buf_t tbuf;                       // 192 KB working set
#pragma HLS ARRAY_PARTITION variable=tbuf complete dim=2
#pragma HLS BIND_STORAGE variable=tbuf type=RAM_2P impl=URAM

		gu_load_tile(in, tbuf);
		gu_compute_tile<SLR>(tbuf, nt, xq, rq_g, rq_u, h_out);
	}
}

// ===========================================================================
// down engine -- one instance per SLR.  SLR s owns W_down[:, 4096s:+4096]
// and h[4096s:+4096]; it produces a full 1536-wide INT32 partial sum.
// ===========================================================================
#define MLP_DN_WORDS  ((MLP_F / MLP_NSLR) / 32)  // 128 x 64-bit words per row
#define MLP_DN_BPROW  (MLP_DN_WORDS / MLP_WPB_AXI)  // 32 AXI beats per row

// working set of one (n_tile, n_pass) block: 128 rows x 4096 K = 128 KB
typedef wbuf_t dn_blk_buf_t[MLP_NLANE][MLP_DN_WORDS];

// scatter one pseudo-channel's 32 B into the down working set
static inline void dn_store_pc(dn_blk_buf_t buf, int p, int r, int wo,
                               hbm_word_t w)
{
#pragma HLS INLINE
	const int nl = p * MLP_CPP_D + r;          // 0..127 inside the block
	for (int u = 0; u < 4; ++u)
		buf[nl][wo + u] = (wbuf_t)w.range(64 * u + 63, 64 * u);
}

// ---- load one 128-row block out of the mover stream ----------------------
static void dn_load_blk(hls::stream<mover_beat_t> &in, dn_blk_buf_t buf)
{
	const int BEATS = MLP_DN_BLK_BYTES / MLP_AXI_BYTES;    // 128 addresses
dn_load:
	for (int b = 0; b < BEATS; ++b) {
#pragma HLS PIPELINE II=2
		mover_beat_t lo = in.read();           // channels 0..15
		mover_beat_t hi = in.read();           // channels 16..31
		const int r  = b / MLP_DN_BPROW;       // row inside the PC block
		const int wo = (b % MLP_DN_BPROW) * MLP_WPB_AXI;
	dn_load_pc:
		for (int pp = 0; pp < MLP_BEAT_PC; ++pp) {
#pragma HLS UNROLL
			dn_store_pc(buf, pp,               r, wo, lo.w[pp]);
			dn_store_pc(buf, pp + MLP_BEAT_PC, r, wo, hi.w[pp]);
		}
	}
}

// ---- 4096 MAC/cycle over one block, K slice of this SLR only -------------
static void dn_compute_blk(const dn_blk_buf_t buf, int blk,
                           const act_t h_in[MLP_F_SLR],
                           acc_t psum[MLP_N_DOWN])
{
	const int nt = blk / MLP_NPASS;                  // n_tile 0..2
	const int np = blk % MLP_NPASS;                  // n_pass 0..3
	const int j0 = nt * MLP_DN + np * MLP_NLANE;

	acc_t acc[MLP_NLANE];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=1
	for (int i = 0; i < MLP_NLANE; ++i) {
#pragma HLS UNROLL
		acc[i] = 0;
	}

dn_ktile:
	for (int kt = 0; kt < MLP_KTILE_DN; ++kt) {
		dsp_t P[MLP_NLANE / MLP_PACK_WAYS][MLP_KLANE];
#pragma HLS ARRAY_PARTITION variable=P complete dim=0
		for (int d = 0; d < MLP_NLANE / MLP_PACK_WAYS; ++d) {
#pragma HLS UNROLL
			for (int kl = 0; kl < MLP_KLANE; ++kl) {
#pragma HLS UNROLL
				P[d][kl] = 0;
			}
		}

	dn_kstep:
		for (int ks = 0; ks < MLP_KSTEP; ++ks) {
#pragma HLS PIPELINE II=1
			const int wi = kt * MLP_KSTEP + ks;
			const int k0 = kt * MLP_DN + ks * MLP_KLANE;

			act_t q[MLP_KLANE];
#pragma HLS ARRAY_PARTITION variable=q complete dim=1
			for (int kl = 0; kl < MLP_KLANE; ++kl) {
#pragma HLS UNROLL
				q[kl] = h_in[k0 + kl];
			}

		dn_dsp:
			for (int d = 0; d < MLP_NLANE / MLP_PACK_WAYS; ++d) {
#pragma HLS UNROLL
				wbuf_t wl = buf[2 * d    ][wi];
				wbuf_t wh = buf[2 * d + 1][wi];
				for (int kl = 0; kl < MLP_KLANE; ++kl) {
#pragma HLS UNROLL
					dsp_a_t a = dsp_pack_a(w_at(wl, kl),
					w_at(wh, kl));
					P[d][kl] += (dsp_t)(a * (dsp_b_t)q[kl]);
				}
			}
		}

	dn_drain:
		for (int d = 0; d < MLP_NLANE / MLP_PACK_WAYS; ++d) {
#pragma HLS UNROLL
			acc_t sl = 0, sh = 0;
			for (int kl = 0; kl < MLP_KLANE; ++kl) {
#pragma HLS UNROLL
				acc_t lo, hi;
				dsp_split(P[d][kl], lo, hi);
				sl += lo;
				sh += hi;
			}
			acc[2 * d    ] += sl;
			acc[2 * d + 1] += sh;
		}
	}

dn_store:
	for (int nl = 0; nl < MLP_NLANE; ++nl) {
#pragma HLS PIPELINE II=1
		psum[j0 + nl] = acc[nl];
	}
}

// ---- the engine: 12 blocks, block load overlapped with the previous
// block's compute (same DATAFLOW ping-pong as the gate/up engine).
// SLR is unused in the body on purpose: the K slice and the h slice are
// already baked into the pointers.  It is kept so that the three calls are
// three distinct template instantiations, i.e. three physical engines.
template <int SLR>
static void dn_engine(hls::stream<mover_beat_t> &in,
                      const act_t h_in[MLP_F_SLR],
                      acc_t psum[MLP_N_DOWN])
{
	const int BLOCKS = MLP_TILES_DN * MLP_NPASS;           // 12
dn_blk:
	for (int blk = 0; blk < BLOCKS; ++blk) {
#pragma HLS DATAFLOW
		dn_blk_buf_t bbuf;                        // 128 KB working set
#pragma HLS ARRAY_PARTITION variable=bbuf complete dim=1
#pragma HLS BIND_STORAGE variable=bbuf type=RAM_2P impl=URAM

		dn_load_blk(in, bbuf);
		dn_compute_blk(bbuf, blk, h_in, psum);
	}
}

// ===========================================================================
// epilogue: cross-SLR add -> c_down dequant -> post_feedforward_layernorm
//           (RMSNorm, fast inverse square root) -> + residual -> Q7.8
// ===========================================================================
static void epilogue(const acc_t ps0[MLP_N_DOWN],
                     const acc_t ps1[MLP_N_DOWN],
                     const acc_t ps2[MLP_N_DOWN],
                     const raw32_t *c_down,
                     const raw32_t *ln_gamma,
                     const raw32_t *resid,
                     raw32_t *y_out)
{
	static norm_t ypre[MLP_N_DOWN];
	ssq_t ss = 0;

ep_dequant:
	for (int j = 0; j < MLP_N_DOWN; ++j) {
#pragma HLS PIPELINE II=1
		// ---- cross-SLR integer reduction (the three K slices) ----
		acc_t p = ps0[j] + ps1[j] + ps2[j];

		// ---- the single dequantization of the whole down_proj ----
		ap_int<64> v = (ap_int<64>)p * (ap_int<64>)(ap_int<32>)c_down[j];
		ap_fixed<64, 64 - MLP_CD_SHIFT> raw;
		raw.range(63, 0) = v;                    // value = v * 2^-24
		norm_t y = (norm_t)raw;
		ypre[j] = y;

		ap_fixed<64, 32> sq = y * y;
		ss += (ssq_t)sq;
	}

	// ---- RMSNorm: 1/sqrt(mean(y^2) + eps) via the shared ISCAS'25 unit ----
	const ap_ufixed<32, 0> INV_N(1.0 / (double)MLP_N_DOWN);
	ms_t  ms = (ms_t)(ss * INV_N) + (ms_t)MLP_LN_EPS;
	rsq_t rs = fast_inv_unit<ms_t, rsq_t>(ms, /*mode_recip=*/false);

ep_norm:
	for (int j = 0; j < MLP_N_DOWN; ++j) {
#pragma HLS PIPELINE II=1
		lnw_t g;                                  // (1 + gamma), Q3.13
		g.range(15, 0) = (ap_uint<16>)(ap_int<16>)(ap_int<32>)ln_gamma[j];

		ap_fixed<48, 24, AP_RND, AP_SAT> n = ypre[j] * rs;
		ap_fixed<48, 24, AP_RND, AP_SAT> w = n * g;

		hid_t r;                                  // residual, Q7.8
		r.range(15, 0) = (ap_uint<16>)(ap_int<16>)(ap_int<32>)resid[j];

		hid_t o = (hid_t)(w + r);
		y_out[j] = (raw32_t)(ap_int<16>)o.range(15, 0);
	}
}

// ===========================================================================
// Phase wrappers: each is one DATAFLOW region (mover + the three SLR engines
// run concurrently; the round-robin tile order keeps all three fed).
// ===========================================================================
#define MLP_HBM_PASS(i) w_hbm_##i,

// every array is written by exactly one process and read by exactly one
// process, as HLS DATAFLOW requires -- which is also the physical truth: the
// activation vector and the requant tables are replicated per SLR.
static void phase_gu(MLP_HBM_PORTS(MLP_HBM_ARG)
                     const act_t xq0[MLP_K],
                     const act_t xq1[MLP_K],
                     const act_t xq2[MLP_K],
                     const acc_t rg0[MLP_F_SLR], const acc_t ru0[MLP_F_SLR],
                     const acc_t rg1[MLP_F_SLR], const acc_t ru1[MLP_F_SLR],
                     const acc_t rg2[MLP_F_SLR], const acc_t ru2[MLP_F_SLR],
                     act_t h0[MLP_F_SLR],
                     act_t h1[MLP_F_SLR],
                     act_t h2[MLP_F_SLR])
{
#pragma HLS DATAFLOW
	hls::stream<mover_beat_t> s0("gu_slr0"), s1("gu_slr1"), s2("gu_slr2");
#pragma HLS STREAM variable=s0 depth=8
#pragma HLS STREAM variable=s1 depth=8
#pragma HLS STREAM variable=s2 depth=8

	mover_gu(MLP_HBM_PORTS(MLP_HBM_PASS) s0, s1, s2);
	gu_engine<0>(s0, xq0, rg0, ru0, h0);
	gu_engine<1>(s1, xq1, rg1, ru1, h1);
	gu_engine<2>(s2, xq2, rg2, ru2, h2);
}

static void phase_dn(MLP_HBM_PORTS(MLP_HBM_ARG)
                     const act_t h0[MLP_F_SLR],
                     const act_t h1[MLP_F_SLR],
                     const act_t h2[MLP_F_SLR],
                     acc_t ps0[MLP_N_DOWN],
                     acc_t ps1[MLP_N_DOWN],
                     acc_t ps2[MLP_N_DOWN])
{
#pragma HLS DATAFLOW
	hls::stream<mover_beat_t> s0("dn_slr0"), s1("dn_slr1"), s2("dn_slr2");
#pragma HLS STREAM variable=s0 depth=8
#pragma HLS STREAM variable=s1 depth=8
#pragma HLS STREAM variable=s2 depth=8

	mover_dn(MLP_HBM_PORTS(MLP_HBM_PASS) s0, s1, s2);
	dn_engine<0>(s0, h0, ps0);
	dn_engine<1>(s1, h1, ps1);
	dn_engine<2>(s2, h2, ps2);
}

// ===========================================================================
// Top-level kernel
// ===========================================================================
extern "C" {
void mlp(MLP_HBM_PORTS(MLP_HBM_ARG)
         const raw32_t *x_q,
         const raw32_t *r_g,
         const raw32_t *r_u,
         const raw32_t *c_down,
         const raw32_t *ln_gamma,
         const raw32_t *resid,
         raw32_t *y_out)
{
	// ---- 32 HBM pseudo-channels, all owned by the SLR0 data mover --------
	// (depth 13824 = MLP_CH_BYTES / 32, the per-channel beat count)
#pragma HLS INTERFACE m_axi port=w_hbm_0  offset=slave bundle=hbm0  depth=13824
#pragma HLS INTERFACE m_axi port=w_hbm_1  offset=slave bundle=hbm1  depth=13824
#pragma HLS INTERFACE m_axi port=w_hbm_2  offset=slave bundle=hbm2  depth=13824
#pragma HLS INTERFACE m_axi port=w_hbm_3  offset=slave bundle=hbm3  depth=13824
#pragma HLS INTERFACE m_axi port=w_hbm_4  offset=slave bundle=hbm4  depth=13824
#pragma HLS INTERFACE m_axi port=w_hbm_5  offset=slave bundle=hbm5  depth=13824
#pragma HLS INTERFACE m_axi port=w_hbm_6  offset=slave bundle=hbm6  depth=13824
#pragma HLS INTERFACE m_axi port=w_hbm_7  offset=slave bundle=hbm7  depth=13824
#pragma HLS INTERFACE m_axi port=w_hbm_8  offset=slave bundle=hbm8  depth=13824
#pragma HLS INTERFACE m_axi port=w_hbm_9  offset=slave bundle=hbm9  depth=13824
#pragma HLS INTERFACE m_axi port=w_hbm_10 offset=slave bundle=hbm10 depth=13824
#pragma HLS INTERFACE m_axi port=w_hbm_11 offset=slave bundle=hbm11 depth=13824
#pragma HLS INTERFACE m_axi port=w_hbm_12 offset=slave bundle=hbm12 depth=13824
#pragma HLS INTERFACE m_axi port=w_hbm_13 offset=slave bundle=hbm13 depth=13824
#pragma HLS INTERFACE m_axi port=w_hbm_14 offset=slave bundle=hbm14 depth=13824
#pragma HLS INTERFACE m_axi port=w_hbm_15 offset=slave bundle=hbm15 depth=13824
#pragma HLS INTERFACE m_axi port=w_hbm_16 offset=slave bundle=hbm16 depth=13824
#pragma HLS INTERFACE m_axi port=w_hbm_17 offset=slave bundle=hbm17 depth=13824
#pragma HLS INTERFACE m_axi port=w_hbm_18 offset=slave bundle=hbm18 depth=13824
#pragma HLS INTERFACE m_axi port=w_hbm_19 offset=slave bundle=hbm19 depth=13824
#pragma HLS INTERFACE m_axi port=w_hbm_20 offset=slave bundle=hbm20 depth=13824
#pragma HLS INTERFACE m_axi port=w_hbm_21 offset=slave bundle=hbm21 depth=13824
#pragma HLS INTERFACE m_axi port=w_hbm_22 offset=slave bundle=hbm22 depth=13824
#pragma HLS INTERFACE m_axi port=w_hbm_23 offset=slave bundle=hbm23 depth=13824
#pragma HLS INTERFACE m_axi port=w_hbm_24 offset=slave bundle=hbm24 depth=13824
#pragma HLS INTERFACE m_axi port=w_hbm_25 offset=slave bundle=hbm25 depth=13824
#pragma HLS INTERFACE m_axi port=w_hbm_26 offset=slave bundle=hbm26 depth=13824
#pragma HLS INTERFACE m_axi port=w_hbm_27 offset=slave bundle=hbm27 depth=13824
#pragma HLS INTERFACE m_axi port=w_hbm_28 offset=slave bundle=hbm28 depth=13824
#pragma HLS INTERFACE m_axi port=w_hbm_29 offset=slave bundle=hbm29 depth=13824
#pragma HLS INTERFACE m_axi port=w_hbm_30 offset=slave bundle=hbm30 depth=13824
#pragma HLS INTERFACE m_axi port=w_hbm_31 offset=slave bundle=hbm31 depth=13824

	// narrow control-plane vectors share one bundle
#pragma HLS INTERFACE m_axi port=x_q      offset=slave bundle=gmemS depth=1536
#pragma HLS INTERFACE m_axi port=r_g      offset=slave bundle=gmemS depth=12288
#pragma HLS INTERFACE m_axi port=r_u      offset=slave bundle=gmemS depth=12288
#pragma HLS INTERFACE m_axi port=c_down   offset=slave bundle=gmemS depth=1536
#pragma HLS INTERFACE m_axi port=ln_gamma offset=slave bundle=gmemS depth=1536
#pragma HLS INTERFACE m_axi port=resid    offset=slave bundle=gmemS depth=1536
#pragma HLS INTERFACE m_axi port=y_out    offset=slave bundle=gmemS depth=1536

	// no explicit s_axilite pragmas: the Vitis flow target builds the single
	// control bundle itself, and requires every scalar/offset port to live in
	// that one bundle (HLS 214-219 otherwise).

	// the activation vector is broadcast to the three SLRs, and each SLR
	// keeps the requant multipliers of the 4096-wide intermediate slice it
	// owns -- so every buffer below has exactly one producer and one consumer
	static act_t xq0[MLP_K], xq1[MLP_K], xq2[MLP_K];
	static acc_t rg0[MLP_F_SLR], rg1[MLP_F_SLR], rg2[MLP_F_SLR];
	static acc_t ru0[MLP_F_SLR], ru1[MLP_F_SLR], ru2[MLP_F_SLR];
	static act_t h0[MLP_F_SLR], h1[MLP_F_SLR], h2[MLP_F_SLR];
	static acc_t ps0[MLP_N_DOWN], ps1[MLP_N_DOWN], ps2[MLP_N_DOWN];

load_x:
	for (int k = 0; k < MLP_K; ++k) {
#pragma HLS PIPELINE II=1
		act_t v = (act_t)(ap_int<8>)(ap_int<32>)x_q[k];
		xq0[k] = v;
		xq1[k] = v;
		xq2[k] = v;
	}

load_rq:
	for (int i = 0; i < MLP_F_SLR; ++i) {
#pragma HLS PIPELINE II=1
		rg0[i] = (acc_t)r_g[i];
		rg1[i] = (acc_t)r_g[i + MLP_F_SLR];
		rg2[i] = (acc_t)r_g[i + 2 * MLP_F_SLR];
		ru0[i] = (acc_t)r_u[i];
		ru1[i] = (acc_t)r_u[i + MLP_F_SLR];
		ru2[i] = (acc_t)r_u[i + 2 * MLP_F_SLR];
	}

	// gate_proj + up_proj + GELU LUT  ->  h (INT8 @ down_proj input scale)
	phase_gu(MLP_HBM_PORTS(MLP_HBM_PASS)
	         xq0, xq1, xq2, rg0, ru0, rg1, ru1, rg2, ru2, h0, h1, h2);

	// down_proj (K split across the three SLRs)
	phase_dn(MLP_HBM_PORTS(MLP_HBM_PASS) h0, h1, h2, ps0, ps1, ps2);

	// cross-SLR add, dequant, post_feedforward_layernorm, residual
	epilogue(ps0, ps1, ps2, c_down, ln_gamma, resid, y_out);
}
}
