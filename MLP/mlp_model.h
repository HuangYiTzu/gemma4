/*******************************************************************************
** mlp_model.h -- offline weight repacker + golden model for the Gemma-4-E2B
**                MLP accelerator.  Shared by tb_MLP.cpp (C simulation) and
**                host.cpp (Vitis / OpenCL) so that the HBM layout contract
**                has exactly ONE definition.
**
** Plain C++ only (no ap_int / ap_fixed): the host must be able to use it
** without the HLS headers.
**
** The real QAT export (google/gemma-4-E2B-it-qat-mobile-transformers) is not
** available yet, so wgen() synthesises deterministic INT2 weights and
** gen_stimulus() picks placeholder activation scales.  Replacing those two
** functions with a checkpoint reader is the only change needed later -- the
** repacking and the golden model are already the real thing.
**
** gen_stimulus() takes a test-case index so that ONE build can sweep many
** input vectors (see the MLP_TC_* block below).  The weights are deliberately
** NOT part of a test case: they live in HBM and are identical for every case,
** exactly as at decode time where only the activation and the scales change
** from token to token.
*******************************************************************************/

#ifndef MLP_MODEL_H
#define MLP_MODEL_H

#include <cmath>
#include <cstdio>

// kernel.h must be included first (it defines the geometry and pulls in the
// auto-generated gelu_luts.h with the exact G' table).

typedef unsigned char mlp_img_t[MLP_NPC][MLP_CH_BYTES];   // 32 x 432 KB

// ===========================================================================
// Placeholder weights and integer helpers
// ===========================================================================

// uniform over the INT2 alphabet {-2,-1,0,1};  kind 0 = gate, 1 = up, 2 = down
static inline int wgen(unsigned kind, unsigned a, unsigned b)
{
	unsigned h = kind * 2654435761u + a * 40503u + b * 2246822519u;
	h ^= h >> 13;  h *= 0x5bd1e995u;  h ^= h >> 15;
	return (int)(h & 3u) - 2;
}

// raw QAT byte code of one INT2 value: u = (v & 3) ^ 2, so that a byte-wide
// ^0xAA turns four packed offset codes into two's complement in one operation
static inline unsigned mlp_raw_code(int v) { return (unsigned)((v & 3) ^ 2); }

// sign-extend a 2-bit field
static inline int mlp_sext2(unsigned u) { int v = (int)(u & 3u); return (v & 2) ? v - 4 : v; }

// round-half-to-even of v * 2^-sh  (bit-identical to the kernel's rne_sh)
static inline long long mlp_rne(long long v, int sh)
{
	long long q    = v >> sh;                  // arithmetic shift == floor
	long long r    = v - (q << sh);
	long long half = 1LL << (sh - 1);
	if (r > half || (r == half && (q & 1))) ++q;
	return q;
}

static inline int mlp_clamp8(long long v)
{
	if (v >  127) return  127;
	if (v < -128) return -128;
	return (int)v;
}

// ===========================================================================
// Offline repacking -- byte-identical to the numpy reference in the spec
// ===========================================================================

// W_gate / W_up : fused[2i] = gate column i, fused[2i+1] = up column i, then
// fused ^= 0xAA, then macro tiles of DN = 512 columns are scattered over the
// 32 pseudo-channels, CPP = 16 columns each.  SLR(t) = t/16, and because the
// tiles are emitted in increasing t the three SLR ranges stay contiguous.
static void mlp_repack_gate_up(mlp_img_t img)
{
	for (int t = 0; t < MLP_TILES_GU; ++t)
		for (int p = 0; p < MLP_NPC; ++p)
			for (int j = 0; j < MLP_CPP; ++j) {
				const int c    = t * MLP_DN + p * MLP_CPP + j;   // fused column
				const int kind = c & 1;                          // 0 gate, 1 up
				const int col  = c >> 1;
				const size_t base = (size_t)t * MLP_GU_TILE_BYTES
				                  + (size_t)j * MLP_BPC;
				for (int b = 0; b < MLP_BPC; ++b) {
					unsigned byte = 0;
					for (int s = 0; s < MLP_WPB; ++s)            // k-major
						byte |= mlp_raw_code(wgen(kind, col, 4 * b + s)) << (2 * s);
					img[p][base + b] = (unsigned char)(byte ^ MLP_XOR_MASK);
				}
			}
}

// W_down : split along K.  SLR s takes the K segment [4096s, 4096s+4096),
// i.e. bytes [1024s, 1024s+1024) of every row; inside a segment the rows are
// walked n_tile / n_pass and handed out CPP_D = 4 rows per pseudo-channel.
static void mlp_repack_down(mlp_img_t img)
{
	const int SEG = (MLP_F / MLP_NSLR) / MLP_WPB;               // 1024 B
	for (int s = 0; s < MLP_NSLR; ++s)
		for (int t = 0; t < MLP_TILES_DN; ++t)
			for (int q = 0; q < MLP_NPASS; ++q) {
				const int blk = t * MLP_NPASS + q;
				for (int p = 0; p < MLP_NPC; ++p)
					for (int r = 0; r < MLP_CPP_D; ++r) {
						const int row = t * MLP_DN + q * MLP_NLANE
						              + p * MLP_CPP_D + r;
						const size_t base = (size_t)MLP_GU_CH_BYTES
						                  + (size_t)s * MLP_DN_SLR_BYTES
						                  + (size_t)blk * MLP_DN_BLK_BYTES
						                  + (size_t)r * SEG;
						for (int b = 0; b < SEG; ++b) {
							unsigned byte = 0;
							const int k0 = s * (MLP_F / MLP_NSLR) + 4 * b;
							for (int u = 0; u < MLP_WPB; ++u)
								byte |= mlp_raw_code(wgen(2, row, k0 + u)) << (2 * u);
							img[p][base + b] = (unsigned char)(byte ^ MLP_XOR_MASK);
						}
					}
			}
}

// ---------------------------------------------------------------------------
// Packing contract self-check: decode weights straight out of the HBM image
// using the kernel's own address arithmetic, compare against the generator.
// Returns the number of mismatches (0 = the layout contract holds).
// ---------------------------------------------------------------------------
static int mlp_check_packing(const mlp_img_t img)
{
	int bad = 0;

	// gate/up: fused column c -> tile c/512, channel (c%512)/16, column c%16
	for (unsigned n = 0; n < 40000u && bad < 5; ++n) {
		const int c = (int)((n * 7919u)   % MLP_N_FUSED);
		const int k = (int)((n * 104729u) % MLP_K);
		const int t = c / MLP_DN,  lc = c % MLP_DN;
		const int p = lc / MLP_CPP, j = lc % MLP_CPP;
		const size_t off = (size_t)t * MLP_GU_TILE_BYTES + (size_t)j * MLP_BPC
		                 + (size_t)(k / MLP_WPB);
		const int got = mlp_sext2(img[p][off] >> (2 * (k % MLP_WPB)));
		const int exp = wgen(c & 1, c >> 1, k);
		if (got != exp) {
			printf("MODEL-Error: gate/up packing c=%d k=%d got %d expected %d\n",
			       c, k, got, exp);
			++bad;
		}
	}

	// down: row j, K index i -> SLR i/4096, channel (j%128)/4, row j%4
	for (unsigned n = 0; n < 40000u && bad < 10; ++n) {
		const int j  = (int)((n * 7919u)   % MLP_N_DOWN);
		const int i  = (int)((n * 104729u) % MLP_F);
		const int s  = i / (MLP_F / MLP_NSLR);
		const int kk = i % (MLP_F / MLP_NSLR);
		const int t  = j / MLP_DN, q = (j % MLP_DN) / MLP_NLANE;
		const int p  = (j % MLP_NLANE) / MLP_CPP_D, r = j % MLP_CPP_D;
		const size_t off = (size_t)MLP_GU_CH_BYTES
		                 + (size_t)s * MLP_DN_SLR_BYTES
		                 + (size_t)(t * MLP_NPASS + q) * MLP_DN_BLK_BYTES
		                 + (size_t)r * ((MLP_F / MLP_NSLR) / MLP_WPB)
		                 + (size_t)(kk / MLP_WPB);
		const int got = mlp_sext2(img[p][off] >> (2 * (kk % MLP_WPB)));
		const int exp = wgen(2, j, i);
		if (got != exp) {
			printf("MODEL-Error: down packing j=%d i=%d got %d expected %d\n",
			       j, i, got, exp);
			++bad;
		}
	}
	return bad;
}

// ===========================================================================
// Stimulus -- one test case = one set of the six runtime vectors
//
// Case 0 is the original vector and stays the regression baseline.  Cases 1..5
// are hand-picked corners the baseline provably cannot reach; every case from
// MLP_TC_NB_FIXED on is a pseudo-random draw whose global dequant scale steps
// one decade, so the RMSNorm / fast-inverse-square-root unit is exercised over
// its range instead of at a single operating point.
//
// What each corner is for:
//   ZERO_X    x = 0 -> ss = 0 -> the rsqrt unit sees LN_EPS alone, the extreme
//             low end of its input range.  That takes BOTH branches the
//             baseline never reaches: pos < 23 in the leading-one detector and
//             e_y >= 0 in the fixed-point rebuild (MLP.cpp:96-97, 111).
//   ALIGN     x[k] = sign(W_gate[I0][k]) * 127 correlates the activation with
//             one weight row, so |acc| for that row reaches ~127*sum|w| ~ 1.9e5,
//             about half the K*2*127 = 390144 hard bound.  The baseline only
//             reaches 3.4% of it because wgen() and x are independent.
//   SMALL_MS  small c_down -> mean-square < 0.5: the same two branches as
//             ZERO_X but with a NON-zero y_pre, so the VALUE of rs is checked
//             instead of being multiplied by zero.
//   LARGE_MS  large c_down -> mean-square ~1e4, the low end of rsq_t.
//   SAT_OUT   |residual| ~ 124 in Q7.8 so the AP_SAT clamp on hid_t fires.
// ===========================================================================
#define MLP_TC_BASELINE   0
#define MLP_TC_ZERO_X     1
#define MLP_TC_ALIGN      2
#define MLP_TC_SMALL_MS   3
#define MLP_TC_LARGE_MS   4
#define MLP_TC_SAT_OUT    5
#define MLP_TC_NB_FIXED   6
#define MLP_TC_ALIGN_ROW  777        // intermediate channel that x is aligned to

// Default number of cases: the 6 corners plus one full pass of the decade
// sweep below.  Overridable at compile time with -DMLP_NB_TESTS=n, and on the
// host also from the command line -- csim and sw_emu are slow, so cutting the
// suite down to the 6 corners (-DMLP_NB_TESTS=6) is a reasonable smoke test.
#ifndef MLP_NB_TESTS
#define MLP_NB_TESTS      14
#endif

static const char *mlp_case_name(int tc)
{
	switch (tc) {
	case MLP_TC_BASELINE: return "baseline";
	case MLP_TC_ZERO_X:   return "zero_x";
	case MLP_TC_ALIGN:    return "align_acc";
	case MLP_TC_SMALL_MS: return "small_ms";
	case MLP_TC_LARGE_MS: return "large_ms";
	case MLP_TC_SAT_OUT:  return "sat_out";
	default:              return "random";
	}
}

// c_down is what sets mean(y^2), so walking its magnitude walks the input of
// the fast inverse-square-root unit: one step per random case, ascending.
//
// The span is bounded on purpose.  The epilogue types put a usable window on
// the RMSNorm operating point that is much narrower than the [2^-16, 2^10]
// range the rsq_t comment in kernel.h claims:
//   high ms -> rs small -> rsq_t (ap_ufixed<32,12>, 20 fractional bits, AP_TRN)
//              quantizes rs with relative step 2^-20 / rs, and that is a
//              MULTIPLICATIVE error on the whole output vector;
//   low  ms -> y_pre small -> norm_t (ap_fixed<32,16>, 16 fractional bits)
//              quantizes y_pre with relative step 2^-16 / rms(y_pre).
// The table below keeps both under ~0.2 LSB.  tb_model.cpp prints the measured
// window; widening the table is how you go looking for the actual cliff.
static const int MLP_CD_DECADE[8] = {
	100, 300, 900, 2700, 5000, 9000, 16000, 27000
};

// same finalizer as wgen(), but seeded per test case
static inline unsigned mlp_hash(unsigned a, unsigned b)
{
	unsigned h = a * 2654435761u + b * 40503u + 0x9E3779B9u;
	h ^= h >> 13;  h *= 0x5bd1e995u;  h ^= h >> 15;
	return h;
}

static void mlp_gen_stimulus(int *x_q, int *r_g, int *r_u,
                             int *c_down, int *ln_gamma, int *resid,
                             int tc = MLP_TC_BASELINE)
{
	// ---- every case starts from the original vector, then overrides -----
	for (int k = 0; k < MLP_K; ++k)
		x_q[k] = (int)(((int)((k * 2654435761u) >> 8) % 255) - 127);

	// requant multipliers, raw * 2^-20: chosen so that q_g / q_u sweep most of
	// the INT8 range and a few channels saturate (exercises the clamps)
	for (int i = 0; i < MLP_F; ++i) {
		r_g[i] = 12000 + (i % 2000);
		r_u[i] = 11000 + ((i * 3) % 2600);
	}
	for (int j = 0; j < MLP_N_DOWN; ++j) {
		c_down[j]   = 100000 + ((j * 37) % 40000);           // raw * 2^-24
		ln_gamma[j] = 8192 + ((j * 11) % 4001) - 2000;       // (1+gamma) Q3.13
		resid[j]    = ((j * 613) % 2048) - 1024;             // residual Q7.8
	}

	switch (tc) {
	case MLP_TC_BASELINE:
		break;

	case MLP_TC_ZERO_X:
		for (int k = 0; k < MLP_K; ++k) x_q[k] = 0;
		break;

	case MLP_TC_ALIGN:
		// |x| is 127 everywhere, so every OTHER channel also accumulates
		// 1.7x wider than the baseline (sigma 5563 vs 3213) -- the baseline
		// r_g / r_u keep those channels sweeping the INT8 range instead of
		// all pinning at the clamp, so the case is not degenerate.
		for (int k = 0; k < MLP_K; ++k)
			x_q[k] = (wgen(0, MLP_TC_ALIGN_ROW, k) < 0) ? -127 : 127;
		break;

	case MLP_TC_SMALL_MS:
		for (int j = 0; j < MLP_N_DOWN; ++j)
			c_down[j] = 2000 + ((j * 37) % 800);
		break;

	case MLP_TC_LARGE_MS:
		for (int j = 0; j < MLP_N_DOWN; ++j)
			c_down[j] = 600000 + ((j * 37) % 240000);
		break;

	case MLP_TC_SAT_OUT:
		for (int j = 0; j < MLP_N_DOWN; ++j)
			resid[j] = ((j & 1) ? 31800 : -31800) + ((j * 613) % 700);
		break;

	default: {                                   // pseudo-random draw
		const unsigned sd  = (unsigned)tc;
		// step one decade per random case, starting at the bottom
		const int      cd0 = MLP_CD_DECADE[(unsigned)(tc - MLP_TC_NB_FIXED) % 8u];
		for (int k = 0; k < MLP_K; ++k)
			x_q[k] = (int)(mlp_hash(sd, (unsigned)k) % 255u) - 127;
		for (int i = 0; i < MLP_F; ++i) {
			r_g[i] = 6000 + (int)(mlp_hash(sd ^ 0x11u, (unsigned)i) % 16000u);
			r_u[i] = 6000 + (int)(mlp_hash(sd ^ 0x22u, (unsigned)i) % 16000u);
		}
		for (int j = 0; j < MLP_N_DOWN; ++j) {
			c_down[j]   = cd0 + (int)(mlp_hash(sd ^ 0x33u, (unsigned)j)
			                          % (unsigned)(cd0 / 3 + 1));
			ln_gamma[j] = 8192 + (int)(mlp_hash(sd ^ 0x44u, (unsigned)j) % 8001u) - 4000;
			resid[j]    = (int)(mlp_hash(sd ^ 0x55u, (unsigned)j) % 8192u) - 4096;
		}
		break;
	}
	}
}

// ===========================================================================
// Datapath coverage of ONE test case.
//
// With several input vectors the interesting question stops being "did it
// pass" and becomes "what did this vector actually reach".  The golden model
// already walks every value, so collecting the counters here is free.
// ===========================================================================
typedef struct {
	long long acc_max;   // max |gate/up accumulator|  (hard bound K*2*127)
	long long ps_max;    // max |down partial sum|     (hard bound F*2*128)
	int  qg_sat, qu_sat; // channels where the INT8 clamp fired after the SRQ
	int  h_sat;          // channels where the INT8 clamp fired on h
	int  lut_hits;       // distinct G' entries read (out of 256)
	double ms, rs;       // RMSNorm operating point: ms and 1/sqrt(ms)
	double yn_max;       // max |y_pre * rs * (1+gamma)|, the NORMALIZED term.
	                     //   The residual is added after the rs multiply, so
	                     //   only this part carries the rs / y_pre rounding
	                     //   error -- it is what the epilogue error scales with.
	double y_min, y_max; // range of the real-valued output
	int  y_sat;          // outputs outside the Q7.8 range (AP_SAT would fire)
} mlp_cov_t;

// ===========================================================================
// Golden model
//   GEMV -> SRQ -> exact G' LUT -> GEMV : integer, bit-exact vs. the kernel
//   epilogue (dequant, RMSNorm, residual)  : double precision
// ===========================================================================
static void mlp_golden(const int *x_q, const int *r_g, const int *r_u,
                       const int *c_down, const int *ln_gamma, const int *resid,
                       int *h_out,          // [MLP_F]        INT8
                       long long *ps_out,   // [MLP_N_DOWN]   INT32 range
                       double *y_out,       // [MLP_N_DOWN]   real value
                       mlp_cov_t *cov = 0)  // optional coverage counters
{
	static unsigned char lut_seen[256];
	if (cov) {
		cov->acc_max = 0;  cov->ps_max = 0;
		cov->qg_sat  = 0;  cov->qu_sat = 0;  cov->h_sat = 0;
		cov->lut_hits = 0; cov->y_sat  = 0;
		for (int n = 0; n < 256; ++n) lut_seen[n] = 0;
	}

	for (int i = 0; i < MLP_F; ++i) {
		long long ag = 0, au = 0;
		for (int k = 0; k < MLP_K; ++k) {
			const long long q = x_q[k];
			ag += (long long)wgen(0, i, k) * q;
			au += (long long)wgen(1, i, k) * q;
		}
		const long long rg_raw = mlp_rne(ag * (long long)r_g[i], MLP_RQ_SHIFT);
		const long long ru_raw = mlp_rne(au * (long long)r_u[i], MLP_RQ_SHIFT);
		const int q_g = mlp_clamp8(rg_raw);
		const int q_u = mlp_clamp8(ru_raw);
		const long long gp = (long long)GELU_LUT_Q313[q_g + 128];
		const long long h_raw = mlp_rne(gp * (long long)q_u, MLP_GLUT_FRAC);
		h_out[i] = mlp_clamp8(h_raw);

		if (cov) {
			const long long am = ag < 0 ? -ag : ag;
			const long long um = au < 0 ? -au : au;
			if (am > cov->acc_max) cov->acc_max = am;
			if (um > cov->acc_max) cov->acc_max = um;
			if (rg_raw != q_g) ++cov->qg_sat;
			if (ru_raw != q_u) ++cov->qu_sat;
			if (h_raw != h_out[i]) ++cov->h_sat;
			lut_seen[q_g + 128] = 1;
		}
	}

	for (int j = 0; j < MLP_N_DOWN; ++j) {
		long long acc = 0;
		for (int i = 0; i < MLP_F; ++i)
			acc += (long long)wgen(2, j, i) * (long long)h_out[i];
		ps_out[j] = acc;
		if (cov) {
			const long long pm = acc < 0 ? -acc : acc;
			if (pm > cov->ps_max) cov->ps_max = pm;
		}
	}

	double ss = 0.0;
	static double ypre[MLP_N_DOWN];
	for (int j = 0; j < MLP_N_DOWN; ++j) {
		ypre[j] = (double)ps_out[j] * (double)c_down[j]
		        / (double)(1 << MLP_CD_SHIFT);
		ss += ypre[j] * ypre[j];
	}
	// ms is the operating point of the fast inverse-square-root unit.  It is
	// reported so a multi-case run can show WHICH part of that unit's input
	// range the test actually covered -- a single case only ever hits one.
	const double ms = ss / (double)MLP_N_DOWN + MLP_LN_EPS;
	const double rs = 1.0 / sqrt(ms);
	double yn_max = 0.0;
	for (int j = 0; j < MLP_N_DOWN; ++j) {
		const double g = (double)ln_gamma[j] / (double)(1 << MLP_LN_FRAC);
		const double r = (double)resid[j]    / (double)MLP_HID_SCALE;
		const double n = ypre[j] * rs * g;
		if (fabs(n) > yn_max) yn_max = fabs(n);
		y_out[j] = n + r;
	}

	if (cov) {
		const double q78_hi = 127.0 + 255.0 / 256.0, q78_lo = -128.0;
		cov->ms = ms;  cov->rs = rs;  cov->yn_max = yn_max;
		cov->y_min = y_out[0];  cov->y_max = y_out[0];
		for (int n = 0; n < 256; ++n) cov->lut_hits += lut_seen[n];
		for (int j = 0; j < MLP_N_DOWN; ++j) {
			if (y_out[j] < cov->y_min) cov->y_min = y_out[j];
			if (y_out[j] > cov->y_max) cov->y_max = y_out[j];
			if (y_out[j] > q78_hi || y_out[j] < q78_lo) ++cov->y_sat;
		}
	}
}

// saturate a real value into the Q7.8 output range (for comparisons)
static inline double mlp_sat_q78(double v)
{
	const double hi = 127.0 + 255.0 / 256.0, lo = -128.0;
	return v > hi ? hi : (v < lo ? lo : v);
}

#endif // MLP_MODEL_H
