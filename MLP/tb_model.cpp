/*******************************************************************************
** tb_model.cpp -- host-only regression + coverage harness for the multi-case
**                 stimulus.  Plain C++: it compiles and runs WITHOUT Vitis,
**                 without HLS headers and without an FPGA.
**
**   g++ -O2 -DMLP_HOST_ONLY -o tb_model tb_model.cpp
**   ./tb_model [nb_cases]
**
** What it does:
**   1) builds the 32 HBM channel images once and runs the packing contract
**      self-check (the weights do not depend on the test case);
**   2) runs the golden model for every test case and prints what that case
**      actually reached in the datapath -- accumulator magnitude, the three
**      INT8 clamps, how many G' entries were read, the RMSNorm operating
**      point, and whether the Q7.8 output saturated;
**   3) prints the union over all cases, i.e. the coverage the whole suite buys.
**
** This is the cheap gate: if a case does not move the numbers here, it will
** not move them in csim or sw_emu either, and there is no point paying the
** simulation time for it.
*******************************************************************************/

#include <cstdio>
#include <cstdlib>
#include <cmath>

#ifndef MLP_HOST_ONLY
#define MLP_HOST_ONLY
#endif
#include "kernel.h"
#include "mlp_model.h"

static mlp_img_t  img;
static int        x_q[MLP_K], r_g[MLP_F], r_u[MLP_F];
static int        c_down[MLP_N_DOWN], ln_gamma[MLP_N_DOWN], resid[MLP_N_DOWN];
static int        g_h[MLP_F];
static long long  g_ps[MLP_N_DOWN];
static double     g_y[MLP_N_DOWN];

// hard bounds the geometry guarantees, independent of the data
static const double ACC_BOUND = (double)MLP_K * 2.0 * 127.0;        // 390144
static const double PS_BOUND  = (double)MLP_F * 2.0 * 128.0;        // 3145728

int main(int argc, char **argv)
{
	const int NB = (argc > 1) ? atoi(argv[1]) : MLP_NB_TESTS;
	if (NB < 1) { printf("MODEL-Error: nb_cases must be >= 1\n"); return 1; }

	printf("MODEL-Info: Gemma-4-E2B MLP  K=%d  F=%d, INT2 weights\n", MLP_K, MLP_F);
	printf("MODEL-Info: host-only golden/coverage harness, %d test cases\n\n", NB);

	// ---- the weights are the same for every case: pack and check once ----
	printf("MODEL-Info: repacking weights into %d HBM images ... ", MLP_NPC);
	fflush(stdout);
	mlp_repack_gate_up(img);
	mlp_repack_down(img);
	printf("done\n");
	if (mlp_check_packing(img)) {
		printf("MODEL-Error: packing contract self-check FAILED\n");
		return 1;
	}
	printf("MODEL-Info: packing contract self-check : ok (80000 probes)\n\n");

	// ---- per-case coverage ------------------------------------------------
	mlp_cov_t  u;                                  // union over all cases
	u.acc_max = 0; u.ps_max = 0;
	u.qg_sat = 0; u.qu_sat = 0; u.h_sat = 0; u.lut_hits = 0; u.y_sat = 0;
	u.ms = 1e300; u.rs = 0.0; u.y_min = 1e300; u.y_max = -1e300; u.yn_max = 0.0;
	double ms_hi = 0.0, rs_lo = 1e300;

	// Predicted epilogue error, in output LSB.  Only the NORMALIZED term
	// carries these: the residual is added after the rs multiply.
	//   rsq_t  ap_ufixed<32,12>: step 2^-20 on rs -> yn * 2^-20 / rs
	//   norm_t ap_fixed<32,16> : step 2^-16 on y_pre, and rms(y_pre) ~ 1/rs,
	//                            so the output error is yn * 2^-16 * rs
	const double LSB_ = 1.0 / (double)MLP_HID_SCALE;
	double e_rsq_hi = 0.0, e_nrm_hi = 0.0;

	printf("MODEL-Info: ----------------------------------------------------------------------------------------------------------\n");
	printf("MODEL-Info: case  name       |acc|max   |ps|max  qg_sat qu_sat h_sat  LUT    mean-sq     rsqrt  y_sat  e_rsq  e_norm\n");
	printf("MODEL-Info:                                                                                            (LSB)  (LSB)\n");
	printf("MODEL-Info: ----------------------------------------------------------------------------------------------------------\n");

	for (int tc = 0; tc < NB; ++tc) {
		mlp_cov_t c;
		mlp_gen_stimulus(x_q, r_g, r_u, c_down, ln_gamma, resid, tc);
		mlp_golden(x_q, r_g, r_u, c_down, ln_gamma, resid, g_h, g_ps, g_y, &c);

		const double e_rsq = c.yn_max * pow(2.0, -20) / c.rs / LSB_;
		const double e_nrm = c.yn_max * pow(2.0, -16) * c.rs / LSB_;
		if (e_rsq > e_rsq_hi) e_rsq_hi = e_rsq;
		if (e_nrm > e_nrm_hi) e_nrm_hi = e_nrm;

		printf("MODEL-Info: %4d  %-10s %8lld  %8lld  %5d  %5d  %4d %3d/256 %10.3e %9.3e %5d %6.3f %6.3f\n",
		       tc, mlp_case_name(tc), c.acc_max, c.ps_max,
		       c.qg_sat, c.qu_sat, c.h_sat, c.lut_hits, c.ms, c.rs, c.y_sat,
		       e_rsq, e_nrm);

		if (c.acc_max > u.acc_max) u.acc_max = c.acc_max;
		if (c.ps_max  > u.ps_max)  u.ps_max  = c.ps_max;
		u.qg_sat += c.qg_sat;  u.qu_sat += c.qu_sat;  u.h_sat += c.h_sat;
		u.y_sat  += c.y_sat;
		if (c.ms < u.ms) u.ms = c.ms;
		if (c.ms > ms_hi) ms_hi = c.ms;
		if (c.rs > u.rs) u.rs = c.rs;
		if (c.rs < rs_lo) rs_lo = c.rs;
		if (c.y_min < u.y_min) u.y_min = c.y_min;
		if (c.y_max > u.y_max) u.y_max = c.y_max;
		if (c.lut_hits > u.lut_hits) u.lut_hits = c.lut_hits;
		if (c.yn_max > u.yn_max) u.yn_max = c.yn_max;
	}

	printf("MODEL-Info: -------------------------------------------------------------------------------------------\n\n");

	printf("MODEL-Info: ===== coverage of the whole suite =====\n");
	printf("MODEL-Info: |acc| reached      : %lld  = %.1f%% of the K*2*127 = %.0f hard bound\n",
	       u.acc_max, 100.0 * (double)u.acc_max / ACC_BOUND, ACC_BOUND);
	printf("MODEL-Info: |psum| reached     : %lld  = %.2f%% of the F*2*128 = %.0f hard bound\n",
	       u.ps_max, 100.0 * (double)u.ps_max / PS_BOUND, PS_BOUND);
	printf("MODEL-Info: INT8 clamps fired  : q_g %d, q_u %d, h %d\n",
	       u.qg_sat, u.qu_sat, u.h_sat);
	printf("MODEL-Info: Q7.8 AP_SAT fired  : %d outputs\n", u.y_sat);
	printf("MODEL-Info: output range       : [%.3f, %.3f]\n", u.y_min, u.y_max);
	printf("MODEL-Info: rsqrt input  (ms)  : %.3e .. %.3e   (%.1f octaves)\n",
	       u.ms, ms_hi, log(ms_hi / u.ms) / log(2.0));
	printf("MODEL-Info: rsqrt output (rs)  : %.3e .. %.3e\n", rs_lo, u.rs);

	// ---- which fast_inv_unit branches the suite reaches --------------------
	// ms_t is ap_ufixed<56,32>, i.e. 24 fractional bits, so the leading-one
	// position is pos = floor(log2(ms)) + 24 and the "pos < 23" branch needs
	// ms < 0.5.  The rebuild takes "e_y >= 0" when rs >= 1, i.e. ms <= 1.
	const bool br_lo = (u.ms   <  0.5);
	const bool br_hi = (ms_hi  >= 0.5);
	const bool br_ey_pos = (u.rs  >= 1.0);
	const bool br_ey_neg = (rs_lo <  1.0);
	printf("MODEL-Info: fast_inv_unit branch coverage\n");
	printf("MODEL-Info:   leading-one  pos <  23 (ms < 0.5) : %s\n", br_lo ? "COVERED" : "not reached");
	printf("MODEL-Info:   leading-one  pos >= 23            : %s\n", br_hi ? "COVERED" : "not reached");
	printf("MODEL-Info:   rebuild      e_y >= 0 (rs >= 1)   : %s\n", br_ey_pos ? "COVERED" : "not reached");
	printf("MODEL-Info:   rebuild      e_y <  0             : %s\n", br_ey_neg ? "COVERED" : "not reached");

	// ---- how much epilogue headroom the suite is using --------------------
	// The Q7.8 rounding floor alone is 0.5 LSB, and the host tolerance is
	// 4 LSB, so anything above ~1 LSB here means the epilogue types, not the
	// output format, have become the dominant error source.
	printf("MODEL-Info: predicted epilogue error over the suite\n");
	printf("MODEL-Info:   rsq_t  (ap_ufixed<32,12>) : worst %.3f LSB  (at rs = %.3e)\n",
	       e_rsq_hi, rs_lo);
	printf("MODEL-Info:   norm_t (ap_fixed<32,16>)  : worst %.3f LSB  (at rs = %.3e)\n",
	       e_nrm_hi, u.rs);
	printf("MODEL-Info:   Q7.8 rounding floor       :       0.500 LSB   <- irreducible\n");
	if (e_rsq_hi > 1.0 || e_nrm_hi > 1.0)
		printf("MODEL-Warn: an epilogue type now dominates the error; widen it or\n"
		       "MODEL-Warn: narrow MLP_CD_DECADE before trusting these cases.\n");

	// The comment on rsq_t in kernel.h claims [2^-16, 2^10].  At rs = 2^-16
	// the relative step of a 20-fraction-bit type is 2^-20/2^-16 = 6.25%, so
	// the low end of that claimed range is not usable at this width.
	printf("MODEL-Info: note: rsq_t is declared for rs in [2^-16, 2^10], but with 20\n");
	printf("MODEL-Info:       fractional bits the relative step at rs = 2^-16 is 6.25%%.\n");
	printf("MODEL-Info:       Keeping the epilogue under 0.5 LSB needs rs >~ %.2e here.\n",
	       u.yn_max * pow(2.0, -20) / (0.5 * LSB_));

	const bool ok = (br_lo && br_hi && br_ey_pos && br_ey_neg &&
	                 u.qg_sat > 0 && u.qu_sat > 0 && u.h_sat > 0 && u.y_sat > 0);
	printf("\nMODEL-Info: %s\n", ok ? "Coverage goals met"
	                               : "Coverage goals NOT met (see above)");
	return ok ? 0 : 1;
}
