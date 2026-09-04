/*******************************************************************************
** tb_MLP.cpp -- Vitis HLS C-simulation testbench for the Gemma-4-E2B MLP layer
**
**   1) builds the 32 HBM pseudo-channel images with the offline repacker in
**      mlp_model.h (fused gate/up zip + ^0xAA bias removal + macro-tile
**      scatter; K-split for down_proj) -- the same code the host uses;
**   2) checks the packing contract directly: 80000 randomly chosen weights
**      are decoded straight out of the HBM image with the kernel's own
**      address arithmetic and compared against the generator;
**   3) runs the kernel ONCE PER TEST CASE and compares each 1536-wide Q7.8
**      output against the golden model.
**
** The weights are packed and converted to the AXI view exactly once: they do
** not depend on the test case (which mirrors decode time, where the weights
** are resident in HBM and only the activation / scales change per token).  A
** test case is therefore just one draw of the six runtime vectors; see the
** MLP_TC_* block in mlp_model.h for what each corner is meant to reach.
**
** The golden GEMV / requant / GELU-LUT path is EXACT integer arithmetic, so
** any layout, DSP field-split or accumulation error becomes a gross mismatch.
** The measured residual is 0.501 LSB, i.e. the Q7.8 output rounding floor
** alone: the fixed-point epilogue (dequant, fast inverse square root) adds
** about 0.001 LSB on top of it.
**
** csim is slow, so the suite size is a compile-time knob:
**     -DMLP_NB_TESTS=6    the six hand-picked corners only (quick)
**     -DMLP_NB_TESTS=14   default: corners + one full decade sweep
** tb_model.cpp runs the same cases through the golden model alone in seconds
** and reports what each one covers -- use that first, this second.
**
** The real QAT weights/scales are not exported yet, so every weight and scale
** is a chosen placeholder value and only functional correctness is verified.
*******************************************************************************/

#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "kernel.h"
#include "mlp_model.h"

// ---------------------------------------------------------------------------
// tolerances on the Q7.8 output (1 LSB = 1/256 = 3.9e-3)
//   measured: MAE = 0.257 LSB, max |err| = 0.501 LSB
// ---------------------------------------------------------------------------
static const double LSB         = 1.0 / MLP_HID_SCALE;
static const double MAX_ABS_TOL = 2.0 * LSB;
static const double MAE_TOL     = 0.5 * LSB;

// ---------------------------------------------------------------------------
// Images and vectors (static: far too large for the stack)
// ---------------------------------------------------------------------------
static mlp_img_t  img;                                    // 32 x 432 KB bytes
static hbm_word_t ch[MLP_NPC][MLP_CH_WORDS];              // 256-bit AXI view

static int        x_q[MLP_K], r_g[MLP_F], r_u[MLP_F];
static int        c_down[MLP_N_DOWN], ln_gamma[MLP_N_DOWN], resid[MLP_N_DOWN];

static raw32_t    k_x[MLP_K], k_rg[MLP_F], k_ru[MLP_F];
static raw32_t    k_cd[MLP_N_DOWN], k_ln[MLP_N_DOWN], k_rs[MLP_N_DOWN];
static raw32_t    k_y [MLP_N_DOWN];

static int        g_h [MLP_F];
static long long  g_ps[MLP_N_DOWN];
static double     g_y [MLP_N_DOWN];

// byte image -> 256-bit AXI words (little endian, as on any AXI master)
static void image_to_axi()
{
	for (int p = 0; p < MLP_NPC; ++p)
		for (int w = 0; w < MLP_CH_WORDS; ++w)
			for (int u = 0; u < 4; ++u) {
				unsigned long long v = 0;
				const unsigned char *s =
				    &img[p][(size_t)w * MLP_AXI_BYTES + 8 * u];
				for (int n = 7; n >= 0; --n) v = (v << 8) | s[n];
				ch[p][w].range(64 * u + 63, 64 * u) = v;
			}
}

// ---- run one test case: stimulus -> golden -> kernel -> compare -----------
// Returns true if the case passes; fills *mae_o / *max_o for the summary.
static bool run_case(int tc, double *mae_o, double *max_o, mlp_cov_t *cov)
{
	mlp_gen_stimulus(x_q, r_g, r_u, c_down, ln_gamma, resid, tc);
	mlp_golden(x_q, r_g, r_u, c_down, ln_gamma, resid, g_h, g_ps, g_y, cov);

	for (int k = 0; k < MLP_K;       ++k) k_x [k] = (raw32_t)x_q[k];
	for (int i = 0; i < MLP_F;       ++i) { k_rg[i] = (raw32_t)r_g[i];
	                                        k_ru[i] = (raw32_t)r_u[i]; }
	for (int j = 0; j < MLP_N_DOWN;  ++j) { k_cd[j] = (raw32_t)c_down[j];
	                                        k_ln[j] = (raw32_t)ln_gamma[j];
	                                        k_rs[j] = (raw32_t)resid[j]; }

#define TB_CH(i) ch[i],
	mlp(MLP_HBM_PORTS(TB_CH) k_x, k_rg, k_ru, k_cd, k_ln, k_rs, k_y);
#undef TB_CH

	double max_err = 0.0, sum_err = 0.0;
	int    worst_j = 0, nb_over = 0, nb_printed = 0;

	for (int j = 0; j < MLP_N_DOWN; ++j) {
		const double hw   = (double)(int)k_y[j] / (double)MLP_HID_SCALE;
		const double gold = mlp_sat_q78(g_y[j]);
		const double err  = fabs(hw - gold);

		sum_err += err;
		if (err > max_err) { max_err = err; worst_j = j; }
		if (err > MAX_ABS_TOL) {
			++nb_over;
			if (nb_printed++ < 10)
				printf("TB-Error: case %d j=%4d  hw=%+.6f  gold=%+.6f  err=%.3e\n",
				       tc, j, hw, gold, err);
		}
	}

	const double mae = sum_err / MLP_N_DOWN;
	*mae_o = mae;  *max_o = max_err;
	(void)worst_j;
	return (nb_over == 0) && (mae <= MAE_TOL);
}

int main()
{
	printf("TB-Info: Gemma-4-E2B MLP  K=%d  F=%d  INT2 weights, decode stage\n",
	       MLP_K, MLP_F);
	printf("TB-Info: %d macro tiles of %dx%d (%d per SLR), "
	       "%d HBM PCs x %d B, %dx%d MAC/cycle per SLR\n",
	       MLP_TILES_GU, MLP_DN, MLP_DN, MLP_TILES_SLR,
	       MLP_NPC, MLP_CH_BYTES, MLP_NLANE, MLP_KLANE);
	printf("TB-Info: %d test cases (build with -DMLP_NB_TESTS=n to change)\n",
	       MLP_NB_TESTS);

	// ---- the weights do not depend on the test case: pack them once ------
	mlp_repack_gate_up(img);
	mlp_repack_down(img);
	if (mlp_check_packing(img)) {
		printf("TB-Error: Test Failed (HBM packing contract)\n");
		return 1;
	}
	printf("TB-Info: packing contract self-check: ok\n");
	image_to_axi();

	int nb_failed = 0;
	double worst_mae = 0.0, worst_max = 0.0;
	int    worst_mae_tc = 0, worst_max_tc = 0;

	printf("TB-Info: -----------------------------------------------------------------------------\n");
	printf("TB-Info: case  name         MAE(LSB)  max(LSB)   mean-sq      rsqrt   clamps(g/u/h)  res\n");
	printf("TB-Info: -----------------------------------------------------------------------------\n");

	for (int tc = 0; tc < MLP_NB_TESTS; ++tc) {
		double mae = 0.0, mx = 0.0;
		mlp_cov_t cov;
		const bool ok = run_case(tc, &mae, &mx, &cov);

		printf("TB-Info: %4d  %-10s %9.3f %9.3f %10.3e %10.3e %5d/%5d/%4d  %s\n",
		       tc, mlp_case_name(tc), mae / LSB, mx / LSB,
		       cov.ms, cov.rs, cov.qg_sat, cov.qu_sat, cov.h_sat,
		       ok ? "PASS" : "FAIL");

		if (!ok) ++nb_failed;
		if (mae / LSB > worst_mae) { worst_mae = mae / LSB; worst_mae_tc = tc; }
		if (mx  / LSB > worst_max) { worst_max = mx  / LSB; worst_max_tc = tc; }
	}

	printf("TB-Info: -----------------------------------------------------------------------------\n");
	printf("TB-Info: outputs per case = %d (Q7.8, 1 LSB = %.3e)\n", MLP_N_DOWN, LSB);
	printf("TB-Info: tolerance        : MAE <= %.3f LSB, max <= %.3f LSB\n",
	       MAE_TOL / LSB, MAX_ABS_TOL / LSB);
	printf("TB-Info: worst MAE        : %.3f LSB (case %d %s)\n",
	       worst_mae, worst_mae_tc, mlp_case_name(worst_mae_tc));
	printf("TB-Info: worst max |err|  : %.3f LSB (case %d %s)\n",
	       worst_max, worst_max_tc, mlp_case_name(worst_max_tc));

	if (nb_failed == 0) printf("TB-Info: Test Successful (%d/%d cases)\n",
	                           MLP_NB_TESTS, MLP_NB_TESTS);
	else                printf("TB-Error: Test Failed (%d/%d cases over tolerance)\n",
	                           nb_failed, MLP_NB_TESTS);
	return nb_failed ? 1 : 0;
}
