/*******************************************************************************
** mlp_model.h -- checkpoint reader + golden model for the Gemma-4-E2B MLP
**                accelerator.  Shared by tb_MLP.cpp (C simulation),
**                tb_model.cpp (host-only harness) and host.cpp (Vitis/OpenCL)
**                so that the HBM layout contract has exactly ONE definition.
**
** Plain C++ only (no ap_int / ap_fixed): the host must be able to use it
** without the HLS headers.  kernel.h must be included first (geometry +
** the auto-generated gelu_luts.h with the exact G' table and the scales).
**
** ============================================================================
** INPUT DATA FORMAT  (bin_packing_mlp.py export of the QAT checkpoint
**                     google/gemma-4-E2B-it-qat-mobile-transformers, "wNa8o8")
** ============================================================================
** Directory : mlp_bin_output/layer_NN/   (NN = two-digit layer index, here 15;
**             layer 15 is an INT2 layer with F = intermediate_size = 12288,
**             K = hidden_size = 1536, N_DOWN = 1536).  manifest.json at the
**             top level repeats per layer: num_bits, F, gate_up_meta {BPC 384,
**             N 24576, CPP 16, DN 512, NPC 32, magic 170}, down_meta {SEG 1024,
**             CPP_D 4, n_lane 128, magic 170} and the activation scales.
**
** All files are raw little-endian byte streams with no header.  The weight
** files are byte arrays (no multi-byte words), the scale files are IEEE-754
** float32 little-endian (numpy '<f4', native on the x86 exporter; the values
** decode to the expected 1e-2 .. 1e-1 range only with that byte order).
**
** 1) gate_up_weight_pcPP.bin      PP = 00..31 (HBM pseudo-channel), 294912 B
**    ----------------------------------------------------------------------
**    The gate_proj.weight and up_proj.weight tensors are stored packed in the
**    checkpoint as uint8 [F, K/4] = [12288, 384] (4 INT2 codes per byte,
**    K-major).  The exporter zips them column-wise into one fused matrix
**        fused[2i]   = gate_proj.weight[i]      (i = intermediate channel)
**        fused[2i+1] = up_proj.weight[i]
**    of N = 2F = 24576 rows x BPC = 384 bytes, XORs every byte with the
**    "magic" 0xAA (= MLP_XOR_MASK) and scatters macro tiles of DN = 512 rows
**    over the 32 pseudo-channels, CPP = 16 rows each.  Tiles are emitted in
**    increasing t (SLR order 0..15, 16..31, 32..47 = identity), so
**        file offset  t*6144 + j*384 + b   <->   fused row t*512 + p*16 + j,
**                                               byte b (K indices 4b .. 4b+3)
**    which is exactly the gate/up region of HBM channel image p in kernel.h
**    (MLP_GU_TILE_BYTES = 6144, MLP_BPC = 384, MLP_GU_CH_BYTES = 294912).
**
**    INT2 packing inside a byte (unpack_to_int in the exporter): field s of
**    byte b, bits [2s+1 : 2s], is K index 4b + s -- i.e. the LOWEST two bits
**    hold the lowest K index (little-endian nibble order).
**    Sign convention: the checkpoint code u in {0,1,2,3} is OFFSET BINARY,
**    value = u - 2.  The ^0xAA flips bit 1 of every field, which turns the
**    offset code into 2-bit TWO'S COMPLEMENT:  00 = 0, 01 = +1, 10 = -2,
**    11 = -1.  The files therefore hold two's complement INT2 already and are
**    consumed by the kernel (w_at) and by this reader (mlp_sext2) with NO
**    further XOR.  Real weight = code * weight_scale[row].
**
** 2) down_weight_slrS_pcPP.bin    S = 0..2, PP = 00..31, 49152 B each
**    ----------------------------------------------------------------------
**    down_proj.weight is uint8 [N_DOWN, F/4] = [1536, 3072], same packing and
**    the same ^0xAA.  It is split along K (= F) into three 1024-byte column
**    segments, SLR s owning bytes [1024s, 1024s+1024) = K indices
**    [4096s, 4096s+4096).  Inside a segment the rows are walked as
**    (n_tile t = 0..2, n_pass q = 0..3) blocks of 128 rows, block index
**    blk = 4t + q, and pseudo-channel p receives CPP_D = 4 rows per block:
**        file offset  blk*4096 + r*1024 + b   <->   row 512t + 128q + 4p + r,
**                                                   K index 4096s + 4b + field
**    Channel image p = gate_up_weight_pcPP ++ down_weight_slr0_pcPP ++
**    down_weight_slr1_pcPP ++ down_weight_slr2_pcPP (294912 + 3 x 49152 =
**    442368 B = MLP_CH_BYTES), so the files ARE the HBM images: nothing is
**    repacked on the host any more.
**
** 3) gate_up_scale_pcPP.bin       PP = 00..31, 3072 B = 768 x float32 LE
**    ----------------------------------------------------------------------
**    Per-output-channel weight scales in the SAME fused / tiled order as the
**    weights (repack_scale_gate_up mirrors repack_and_compile_generic):
**        float index  t*16 + j   <->   fused column t*512 + p*16 + j,
**    even fused columns = gate_proj.weight_scale[c/2], odd = up_proj.weight_
**    scale[c/2].  The exporter builds the fused vector as float32 [N, 1]
**    regardless of the checkpoint dtype.  Layer 15: ws_gate in [0.011, 0.084],
**    ws_up in [0.014, 0.064].
**
** 4) down_scale_slrS.bin          S = 0..2, 6144 B = 1536 x float32 LE
**    ----------------------------------------------------------------------
**    down_proj.weight_scale[j], j = output row 0..1535, natural order (the
**    tensor's own bytes: 6144 / 1536 = 4 B -> float32).  Three IDENTICAL
**    copies, one per SLR ("option A" in the exporter); the reader checks that
**    they agree.  Layer 15: ws_down in [0.0055, 0.038].
**
** 5) activation_scales.json / activation_scales.bin
**    ----------------------------------------------------------------------
**    Six per-tensor activation scales (scalar float32 tensors of the
**    checkpoint).  JSON key order == .bin order, 6 x float32 LE = 24 B:
**        [0] gate_proj_input   = S_IN  (INT8 grid of x_q; == [2])
**        [1] gate_proj_output  = S_G   (grid of q_g)
**        [2] up_proj_input     = S_IN
**        [3] up_proj_output    = S_U   (grid of q_u)
**        [4] down_proj_input   = S_H   (grid of h)
**        [5] down_proj_output  = S_Y   (INT8 grid of y; kernel emits S_Y/256)
**    gen_luts.py bakes these five values into gelu_luts.h (MLP_S_*) because
**    the G' table depends on them; the reader refuses a directory whose
**    scales differ from the compiled header.
**
** ============================================================================
** Every stage from x_q to y_q is integer arithmetic, so the golden model
** predicts the kernel output EXACTLY.  The real-valued computations here are
** (a) the semantic check that y_q * S_Y16 really is W_down . h, and (b) a
** floating-point FAKE-QUANT reference of the checkpoint's own quantized
** forward pass, which differs from the integer datapath only by the 2^-20
** rounding of the SRQ multipliers.
**
** A test case is one activation vector x_q; the weights AND scales come from
** the checkpoint and are identical for every case, exactly as at decode time
** where only the activation changes from token to token.
*******************************************************************************/

#ifndef MLP_MODEL_H
#define MLP_MODEL_H

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

typedef unsigned char mlp_img_t[MLP_NPC][MLP_CH_BYTES];   // 32 x 432 KB

// ===========================================================================
// Integer helpers
// ===========================================================================

// sign-extend a 2-bit two's complement field (the file convention, see above)
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

static inline int mlp_clamp16(long long v)
{
	if (v > MLP_Y16_MAX) return MLP_Y16_MAX;
	if (v < MLP_Y16_MIN) return MLP_Y16_MIN;
	return (int)v;
}

// GELU tanh approximation (gelu_pytorch_tanh, as used by Gemma / gen_luts.py)
static inline double mlp_gelu_tanh(double x)
{
	return 0.5 * x * (1.0 + tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)));
}

// Host side of the SRQ requant:  r = round( ws * S_IN / S_out * 2^MLP_RQ_SHIFT )
// and of the down_proj requant:  c_down = round( ws_down * S_H / S_Y16 * 2^24 )
// Both return -1 when the multiplier does not fit INT32 -- a configuration
// error (scale grid too coarse for the weight scale), not a runtime condition.
static inline int mlp_rq_from_ws(double ws, double s_in, double s_out)
{
	const double v = ws * s_in / s_out * (double)(1 << MLP_RQ_SHIFT);
	if (!(v >= 0.0) || v > 2147483647.0) return -1;
	return (int)floor(v + 0.5);
}

static inline int mlp_cd_from_ws(double ws_down)
{
	const double v = ws_down * MLP_S_H / MLP_S_Y16 * (double)(1 << MLP_CD_SHIFT);
	if (!(v >= 0.0) || v > 2147483647.0) return -1;
	return (int)floor(v + 0.5);
}

// ===========================================================================
// The decoded layer
// ===========================================================================
typedef struct {
	signed char *w_gate;          // [MLP_F][MLP_K]       INT2 codes -2..1
	signed char *w_up;            // [MLP_F][MLP_K]
	signed char *w_down;          // [MLP_N_DOWN][MLP_F]
	float  ws_gate[MLP_F];        // gate_proj.weight_scale   (per channel)
	float  ws_up[MLP_F];          // up_proj.weight_scale
	float  ws_down[MLP_N_DOWN];   // down_proj.weight_scale
	float  act[6];                // activation_scales.bin, JSON key order
	char   dir[1024];             // where it came from
} mlp_layer_t;

static inline int mlp_wg(const mlp_layer_t *L, int i, int k) { return L->w_gate[(size_t)i * MLP_K + k]; }
static inline int mlp_wu(const mlp_layer_t *L, int i, int k) { return L->w_up  [(size_t)i * MLP_K + k]; }
static inline int mlp_wd(const mlp_layer_t *L, int j, int i) { return L->w_down[(size_t)j * MLP_F + i]; }

// ---------------------------------------------------------------------------
// Locate the export directory.  Order: $MLP_LAYER_DIR, the compile-time
// -DMLP_LAYER_DIR=path (UNQUOTED, no spaces -- Vitis strips quotes from
// -cflags on Windows, so the macro is stringified here; run_hls.tcl passes
// the absolute path), then mlp_bin_output/layer_NN relative to the cwd and
// up to six parent levels (Vitis csim runs in proj/sol/csim/build).
// ---------------------------------------------------------------------------
static bool mlp_dir_ok(const char *dir)
{
	char p[1200];
	snprintf(p, sizeof p, "%s/activation_scales.bin", dir);
	FILE *f = fopen(p, "rb");
	if (!f) return false;
	fclose(f);
	return true;
}

static const char *mlp_find_layer_dir(char *out, size_t n)
{
	const char *env = getenv("MLP_LAYER_DIR");
	if (env && mlp_dir_ok(env)) { snprintf(out, n, "%s", env); return out; }
#ifdef MLP_LAYER_DIR
#define MLP_STR2_(x) #x
#define MLP_STR_(x)  MLP_STR2_(x)
	if (mlp_dir_ok(MLP_STR_(MLP_LAYER_DIR))) {
		snprintf(out, n, "%s", MLP_STR_(MLP_LAYER_DIR));
		return out;
	}
#endif
	char rel[256] = "";
	for (int up = 0; up <= 6; ++up) {
		snprintf(out, n, "%smlp_bin_output/layer_%02d", rel, MLP_LAYER_IDX);
		if (mlp_dir_ok(out)) return out;
		strncat(rel, "../", sizeof rel - strlen(rel) - 1);
	}
	snprintf(out, n, "mlp_bin_output/layer_%02d", MLP_LAYER_IDX);
	return out;                                // caller reports the failure
}

// ---------------------------------------------------------------------------
// read exactly `bytes` bytes of a file (the file must be exactly that long)
// ---------------------------------------------------------------------------
static int mlp_read_exact(const char *dir, const char *name, void *dst, size_t bytes)
{
	char p[1300];
	snprintf(p, sizeof p, "%s/%s", dir, name);
	FILE *f = fopen(p, "rb");
	if (!f) { printf("MODEL-Error: cannot open %s\n", p); return -1; }
	fseek(f, 0, SEEK_END);
	const long sz = ftell(f);
	fseek(f, 0, SEEK_SET);
	if (sz != (long)bytes) {
		printf("MODEL-Error: %s is %ld B, expected %lu B\n", p, sz, (unsigned long)bytes);
		fclose(f);
		return -1;
	}
	const size_t got = fread(dst, 1, bytes, f);
	fclose(f);
	if (got != bytes) { printf("MODEL-Error: short read on %s\n", p); return -1; }
	return 0;
}

// ---------------------------------------------------------------------------
// Decode the INT2 matrices out of the channel images by walking the files in
// the exporter's WRITE order (sequential).  mlp_check_packing() below re-reads
// random weights through the kernel's closed-form address arithmetic instead,
// so the two derivations of the layout cross-check each other.
// ---------------------------------------------------------------------------
static void mlp_decode_weights(const mlp_img_t img, mlp_layer_t *L)
{
	// gate/up: tile t, channel p, row j, byte b  (see format item 1)
	for (int t = 0; t < MLP_TILES_GU; ++t)
		for (int p = 0; p < MLP_NPC; ++p)
			for (int j = 0; j < MLP_CPP; ++j) {
				const int c   = t * MLP_DN + p * MLP_CPP + j;   // fused column
				const int col = c >> 1;
				signed char *dst = (c & 1) ? L->w_up + (size_t)col * MLP_K
				                           : L->w_gate + (size_t)col * MLP_K;
				const unsigned char *src = &img[p][(size_t)t * MLP_GU_TILE_BYTES
				                                   + (size_t)j * MLP_BPC];
				for (int b = 0; b < MLP_BPC; ++b)
					for (int s = 0; s < MLP_WPB; ++s)
						dst[MLP_WPB * b + s] = (signed char)mlp_sext2(src[b] >> (2 * s));
			}

	// down: SLR s, block (t,q), channel p, row r, byte b  (format item 2)
	const int SEG = (MLP_F / MLP_NSLR) / MLP_WPB;                // 1024 B
	for (int s = 0; s < MLP_NSLR; ++s)
		for (int t = 0; t < MLP_TILES_DN; ++t)
			for (int q = 0; q < MLP_NPASS; ++q) {
				const int blk = t * MLP_NPASS + q;
				for (int p = 0; p < MLP_NPC; ++p)
					for (int r = 0; r < MLP_CPP_D; ++r) {
						const int row = t * MLP_DN + q * MLP_NLANE + p * MLP_CPP_D + r;
						const unsigned char *src = &img[p][(size_t)MLP_GU_CH_BYTES
						    + (size_t)s * MLP_DN_SLR_BYTES
						    + (size_t)blk * MLP_DN_BLK_BYTES + (size_t)r * SEG];
						signed char *dst = L->w_down + (size_t)row * MLP_F
						                 + (size_t)s * (MLP_F / MLP_NSLR);
						for (int b = 0; b < SEG; ++b)
							for (int u = 0; u < MLP_WPB; ++u)
								dst[MLP_WPB * b + u] = (signed char)mlp_sext2(src[b] >> (2 * u));
					}
			}
}

// ---------------------------------------------------------------------------
// Load one exported layer: the 32 channel images (weights), the per-channel
// weight scales and the activation scales.  Returns 0 on success.
// ---------------------------------------------------------------------------
static int mlp_load_layer(const char *dir, mlp_img_t img, mlp_layer_t *L)
{
	char name[64];
	snprintf(L->dir, sizeof L->dir, "%s", dir);

	// ---- weights: the files are the HBM channel images ----
	for (int p = 0; p < MLP_NPC; ++p) {
		snprintf(name, sizeof name, "gate_up_weight_pc%02d.bin", p);
		if (mlp_read_exact(dir, name, img[p], MLP_GU_CH_BYTES)) return -1;
		for (int s = 0; s < MLP_NSLR; ++s) {
			snprintf(name, sizeof name, "down_weight_slr%d_pc%02d.bin", s, p);
			if (mlp_read_exact(dir, name,
			                   img[p] + MLP_GU_CH_BYTES + (size_t)s * MLP_DN_SLR_BYTES,
			                   MLP_DN_SLR_BYTES)) return -1;
		}
	}

	// ---- gate/up weight scales: fused/tiled order, float32 LE ----
	{
		static float pc[MLP_TILES_GU * MLP_CPP];
		for (int p = 0; p < MLP_NPC; ++p) {
			snprintf(name, sizeof name, "gate_up_scale_pc%02d.bin", p);
			if (mlp_read_exact(dir, name, pc, sizeof pc)) return -1;
			for (int t = 0; t < MLP_TILES_GU; ++t)
				for (int j = 0; j < MLP_CPP; ++j) {
					const int c = t * MLP_DN + p * MLP_CPP + j;
					if (c & 1) L->ws_up  [c >> 1] = pc[t * MLP_CPP + j];
					else       L->ws_gate[c >> 1] = pc[t * MLP_CPP + j];
				}
		}
	}

	// ---- down weight scale: three identical copies, natural row order ----
	{
		static float cp[MLP_N_DOWN];
		if (mlp_read_exact(dir, "down_scale_slr0.bin", L->ws_down, sizeof cp)) return -1;
		for (int s = 1; s < MLP_NSLR; ++s) {
			snprintf(name, sizeof name, "down_scale_slr%d.bin", s);
			if (mlp_read_exact(dir, name, cp, sizeof cp)) return -1;
			if (memcmp(cp, L->ws_down, sizeof cp)) {
				printf("MODEL-Error: %s differs from down_scale_slr0.bin\n", name);
				return -1;
			}
		}
	}

	// ---- activation scales: must be the ones gelu_luts.h was built from ----
	if (mlp_read_exact(dir, "activation_scales.bin", L->act, sizeof L->act)) return -1;
	{
		const float want[6] = { (float)MLP_S_IN, (float)MLP_S_G, (float)MLP_S_IN,
		                        (float)MLP_S_U,  (float)MLP_S_H, (float)MLP_S_Y };
		static const char *nm[6] = { "gate_proj_input", "gate_proj_output",
		                             "up_proj_input",   "up_proj_output",
		                             "down_proj_input", "down_proj_output" };
		for (int n = 0; n < 6; ++n)
			if (L->act[n] != want[n]) {
				printf("MODEL-Error: %s = %.9g in %s, but gelu_luts.h was generated "
				       "for %.9g -- run: python gen_luts.py --layer-dir %s\n",
				       nm[n], L->act[n], dir, want[n], dir);
				return -1;
			}
	}

	// ---- weight matrices for the golden model ----
	if (!L->w_gate) L->w_gate = (signed char *)malloc((size_t)MLP_F * MLP_K);
	if (!L->w_up)   L->w_up   = (signed char *)malloc((size_t)MLP_F * MLP_K);
	if (!L->w_down) L->w_down = (signed char *)malloc((size_t)MLP_N_DOWN * MLP_F);
	if (!L->w_gate || !L->w_up || !L->w_down) {
		printf("MODEL-Error: out of memory for the decoded weights\n");
		return -1;
	}
	mlp_decode_weights(img, L);
	return 0;
}

static void mlp_free_layer(mlp_layer_t *L)
{
	free(L->w_gate); free(L->w_up); free(L->w_down);
	L->w_gate = L->w_up = L->w_down = 0;
}

// ---------------------------------------------------------------------------
// Packing contract self-check: decode weights straight out of the HBM image
// using the kernel's own (closed-form) address arithmetic and compare them
// with the sequentially decoded matrices.  Returns the number of mismatches
// (0 = both derivations of the layout agree).
// ---------------------------------------------------------------------------
static int mlp_check_packing(const mlp_img_t img, const mlp_layer_t *L)
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
		const int exp = (c & 1) ? mlp_wu(L, c >> 1, k) : mlp_wg(L, c >> 1, k);
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
		const int exp = mlp_wd(L, j, i);
		if (got != exp) {
			printf("MODEL-Error: down packing j=%d i=%d got %d expected %d\n",
			       j, i, got, exp);
			++bad;
		}
	}
	return bad;
}

// ===========================================================================
// Runtime vectors derived from the checkpoint scales (once per layer)
//
//     r_g[i]    = round( ws_gate[i] * S_IN / S_G * 2^20 )
//     r_u[i]    = round( ws_up[i]   * S_IN / S_U * 2^20 )
//     c_down[j] = round( ws_down[j] * S_H / S_Y16 * 2^24 )
//
// Returns the number of multipliers that do not fit INT32 (0 = ok).
// ===========================================================================
static int mlp_layer_requant(const mlp_layer_t *L, int *r_g, int *r_u, int *c_down)
{
	int bad = 0;
	for (int i = 0; i < MLP_F; ++i) {
		r_g[i] = mlp_rq_from_ws(L->ws_gate[i], MLP_S_IN, MLP_S_G);
		r_u[i] = mlp_rq_from_ws(L->ws_up[i],   MLP_S_IN, MLP_S_U);
		if (r_g[i] < 0 || r_u[i] < 0) ++bad;
	}
	for (int j = 0; j < MLP_N_DOWN; ++j) {
		c_down[j] = mlp_cd_from_ws(L->ws_down[j]);
		if (c_down[j] < 0) ++bad;
	}
	return bad;
}

// ===========================================================================
// Stimulus -- one test case = one INT8 activation vector x_q (grid S_IN)
//
// Case 0 is the regression baseline.  Cases 1..4 are hand-picked corners,
// every case from MLP_TC_NB_FIXED on is a pseudo-random draw whose amplitude
// steps through MLP_AMP_DECADE.
//
//   BASELINE  deterministic hash, full INT8 range: p ~ 1e4 .. 1e5, most
//             channels reach the INT16 rails through the layer's gains.
//   ZERO_X    x = 0 -> p = 0 -> y_q = 0 on every channel.
//   ALIGN     x[k] = sign(W_gate[I0][k]) * 127 correlates the activation with
//             one real weight row, so |acc| for that row is ~127*sum|w|, i.e.
//             a good fraction of the K*2*127 = 390144 hard bound.
//   SMALL_X   |x| <= 4: q_g / q_u stay within a few codes of zero, no INT8
//             clamp fires, and the products / roundings near zero (negative
//             half-ties of the SRQ and of G'*q_u) are what gets checked.
//   GAUSS     sum of four uniforms, sigma ~ 49 codes (~1.1 real): the shape
//             of a post-RMSNorm activation at the calibrated S_IN, i.e. the
//             operating point of the layer rather than a stress test.
// ===========================================================================
#define MLP_TC_BASELINE   0
#define MLP_TC_ZERO_X     1
#define MLP_TC_ALIGN      2
#define MLP_TC_SMALL_X    3
#define MLP_TC_GAUSS      4
#define MLP_TC_NB_FIXED   5
#define MLP_TC_ALIGN_ROW  777        // intermediate channel that x is aligned to

// Default number of cases: the 5 corners plus one full pass of the amplitude
// sweep below.  Overridable at compile time with -DMLP_NB_TESTS=n, and on the
// host also from the command line -- csim and sw_emu are slow, so cutting the
// suite down to the 5 corners (-DMLP_NB_TESTS=5) is a reasonable smoke test.
#ifndef MLP_NB_TESTS
#define MLP_NB_TESTS      11
#endif

static const char *mlp_case_name(int tc)
{
	switch (tc) {
	case MLP_TC_BASELINE: return "baseline";
	case MLP_TC_ZERO_X:   return "zero_x";
	case MLP_TC_ALIGN:    return "align_acc";
	case MLP_TC_SMALL_X:  return "small_x";
	case MLP_TC_GAUSS:    return "gauss";
	default:              return "random";
	}
}

// amplitude (max |x_q|) per random case, ascending
static const int MLP_AMP_DECADE[6] = { 8, 16, 32, 64, 96, 127 };

static inline unsigned mlp_hash(unsigned a, unsigned b)
{
	unsigned h = a * 2654435761u + b * 40503u + 0x9E3779B9u;
	h ^= h >> 13;  h *= 0x5bd1e995u;  h ^= h >> 15;
	return h;
}

static void mlp_gen_x(const mlp_layer_t *L, int *x_q, int tc = MLP_TC_BASELINE)
{
	for (int k = 0; k < MLP_K; ++k)
		x_q[k] = (int)(((int)((k * 2654435761u) >> 8) % 255) - 127);

	switch (tc) {
	case MLP_TC_BASELINE:
		break;

	case MLP_TC_ZERO_X:
		for (int k = 0; k < MLP_K; ++k) x_q[k] = 0;
		break;

	case MLP_TC_ALIGN:
		for (int k = 0; k < MLP_K; ++k)
			x_q[k] = (mlp_wg(L, MLP_TC_ALIGN_ROW, k) < 0) ? -127 : 127;
		break;

	case MLP_TC_SMALL_X:
		for (int k = 0; k < MLP_K; ++k)
			x_q[k] = (int)(mlp_hash(3u, (unsigned)k) % 9u) - 4;
		break;

	case MLP_TC_GAUSS:
		for (int k = 0; k < MLP_K; ++k) {
			int s = 0;                       // 4 x U(-42,42): sigma ~ 49
			for (unsigned n = 0; n < 4; ++n)
				s += (int)(mlp_hash(4u + 16u * n, (unsigned)k) % 85u) - 42;
			x_q[k] = mlp_clamp8(s);
		}
		break;

	default: {                                   // pseudo-random draw
		const unsigned sd  = (unsigned)tc;
		const int      amp = MLP_AMP_DECADE[(unsigned)(tc - MLP_TC_NB_FIXED) % 6u];
		for (int k = 0; k < MLP_K; ++k)
			x_q[k] = (int)(mlp_hash(sd, (unsigned)k) % (unsigned)(2 * amp + 1)) - amp;
		break;
	}
	}
}

// ===========================================================================
// Datapath coverage of ONE test case.
// ===========================================================================
typedef struct {
	long long acc_max;    // max |gate/up accumulator|  (hard bound K*2*127)
	long long ps_max;     // max |down partial sum|     (hard bound F*2*128)
	int  qg_sat, qu_sat;  // channels where the INT8 clamp fired after the SRQ
	int  h_sat;           // channels where the INT8 clamp fired on h
	int  lut_hits;        // distinct G' entries read (out of 256)
	int  y_min, y_max;    // range of the INT16 output codes
	int  ysat_pos;        // outputs clamped to +32767
	int  ysat_neg;        // outputs clamped to -32768
	int  tie_up, tie_dn;  // exact .5 ties rne rounded up (odd floor) / kept (even)
	double sem_err;       // max |y_q - p*ws_down*S_H/S_Y16| over unclamped
	                      //   outputs, in output LSB
	double sem_bound;     // what sem_err may reach: 0.5 (output rounding)
	                      //   + |p|max * 2^-25 (c_down rounded to an integer)
	// fake-quant reference (double precision forward pass of the checkpoint's
	// quantized model: round(acc*ws*S_IN/S_G) etc., same clamps)
	int    fq_h_mis;      // channels where h differs from the reference
	int    fq_h_maxd;     // largest |h - h_ref| among them
	double fq_y_err;      // max |y_q - y_ref/S_Y16| over unclamped outputs, LSB
	double fq_y_rms;      // rms of the same, LSB
} mlp_cov_t;

// ===========================================================================
// Golden model -- integer end to end, bit-exact vs. the kernel
//   GEMV -> SRQ -> exact G' LUT -> GEMV -> x c_down -> rne -> INT16 clamp
// ===========================================================================
static void mlp_golden(const mlp_layer_t *L,
                       const int *x_q, const int *r_g, const int *r_u,
                       const int *c_down,
                       int *h_out,                 // [MLP_F]       INT8
                       long long *ps_out,          // [MLP_N_DOWN]  INT32 range
                       int *y_code,                // [MLP_N_DOWN]  INT16 code
                       mlp_cov_t *cov = 0)         // optional coverage counters
{
	static unsigned char lut_seen[256];
	static signed char   h_ref[MLP_F];             // fake-quant reference h
	if (cov) {
		cov->acc_max = 0;  cov->ps_max = 0;
		cov->qg_sat  = 0;  cov->qu_sat = 0;  cov->h_sat = 0;
		cov->lut_hits = 0;
		cov->y_min = MLP_Y16_MAX;  cov->y_max = MLP_Y16_MIN;
		cov->ysat_pos = 0;  cov->ysat_neg = 0;
		cov->tie_up = 0;    cov->tie_dn = 0;
		cov->sem_err = 0.0; cov->sem_bound = 0.0;
		cov->fq_h_mis = 0;  cov->fq_h_maxd = 0;
		cov->fq_y_err = 0.0; cov->fq_y_rms = 0.0;
		for (int n = 0; n < 256; ++n) lut_seen[n] = 0;
	}

	for (int i = 0; i < MLP_F; ++i) {
		long long ag = 0, au = 0;
		const signed char *wg = L->w_gate + (size_t)i * MLP_K;
		const signed char *wu = L->w_up   + (size_t)i * MLP_K;
		for (int k = 0; k < MLP_K; ++k) {
			ag += (long long)wg[k] * x_q[k];
			au += (long long)wu[k] * x_q[k];
		}
		const long long rg_raw = mlp_rne(ag * (long long)r_g[i], MLP_RQ_SHIFT);
		const long long ru_raw = mlp_rne(au * (long long)r_u[i], MLP_RQ_SHIFT);
		const int q_g = mlp_clamp8(rg_raw);
		const int q_u = mlp_clamp8(ru_raw);
		const long long gp = (long long)GELU_LUT_I16[q_g + 128];
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

			// fake-quant reference: the checkpoint's quantized forward in
			// double precision (round half to even, same INT8 clamps)
			const double g  = (double)ag * L->ws_gate[i] * MLP_S_IN;
			const double u  = (double)au * L->ws_up[i]   * MLP_S_IN;
			const int qg_r  = mlp_clamp8((long long)nearbyint(g / MLP_S_G));
			const int qu_r  = mlp_clamp8((long long)nearbyint(u / MLP_S_U));
			const double hr = mlp_gelu_tanh(qg_r * MLP_S_G) * (qu_r * MLP_S_U) / MLP_S_H;
			h_ref[i] = (signed char)mlp_clamp8((long long)nearbyint(hr));
			const int d = abs(h_out[i] - h_ref[i]);
			if (d) { ++cov->fq_h_mis; if (d > cov->fq_h_maxd) cov->fq_h_maxd = d; }
		}
	}

	for (int j = 0; j < MLP_N_DOWN; ++j) {
		long long acc = 0;
		const signed char *wd = L->w_down + (size_t)j * MLP_F;
		for (int i = 0; i < MLP_F; ++i)
			acc += (long long)wd[i] * (long long)h_out[i];
		ps_out[j] = acc;
		if (cov) {
			const long long pm = acc < 0 ? -acc : acc;
			if (pm > cov->ps_max) cov->ps_max = pm;
		}
	}

	// ---- down_proj requant: INT32 partial sum -> INT16 code at S_Y16 ----
	const long long HALF = 1LL << (MLP_CD_SHIFT - 1);
	const long long MASK = (1LL << MLP_CD_SHIFT) - 1;
	int    n_unclamped = 0;
	double fq_sq = 0.0;
	for (int j = 0; j < MLP_N_DOWN; ++j) {
		const long long v   = ps_out[j] * (long long)c_down[j];
		const long long raw = mlp_rne(v, MLP_CD_SHIFT);
		y_code[j] = mlp_clamp16(raw);

		if (!cov) continue;
		if ((v & MASK) == HALF) {
			if ((v >> MLP_CD_SHIFT) & 1) ++cov->tie_up;
			else                         ++cov->tie_dn;
		}
		if (raw > MLP_Y16_MAX) ++cov->ysat_pos;
		if (raw < MLP_Y16_MIN) ++cov->ysat_neg;
		if (y_code[j] < cov->y_min) cov->y_min = y_code[j];
		if (y_code[j] > cov->y_max) cov->y_max = y_code[j];

		if (raw == y_code[j]) {
			// semantic check: y_q * S_Y16 must be the real down_proj output
			// p * ws_down * S_H (h is at S_H, W_down's real value is w * ws)
			const double ref = (double)ps_out[j] * L->ws_down[j] * MLP_S_H / MLP_S_Y16;
			const double e   = fabs((double)y_code[j] - ref);
			if (e > cov->sem_err) cov->sem_err = e;

			// fake-quant reference: W_down . h_ref, real value in LSB16
			long long pr = 0;
			const signed char *wd = L->w_down + (size_t)j * MLP_F;
			for (int i = 0; i < MLP_F; ++i) pr += (long long)wd[i] * h_ref[i];
			const double yr = (double)pr * L->ws_down[j] * MLP_S_H / MLP_S_Y16;
			const double f  = fabs((double)y_code[j] - yr);
			if (f > cov->fq_y_err) cov->fq_y_err = f;
			fq_sq += f * f;
			++n_unclamped;
		}
	}
	if (cov) {
		for (int n = 0; n < 256; ++n) cov->lut_hits += lut_seen[n];
		cov->sem_bound = 0.5 + (double)cov->ps_max * pow(2.0, -(MLP_CD_SHIFT + 1))
		               + 1e-6;
		cov->fq_y_rms = n_unclamped ? sqrt(fq_sq / n_unclamped) : 0.0;
	}
}

#endif // MLP_MODEL_H
