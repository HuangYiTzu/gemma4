/*******************************************************************************
** mlp_mover.cpp -- KERNEL 1 of the Gemma-4-E2B MLP accelerator: the weight
**                  data mover.  MLP_NMV = 4 IDENTICAL COMPUTE UNITS in SLR0.
**
** CU j owns MLP_NPC_MV = 8 HBM pseudo-channels (local AXI, zero SLL) and hands
** every engine MLP_WSTREAMS_MV = 1 of its MLP_WSTREAMS = 4 weight lanes.  The
** four CUs together deliver every HBM address of the three SLRs, interleaved,
** at 768 B/cycle: 3/4 of the 1024 B/cycle the 32 pseudo-channels could
** sustain, because the 4-lane link (kernel.h, FOUR LANES PER ENGINE) needs
** 4 cycles per address where the HBM ports need 3.
**
** NOTHING here knows which CU it is.  Which HBM banks a CU reads and which
** engine lane it drives are decided entirely by sp= and stream_connect= in
** mlp_link.cfg, so all four CUs are one netlist:
**
**     CU j  ->  HBM[8j .. 8j+7]  ->  engine lane j of engines 1..3
**
** WHY FOUR AND NOT ONE.  A single mover put every SLR-crossing net in the same
** place, and SLLs are allocated per COLUMN: route_design measured 207 % demand
** on one column against 14 % on another and refused the design (VPL 35-3,
** global congestion level 7) even though the boundary TOTAL was only 83 %.
** Four CUs, each anchored to its own contiguous block of HBM banks, start
** their crossings from four points spread across the die width.  kernel.h has
** the full argument; mlp_link.cfg pins the placement down with a pblock per CU.
**
** The split itself costs no throughput.  The lane count does: a CU issues 3
** reads per HBM port (one address per SLR) but MLP_WSUB = 4 writes per lane,
** so mv_beat runs at II = 4 (it was 3 with 8 lanes and MLP_WSUB = 2).
**
** The control plane (x_q / r_g / r_u -> the ctrl lanes) lives in its own
** kernel, mlp_ctrl.cpp: if it stayed here, CU 1 would need three AXIS ports
** the other three CUs do not have and the four could not be one kernel.
**
** One kernel per source file: v++ copies the source into every .xo for
** software emulation, so kernels sharing a .cpp collide at link time.
*******************************************************************************/

#include "mlp_common.h"

// ###########################################################################
// KERNEL 1: mlp_mover  (MLP_NMV CUs, SLR0)
// ###########################################################################

// ---- the job list: every HBM address of the three SLRs, interleaved -------
//
// The addresses are the same in every channel image and therefore in every
// mover CU -- only the SET of channels read differs, and that is a link-time
// decision.  b0 / b1 / b2 are the byte offsets of this job inside the SLR 0 /
// 1 / 2 weight region, unchanged from the single-mover version.
// the literal II=4 of mv_beat below (HLS pragmas cannot take a #define)
typedef char mlp_mv_ii_assert[(MLP_MV_II == 4) ? 1 : -1];

static void send_jobs(MLP_HBM_PORTS(MLP_HBM_ARG)
                      hls::stream<wstrm_t> &w0_0,
                      hls::stream<wstrm_t> &w1_0,
                      hls::stream<wstrm_t> &w2_0)
{
	const int CH0    = MLP_GU_CH_BYTES  / MLP_AXI_BYTES;   // down region base
	const int DN_SLR = MLP_DN_SLR_BYTES / MLP_AXI_BYTES;   // per-SLR down stride

mv_job:
	for (int job = 0; job < MLP_JOBS; ++job) {
		const bool is_gu = (job < MLP_JOBS_GU);
		const int  beats = is_gu ? MLP_GU_BEATS : MLP_DN_BEATS;

		// base address of this job in each SLR's weight region
		const int b0 = is_gu ? (MLP_TILES_SLR * 0 + job) * MLP_GU_BEATS
		                     : CH0 + 0 * DN_SLR + (job - MLP_JOBS_GU) * MLP_DN_BEATS;
		const int b1 = is_gu ? (MLP_TILES_SLR * 1 + job) * MLP_GU_BEATS
		                     : CH0 + 1 * DN_SLR + (job - MLP_JOBS_GU) * MLP_DN_BEATS;
		const int b2 = is_gu ? (MLP_TILES_SLR * 2 + job) * MLP_GU_BEATS
		                     : CH0 + 2 * DN_SLR + (job - MLP_JOBS_GU) * MLP_DN_BEATS;

		// one address for every SLR per iteration: 3 reads per HBM port and
		// MLP_WSUB = 4 writes per lane, so the lane sets II = 4 (MLP_MV_II)
		// -- 8 ports x 32 B x 3/4 = 192 B/cycle here, 768 B/cycle once the
		// four CUs are counted.  (With 8 lanes: II = 3, 1024 B/cycle.)
	mv_beat:
		for (int b = 0; b < beats; ++b) {
#pragma HLS PIPELINE II=4
#pragma HLS LOOP_TRIPCOUNT min=128 max=192
			hbm_word_t c0[MLP_NPC_MV], c1[MLP_NPC_MV], c2[MLP_NPC_MV];
#pragma HLS ARRAY_PARTITION variable=c0 complete dim=1
#pragma HLS ARRAY_PARTITION variable=c1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=c2 complete dim=1

#define MLP_RD(i) c0[i] = w_hbm_##i[b0 + b]; \
                  c1[i] = w_hbm_##i[b1 + b]; \
                  c2[i] = w_hbm_##i[b2 + b];
			MLP_HBM_PORTS(MLP_RD)
#undef MLP_RD

			send_addr(c0, w0_0);
			send_addr(c1, w1_0);
			send_addr(c2, w2_0);
		}
	}
}

extern "C" void mlp_mover(MLP_HBM_PORTS(MLP_HBM_ARG)
                          hls::stream<wstrm_t> &w0_0,
                          hls::stream<wstrm_t> &w1_0,
                          hls::stream<wstrm_t> &w2_0)
{
	// ---- this CU's MLP_NPC_MV pseudo-channels, all local to SLR0 ---------
	// The ports are numbered from zero in EVERY CU; mlp_link.cfg decides
	// which HBM banks CU j actually sees.  depth still counts the whole
	// channel image (13824 = MLP_CH_BYTES / 32): the split is across
	// channels, not within one.
#pragma HLS INTERFACE m_axi port=w_hbm_0 offset=slave bundle=hbm0 depth=13824
#pragma HLS INTERFACE m_axi port=w_hbm_1 offset=slave bundle=hbm1 depth=13824
#pragma HLS INTERFACE m_axi port=w_hbm_2 offset=slave bundle=hbm2 depth=13824
#pragma HLS INTERFACE m_axi port=w_hbm_3 offset=slave bundle=hbm3 depth=13824
#pragma HLS INTERFACE m_axi port=w_hbm_4 offset=slave bundle=hbm4 depth=13824
#pragma HLS INTERFACE m_axi port=w_hbm_5 offset=slave bundle=hbm5 depth=13824
#pragma HLS INTERFACE m_axi port=w_hbm_6 offset=slave bundle=hbm6 depth=13824
#pragma HLS INTERFACE m_axi port=w_hbm_7 offset=slave bundle=hbm7 depth=13824

	// CU-to-CU links (v++ stream_connect, see mlp_link.cfg)
#pragma HLS INTERFACE axis port=w0_0
#pragma HLS INTERFACE axis port=w1_0
#pragma HLS INTERFACE axis port=w2_0

	send_jobs(MLP_HBM_PORTS(MLP_HBM_PASS)
	          w0_0, w1_0, w2_0);
}
