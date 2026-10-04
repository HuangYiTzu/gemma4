/*******************************************************************************
** mlp_collect.cpp -- KERNEL 3: the cross-SLR reduction, 1 compute unit, SLR0.
**
** Reads the three INT32 partial-sum lanes in lockstep (all engines emit the
** same block order), adds them, applies the per-channel c_down multiplier,
** rounds half to even, clamps to INT16 and writes y_q.
**
** One kernel per source file: v++ copies the source into every .xo for
** software emulation, so kernels sharing a .cpp collide at link time.
*******************************************************************************/

#include "mlp_common.h"

// ###########################################################################
// KERNEL 3: mlp_collect  (1 CU, SLR0)
//
//   y_q[j] = clamp_int16( rne( (ps0+ps1+ps2)[j] * c_down[j], 2^-MLP_CD_SHIFT ) )
//
// c_down[j] = round(ws_down[j] * S_H / S_Y16 * 2^24) folds the per-channel
// down_proj weight scale and the output activation scale into one INT32
// multiplier, so no real value exists anywhere in the kernel.  |p| < 2^22 and
// c_down < 2^31, so the 64-bit product never wraps.
// ###########################################################################
extern "C" void mlp_collect(const raw32_t *c_down,
                            raw32_t *y_q,
                            hls::stream<pstrm_t> &ps0,
                            hls::stream<pstrm_t> &ps1,
                            hls::stream<pstrm_t> &ps2)
{
#pragma HLS INTERFACE axis port=ps0
#pragma HLS INTERFACE axis port=ps1
#pragma HLS INTERFACE axis port=ps2
#pragma HLS INTERFACE m_axi port=c_down offset=slave bundle=gmemC depth=1536
#pragma HLS INTERFACE m_axi port=y_q    offset=slave bundle=gmemC depth=1536

rq_blk:
	for (int blk = 0; blk < MLP_JOBS_DN; ++blk) {
		// the engines walk the down blocks in this order, 128 lanes each
		const int j0 = (blk / MLP_NPASS) * MLP_DN
		             + (blk % MLP_NPASS) * MLP_NLANE;
	rq_down:
		for (int nl = 0; nl < MLP_NLANE; ++nl) {
#pragma HLS PIPELINE II=1
			// ---- cross-SLR integer reduction (the three K slices) ----
			acc_t p = (acc_t)(ap_int<32>)ps0.read()
			        + (acc_t)(ap_int<32>)ps1.read()
			        + (acc_t)(ap_int<32>)ps2.read();

			// ---- requant: INT32 partial sum -> INT16 code at S_Y16 ----
			ap_int<64> v = (ap_int<64>)p
			             * (ap_int<64>)(ap_int<32>)c_down[j0 + nl];
			y16_t      y = clamp_i16(rne_sh<MLP_CD_SHIFT>(v));
			y_q[j0 + nl] = (raw32_t)y;           // sign-extended on the bus
		}
	}
}
