/*******************************************************************************
** mlp_ctrl.cpp -- KERNEL 0 of the Gemma-4-E2B MLP accelerator: the control
**                 plane.  1 compute unit, SLR0.
**
** Owns the x_q / r_g / r_u pointers and hands every engine one control block:
** x_q (identical for the three), then that engine's 4096-entry r_g slice, then
** its r_u slice, 16 INT32 per 512-bit AXIS word.  Global memory already holds
** exactly that layout, so a word is read and forwarded untouched.
**
** WHY THIS IS ITS OWN KERNEL.  It used to be a DATAFLOW process inside
** mlp_mover.  The mover is now MLP_NMV = 4 compute units of ONE kernel (see
** kernel.h), and only one of them may drive the ctrl lanes -- but v++ builds
** every CU of a kernel from the same netlist, so there is no way to give CU 1
** three AXIS ports the other three do not have.  Splitting the control path
** out is what lets the four movers stay identical.
**
** Nothing else changed: the per-stream ORDER is still x_q, r_g slice, r_u
** slice, which is what recv_ctrl in the engine expects, and the kernel still
** runs concurrently with the movers (they are separate hardware now instead of
** two processes of one DATAFLOW region).
**
** One kernel per source file: v++ copies the source into every .xo for
** software emulation, so kernels sharing a .cpp collide at link time.
*******************************************************************************/

#include "mlp_common.h"

// ###########################################################################
// KERNEL 0: mlp_ctrl  (1 CU, SLR0)
// ###########################################################################

// Everything moves in whole 512-bit words.  The previous version walked one
// INT32 per iteration and paid 16x twice:
//   o) the loop ran MLP_CTRL_VALS times instead of MLP_CTRL_WORDS times
//      (24576 cycles for r_g/r_u alone, ~60% of a whole token);
//   o) HLS could not auto-widen the accesses, because "half ? r_u[..] :
//      r_g[..]" picks the pointer at RUN TIME and the burst analysis gave up
//      ("Could not analyze pattern", 214-229).  Every 32-bit value therefore
//      consumed a full 64 B beat: 1.58 MB per token on a bus that only had to
//      carry 102 KB.
// Each loop below touches ONE pointer with a plainly increasing index, which
// is what lets HLS infer the burst AND widen it.
static void send_ctrl(const axis_word_t *x_q,
                      const axis_word_t *r_g,
                      const axis_word_t *r_u,
                      hls::stream<wstrm_t> &c0,
                      hls::stream<wstrm_t> &c1,
                      hls::stream<wstrm_t> &c2)
{
	// x_q: identical for the three engines
ctrl_x:
	for (int w = 0; w < MLP_XQ_WORDS; ++w) {
#pragma HLS PIPELINE II=1
		const axis_word_t d = x_q[w];
		axis_put(c0, d);
		axis_put(c1, d);
		axis_put(c2, d);
	}

	// r_g then r_u.  Engine s owns words [s*256, s*256+256), so ONE flat
	// sweep per vector keeps the address strictly increasing across all
	// three slices and only the destination stream changes.
ctrl_rg:
	for (int w = 0; w < MLP_RQ_WORDS; ++w) {
#pragma HLS PIPELINE II=1
		const axis_word_t d = r_g[w];
		const int s = w / MLP_RQ_WORDS_SLR;
		if      (s == 0) axis_put(c0, d);
		else if (s == 1) axis_put(c1, d);
		else             axis_put(c2, d);
	}
ctrl_ru:
	for (int w = 0; w < MLP_RQ_WORDS; ++w) {
#pragma HLS PIPELINE II=1
		const axis_word_t d = r_u[w];
		const int s = w / MLP_RQ_WORDS_SLR;
		if      (s == 0) axis_put(c0, d);
		else if (s == 1) axis_put(c1, d);
		else             axis_put(c2, d);
	}
}

extern "C" void mlp_ctrl(const axis_word_t *x_q,
                         const axis_word_t *r_g,
                         const axis_word_t *r_u,
                         hls::stream<wstrm_t> &ctrl0,
                         hls::stream<wstrm_t> &ctrl1,
                         hls::stream<wstrm_t> &ctrl2)
{
	// The three control vectors share one bundle.  depth counts POINTER
	// ELEMENTS, and the pointers are 512-bit words, so the same host buffers
	// are 96 / 768 / 768 deep instead of 1536 / 12288 / 12288.  The bundle is
	// natively 512 bit wide here, so no widening is needed at all -- every
	// beat is full by construction.
#pragma HLS INTERFACE m_axi port=x_q offset=slave bundle=gmemS depth=96
#pragma HLS INTERFACE m_axi port=r_g offset=slave bundle=gmemS depth=768
#pragma HLS INTERFACE m_axi port=r_u offset=slave bundle=gmemS depth=768

	// CU-to-CU links (v++ stream_connect, see mlp_link.cfg)
#pragma HLS INTERFACE axis port=ctrl0
#pragma HLS INTERFACE axis port=ctrl1
#pragma HLS INTERFACE axis port=ctrl2

	send_ctrl(x_q, r_g, r_u, ctrl0, ctrl1, ctrl2);
}
