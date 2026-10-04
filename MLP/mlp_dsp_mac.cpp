/*******************************************************************************
** mlp_dsp_mac.cpp -- C side of the mlp_dsp_mac RTL blackbox (mlp_dsp_mac.v).
**
** Listed as "c_files" in the blackbox JSON that mlp_blackbox.tcl writes.  It
** is NOT one of the kernel sources and must not be added with add_files.
**
** The blackbox keeps its accumulator INSIDE the DSP, so the real behaviour is
** stateful (P <= (first ? 0 : P) + a*q) and cannot be written as a function
** of the call's arguments alone.  The simulation model that the C simulation,
** sw_emu and the host-side half of hw_emu actually run is therefore
** mlp_dsp_mac_model() in mlp_common.h, which carries the accumulator in an
** explicit per-DSP state variable; mlp_engine.cpp reaches either one through
** the MLP_MAC() macro (model when __SYNTHESIS__ is undefined, blackbox when it
** is defined).
**
** This definition only exists because the blackbox flow wants a C function
** with the blackbox's signature.  Nothing calls it.  It returns one step of a
** freshly started accumulator (the "first" case), which is also what the RTL
** returns for any call with first = 1.
*******************************************************************************/

#include <ap_int.h>

ap_int<32> mlp_dsp_mac(ap_int<2> w_lo, ap_int<2> w_hi, ap_int<8> q,
                       ap_uint<1> first)
{
	(void)first;
	const ap_int<27> a = ((ap_int<27>)w_hi << 18) + (ap_int<27>)w_lo;
	return (ap_int<32>)(a * q);
}
