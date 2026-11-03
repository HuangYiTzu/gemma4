#include "aux_top.h"

// 暫定:ap_memory 介面、PIPELINE II=1(吞吐量與 DSP 預算未決)。要改 stream/AXI 只動這個檔。

void exp2_top(const ap_int<16> d[AUX_N], ap_uint<17> out[AUX_N]) {
#pragma HLS INTERFACE ap_memory port = d
#pragma HLS INTERFACE ap_memory port = out
exp2_loop:
    for (int i = 0; i < AUX_N; i++) {
#pragma HLS PIPELINE II = 1
        out[i] = aux::Exp2<>::eval(d[i]);
    }
}

void recip_top(const ap_uint<25> l[AUX_N], ap_uint<13> R[AUX_N], ap_int<6> w[AUX_N]) {
#pragma HLS INTERFACE ap_memory port = l
#pragma HLS INTERFACE ap_memory port = R
#pragma HLS INTERFACE ap_memory port = w
recip_loop:
    for (int i = 0; i < AUX_N; i++) {
#pragma HLS PIPELINE II = 1
        ap_uint<13> r;
        ap_int<6> ww;
        aux::Recip<>::eval(l[i], r, ww);
        R[i] = r;
        w[i] = ww;
    }
}
