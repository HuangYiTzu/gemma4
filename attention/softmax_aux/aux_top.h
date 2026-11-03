// exp2 / 1/ℓ 查表的綜合測試用 top(暫定:介面型式與吞吐量未決,spec §8 ⛔)。
// AUX_N = 一次處理的元素數,暫定 64(= Bc,一個 tile 的 P̃ 個數)。
#pragma once
#include "recip.h"

#define AUX_N 64

void exp2_top(const ap_int<16> d[AUX_N], ap_uint<17> out[AUX_N]);
void recip_top(const ap_uint<25> l[AUX_N], ap_uint<13> R[AUX_N], ap_int<6> w[AUX_N]);
