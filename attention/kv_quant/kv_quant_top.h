// kv_quant 的綜合測試用 top(暫定:介面型式未決,spec §8 ⛔)。
// layer = KvCache 的 layer 索引(sliding:0..11 對應 layer 0,1,2,3,5,6,7,8,10,11,12,13;global:0..2 對應 layer 4,9,14),
// 每層的 (M, B) 由 kvq_consts.h 的唯讀表查出。
#pragma once
#include "kv_quant.h"

void kv_quant_sliding(const ap_int<16> k_in[256], const ap_int<16> v_in[256], int layer, kv::elem_t k_out[256], kv::elem_t v_out[256]);
void kv_quant_global(const ap_int<16> k_in[512], const ap_int<16> v_in[512], int layer, kv::elem_t k_out[512], kv::elem_t v_out[512]);
