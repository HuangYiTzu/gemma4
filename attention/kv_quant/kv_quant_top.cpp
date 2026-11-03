#include "kv_quant_top.h"

// 暫定:ap_memory 介面;每層的 (M, B) 從常數表查(HLS 推成 ROM)。

void kv_quant_sliding(const ap_int<16> k_in[256], const ap_int<16> v_in[256], int layer, kv::elem_t k_out[256], kv::elem_t v_out[256]) {
#pragma HLS INTERFACE ap_memory port = k_in
#pragma HLS INTERFACE ap_memory port = v_in
#pragma HLS INTERFACE ap_memory port = k_out
#pragma HLS INTERFACE ap_memory port = v_out
    assert(layer >= 0 && layer < 12);
    kvq::KvQuant<256>::quant_row(k_in, v_in, KVQ_SLIDING_K_M[layer], KVQ_SLIDING_K_B[layer], KVQ_SLIDING_V_M[layer],
                                 KVQ_SLIDING_V_B[layer], k_out, v_out);
}

void kv_quant_global(const ap_int<16> k_in[512], const ap_int<16> v_in[512], int layer, kv::elem_t k_out[512], kv::elem_t v_out[512]) {
#pragma HLS INTERFACE ap_memory port = k_in
#pragma HLS INTERFACE ap_memory port = v_in
#pragma HLS INTERFACE ap_memory port = k_out
#pragma HLS INTERFACE ap_memory port = v_out
    assert(layer >= 0 && layer < 3);
    kvq::KvQuant<512>::quant_row(k_in, v_in, KVQ_GLOBAL_K_M[layer], KVQ_GLOBAL_K_B[layer], KVQ_GLOBAL_V_M[layer], KVQ_GLOBAL_V_B[layer],
                                 k_out, v_out);
}
