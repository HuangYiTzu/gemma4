#include "kv_cache_top.h"

// 暫定:ap_memory 介面;記憶體為 static 陣列,依 bank(dim 2)完全分割以便一次平行讀 NB 個 word。
// 記憶體型態(URAM/BRAM)尚未綁定,等 URAM 預算與 pragma 寫法確認後再加。

void kv_cache_sliding(ap_uint<1> wr, int layer, int pos, const kv::elem_t k_in[256], const kv::elem_t v_in[256], int g,
                      int chunk, kv_word_t k_out[8], kv_word_t v_out[8]) {
#pragma HLS INTERFACE ap_memory port = k_in
#pragma HLS INTERFACE ap_memory port = v_in
#pragma HLS INTERFACE ap_memory port = k_out
#pragma HLS INTERFACE ap_memory port = v_out
    typedef kv::KvCache<256, 512, 12, 1> C;
    static C::word_t k_mem[12][8][C::DEPTH];
    static C::word_t v_mem[12][8][C::DEPTH];
#pragma HLS ARRAY_PARTITION variable = k_mem complete dim = 2
#pragma HLS ARRAY_PARTITION variable = v_mem complete dim = 2
    if (wr) {
        C::write_row(k_mem, v_mem, layer, pos, k_in, v_in);
    } else {
        C::read_group(k_mem, layer, g, chunk, k_out);
        C::read_group(v_mem, layer, g, chunk, v_out);
    }
}

void kv_cache_global(ap_uint<1> wr, int layer, int pos, const kv::elem_t k_in[512], const kv::elem_t v_in[512], int g,
                     int chunk, kv_word_t k_out[8], kv_word_t v_out[8]) {
#pragma HLS INTERFACE ap_memory port = k_in
#pragma HLS INTERFACE ap_memory port = v_in
#pragma HLS INTERFACE ap_memory port = k_out
#pragma HLS INTERFACE ap_memory port = v_out
    typedef kv::KvCache<512, 4096, 3, 0> C;
    static C::word_t k_mem[3][8][C::DEPTH];
    static C::word_t v_mem[3][8][C::DEPTH];
#pragma HLS ARRAY_PARTITION variable = k_mem complete dim = 2
#pragma HLS ARRAY_PARTITION variable = v_mem complete dim = 2
    if (wr) {
        C::write_row(k_mem, v_mem, layer, pos, k_in, v_in);
    } else {
        C::read_group(k_mem, layer, g, chunk, k_out);
        C::read_group(v_mem, layer, g, chunk, v_out);
    }
}
