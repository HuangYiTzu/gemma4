// KV cache 的綜合測試用 top(暫定:介面型式未決,spec §8 ⛔)。
// 記憶體是 top 內的 static 陣列;wr = 1 寫入一列,wr = 0 平行讀取位置群組 g 的第 chunk 個 word(K 與 V)。
// 預設 LANES = 8、NB = 8(每週期 64 個元素,暫定)。記憶體型態(URAM/BRAM)尚未綁定。
#pragma once
#include "kv_cache.h"

typedef kv::KvCache<256, 512, 12, 1>::word_t kv_word_t;   // 64 bit

// sliding:D = 256、環形緩衝 512 筆、12 個 producer 層
void kv_cache_sliding(ap_uint<1> wr, int layer, int pos, const kv::elem_t k_in[256], const kv::elem_t v_in[256], int g,
                      int chunk, kv_word_t k_out[8], kv_word_t v_out[8]);
// global:D = 512、4096 筆、3 個 producer 層
void kv_cache_global(ap_uint<1> wr, int layer, int pos, const kv::elem_t k_in[512], const kv::elem_t v_in[512], int g,
                     int chunk, kv_word_t k_out[8], kv_word_t v_out[8]);
