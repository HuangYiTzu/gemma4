#include "attn_top.h"

// 暫定:ap_memory 介面;q、code 的 head 維度(dim 1)完全分割,讓 HEAD_PAR 個 head 同時載入與寫出;
// K/V cache 的 bank 維度(dim 2)完全分割(NB = 1 時只有一個 bank)。
// 記憶體型態(URAM/BRAM)與跨層共用由 KV cache 所屬的那一層決定,這裡不綁。

void attn_core_sliding(const attn::q_t q[8][256], const AttnSliding::word_t k_mem[12][1][AttnSliding::DEPTH],
                       const AttnSliding::word_t v_mem[12][1][AttnSliding::DEPTH], int layer, int p, attn::ms_t m_s,
                       attn::c2_t c2, attn::code_t code[8][256]) {
#pragma HLS INTERFACE ap_memory port = q
#pragma HLS INTERFACE ap_memory port = k_mem
#pragma HLS INTERFACE ap_memory port = v_mem
#pragma HLS INTERFACE ap_memory port = code
#pragma HLS ARRAY_PARTITION variable = q complete dim = 1
#pragma HLS ARRAY_PARTITION variable = code complete dim = 1
#pragma HLS ARRAY_PARTITION variable = k_mem complete dim = 2
#pragma HLS ARRAY_PARTITION variable = v_mem complete dim = 2
    AttnSliding::run(q, k_mem, v_mem, layer, p, m_s, c2, code);
}

void attn_core_global(const attn::q_t q[8][512], const AttnGlobal::word_t k_mem[3][1][AttnGlobal::DEPTH],
                      const AttnGlobal::word_t v_mem[3][1][AttnGlobal::DEPTH], int layer, int p, attn::ms_t m_s,
                      attn::c2_t c2, attn::code_t code[8][512]) {
#pragma HLS INTERFACE ap_memory port = q
#pragma HLS INTERFACE ap_memory port = k_mem
#pragma HLS INTERFACE ap_memory port = v_mem
#pragma HLS INTERFACE ap_memory port = code
#pragma HLS ARRAY_PARTITION variable = q complete dim = 1
#pragma HLS ARRAY_PARTITION variable = code complete dim = 1
#pragma HLS ARRAY_PARTITION variable = k_mem complete dim = 2
#pragma HLS ARRAY_PARTITION variable = v_mem complete dim = 2
    AttnGlobal::run(q, k_mem, v_mem, layer, p, m_s, c2, code);
}
