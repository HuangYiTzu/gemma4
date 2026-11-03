// attn_core 的綜合用 top(暫定:介面型式、並行度與 cache 組織未決,spec §8 ⛔)。
// K/V 來自 KV cache 的 bank 陣列(ap_memory 埠)。預設並行度(暫定):HEAD_PAR = 8、LANES = 8、NB = 1、EXP_PAR = 8
//   → 乘加 / 週期 = 8·1·8 = 64,cache 每週期只讀 NB·LANES = 8 個元素,cache 不需要分 bank。
#pragma once
#include "attn_core.h"

// sliding:D = 256、環形緩衝 512、window 512、12 個 producer 層;global:D = 512、4096 筆、純 causal、3 個 producer 層
typedef attn::AttnCore<256, 512, 512, 12, 8, 8, 1, 8, 8> AttnSliding;   // <D, CACHE_LEN, WINDOW, NLAYERS, H, LANES, NB, EXP_PAR, HEAD_PAR>
typedef attn::AttnCore<512, 4096, 0, 3, 8, 8, 1, 8, 8> AttnGlobal;

void attn_core_sliding(const attn::q_t q[8][256], const AttnSliding::word_t k_mem[12][1][AttnSliding::DEPTH],
                       const AttnSliding::word_t v_mem[12][1][AttnSliding::DEPTH], int layer, int p, attn::ms_t m_s,
                       attn::c2_t c2, attn::code_t code[8][256]);
void attn_core_global(const attn::q_t q[8][512], const AttnGlobal::word_t k_mem[3][1][AttnGlobal::DEPTH],
                      const AttnGlobal::word_t v_mem[3][1][AttnGlobal::DEPTH], int layer, int p, attn::ms_t m_s,
                      attn::c2_t c2, attn::code_t code[8][512]);
