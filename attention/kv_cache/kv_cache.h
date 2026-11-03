// KV cache(spec §6):int8 的 K、V 儲存,依 bank 切分以便一次平行讀取多個元素。位元精確(只是儲存與位址)。
//   元素格式:int8(寫入前的 INT16 → int8 量化不在本模組:spec §8 🟡 暫定、golden 尚未建模)。
//   位址:ring(sliding)  a = pos mod CACHE_LEN(環形緩衝,CACHE_LEN = 512);
//         global         a = pos(必須 < CACHE_LEN = 4096)。
//   bank = a mod NB;row = a / NB;記憶體內的 word 位址 = row·CHUNKS + chunk。
//   一個 word = LANES 個 int8(元素 d 在 chunk d / LANES、lane d % LANES,位元 [8·lane+7 : 8·lane])。
//   對齊的位置群組 g(位置 NB·g .. NB·g+NB−1)落在 NB 個不同的 bank、row 相同 → 一次平行讀 NB 個 word = NB·LANES 個元素。
//   一個 tile(Bc = 64)的 64 個位置每個 bank 恰好 64/NB 次,沒有 bank 衝突(Bc 整除 512,✅ spec §6)。
//   layer 是高位位址:同類型的 producer 層放進同一塊記憶體,同一時間只有一層在讀,容量由全部層共用。
//     sliding 12 層:layer 0,1,2,3,5,6,7,8,10,11,12,13 → 索引 0..11(consumer 的 sliding 層讀 layer 13 = 索引 11);
//     global   3 層:layer 4,9,14 → 索引 0..2(consumer 的 global 層讀 layer 14 = 索引 2)。
//   LANES、NB 為暫定(吞吐量與 URAM 預算未決,spec §8 ⛔):預設 8 × 8 = 每週期 64 個元素。
//   記憶體型態(URAM/BRAM)由使用這個結構的 top 決定(暫定,未綁定)。C++98 相容(Vivado HLS 2019.2)。
#pragma once
#include <ap_int.h>
#include <cassert>

#include "../softmax_aux/exp2.h"   // AUX_STATIC_ASSERT

namespace kv {

typedef ap_int<8> elem_t;

// RING = 1:sliding 環形緩衝;RING = 0:global 線性位址
template <int D, int CACHE_LEN, int NLAYERS, int RING, int LANES = 8, int NB = 8>
struct KvCache {
    static const int CHUNKS = D / LANES;             // 一列有幾個 word
    static const int ROWS = CACHE_LEN / NB;          // 每個 bank 的列數
    static const int DEPTH = ROWS * CHUNKS;          // 每個 bank 的 word 數
    AUX_STATIC_ASSERT(LANES >= 1 && NB >= 1 && NLAYERS >= 1, "kv_cache: LANES、NB、NLAYERS 必須 >= 1");
    AUX_STATIC_ASSERT(D % LANES == 0, "kv_cache: D 必須能被 LANES 整除");
    AUX_STATIC_ASSERT(CACHE_LEN % NB == 0 && 64 % NB == 0, "kv_cache: NB 必須整除 CACHE_LEN 與 Bc = 64");
    AUX_STATIC_ASSERT((NB & (NB - 1)) == 0 && (CACHE_LEN & (CACHE_LEN - 1)) == 0, "kv_cache: NB、CACHE_LEN 必須是 2 的冪");

    typedef ap_uint<8 * LANES> word_t;

    static int phys_addr(int pos) {
        if (RING) return pos % CACHE_LEN;
        assert(pos >= 0 && pos < CACHE_LEN);          // global:位置超出是錯誤(C 模擬檢查,硬體不檢查,暫定)
        return pos;
    }

    // 寫入一列(K 與 V 各 D 個 int8):D / LANES 個週期;sliding 超過 CACHE_LEN 後覆蓋最舊的位置。
    static void write_row(word_t k_mem[NLAYERS][NB][DEPTH], word_t v_mem[NLAYERS][NB][DEPTH], int layer, int pos,
                          const elem_t k[D], const elem_t v[D]) {
        assert(layer >= 0 && layer < NLAYERS);
        const int a = phys_addr(pos);
        const int bank = a % NB;
        const int row = a / NB;
    write_chunk:
        for (int c = 0; c < CHUNKS; c++) {
#pragma HLS PIPELINE II = 1   // 暫定
            word_t kw = 0, vw = 0;
        pack:
            for (int l = 0; l < LANES; l++) {
                kw.range(8 * l + 7, 8 * l) = k[c * LANES + l].range(7, 0);
                vw.range(8 * l + 7, 8 * l) = v[c * LANES + l].range(7, 0);
            }
            k_mem[layer][bank][row * CHUNKS + c] = kw;
            v_mem[layer][bank][row * CHUNKS + c] = vw;
        }
    }

    // 平行讀取:位置群組 g(位置 NB·g .. NB·g+NB−1)的第 chunk 個 word,NB 個 bank 各一個。
    // out[b] 是位置 NB·g + b 的 word(sliding:g 以 ROWS 為週期,與環形緩衝一致)。
    static void read_group(const word_t mem[NLAYERS][NB][DEPTH], int layer, int g, int chunk, word_t out[NB]) {
        assert(layer >= 0 && layer < NLAYERS && chunk >= 0 && chunk < CHUNKS);
        const int row = RING ? (g % ROWS) : g;
        assert(row >= 0 && row < ROWS);
    read_bank:
        for (int b = 0; b < NB; b++) {
#pragma HLS UNROLL
            out[b] = mem[layer][b][row * CHUNKS + chunk];
        }
    }

    // 取出 word 中的第 lane 個 int8(帶號)
    static elem_t unpack(const word_t &w, int lane) {
#pragma HLS INLINE
        ap_uint<8> u = w.range(8 * lane + 7, 8 * lane);
        elem_t e;
        e.range(7, 0) = u;
        return e;
    }
};

}  // namespace kv
