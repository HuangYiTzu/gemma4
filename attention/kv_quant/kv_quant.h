// K/V 寫入 cache 前的 INT16 → int8 量化(spec §8 🟡 暫定;標準答案:golden 既有的 attention_golden.quantize,不修改 golden)。位元精確。
//   code = clip(round_half_up(x · M / 2^B), −127, 127)
//   x:INT16(K:RoPE 輸出,小數 14 位;V:norm 輸出,小數 10 位 —— 🟡 norm 格式);M = round(2^−F / scale · 2^B),M < 2^MW;
//   每層、每種(K、V)一組 (M, B),由 golden/export_kv_quant_vectors.py 對 INT16 全部 65536 個輸入窮舉驗證與 golden 逐位元一致後
//   產生 kvq_consts.h(HLS 不做浮點運算)。
//   暫定:乘數位寬 MW = 24(spec 暫定值 16 位元無法與 golden 逐位元一致:窮舉顯示 24/30 組各有 2~18 個輸入差 1 LSB)。
//   乘積保留完整位寬(INT16 × 24-bit = 40 bit),+ 2^(B−1) 後算術右移(round-half-up,負數也一樣),飽和 ±127(不是 −128)。
//   輸出直接接 KvCache::write_row(元素格式同為 ap_int<8>)。C++98 相容(Vivado HLS 2019.2)。
#pragma once
#include <ap_int.h>
#include <cassert>

#include "../kv_cache/kv_cache.h"   // kv::elem_t、AUX_STATIC_ASSERT
#include "kvq_consts.h"

namespace kvq {

// F_K / F_V:輸入小數位(🟡 暫定,隨 norm 輸出格式);(M, B) 表只對應 kvq_consts.h 產生時的 F、MW,參數不符則編譯期擋下。
template <int D, int F_K = 14, int F_V = 10, int MW = 24>
struct KvQuant {
    AUX_STATIC_ASSERT(F_K == KVQ_F_K && F_V == KVQ_F_V && MW == KVQ_MW,
                      "kv_quant: 參數與 kvq_consts.h 不符,請重新用 export_kv_quant_vectors.py 產生常數表");

    typedef ap_int<16> in_t;
    typedef ap_uint<MW> m_t;
    typedef ap_uint<6> b_t;      // B 最大 30(實測 28..30)

    // 一個元素:round_half_up(x·M / 2^B) 後飽和到 ±127
    static kv::elem_t quant_one(in_t x, m_t M, b_t B) {
#pragma HLS INLINE
        assert(B >= 1);
        const ap_int<MW + 1> m1 = M;                                   // 無號 MW 位元 → 有號 MW+1 位元
        const ap_int<MW + 17> prod = x * m1;                           // 完整位寬乘積(INT16 × 24 bit = 40 bit)
        const ap_int<MW + 18> rnd = ap_int<MW + 18>(1) << (B - 1);     // 2^(B−1)
        const ap_int<MW + 18> r = ap_int<MW + 18>(ap_int<MW + 18>(prod) + rnd) >> B;   // 算術右移
        kv::elem_t out;
        if (r > 127) out = 127;
        else if (r < -127) out = -127;
        else out = r;
        return out;
    }

    // 一列(K 與 V 各 D 個元素):D 個週期,II = 1(暫定:一次一個元素,遠小於 attention 的週期數,不是瓶頸)
    static void quant_row(const in_t k_in[D], const in_t v_in[D], m_t m_k, b_t b_k, m_t m_v, b_t b_v, kv::elem_t k_out[D],
                          kv::elem_t v_out[D]) {
    quant_loop:
        for (int d = 0; d < D; d++) {
#pragma HLS PIPELINE II = 1   // 暫定
            k_out[d] = quant_one(k_in[d], m_k, b_k);
            v_out[d] = quant_one(v_in[d], m_v, b_v);
        }
    }
};

}  // namespace kvq
