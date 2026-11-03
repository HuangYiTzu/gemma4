// 1/ℓ 查表(spec §5;golden:attention_golden.py 的 recip_int,參數 HW_D1)。位元精確。
//   ℓ:無號 L_BITS 位元、小數 L_FRAC 位。ℓ = 2^top · x,x ∈ [1,2);idx = ℓ 最高位「之後」的 LUT_BITS 位元;
//   R = LUT[idx],LUT[i] = round(2^R_FRAC / (1 + (i+0.5)/2^LUT_BITS))(中點);w = top − L_FRAC。
//   1/ℓ = R · 2^−R_FRAC · 2^−w。ℓ = 0 時 R = 0、w = −L_FRAC(與 golden 相同;實際不會出現,至少有一個位置 P̃ = 1.0)。
//   查表來自 aux_tables.h(golden 匯出腳本產生)。C++98 相容。
#pragma once
#include <ap_int.h>

#include "exp2.h"   // 取得 AUX_STATIC_ASSERT 與 aux_tables.h

namespace aux {

template <int L_FRAC = 12, int LUT_BITS = 8, int R_FRAC = 12, int L_BITS = 25>
struct Recip {
    AUX_STATIC_ASSERT(L_FRAC == AUX_L_FRAC && LUT_BITS == AUX_LUT_BITS && R_FRAC == AUX_R_FRAC,
                      "recip: 參數與 aux_tables.h(D1)不符,請重新用 export_softmax_aux_vectors.py 產生表");
    AUX_STATIC_ASSERT(L_BITS > LUT_BITS, "recip: L_BITS 必須大於 LUT_BITS");

    typedef ap_uint<L_BITS> l_t;
    typedef ap_uint<R_FRAC + 1> r_t;   // 13 位元(實際最大 4088)
    typedef ap_int<6> w_t;             // −12 .. 12

    static void eval(l_t l, r_t &R, w_t &w) {
#pragma HLS INLINE
        if (l == 0) {
            R = 0;
            w = -L_FRAC;
            return;
        }
        // top = 最高位元的位置(前導 1 偵測)
        int top = 0;
    find_top:
        for (int i = 0; i < L_BITS; i++) {
#pragma HLS UNROLL
            if (l[i]) top = i;
        }
        // 正規化:左移使最高位元落在 bit (L_BITS-1);idx = 其下方的 LUT_BITS 位元
        const ap_uint<L_BITS> norm = l << ((L_BITS - 1) - top);
        const ap_uint<LUT_BITS> idx = norm.range(L_BITS - 2, L_BITS - 1 - LUT_BITS);
        R = AUX_RECIP_LUT[(int)idx];
        w = top - L_FRAC;
    }
};

}  // namespace aux
