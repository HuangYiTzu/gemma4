// exp2(spec §5;golden:attention_golden.py 的 exp2_int,參數 HW_D1)。位元精確。
//   輸入 d:≤ 0 的整數(S 的格式,小數 S_FRAC 位);輸出 UQ1.E_FRAC(0 .. 2^E_FRAC)。
//     t   = round_half_up(d · LG_MULT / 2^(S_FRAC + LG_BITS − T_FRAC))      log2e ≈ LG_MULT / 2^LG_BITS
//     u   = t >> T_FRAC(算術右移,floor);v = t − (u << T_FRAC)
//     idx = v >> (T_FRAC − log2 SEG);fr = v & (2^(T_FRAC − log2 SEG) − 1)
//     f   = A[idx] + round_half_up((A[idx+1] − A[idx]) · fr / 2^(T_FRAC − log2 SEG))      A[k] = round(2^(k/SEG)·2^A_FRAC)
//     out = round_half_up(f / 2^(E_FRAC − A_FRAC + (−u)))    (A_FRAC = E_FRAC 時為 f / 2^(−u))
//   用於 P̃(d = S − m_new)與 α(d = m − m_new)。查表來自 aux_tables.h(由 golden 匯出腳本產生,HLS 不算 pow)。
//   C++98 相容(Vivado HLS 2019.2 前端預設不是 C++11)。
#pragma once
#include <ap_int.h>
#include <cassert>

#include "aux_tables.h"

#ifndef AUX_STATIC_ASSERT
#define AUX_CAT2(a, b) a##b
#define AUX_CAT(a, b) AUX_CAT2(a, b)
#if __cplusplus >= 201103L
#define AUX_STATIC_ASSERT(cond, msg) static_assert(cond, msg)
#else
#define AUX_STATIC_ASSERT(cond, msg) typedef char AUX_CAT(aux_static_assert_, __LINE__)[(cond) ? 1 : -1]
#endif
#endif

namespace aux {

// 預設值 = HW_D1(✅ 凍結);仍做成 template 參數。目前只有對應 aux_tables.h 的組態能編譯(表只對應 D1)。
template <int S_FRAC = 8, int LG_MULT = 1477, int LG_BITS = 10, int T_FRAC = 10, int SEG_LOG2 = 3, int A_FRAC = 16,
          int E_FRAC = 16>
struct Exp2 {
    static const int SH_T = S_FRAC + LG_BITS - T_FRAC;     // t 的右移量
    static const int SH_FR = T_FRAC - SEG_LOG2;            // 分段內插的右移量
    static const int OUT_W = E_FRAC + 1;                   // UQ1.E_FRAC
    AUX_STATIC_ASSERT(SH_T > 0 && SH_FR > 0, "exp2: 右移量必須 > 0");
    AUX_STATIC_ASSERT(S_FRAC == AUX_S_FRAC && LG_MULT == AUX_LG_MULT && LG_BITS == AUX_LG_BITS && T_FRAC == AUX_T_FRAC &&
                          (1 << SEG_LOG2) == AUX_SEG && A_FRAC == AUX_A_FRAC && E_FRAC == AUX_E_FRAC,
                      "exp2: 參數與 aux_tables.h(D1)不符,請重新用 export_softmax_aux_vectors.py 產生表");
    AUX_STATIC_ASSERT(A_FRAC >= E_FRAC, "exp2: 輸出小數位不可大於節點表小數位");

    typedef ap_int<16> d_t;            // −32767..0
    typedef ap_uint<OUT_W> out_t;      // 0..2^E_FRAC

    static out_t eval(d_t d) {
#pragma HLS INLINE
        assert(d <= 0);
        // t = round_half_up(d·LG_MULT / 2^SH_T);d·LG_MULT 最大約 2^15·1477 < 2^26,32 位元足夠
        const ap_int<32> prod = ap_int<32>(d) * LG_MULT;
        const ap_int<32> t = ap_int<32>(prod + (1 << (SH_T - 1))) >> SH_T;
        const ap_int<32> u = t >> T_FRAC;                       // floor(算術右移)
        const ap_int<32> v = t - (u << T_FRAC);                 // 0 .. 2^T_FRAC − 1
        const int idx = (int)(v >> SH_FR);                      // 0 .. SEG − 1
        const ap_uint<SH_FR> fr = v & ((1 << SH_FR) - 1);
        const ap_uint<A_FRAC + 2> a0 = AUX_EXP2_A[idx];
        const ap_uint<A_FRAC + 2> a1 = AUX_EXP2_A[idx + 1];
        const ap_uint<A_FRAC + 1> diff = a1 - a0;               // A 為遞增,差為正
        const ap_uint<A_FRAC + 1 + SH_FR> prod2 = diff * fr;
        const ap_uint<A_FRAC + 2> f = a0 + ap_uint<A_FRAC + 2>((prod2 + (1 << (SH_FR - 1))) >> SH_FR);  // < 2^(A_FRAC+1)

        // out = round_half_up(f / 2^s),s = A_FRAC − E_FRAC − u ≥ 0。f < 2^(A_FRAC+1),所以 s > A_FRAC + 1 時結果為 0
        // (golden 把位移量 clip 到 62,結果相同),硬體把位移量飽和在 A_FRAC + 2。
        const ap_int<32> s32 = (A_FRAC - E_FRAC) - u;
        const int s = (s32 > A_FRAC + 2) ? (A_FRAC + 2) : (int)s32;
        ap_uint<A_FRAC + 3> r;
        if (s == 0)
            r = f;
        else
            r = (ap_uint<A_FRAC + 3>(f) + (ap_uint<A_FRAC + 3>(1) << (s - 1))) >> s;
        return out_t(r);
    }
};

}  // namespace aux
