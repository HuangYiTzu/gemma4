// =====================================================================
// fast_inv_unit.h — 共用倒數/倒數平方根核心
// =====================================================================
// 來源：An Accurate and Compact Design Integrating Seven Common Nonlinear
//       Functions in Deep Learning (ISCAS'25), Fig. 2:
//         1) leading-one detector 建出FP32-style的I_x
//         2) magic seed   I_y = a2 - I_x        (1/x)
//                         I_y = a1 - (I_x >> 1)  (1/sqrt(x))
//         3) FP-to-INT 轉回定點
//         4) 兩次Newton-Raphson修正
//
// 這份檔案是從MLP.cpp第63~130行原地搬出來的，函式本體逐位元跟原本
// MLP.cpp裡的版本相同（不是Python golden model那份，是真正拿去C-
// synthesis的HLS版本），目的是讓RMSNorm跟MLP的epilogue共用同一個
// 物理實體，而不是各自#include一份原始碼被綜合成兩顆電路。
//
// MLP.cpp需要同步的修改（這份檔案不會自動幫你改MLP.cpp，需要手動
// 或請寫MLP.cpp的同學執行）：
//   1. 刪掉MLP.cpp第63~130行原本的fast_inv_unit定義（含前面的doc
//      comment），改成 #include "fast_inv_unit.h"
//   2. 確認kernel.h本身不要再重複#include這份檔案（目前kernel.h
//      只include gelu_luts.h，沒有牽涉fast_inv_unit，不用動kernel.h）
//   3. MLP.cpp原本用到的FASTINV_MAGIC_RECIP/FASTINV_MAGIC_RSQRT兩個
//      巨集，在MLP.cpp裡目前是沒有定義的（呼叫端假設這兩個巨集已經
//      存在，可能之前是打算之後補進kernel.h/gelu_luts.h，但實際上
//      一直沒有補），現在改成從這份檔案取得
//
// magic constant的推導過程（不是憑空給的兩個數字，可回頭核對，見
// rmsnorm_verification/golden_model/rmsnorm_golden_model.py裡
// FASTINV_MAGIC_RSQRT的同一段推導，Python golden model跟這裡是
// 同一組數字，已經用rmsnorm_verify_pre_post_mlp.py的(E)節比對過
// float backend跟fast_inv_unit backend，cos_sim=1.000000）：
//   B（exponent bias）= 127
//   L（scaling factor）= 2^23 = 8388608（23-bit mantissa）
//   σ = 0.0450465（論文原文給的修正常數）
//   FASTINV_MAGIC_RSQRT = floor(1.5 * (B - σ) * L) = 0x5f3759df
//     -> 這個值與業界公開、行之有年的Quake fast inverse square root
//        magic number逐位元相同，可視為推導過程的獨立交叉驗證
//   FASTINV_MAGIC_RECIP  = floor(2.0 * (B - σ) * L) = 0x7ef477d5
//     -> 依論文式10直接推導，RMSNorm目前只用rsqrt這個mode，RECIP
//        暫時用不到，留給之後softmax/sigmoid等模組使用；這個值沒有
//        像RSQRT那樣能對照到獨立公開來源核對，建議跟寫這段硬體邏輯
//        的同學再confirm一次燒進ROM的實際數值是否一致
// =====================================================================

#ifndef FAST_INV_UNIT_H
#define FAST_INV_UNIT_H

#include <ap_fixed.h>
#include <ap_int.h>

// ---------------------------------------------------------------------
// magic constant
// ---------------------------------------------------------------------
#define FASTINV_MAGIC_RECIP  0x7ef477d5u   // 1/x 模式（目前RMSNorm用不到）
#define FASTINV_MAGIC_RSQRT  0x5f3759dfu   // 1/sqrt(x) 模式（RMSNorm Stage2用這個）

// ===========================================================================
// fast_inv_unit -- shared reciprocal / inverse-square-root unit, ISCAS'25
// "An Accurate and Compact Design Integrating Seven Common Nonlinear
//  Functions in Deep Learning", Fig. 2:
//   1) leading-one detector builds an FP32-style word I_x
//   2) magic seed   I_y = a2 - I_x         (1/x)
//                   I_y = a1 - (I_x >> 1)  (1/sqrt(x))
//   3) FP-to-INT back to fixed point
//   4) two Newton-Raphson iterations
// Shared across MLP epilogue (1/sqrt(x) mode for post_feedforward_
// layernorm), RMSNorm (same mode, all Post-Norm/Pre-Norm call points),
// and reserved for softmax/sigmoid (1/x mode) in other layers. The
// mode_recip branch is pruned by HLS whenever it's called with a
// compile-time-constant argument, so a caller that only ever needs
// rsqrt does not pay for the reciprocal datapath.
// ===========================================================================
template <typename DT, typename RT>
static RT fast_inv_unit(DT x, bool mode_recip)
{
#pragma HLS INLINE
	const int W = DT::width;
	const int F = DT::width - DT::iwidth;

	// ---- 1) leading-one detector (priority encoder) ----
	ap_uint<W> bits = x.range(W - 1, 0);
	ap_uint<7> pos = 0;
lod:
	for (int i = 0; i < W; ++i) {
#pragma HLS UNROLL
		if (bits[i]) pos = i;
	}
	ap_int<12> e_x = (ap_int<12>)pos - F;

	ap_uint<W> mb = bits;
	mb[pos] = 0;                              // strip the implicit leading 1
	ap_uint<23> M_x;
	if (pos >= 23) M_x = (ap_uint<23>)(mb >> (int)(pos - 23));
	else           M_x = (ap_uint<23>)(((ap_uint<W + 24>)mb) << (int)(23 - pos));

	// ---- 2) magic-constant seed in the INT32 domain ----
	ap_uint<32> I_x = (((ap_uint<32>)(ap_uint<9>)(e_x + 127)) << 23) | M_x;
	ap_uint<32> I_y = mode_recip
	                ? (ap_uint<32>)(FASTINV_MAGIC_RECIP - I_x)
	                : (ap_uint<32>)(FASTINV_MAGIC_RSQRT - (I_x >> 1));

	// ---- 3) FP-to-INT: rebuild y0 = 1.M * 2^e_y as fixed point ----
	ap_int<12>  e_y = (ap_int<12>)(I_y >> 23) - 127;
	ap_uint<23> M_y = I_y.range(22, 0);
	ap_ufixed<56, 25> yb = 0;                 // 2^24 .. 2^-31
	yb[31] = 1;                               // implicit leading 1 (2^0)
	yb.range(30, 8) = M_y;
	RT y = (e_y >= 0) ? (RT)(yb << (int)e_y) : (RT)(yb >> (int)(-e_y));

	// ---- 4) two Newton-Raphson refinement steps ----
	// the intermediates are cut back to narrow types on purpose: x*y is ~1
	// (reciprocal) or ~sqrt(x) (rsqrt), so a full-width product would cost a
	// huge multiplier for bits that are structurally zero.
	const ap_ufixed<2, 1> B1(1.5);
	typedef ap_ufixed<32, 18, AP_RND> xy_t;   // x*y
	typedef ap_ufixed<26, 2,  AP_RND> t_t;
nr:
	for (int it = 0; it < 2; ++it) {
#pragma HLS UNROLL
		xy_t xy = (xy_t)(x * y);
		t_t  t;
		if (mode_recip) t = (t_t)(2  - xy);                 // y*(2 - x*y)
		else            t = (t_t)(B1 - (xy * y >> 1));      // y*(1.5-0.5xy^2)
		y = (RT)(y * t);
	}
	return y;
}

#endif // FAST_INV_UNIT_H
