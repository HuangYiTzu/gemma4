// =====================================================================
// kernel_stub.h — 給c-simulation testbench用的最小kernel.h替身
// =====================================================================
// rmsnorm_kernel.h #include "kernel.h" 只是要拿ms_t/rsq_t/ssq_t這三個
// typedef（見rmsnorm_kernel.h開頭的說明），不需要MLP.cpp那份kernel.h
// 裡其他跟RMSNorm無關的東西。這裡把跑testbench需要的最小子集抽出來，
// 方便在還沒整合進完整專案（沒有MLP.cpp那份真正的kernel.h）之前先
// 用g++單獨compile、跑c-simulation驗證RMSNorm這一塊。
//
// ⚠️ 正式整合進專案時，這個檔案不需要，直接用MLP.cpp那份真正的
// kernel.h即可（typedef名稱/數值要跟這裡一致，已經跟
// rmsnorm_kernel.h/generate_rmsnorm_const_tables.py的位元預算推導
// 對齊過，見hls/stage3_bitwidth_review.md）。
// =====================================================================

#ifndef KERNEL_STUB_H
#define KERNEL_STUB_H

#include <ap_fixed.h>

// Stage1輸出、Stage2輸入：mean(x^2)
typedef ap_ufixed<56, 32, AP_RND> ms_t;

// Stage1內部的平方和累加器
typedef ap_ufixed<64, 42> ssq_t;

// Stage2輸出：1/sqrt(mean_sq)
typedef ap_ufixed<32, 12> rsq_t;

// residual/輸出Q-format（對應golden model checklist記錄的hid_t，
// Q7.8，frac=8，對應目前s_res=2^-5尚未更新的舊註解——這裡先保留
// typedef供kernel_stub.h完整性，實際residual寬度以hid16_t為準，
// 見rmsnorm_kernel.h）
typedef ap_fixed<16, 8, AP_RND, AP_SAT> hid_t;

#endif // KERNEL_STUB_H
