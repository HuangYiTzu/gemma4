// =====================================================================
// tb_rmsnorm.cpp — RMSNorm HLS C-simulation testbench
// =====================================================================
// 驗證層次（跟之前golden model那幾輪驗證是不同層次，見hls/
// stage3_bitwidth_review.md開頭說明）：
//
//   golden model（Python）那幾輪驗證的問題是「這樣量化，跟真實QAT
//   浮點輸出比，誤差有多大」（cos_sim，允許誤差）。
//
//   這支testbench驗證的問題是「hls/rmsnorm_kernel.h這份C++寫的整數
//   運算，是否跟設計時講好的整數運算完全一致」——測試向量的期望值
//   （tb_rmsnorm_vectors.h的TB_PRE_EXPECTED/TB_POST_EXPECTED）是用
//   跟rmsnorm_kernel.h逐bit對應的Python模擬算出來的（見
//   generate_stage3_testbench_vectors.py），理論上C++算出來的結果
//   應該要**完全相同**，差1都算fail——這是在抓C++實作筆誤（型別轉換
//   錯誤、shift方向錯、忘記round的+0.5之類），不是在抓量化精度問題
//   （精度問題已經在golden model那幾輪處理過了）。
//
// 編譯方式（不需要完整Vitis HLS工具鏈，只需要ap_fixed.h/ap_int.h這兩個
//標頭——已實測用g++直接編譯執行成功，見hls/stage3_bitwidth_review.md
// 第6.1節）：
//   1. 把kernel_stub.h複製/改名成kernel.h，放在跟這個檔案同一個
//      include路徑下（rmsnorm_kernel.h寫死#include "kernel.h"，
//      正式整合進專案時改用MLP.cpp那份真正的kernel.h，這裡先用
//      stub讓RMSNorm這塊能獨立測試）：
//        cp kernel_stub.h kernel.h
//   2. 取得ap_fixed.h/ap_int.h：
//      (a) 有Vitis HLS安裝：用$XILINX_HLS/include
//      (b) 沒有的話，Xilinx有開源到GitHub：
//          git clone --depth 1 https://github.com/Xilinx/HLS_arbitrary_Precision_Types.git
//          用該repo的include/目錄即可，不需要完整Vitis安裝
//   3. g++ -I. -I<上面include路徑> -std=c++14 tb_rmsnorm.cpp -o tb_rmsnorm
//   4. ./tb_rmsnorm
//      實測結果：PRE 104/105、POST 105/105 PASS（逐bit相同），共
//      209/210。剩1個PRE case是fast_inv_unit_python()翻譯的1-ULP
//      已知narrow-scope issue，見review文件第6.1節，不影響整體結論。
// =====================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>

#include "rmsnorm_kernel.h"
#include "tb_rmsnorm_vectors.h"

// -----------------------------------------------------------------
// PRE路徑測試：pre_feedforward_layernorm_hls()
// -----------------------------------------------------------------
static int run_pre_tests()
{
    int num_pass = 0;
    int num_fail = 0;
    int max_abs_diff_seen = 0;

    for (int c = 0; c < TB_NUM_PRE_CASES; c++) {
        int layer_idx = TB_PRE_LAYER_IDX[c];

        hid16_t residual_in[TB_HIDDEN_SIZE];
        for (int i = 0; i < TB_HIDDEN_SIZE; i++) {
            residual_in[i] = (hid16_t)TB_PRE_INPUT[c][i];
        }

        act8_t y_out[TB_HIDDEN_SIZE];
        pre_feedforward_layernorm_hls(residual_in, layer_idx, y_out);

        bool case_ok = true;
        int case_max_diff = 0;
        for (int i = 0; i < TB_HIDDEN_SIZE; i++) {
            int got = (int)y_out[i];
            int expected = TB_PRE_EXPECTED[c][i];
            int diff = got - expected;
            if (diff < 0) diff = -diff;
            if (diff > case_max_diff) case_max_diff = diff;
            if (diff != 0) {
                case_ok = false;
#ifdef TB_DEBUG_DUMP
                if (c == TB_DEBUG_CASE) {
                    std::printf("  ch=%d got=%d expected=%d diff=%d residual_in=%d\n",
                                i, got, expected, got - expected, (int)TB_PRE_INPUT[c][i]);
                }
#endif
            }
        }
        if (case_max_diff > max_abs_diff_seen) max_abs_diff_seen = case_max_diff;

        if (case_ok) {
            num_pass++;
        } else {
            num_fail++;
            if (num_fail <= 10) {  // 只印前10個fail，避免洗版
                std::printf("[PRE FAIL] case=%d layer=%d max_abs_diff=%d\n",
                            c, layer_idx, case_max_diff);
            }
        }
    }

    std::printf("\n[PRE] %d/%d PASS（逐bit相同），max_abs_diff（所有case中最大）=%d\n",
                num_pass, TB_NUM_PRE_CASES, max_abs_diff_seen);
    return num_fail;
}

// -----------------------------------------------------------------
// POST路徑測試：post_feedforward_layernorm_hls()
// -----------------------------------------------------------------
static int run_post_tests()
{
    int num_pass = 0;
    int num_fail = 0;
    int max_abs_diff_seen = 0;

    for (int c = 0; c < TB_NUM_POST_CASES; c++) {
        int layer_idx = TB_POST_LAYER_IDX[c];

        acc32_t down_proj_mac_out[TB_HIDDEN_SIZE];
        for (int i = 0; i < TB_HIDDEN_SIZE; i++) {
            down_proj_mac_out[i] = (acc32_t)TB_POST_INPUT[c][i];
        }

        hid16_t y_out[TB_HIDDEN_SIZE];
        post_feedforward_layernorm_hls(down_proj_mac_out, layer_idx, y_out);

        bool case_ok = true;
        int case_max_diff = 0;
        for (int i = 0; i < TB_HIDDEN_SIZE; i++) {
            int got = (int)y_out[i];
            int expected = TB_POST_EXPECTED[c][i];
            int diff = got - expected;
            if (diff < 0) diff = -diff;
            if (diff > case_max_diff) case_max_diff = diff;
            if (diff != 0) case_ok = false;
        }
        if (case_max_diff > max_abs_diff_seen) max_abs_diff_seen = case_max_diff;

        if (case_ok) {
            num_pass++;
        } else {
            num_fail++;
            if (num_fail <= 10) {
                std::printf("[POST FAIL] case=%d layer=%d max_abs_diff=%d\n",
                            c, layer_idx, case_max_diff);
            }
        }
    }

    std::printf("\n[POST] %d/%d PASS（逐bit相同），max_abs_diff（所有case中最大）=%d\n",
                num_pass, TB_NUM_POST_CASES, max_abs_diff_seen);
    return num_fail;
}

int main()
{
    std::printf("=========================================================\n");
    std::printf("RMSNorm HLS C-simulation testbench\n");
    std::printf("全35層 x 3個token的真實QAT activation測試向量\n");
    std::printf("=========================================================\n\n");

    int pre_fail = run_pre_tests();
    int post_fail = run_post_tests();

    std::printf("\n=========================================================\n");
    if (pre_fail == 0 && post_fail == 0) {
        std::printf("全部測試PASS。\n");
        return 0;
    } else {
        std::printf("有測試FAIL：PRE fail=%d, POST fail=%d\n", pre_fail, post_fail);
        std::printf("如果PRE/POST某一邊大量fail、另一邊全過，先查那一路徑；\n");
        std::printf("如果所有case都fail且max_abs_diff很大，很可能是shift方向、\n");
        std::printf("型別寬度、或忘記round的+0.5這類系統性錯誤，不是單一case的問題。\n");
        return 1;
    }
}
