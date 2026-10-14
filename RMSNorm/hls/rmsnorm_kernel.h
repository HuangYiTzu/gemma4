// =====================================================================
// rmsnorm_kernel.h — RMSNorm HLS 核心骨架
// =====================================================================
// 對應：rmsnorm_golden_model.py 的 rmsnorm_core() / rmsnorm_call()
//       跟 rmsnorm_const_table_redesign.md 第15~16節定案的常數表格式。
//
// 這一版先落地 D=1536 的兩個實例：
//     pre_feedforward_layernorm   -> rmsnorm_core<1536, false, true>
//     post_feedforward_layernorm  -> rmsnorm_core<1536, true,  true>
// q/k/v_norm（D=256/512）待Attention側INT8/INT16位寬拍板（A1）後，
// 直接多兩個template實例，這份骨架不需要改。
//
// 依賴的常數表（由generate_rmsnorm_const_tables.py產生，需與此檔案
// 一併#include）：
//     rmsnorm_gamma_table.h       : RMSNORM_GAMMA_PRE[35][1536]
//                                    RMSNORM_GAMMA_POST[35][1536]
//     rmsnorm_layer_scale_table.h : RMSNORM_INV_GATE_UP_SCALE[35]
//                                    RMSNORM_INV_S_RES_SHIFT
//     rmsnorm_step0_table.h       : RMSNORM_STEP0_M_POST[35][1536]
//
// 依賴的kernel.h型別（fast_inv_unit的DT/RT）：
//     ms_t  = ap_ufixed<56, 32, AP_RND>   （Stage1輸出、Stage2輸入）
//     rsq_t = ap_ufixed<32, 12>           （Stage2輸出）
//
// ⚠️ 2026-09-20 c-simulation發現：RMSNORM_GAMMA_PRE/POST跟
// RMSNORM_STEP0_M_POST這幾張大常數表（35*1536個元素）**故意宣告成
// 純C的int16_t/int32_t，不是ap_int<16>/ap_int<32>**。原本想直接用
// ap_int存ROM表，型別跟runtime使用端完全對齊比較乾淨，但實測發現
// 這台機器的g++ 8.1.0 (MinGW) 編譯超過~31列*1536的ap_int<16>靜態
// 陣列初始化會直接crash（同樣大小的plain int16_t完全沒事，已用
// 二分搜尋確認崩潰門檻在30~32列之間，不是我們的設計問題，是這個
// 舊版編譯器處理Xilinx ap_int模板的已知類限制）。改成常數表用plain
// int16_t/int32_t存，函式介面(rmsnorm_step0/rmsnorm_stage3/
// rmsnorm_core)的gamma_table/step0_m參數對應改成int16_t/int32_t，
// 讀取單一元素時才轉型成ap_int（scalar轉型不會觸發這個crash）。
// 這其實也是更常見的HLS實務作法：ROM表存plain int，讀取端才轉型，
// 不需要ROM本身就是ap_int型別。
// =====================================================================

#ifndef RMSNORM_KERNEL_H
#define RMSNORM_KERNEL_H

#include <cstdint>
#include <ap_fixed.h>
#include <ap_int.h>
#include "kernel.h"                    // ms_t / rsq_t / hid_t typedef 來源
#include "fast_inv_unit.h"             // fast_inv_unit本體 + magic constant
#include "rmsnorm_gamma_table.h"
#include "rmsnorm_layer_scale_table.h"
#include "rmsnorm_step0_table.h"

// ---------------------------------------------------------------------
// fast_inv_unit：全系統唯一一份，定義在fast_inv_unit.h（原本從MLP.cpp
// 第63~130行原地搬出），這裡不再自己宣告，直接吃上面那個#include。
// MLP.cpp那邊也要改成#include這份檔案、刪掉自己原本那份定義，才能
// 確保RMSNorm跟MLP的epilogue真的共用同一顆物理電路，而不是各自
// synthesize出兩份（見fast_inv_unit.h開頭的MLP.cpp同步修改說明）。
// ---------------------------------------------------------------------

// ---------------------------------------------------------------------
// Q-format常數：全部由generate_rmsnorm_const_tables.py產生，定義在
// rmsnorm_gamma_table.h / rmsnorm_layer_scale_table.h裡（見那兩個檔案
// 開頭的RMSNORM_GAMMA_PRE_FRAC / RMSNORM_GAMMA_POST_FRAC /
// RMSNORM_INV_GATE_UP_SCALE_FRAC / RMSNORM_INV_RMS_SCALED_FRAC /
// RMSNORM_INV_S_RES_SHIFT），這裡不再手動定義、也不再各自為政地選
// FRAC——這正是第一版的問題：每張表各自「在自己容器裡塞最大
// 精度」，卻沒人往下檢查Stage3 multiply chain的總位元預算，四項乘
// 起來最壞情況遠超過ap_int<64>。
//
// 2026-09-20第二版修正（方案1+方案2混合）：
//   方案1（重新分配FRAC預算）單獨不夠救——實際把位元算完才發現，
//   rsq_t自己的20-bit frac（kernel.h共用型別，不能動）已經逼近
//   ap_int<64>能扛的上限，不管INV_GATE_UP_SCALE的FRAC壓多低，
//   「三個乘法連續相乘、最後才一次捨入」這個資料流結構本身就是
//   不可行的——所以還要加方案2（中途插一次捨入窄化）才夠。
//
// 現在的資料流（見rmsnorm_stage3()）：
//   Step A: inv_rms_scaled_wide = inv_rms(32-bit,frac=RSQ_T_FRAC=20)
//                                × inv_scale(16-bit,frac=每層各自的
//                                  RMSNORM_INV_GATE_UP_SCALE_FRAC[layer])
//           -> 安全落在ap_int<64>內（單一乘法，不需要先窄化）
//   Step B: 中途捨入！把Step A的結果捨進較窄的frac，才能繼續往下乘
//           x_uniform、gamma而不爆位元。
//   Step C: prod = x_uniform × inv_rms_scaled（窄化後）
//   Step D: prod = prod × gamma
//   最終shift = 中途捨入後留下的frac + GAMMA_FRAC
//
// ⚠️ 2026-09-20第三版修正（per-layer drop_bits/shift，取代第二版的
// 單一#define）：第二版用「全35層共用同一組frac/drop_bits」，實測
// 發現gate_up_input_scale跨層差異達30倍以上（Layer 0重新校準後0.70，
// Layer 13只有0.0083），套用「照顧最壞情況（inv_scale最大的層）」
// 算出來的drop_bits到inv_scale本身很小的層（例如Layer 0），Stage3
// 中途捨入直接underflow——cos_sim崩潰到0.000000（不是精度下降，是
// 資訊整個被捨光），見qat_verification/rmsnorm_verify_stage3_
// bitwidth.py的實測結果。
//
// 修正：RMSNORM_PRE_DROP_BITS[layer]跟RMSNORM_STAGE3_SHIFT_PRE[layer]
// 都改成35個值的表（定義在rmsnorm_layer_scale_table.h），不再是
// compile-time常數——PRE路徑Stage3現在需要一個per-layer的shift量
// （runtime從表查出來，不是固定的位移量），POST路徑因為沒有跨層
// inv_scale量級差30倍的問題，繼續維持全域共用一個值即可（見
// generate_rmsnorm_const_tables.py的bit-width檢查結果）。
#define RMSNORM_STEP0_M_FRAC     20    // STEP0_M的小數位元數（未變動）

// POST路徑：2026-09-20修正版，inv_rms_scaled = round(inv_rms <<
// RMSNORM_INV_S_RES_SHIFT)，位移後frac仍從RSQ_T_FRAC(20)中途捨入到
// RMSNORM_POST_INV_RMS_SCALED_FRAC（見rmsnorm_stage3()裡的post_drop_bits
// 那段），最終shift = 中途捨入後留下的frac + gamma的frac，跟PRE路徑
// 同一套邏輯，不再是「20 + GAMMA_FRAC」這種假設位移不需要窄化的算法
// （那是第一版的bug：位移雖然不損失精度，但int_bits跟着變大，整體
// 位元數還是會撐爆後面的乘法鏈，實測Step C'需要76-bit，已修正）。
#define RSQ_T_FRAC 20   // 對應kernel.h的rsq_t = ap_ufixed<32,12>，frac=32-12
#define RMSNORM_STAGE3_SHIFT_POST (RMSNORM_POST_INV_RMS_SCALED_FRAC + RMSNORM_GAMMA_POST_FRAC)

// ---------------------------------------------------------------------
// 型別
// ---------------------------------------------------------------------
typedef ap_int<8>   act8_t;    // INT8 activation（pre_ln輸出、matmul輸入）
typedef ap_int<16>  hid16_t;   // INT16 residual/gamma-table條目
typedef ap_int<24>  step0_t;   // Step0中繼值（文件7節記錄：16-bit會在
                                // Layer 0飽和，已改24-bit，見redesign doc第7節）
typedef ap_int<32>  acc32_t;   // 中間累乘暫存

// =====================================================================
// Stage0（僅Post-Norm用）：per-channel requantize
// 對應golden model的requantize_m_k() / Step0
// =====================================================================
template<int D>
void rmsnorm_step0(
    const acc32_t   matmul_raw[D],          // matmul MAC後的原始整數
    const int32_t   step0_m[D],             // RMSNORM_STEP0_M_POST[layer]，
                                              // 存plain int32_t（不是ap_int），
                                              // 見下方2026-09-20 c-sim發現的
                                              // 編譯器限制說明
    step0_t          x_uniform[D])
{
#pragma HLS INLINE off
STEP0_LOOP:
    for (int i = 0; i < D; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=1        // 先求正確，並行度留給之後integration調整
        ap_int<64> prod = (ap_int<64>)matmul_raw[i] * (ap_int<64>)step0_m[i];
        ap_int<64> rounded = prod + (ap_int<64>)(1 << (RMSNORM_STEP0_M_FRAC - 1));
        ap_int<64> shifted = rounded >> RMSNORM_STEP0_M_FRAC;
        // 飽和到step0_t範圍（文件2.1節：Step0輸出仍需clip，Stage3輸出
        // 才是「不做飽和，留給residual add之後」）
        const ap_int<64> qmax = (1 << (24 - 1)) - 1;
        const ap_int<64> qmin = -(1 << (24 - 1));
        if (shifted > qmax) shifted = qmax;
        if (shifted < qmin) shifted = qmin;
        x_uniform[i] = (step0_t)shifted;
    }
}

// =====================================================================
// rmsnorm_step0_dup：跟rmsnorm_step0邏輯完全一致，差別只在於requantize
// 算出來的值同時寫進兩份輸出（x_uniform_s1/x_uniform_s3），不是先算出
// tmp[]再跑一次獨立的複製迴圈——避免兩個1536元素的迴圈疊加成2倍
// latency（見2026-09-22 code review的效能分析）。
// =====================================================================
template<int D>
void rmsnorm_step0_dup(
    const acc32_t   matmul_raw[D],
    const int32_t   step0_m[D],
    step0_t          x_uniform_s1[D],
    step0_t          x_uniform_s3[D])
{
#pragma HLS INLINE off
STEP0_DUP_LOOP:
    for (int i = 0; i < D; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=1
        ap_int<64> prod = (ap_int<64>)matmul_raw[i] * (ap_int<64>)step0_m[i];
        ap_int<64> rounded = prod + (ap_int<64>)(1 << (RMSNORM_STEP0_M_FRAC - 1));
        ap_int<64> shifted = rounded >> RMSNORM_STEP0_M_FRAC;
        const ap_int<64> qmax = (1 << (24 - 1)) - 1;
        const ap_int<64> qmin = -(1 << (24 - 1));
        if (shifted > qmax) shifted = qmax;
        if (shifted < qmin) shifted = qmin;
        step0_t v = (step0_t)shifted;
        x_uniform_s1[i] = v;
        x_uniform_s3[i] = v;
    }
}

// =====================================================================
// Stage1：sum of squares
// =====================================================================
template<int D, typename IN_T>
ms_t rmsnorm_stage1(const IN_T x[D])
{
#pragma HLS INLINE off
    ssq_t ssq = 0;   // ap_ufixed<64,42>，見kernel.h
STAGE1_LOOP:
    for (int i = 0; i < D; i++) {
#pragma HLS PIPELINE II=1
        ap_int<48> sq = (ap_int<48>)x[i] * (ap_int<48>)x[i];
        ssq += (ssq_t)sq;
    }
    // mean(x^2) = ssq / D；D是compile-time常數，除法在綜合期就是乘上
    // 1/D這個常數（不是runtime除法器），HLS對「除以compile-time常數」
    // 通常能自動優化成乘法，但保險起見這裡明寫成乘法，不依賴工具優化
    const ms_t inv_d = (ms_t)(1.0 / D);
    ms_t mean_sq = (ms_t)(ssq * inv_d);
    // epsilon簡化：clamp下限（文件2.1節），避免除以零
    if (mean_sq < (ms_t)1.0) mean_sq = (ms_t)1.0;
    return mean_sq;
}

// =====================================================================
// Stage2：rsqrt，呼叫全系統共用的fast_inv_unit
// =====================================================================
inline rsq_t rmsnorm_stage2(ms_t mean_sq)
{
#pragma HLS INLINE off
    return fast_inv_unit<ms_t, rsq_t>(mean_sq, /*mode_recip=*/false);
}

// =====================================================================
// Stage3：乘gamma、對齊輸出scale
// 依is_post_norm分兩條路徑（對應golden model rmsnorm_call()的判準）：
//   Pre-Norm ：inv_rms_scaled = round(inv_rms * INV_GATE_UP_SCALE[layer])
//              （乘法 + 中途捨入窄化，見下方詳細註解——這是方案2新增
//              的步驟，用來避免Stage3後續乘法鏈爆位元）
//   Post-Norm：inv_rms_scaled = round(inv_rms << RMSNORM_INV_S_RES_SHIFT)
//              （位移 + 中途捨入窄化）
//              ⚠️ 2026-09-20修正：第一版以為「位移不改變frac，不需要
//              中途捨入」，但實測Step C'（x_uniform x inv_rms_scaled x
//              gamma_post）需要76-bit，超過ap_int<64>——位移雖然不損失
//              精度，但會把inv_rms_scaled的總位元數（int_bits+frac）
//              越撐越大，後面乘法鏈一樣會爆。Post-Norm也需要跟Pre-Norm
//              對稱地插入中途捨入，把frac從RSQ_T_FRAC(20)砍到
//              RMSNORM_POST_INV_RMS_SCALED_FRAC，才能落在ap_int<64>內
//              （見generate_rmsnorm_const_tables.py的位元寬度檢查）。
//
// ⚠️ inv_rms（ap_ufixed類型）轉成ap_int時，Vitis HLS的轉換是「數值
// 保留」（value-preserving），不是bit pattern reinterpret——直接
// (ap_int<N>)一個小數值的ap_ufixed會把小數部分捨去成整數，等於把
// 精度丟光。下面用.range()明確取出底層bit pattern，當成「已經隱含
// RSQ_T_FRAC個小數位元的scaled integer」來處理，這是這次修正的
// 另一個關鍵點（不是只有位元預算不夠，cast語意本身也要修對）。
// =====================================================================
template<int D, bool IS_POST_NORM, bool HAS_GAMMA, typename OUT_T>
void rmsnorm_stage3(
    const step0_t x_uniform[D],      // Stage1的輸入來源（Post-Norm: Step0
                                       // 輸出；Pre-Norm: residual直接進來，
                                       // 呼叫端負責塞進同一個buffer型別）
    rsq_t         inv_rms,
    int           layer_idx,
    const int16_t gamma_table[35][D],       // GAMMA_PRE 或 GAMMA_POST，
                                              // 存plain int16_t（不是
                                              // ap_int），見下方2026-09-20
                                              // c-sim發現的編譯器限制說明
    OUT_T         y_out[D])
{
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=gamma_table cyclic factor=1 dim=2

    // inv_rms的raw bit pattern（frac=RSQ_T_FRAC=20的scaled integer），
    // 用.range()重新解讀bit pattern，不是C-style數值轉型
    ap_uint<32> inv_rms_raw = inv_rms.range(31, 0);

    ap_int<64> inv_rms_scaled;
    if (IS_POST_NORM) {
        // post_feedforward_layernorm：s_res是2的冪次，左移對齊、不需要
        // 乘法器，但左移後frac仍是RSQ_T_FRAC(20)、int_bits變大，
        // 整體位元數一樣要窄化才塞得進後面的乘法鏈（見上方2026-09-20
        // 修正說明）。
        ap_int<64> inv_rms_shifted =
            (ap_int<64>)inv_rms_raw << RMSNORM_INV_S_RES_SHIFT;

        const int post_drop_bits = RSQ_T_FRAC - RMSNORM_POST_INV_RMS_SCALED_FRAC;
        ap_int<64> post_rounded = inv_rms_shifted
                                 + ((ap_int<64>)1 << (post_drop_bits - 1));
        inv_rms_scaled = post_rounded >> post_drop_bits;
    } else {
        // Step A：inv_rms(frac=RSQ_T_FRAC=20) x inv_scale（frac=每層
        //         各自的RMSNORM_INV_GATE_UP_SCALE_FRAC[layer]，不是
        //         全域共用值），用64-bit算，這一步本身安全（見
        //         generate_rmsnorm_const_tables.py印出的Step A
        //         位元寬度檢查，取全部35層裡最壞的組合仍在ap_int<64>內）
        ap_int<16> inv_scale = RMSNORM_INV_GATE_UP_SCALE[layer_idx];
        ap_int<64> inv_rms_scaled_wide =
            (ap_int<64>)inv_rms_raw * (ap_int<64>)inv_scale;

        // Step B：中途捨入！2026-09-20第三版修正：drop_bits改成
        // per-layer查表（RMSNORM_PRE_DROP_BITS[layer]），不是compile-
        // time常數——第二版用全域共用的drop_bits，對inv_scale遠小於
        // 全域最大值的層（例如Layer 0）會直接underflow到輸出全部
        // 是0（cos_sim=0.000000，不是精度下降），已用golden model
        // 實測驗證過，詳見rmsnorm_layer_scale_table.h開頭的說明。
        const int drop_bits = RMSNORM_PRE_DROP_BITS[layer_idx];
        ap_int<64> rounded = inv_rms_scaled_wide
                            + ((ap_int<64>)1 << (drop_bits - 1));
        inv_rms_scaled = rounded >> drop_bits;
    }

    // 2026-09-20第三版修正：PRE路徑的最終shift量改成per-layer查表
    // （RMSNORM_STAGE3_SHIFT_PRE[layer]，35個值的陣列），不再是單一
    // compile-time #define——理由跟上面drop_bits一致，每層的
    // INV_GATE_UP_SCALE_FRAC不同，捨入後留下的frac自然也不同。
    // POST路徑維持compile-time常數（RMSNORM_STAGE3_SHIFT_POST），
    // 因為POST路徑沒有跨層inv_scale量級差30倍的問題。
    const int shift = IS_POST_NORM ? RMSNORM_STAGE3_SHIFT_POST
                                     : RMSNORM_STAGE3_SHIFT_PRE[layer_idx];
    // qmax/qmin目前沒有實際套用在輸出上（見下方迴圈註解：Stage3依規格
    // 刻意不做飽和），保留這兩個變數只是方便之後debug時加assertion用，
    // 目前若編譯器對未使用變數提出warning，屬預期行為，可以先拿掉或
    // 包進#ifdef RMSNORM_DEBUG
    const ap_int<64> qmax = ((ap_int<64>)1 << (OUT_T::width - 1)) - 1;
    const ap_int<64> qmin = -((ap_int<64>)1 << (OUT_T::width - 1));

STAGE3_LOOP:
    for (int i = 0; i < D; i++) {
#pragma HLS PIPELINE II=1
        ap_int<64> prod = (ap_int<64>)x_uniform[i] * (ap_int<64>)inv_rms_scaled;
        if (HAS_GAMMA) {
            // v_norm的無gamma特例（見場合C）在這個D=1536的實例用不到，
            // 但HAS_GAMMA做成template bool，讓HLS在綜合期就把這個乘法
            // 連同gamma_table的讀取一起剪掉，不留下runtime mux
            prod = prod * (ap_int<64>)gamma_table[layer_idx][i];
        }
        ap_int<64> rounded = prod + ((ap_int<64>)1 << (shift - 1));
        ap_int<64> shifted = rounded >> shift;
        // 文件6節：Stage3輸出「不做飽和，飽和留給residual add之後」——
        // 這裡刻意不clip，讓下游residual_add()的飽和邏輯是整條pipeline
        // 唯一的飽和點，避免兩處都做飽和導致行為不一致
        y_out[i] = (OUT_T)shifted;
    }
}

// =====================================================================
// Stage0前置：DATAFLOW canonical檢查要求區域內只能是「宣告 + 函式呼叫」，
// 不能直接出現if/for語句本體，即使是template bool分支、綜合期就會被
// 消掉——把if/for邏輯包進獨立函式，讓rmsnorm_core的DATAFLOW區域內只留
// 函式呼叫（見2026-09-22 code review）。
//
// x_uniform輸出成兩份（x_uniform_s1/x_uniform_s3）：canonical DATAFLOW
// 規定區域內每個陣列只能有一個reader，但x_uniform同時要餵rmsnorm_stage1
// （算平方和）跟rmsnorm_stage3（乘gamma/inv_rms）這兩個consumer，是
// fan-out，所以在producer端就把內容複製兩份，個別對應一個consumer
// （見2026-09-22 code review："single reader and a single writer"）。
// =====================================================================
template<int D, bool IS_POST_NORM, typename IN_RAW_T>
void rmsnorm_step0_or_copy(
    const IN_RAW_T x_in[D],
    const int32_t  step0_m[D],
    step0_t        x_uniform_s1[D],
    step0_t        x_uniform_s3[D])
{
#pragma HLS INLINE off
    if (IS_POST_NORM) {
        // matmul MAC原始整數是acc32_t，這裡假設x_in已經是這個型別，
        // 呼叫端(pre_feedforward_layernorm那條路徑)不會用到這個分支。
        // 用rmsnorm_step0_dup在單一迴圈裡同時算requantize、寫兩份輸出，
        // 不要先算tmp[]再跑一次獨立複製迴圈（那樣latency會疊加成2倍，
        // 見2026-09-22 code review）。
        rmsnorm_step0_dup<D>((const acc32_t*)x_in, step0_m, x_uniform_s1, x_uniform_s3);
    } else {
        // Pre-Norm：residual本身已是單一scale，不需要Step0，直接複製
        // （HLS的if(IS_POST_NORM)在template bool下是綜合期常數分支，
        // 兩條路徑只會有一條真的被實現，不是runtime if）
    COPY_LOOP:
        for (int i = 0; i < D; i++) {
#pragma HLS PIPELINE II=1
            step0_t v = (step0_t)x_in[i];
            x_uniform_s1[i] = v;
            x_uniform_s3[i] = v;
        }
    }
}

// =====================================================================
// 頂層：rmsnorm_core<D, IS_POST_NORM, HAS_GAMMA>
// 對應golden model的rmsnorm_call()，把Step0/Stage1/2/3串起來
// =====================================================================
template<int D, bool IS_POST_NORM, bool HAS_GAMMA, typename IN_RAW_T, typename OUT_T>
void rmsnorm_core(
    const IN_RAW_T   x_in[D],               // Pre-Norm: residual INT16
                                              // Post-Norm: matmul MAC原始整數
    const int32_t    step0_m[D],            // 僅Post-Norm用，Pre-Norm傳nullptr
    int              layer_idx,
    const int16_t    gamma_table[35][D],
    OUT_T            y_out[D])
{
#pragma HLS DATAFLOW

    step0_t x_uniform_s1[D];
    step0_t x_uniform_s3[D];
#pragma HLS ARRAY_PARTITION variable=x_uniform_s1 cyclic factor=1 dim=1
#pragma HLS ARRAY_PARTITION variable=x_uniform_s3 cyclic factor=1 dim=1
#pragma HLS BIND_STORAGE variable=x_uniform_s1 type=RAM_2P impl=BRAM
#pragma HLS BIND_STORAGE variable=x_uniform_s3 type=RAM_2P impl=BRAM

    rmsnorm_step0_or_copy<D, IS_POST_NORM, IN_RAW_T>(x_in, step0_m, x_uniform_s1, x_uniform_s3);

    ms_t mean_sq = rmsnorm_stage1<D, step0_t>(x_uniform_s1);
    rsq_t inv_rms = rmsnorm_stage2(mean_sq);
    rmsnorm_stage3<D, IS_POST_NORM, HAS_GAMMA, OUT_T>(
        x_uniform_s3, inv_rms, layer_idx, gamma_table, y_out);
}

// =====================================================================
// 兩個具名實例（給MLP.cpp/top function呼叫，不要直接呼叫上面的template）
//
// 這裡只留宣告，本體搬到rmsnorm_hls.cpp——因為這兩個函式已經拿掉inline
// （見synthesis階段"Top function not found"的修法），header若同時被
// rmsnorm_hls.cpp（design file）跟tb_rmsnorm.cpp（testbench，也
// #include這份header來呼叫它們）各自include，就會各編出一份非inline
// 函式本體，co-simulation連結design/testbench兩邊object檔時就會撞成
// multiple definition。只留宣告、本體只在rmsnorm_hls.cpp定義一次，才
// 不會重複（見2026-10-13 code review）。
// =====================================================================

// extern "C"：v++/XRT用符號名稱找kernel，C++預設name mangling會把
// 這兩個top function的符號改名，封裝成XCLBIN kernel時對不起來，跟
// MLP.cpp的top function慣例一致，統一用extern "C" linkage。
extern "C" {
void pre_feedforward_layernorm_hls(
    const hid16_t residual_in[1536],
    int           layer_idx,
    act8_t        y_out[1536]);

void post_feedforward_layernorm_hls(
    const acc32_t down_proj_mac_out[1536],
    int           layer_idx,
    hid16_t       y_out[1536]);
}

#endif // RMSNORM_KERNEL_H
