#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_rmsnorm_const_tables.py
=====================================================================
把pre/post_feedforward_layernorm這兩個呼叫點、全35層的常數表，離線
算好，輸出成HLS可以直接#include的C陣列。

⚠️ 2026-09-20第二版修正（方案1+方案2混合，取代9/18的第一版跟9/19的
「純方案1」版本）：

第一版（2026-09-18）把 gamma_i / S_target 直接折成單一常數
WB[layer][ch]，16-bit(Q3.13) ROM，全部35層普遍溢位，改成「raw gamma
+ per-layer純量」架構（見下方「架構」段落）解決。

第一次修正之後才發現：Stage3要把inv_rms、INV_GATE_UP_SCALE、
x_uniform、gamma四項連續相乘、最後才一次捨入，即使INV_GATE_UP_SCALE
自己的FRAC選得再精簡，rsq_t（kernel.h共用型別，frac=20，不可動）自己
的frac就已經逼近ap_int<64>能扛的上限——這不是哪張表FRAC選太大的問題，
是「四項連乘、一次捨入」這個資料流結構本身撐不住，單靠重新分配FRAC
預算（方案1）救不了，需要在inv_rms×INV_GATE_UP_SCALE算完後，插一次
中途捨入窄化（方案2），才能繼續往下乘x_uniform、gamma而不爆位元。
方案1+方案2混合才是這次真正採用的設計，對應hls/rmsnorm_kernel.h的
Stage3實作（Step A/B/C/D）。

架構（「raw gamma + per-layer純量」，數學上 gamma_i/S_target =
gamma_i * (1/S_target)，拆成兩次乘法）：
    1) per-channel: 直接存raw gamma_i（量級小，35層掃過實測
       gamma_pre_ln最大496、gamma_post_ln最大45.5，各自能塞進
       窄位寬INT16，不需要跟著S_target一起放大）
    2) 1/S_target 分開處理：
       - pre_feedforward_layernorm的S_target=gate_up_input_scale，
         每層不同，做成INV_GATE_UP_SCALE[layer]這張「per-layer純量
         表」（只有35個值），這次改用16-bit容器（不是原本硬塞32-bit
         最大化），FRAC由實測動態範圍反推，見derive_inv_scale_frac()
       - post_feedforward_layernorm的S_target固定是S_RES（全域常數，
         不隨layer變動！2^-5是2的冪次），1/S_RES是編譯期左移量，
         連乘法器都不用

輸出三個檔案：
    rmsnorm_gamma_table.h        : RMSNORM_GAMMA_PRE[35][1536]（INT16, frac=5）
                                    RMSNORM_GAMMA_POST[35][1536]（INT16, frac=8）
    rmsnorm_layer_scale_table.h  : RMSNORM_INV_GATE_UP_SCALE[35]（INT16，
                                    frac由derive_inv_scale_frac()回推）
                                    RMSNORM_INV_S_RES_SHIFT（編譯期左移量）
                                    RMSNORM_INV_RMS_SCALED_FRAC（Stage3中途
                                    捨入用的frac，方案2新增）
    rmsnorm_step0_table.h        : RMSNORM_STEP0_M_POST[35][1536]（未變動）

⚠️ 前置需求：先跑qat_verification/rmsnorm_verify_batch.py確認全部
LAYER_LIST層的cos_sim都在可接受範圍，再產生這張表；產生完後務必再跑
qat_verification/rmsnorm_verify_stage3_bitwidth.py，確認方案2新增的
中途捨入沒有把cos_sim拉低太多——位元預算過得了關，不代表精度過得了
關，兩件事要分開驗證（教訓來自residual那邊的Plan A/B經驗）。
"""

import os
import sys
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_QAT_DIR = os.path.join(_THIS_DIR, "..", "qat_verification")
sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, _QAT_DIR)

from rmsnorm_golden_model import S_RES
from rmsnorm_verify_pre_post_mlp import read_layer_constants
from layer_scale_calibration import get_all_recalibrated_gate_up_scales

# 2026-09-20 c-sim發現：這支腳本原本用checkpoint原始的gate_up_input_scale
# 算INV_GATE_UP_SCALE，但qat_verification/那邊的整套驗證管線（batch
# verify、perplexity、stage3 bitwidth）早就改用WikiText重新校準的值
# （見rmsnorm_const_table_redesign.md第15.5節），兩邊沒對齊——這支
# 腳本產生的常數表用的是「沒校準過」的scale，跟testbench/驗證腳本
# 假設的scale不一致，導致c-simulation testbench全部fail（PRE 0/105
# pass）。這裡改成統一用同一份校準結果，不能各腳本各自為政。
RECALIBRATED_GATE_UP_SCALE = get_all_recalibrated_gate_up_scales()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# 實際本機快取的safetensors路徑（2026-09-16確認，HF_HOME=~/.cache/huggingface）。
SAFETENSORS_PATH = (
    "C:/Users/eva huang/.cache/huggingface/hub/"
    "models--google--gemma-4-E2B-it-qat-mobile-transformers/"
    "snapshots/dd693ff40353f057ca5f07e945ad867f4afbf2ec/model.safetensors"
)
NUM_LAYERS = 35
HIDDEN_SIZE = 1536

# ---------------------------------------------------------------------
# Q-format：資料驅動選出來的值（沿用redesign doc第5節的結論，這兩個
# 不是這次要修的部分，維持不變）
# ---------------------------------------------------------------------
GAMMA_PRE_FRAC = 5     # 實測RMSNORM_GAMMA_PRE max≈496 反推
GAMMA_POST_FRAC = 8    # 實測RMSNORM_GAMMA_POST max≈45.5 反推
M_FRAC = 20            # Step0 M常數表，未變動，延續MLP_RQ_SHIFT風格
GAMMA_BITS = 16        # raw gamma表的總位元數（int16_t）

# ---------------------------------------------------------------------
# Stage3 multiply chain的全域位元預算（這次修正的核心）
# ---------------------------------------------------------------------
# rsq_t = ap_ufixed<32,12>（kernel.h共用型別，iwidth=12, frac=20）
RSQ_T_IWIDTH = 12
RSQ_T_FRAC = 20

# Stage3中途捨入的容器：24-bit（沿用Step0已驗證過的24-bit前例，見
# redesign doc第7節：16-bit在Layer0會飽和，24-bit沒問題）
INV_RMS_SCALED_CONTAINER_BITS = 24

MULTIPLY_CHAIN_BUDGET_BITS = 64  # ap_int<64>，Stage3所有中間乘法的容量上限


def derive_inv_scale_frac(inv_gate_up_scale_all_layers: np.ndarray) -> tuple:
    """
    ⚠️ 2026-09-20發現：這個函式用「全35層共用同一個frac」算出來的
    drop_bits，會讓inv_scale本身偏小的層（例如Layer 0重新校準後
    gate_up_input_scale=0.70，inv_scale只有1.42，是全域最大值的
    1/33）在Stage3中途捨入這一步直接underflow到cos_sim=0——不是精度
    輕微下降，是資訊整個被捨光。實測驗證見
    qat_verification/rmsnorm_verify_stage3_bitwidth.py。

    根因：全域共用的frac/drop_bits是照「最壞情況（inv_scale最大的
    層）」算的，套用到「inv_scale很小的層」時，等於用照顧大數字的
    粗粒度去捨入一個本來就很小的數字。真正的修正是改成
    derive_inv_scale_frac_per_layer()（per-layer各自算），這個函式
    保留下來只給沒有per-layer需求、或想快速看「全域共用會多壞」的
    對照組使用，**不要在正式流程裡用這個函式決定drop_bits**。
    """
    max_val = float(np.abs(inv_gate_up_scale_all_layers).max())
    int_bits = int(np.ceil(np.log2(max_val))) + 1  # +1 sign bit
    container_bits = 16   # INV_GATE_UP_SCALE改用16-bit容器（原本是32-bit）
    frac = container_bits - int_bits
    return max(frac, 0), int_bits


def derive_inv_scale_frac_per_layer(inv_gate_up_scale_all_layers: np.ndarray,
                                      container_bits: int = 16) -> tuple:
    """
    2026-09-20修正版：INV_GATE_UP_SCALE每一層各自用自己的實測值決定
    frac/int_bits，不再全部共用同一組「照顧最壞情況」的格式。

    回傳：
        inv_scale_frac  : (35,) int陣列，每層自己的frac
        inv_scale_int_bits : (35,) int陣列，每層自己的int_bits
    """
    n = len(inv_gate_up_scale_all_layers)
    frac = np.zeros(n, dtype=np.int64)
    int_bits = np.zeros(n, dtype=np.int64)
    for i, val in enumerate(inv_gate_up_scale_all_layers):
        v = float(abs(val))
        ib = int(np.ceil(np.log2(max(v, 1e-9)))) + 1  # +1 sign bit
        ib = max(ib, 1)
        int_bits[i] = ib
        frac[i] = max(container_bits - ib, 0)
    return frac, int_bits


def clip_signed(x, total_bits: int):
    qmax = 2 ** (total_bits - 1) - 1
    qmin = -2 ** (total_bits - 1)
    return np.clip(np.round(x), qmin, qmax).astype(np.int64)


def emit_c_array_2d(name: str, arr: np.ndarray, dtype: str = "int16_t") -> str:
    lines = [f"static const {dtype} {name}[{arr.shape[0]}][{arr.shape[1]}] = {{"]
    for layer_idx in range(arr.shape[0]):
        row = ", ".join(str(int(v)) for v in arr[layer_idx])
        lines.append(f"    {{ {row} }},  // layer {layer_idx}")
    lines.append("};")
    return "\n".join(lines)


def main():
    gamma_pre = np.zeros((NUM_LAYERS, HIDDEN_SIZE), dtype=np.int64)
    gamma_post = np.zeros((NUM_LAYERS, HIDDEN_SIZE), dtype=np.int64)
    step0_m_post = np.zeros((NUM_LAYERS, HIDDEN_SIZE), dtype=np.int64)
    inv_gate_up_scale_float = np.zeros(NUM_LAYERS, dtype=np.float64)
    gamma_pre_raw_max = 0.0
    gamma_post_raw_max = 0.0

    for layer_idx in range(NUM_LAYERS):
        consts = read_layer_constants(layer_idx, SAFETENSORS_PATH, manifest_path=None)

        gamma_pre_raw_max = max(gamma_pre_raw_max, np.abs(consts["gamma_pre_ln"]).max())
        gamma_post_raw_max = max(gamma_post_raw_max, np.abs(consts["gamma_post_ln"]).max())
        gamma_pre[layer_idx] = clip_signed(
            consts["gamma_pre_ln"] * (2 ** GAMMA_PRE_FRAC), GAMMA_BITS)
        gamma_post[layer_idx] = clip_signed(
            consts["gamma_post_ln"] * (2 ** GAMMA_POST_FRAC), GAMMA_BITS)

        inv_gate_up_scale_float[layer_idx] = 1.0 / RECALIBRATED_GATE_UP_SCALE[layer_idx]

        S_down_in = consts["down_proj_input_scale"]
        S_down = consts["down_proj_weight_scale"]
        s_eff = (S_down_in * S_down) / S_RES
        step0_m_post[layer_idx] = clip_signed(s_eff * (2 ** M_FRAC), 32)

    # ---- 2026-09-20修正：INV_GATE_UP_SCALE跟中途捨入的drop_bits/最終
    #      shift全部改成per-layer，不再全35層共用同一組frac ----
    #
    # 全域共用frac的問題（實測發現，見上方derive_inv_scale_frac的
    # docstring）：gate_up_input_scale跨層差異可達30倍以上（Layer 0
    # 重新校準後0.70，Layer 13只有0.0083），若全部層共用「照顧最壞
    # 情況（inv_scale最大的層）」算出來的drop_bits，inv_scale本身偏
    # 小的層（例如Layer 0）會在中途捨入這一步直接underflow——cos_sim
    # 崩潰到0.000000（不是精度下降，是資訊整個被捨光），已用
    # qat_verification/rmsnorm_verify_stage3_bitwidth.py實測驗證過。
    #
    # 修正：每一層各自算自己的inv_scale_frac（用自己的實測值決定int_bits，
    # 不是全域最大值），對應的drop_bits/最終Stage3 shift也變成per-layer
    # （35個值的小表，不是單一#define常數）——這跟INV_GATE_UP_SCALE本身
    # 存成per-layer表是同一個道理，只是這次連「怎麼処理這張表」的shift
    # 邏輯也要per-layer，不能只有表格資料per-layer、處理邏輯卻共用。
    inv_scale_frac, inv_scale_int_bits = derive_inv_scale_frac_per_layer(inv_gate_up_scale_float)
    inv_gate_up_scale_int = clip_signed(
        inv_gate_up_scale_float * (2.0 ** inv_scale_frac), 16)

    # ---- Stage3中途捨入的frac（PRE路徑，per-layer）：每一層各自的
    #      int_bits決定中途捨入後還能留多少frac ----
    inv_rms_scaled_int_bits = RSQ_T_IWIDTH + inv_scale_int_bits  # (35,) 陣列
    inv_rms_scaled_frac = INV_RMS_SCALED_CONTAINER_BITS - inv_rms_scaled_int_bits  # (35,) 陣列
    if np.any(inv_rms_scaled_frac <= 0):
        bad_layers = np.where(inv_rms_scaled_frac <= 0)[0].tolist()
        raise RuntimeError(
            f"INV_RMS_SCALED_CONTAINER_BITS={INV_RMS_SCALED_CONTAINER_BITS}不夠裝下"
            f"以下層的整數部分：{bad_layers}，需要加寬中繼容器")

    # per-layer drop_bits（Step A算完後要捨去多少bit）跟per-layer最終
    # Stage3 shift量（給rmsnorm_stage3()在乘完gamma之後用）
    pre_drop_bits = (RSQ_T_FRAC + inv_scale_frac) - inv_rms_scaled_frac  # (35,)
    pre_final_shift = inv_rms_scaled_frac + GAMMA_PRE_FRAC  # (35,)

    S_RES_EXP = int(round(-np.log2(S_RES)))  # S_RES=2^-5 -> 5

    # ---- Post-Norm路徑也需要中途捨入（2026-09-20新發現，跟Pre-Norm
    #      對稱補上）：inv_rms左移S_RES_EXP位對齊S_RES後，int_bits跟
    #      frac(=RSQ_T_FRAC=20)全部原封不動帶進後面的乘法，Step C'
    #      （x_uniform x inv_rms_scaled x gamma_post）實測需要76-bit，
    #      超過ap_int<64>。用跟Pre-Norm一樣的回推邏輯：先扣掉x_uniform
    #      跟gamma_post各自需要的位元數，剩下的budget才是inv_rms_scaled
    #      能留的frac（int_bits=RSQ_T_IWIDTH+S_RES_EXP是硬性需求，不能
    #      再壓）。----
    X_UNIFORM_BITS = 24  # step0_t容器本身的位元數，PRE/POST共用
    gamma_post_int_bits_for_budget = int(np.ceil(np.log2(gamma_post_raw_max))) + 1
    gamma_post_total_bits = gamma_post_int_bits_for_budget + GAMMA_POST_FRAC
    post_inv_rms_scaled_int_bits = RSQ_T_IWIDTH + S_RES_EXP
    # -1 bit安全餘裕：MULTIPLY_CHAIN_BUDGET_BITS(64)本身是ap_int<64>的
    # 總位元數（含符號位），乘法鏈剛好卡到64-bit邊界時ap_int<64>裝不下
    # （有號數最大只能表示到2^63-1），這裡先扣1-bit margin避免卡邊界。
    post_available_bits = MULTIPLY_CHAIN_BUDGET_BITS - 1 - X_UNIFORM_BITS - gamma_post_total_bits
    post_inv_rms_scaled_frac = post_available_bits - post_inv_rms_scaled_int_bits
    if post_inv_rms_scaled_frac <= 0:
        raise RuntimeError(
            f"Post-Norm的multiply chain budget不夠：需要加寬"
            f"MULTIPLY_CHAIN_BUDGET_BITS或縮減gamma_post/x_uniform的位元數")

    layer_scale_header = f"""\
// 自動產生檔案，請勿手動修改。由 generate_rmsnorm_const_tables.py 產生。
// 2026-09-20第三版（per-layer drop_bits/shift，取代全域共用版本）：
// 全域共用一組frac/drop_bits時，Layer 0這種inv_scale遠小於全域最大值
// 的層，Stage3中途捨入會直接underflow到cos_sim=0.000000（不是精度
// 下降，是資訊整個被捨光，見qat_verification/
// rmsnorm_verify_stage3_bitwidth.py的實測結果）。改成每一層各自的
// int_bits/frac/drop_bits/最終shift，不再共用單一#define。
//
// RMSNORM_INV_GATE_UP_SCALE_FRAC[layer]：每層自己的frac（不是共用值）
// RMSNORM_PRE_DROP_BITS[layer]         ：Step A算完後要捨去的bit數
// RMSNORM_STAGE3_SHIFT_PRE[layer]      ：Stage3最終shift量（取代原本的
//                                        單一#define RMSNORM_STAGE3_SHIFT_PRE）
#ifndef RMSNORM_LAYER_SCALE_TABLE_H
#define RMSNORM_LAYER_SCALE_TABLE_H

#define RMSNORM_INV_S_RES_SHIFT        {S_RES_EXP}
#define RMSNORM_POST_INV_RMS_SCALED_FRAC {post_inv_rms_scaled_frac}   // Post-Norm中途捨入用（全域共用，POST路徑沒有跨層量級差30倍的問題，見bit-width check）

static const int16_t RMSNORM_INV_GATE_UP_SCALE[{NUM_LAYERS}] = {{
    {", ".join(str(int(v)) for v in inv_gate_up_scale_int)}
}};

// 每層自己的frac，用來interpret上面那張表的每一個值（不是共用單一frac）
static const int8_t RMSNORM_INV_GATE_UP_SCALE_FRAC[{NUM_LAYERS}] = {{
    {", ".join(str(int(v)) for v in inv_scale_frac)}
}};

// Step A（inv_rms x inv_scale）算完後，per-layer要右移多少bit才落進
// {INV_RMS_SCALED_CONTAINER_BITS}-bit中繼容器
static const int8_t RMSNORM_PRE_DROP_BITS[{NUM_LAYERS}] = {{
    {", ".join(str(int(v)) for v in pre_drop_bits)}
}};

// Stage3最終shift量（乘完gamma之後要右移多少bit），per-layer
static const int8_t RMSNORM_STAGE3_SHIFT_PRE[{NUM_LAYERS}] = {{
    {", ".join(str(int(v)) for v in pre_final_shift)}
}};

#endif // RMSNORM_LAYER_SCALE_TABLE_H
"""

    gamma_header = f"""\
// 自動產生檔案，請勿手動修改。由 generate_rmsnorm_const_tables.py 產生。
#ifndef RMSNORM_GAMMA_TABLE_H
#define RMSNORM_GAMMA_TABLE_H

#define RMSNORM_GAMMA_PRE_FRAC  {GAMMA_PRE_FRAC}
#define RMSNORM_GAMMA_POST_FRAC {GAMMA_POST_FRAC}

// pre_feedforward_layernorm的raw gamma，round(gamma * 2^{GAMMA_PRE_FRAC})
{emit_c_array_2d("RMSNORM_GAMMA_PRE", gamma_pre, dtype="int16_t")}

// post_feedforward_layernorm的raw gamma，round(gamma * 2^{GAMMA_POST_FRAC})
{emit_c_array_2d("RMSNORM_GAMMA_POST", gamma_post, dtype="int16_t")}

#endif // RMSNORM_GAMMA_TABLE_H
"""

    step0_header = f"""\
// 自動產生檔案，請勿手動修改。由 generate_rmsnorm_const_tables.py 產生。
// post_feedforward_layernorm的Step0 M[layer][ch]（未變動）
#ifndef RMSNORM_STEP0_TABLE_H
#define RMSNORM_STEP0_TABLE_H

{emit_c_array_2d("RMSNORM_STEP0_M_POST", step0_m_post, dtype="int32_t")}

#endif // RMSNORM_STEP0_TABLE_H
"""

    out_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(out_dir, "rmsnorm_layer_scale_table.h"), "w", encoding="utf-8") as f:
        f.write(layer_scale_header)
    with open(os.path.join(out_dir, "rmsnorm_gamma_table.h"), "w", encoding="utf-8") as f:
        f.write(gamma_header)
    with open(os.path.join(out_dir, "rmsnorm_step0_table.h"), "w", encoding="utf-8") as f:
        f.write(step0_header)

    print("已產生 rmsnorm_layer_scale_table.h、rmsnorm_gamma_table.h、rmsnorm_step0_table.h\n")
    print(f"INV_GATE_UP_SCALE_FRAC（per-layer）: min={inv_scale_frac.min()}, max={inv_scale_frac.max()}"
          f"（Layer 0 frac={inv_scale_frac[0]}, Layer 13 frac={inv_scale_frac[13]}——"
          f"跨層frac不再共用同一個值，這是這次修正的重點）")
    print(f"RMSNORM_PRE_DROP_BITS（per-layer）: min={pre_drop_bits.min()}, max={pre_drop_bits.max()}")
    print(f"RMSNORM_STAGE3_SHIFT_PRE（per-layer最終shift）: min={pre_final_shift.min()}, "
          f"max={pre_final_shift.max()}（取代原本單一#define，這裡是35個值的表）")
    print(f"最終Stage3總shift（Post-Norm）= RMSNORM_POST_INV_RMS_SCALED_FRAC"
          f"({post_inv_rms_scaled_frac}) + GAMMA_POST_FRAC({GAMMA_POST_FRAC}) = "
          f"{post_inv_rms_scaled_frac + GAMMA_POST_FRAC}"
          f"（POST路徑沒有跨層inv_scale量級差30倍的問題，繼續維持全域共用一個值）")

    def bits_needed(max_abs_value: float) -> int:
        if max_abs_value <= 0:
            return 1
        return int(np.ceil(np.log2(max_abs_value))) + 1  # +1 sign bit

    print("\n" + "=" * 78)
    print("Stage3 multiply chain位元寬度檢查（Pre-Norm路徑，取全部35層裡最壞的組合）")
    print("=" * 78)
    step_a_bits = int(RSQ_T_IWIDTH + RSQ_T_FRAC + inv_scale_int_bits.max() + inv_scale_frac.max())
    print(f"  Step A（inv_rms x inv_scale）: {step_a_bits}-bit "
          f"({'OK, 落在ap_int<64>內' if step_a_bits <= MULTIPLY_CHAIN_BUDGET_BITS - 1 else '⚠️ 超過ap_int<64>'})")
    x_uniform_int_bits = 24  # step0_t容器本身的位元數（Pre-Norm這裡x_uniform=residual，直接沿用容器寬度）
    # ⚠️ 逐層檢查，不能把不同層的int_bits.max()跟frac.max()分開加總——
    # 兩者是同一層內互補的關係（int_bits[i]+frac[i]恆等於
    # INV_RMS_SCALED_CONTAINER_BITS），但int_bits最大的那一層跟frac
    # 最大的那一層通常不是同一層，分開取max()會虛報一個不存在的組合。
    step_b_bits_per_layer = x_uniform_int_bits + inv_rms_scaled_int_bits + inv_rms_scaled_frac  # 恆為常數
    step_b_bits = int(step_b_bits_per_layer.max())
    print(f"  Step B（x_uniform x inv_rms_scaled，中途捨入後）: {step_b_bits}-bit "
          f"({'OK' if step_b_bits <= MULTIPLY_CHAIN_BUDGET_BITS - 1 else '⚠️ 超過ap_int<64>'})"
          f"  （逐層檢查：每層各自的int_bits+frac固定等於"
          f"{INV_RMS_SCALED_CONTAINER_BITS}-bit中繼容器，by construction不會超）")
    gamma_pre_int_bits = int(np.ceil(np.log2(gamma_pre_raw_max))) + 1
    step_c_bits = step_b_bits + gamma_pre_int_bits + GAMMA_PRE_FRAC
    print(f"  Step C（prod x gamma）: {step_c_bits}-bit "
          f"({'OK' if step_c_bits <= MULTIPLY_CHAIN_BUDGET_BITS - 1 else '⚠️ 超過ap_int<64>'})"
          f"  (gamma_pre實測最大值={gamma_pre_raw_max:.1f}, 整數位元={gamma_pre_int_bits})")

    print("\n" + "=" * 78)
    print("Stage3 multiply chain位元寬度檢查（Post-Norm路徑，2026-09-20修正版）")
    print("=" * 78)
    # 修正前（inv_rms純位移、frac維持RSQ_T_FRAC=20不變）：
    naive_int_bits = RSQ_T_IWIDTH + S_RES_EXP
    naive_total_bits = naive_int_bits + RSQ_T_FRAC
    naive_step_c = X_UNIFORM_BITS + naive_total_bits + gamma_post_int_bits_for_budget + GAMMA_POST_FRAC
    print(f"  [修正前，純位移無中途捨入] inv_rms_scaled: 整數位元={naive_int_bits}, "
          f"frac={RSQ_T_FRAC}, 總位元={naive_total_bits}-bit")
    print(f"  [修正前] Step C'（x_uniform x inv_rms_scaled x gamma）: {naive_step_c}-bit "
          f"({'OK' if naive_step_c <= MULTIPLY_CHAIN_BUDGET_BITS - 1 else '⚠️ 超過ap_int<64>，證實需要修正'})")

    print(f"\n  [修正後，比照Pre-Norm插入中途捨入] "
          f"inv_rms_scaled: 整數位元={post_inv_rms_scaled_int_bits}（固定，= "
          f"RSQ_T_IWIDTH({RSQ_T_IWIDTH})+S_RES_EXP({S_RES_EXP})）, "
          f"frac={post_inv_rms_scaled_frac}（中途捨入後）")
    step_b_post_bits = X_UNIFORM_BITS + post_inv_rms_scaled_int_bits + post_inv_rms_scaled_frac
    print(f"  [修正後] Step B'（x_uniform x inv_rms_scaled，中途捨入後）: {step_b_post_bits}-bit "
          f"({'OK' if step_b_post_bits <= MULTIPLY_CHAIN_BUDGET_BITS - 1 else '⚠️ 超過ap_int<64>'})")
    step_c_post_bits = step_b_post_bits + gamma_post_int_bits_for_budget + GAMMA_POST_FRAC
    print(f"  [修正後] Step C'（prod x gamma）: {step_c_post_bits}-bit "
          f"({'OK' if step_c_post_bits <= MULTIPLY_CHAIN_BUDGET_BITS - 1 else '⚠️ 超過ap_int<64>'})"
          f"  (gamma_post實測最大值={gamma_post_raw_max:.1f}, 整數位元={gamma_post_int_bits_for_budget})")
    print(f"\n結論：Post-Norm路徑{'不需要' if naive_step_c <= MULTIPLY_CHAIN_BUDGET_BITS-1 else '也需要（已修正）'}"
          f"像Pre-Norm路徑那樣插入中途捨入——修正前76-bit會爆ap_int<64>，"
          f"修正後（新增RMSNORM_POST_INV_RMS_SCALED_FRAC中途捨入）落在"
          f"{step_c_post_bits}-bit，安全。rmsnorm_kernel.h的IS_POST_NORM分支"
          f"需要同步更新，不能再是純位移、沒有Step B窄化。")

    print("\n⚠️ 這次新增的中途捨入（INV_RMS_SCALED_FRAC）多了一道捨入步驟，")
    print("   精度影響需要拿golden model重新驗證，不能只看位元寬度算得過")
    print("   去就直接信任數值正確——這點跟residual那邊Plan A/B的教訓一樣：")
    print("   位元預算過得了關，不代表精度過得了關，兩件事要分開驗證。")
    print("   請接著跑 qat_verification/rmsnorm_verify_stage3_bitwidth.py")


if __name__ == "__main__":
    main()
