#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_stage3_testbench_vectors.py
=====================================================================
把真實QAT activation（outputs/real_activations_layer{L}.npz）轉成
C-simulation testbench能吃的測試向量：每一筆包含
    - layer_idx
    - 輸入（Pre-Norm: residual INT16 @ S_RES；Post-Norm: down_proj
      MAC原始整數）
    - 期望輸出（用跟rmsnorm_verify_stage3_bitwidth.py完全相同的
      bit-accurate Python模擬算出來的整數結果——不是float golden
      model、不是cos_sim近似，是逐bit對應hls/rmsnorm_kernel.h實際
      運算步驟的精確整數值）

跟rmsnorm_verify_stage3_bitwidth.py的差異：那支腳本比對的是
「bit-accurate模擬 vs 真實QAT浮點輸出」（cos_sim，允許量化誤差）；
這支腳本產生的testbench要比對的是「C++硬體實作 vs Python bit-accurate
模擬」，兩者做的是完全相同的整數運算，理論上應該**逐bit相同**（差1都
算testbench fail），這是完全不同層次的驗證——上一層驗證「量化這樣做
對不對」，這一層驗證「C++寫的跟設計的整數運算是否一致」。

每層取前N_TOKENS_PER_LAYER個token（不需要全部75個，testbench只是
要驗證運算邏輯正確，不是統計精度），全35層都涵蓋（呼應上一輪「6層
抽樣會漏掉離散failure mode」的教訓，這裡也採全35層）。
"""

import os
import sys
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_GOLDEN_MODEL_DIR = os.path.join(_THIS_DIR, "..", "golden_model")
_QAT_DIR = os.path.join(_THIS_DIR, "..", "qat_verification")
sys.path.insert(0, _GOLDEN_MODEL_DIR)
sys.path.insert(0, _QAT_DIR)

from rmsnorm_golden_model import S_RES
from rmsnorm_verify_pre_post_mlp import read_layer_constants
from layer_scale_calibration import get_all_recalibrated_gate_up_scales
from generate_rmsnorm_const_tables import (
    derive_inv_scale_frac_per_layer, RSQ_T_IWIDTH, GAMMA_PRE_FRAC, GAMMA_POST_FRAC,
    INV_RMS_SCALED_CONTAINER_BITS, MULTIPLY_CHAIN_BUDGET_BITS,
)
from rmsnorm_verify_stage3_bitwidth import simulate_pre_stage3, simulate_post_stage3, RSQ_T_FRAC

M_FRAC = 20  # 跟generate_rmsnorm_const_tables.py的M_FRAC/rmsnorm_kernel.h的RMSNORM_STEP0_M_FRAC一致


def bit_accurate_step0(matmul_raw: np.ndarray, step0_m: np.ndarray) -> np.ndarray:
    """逐bit對應rmsnorm_kernel.h的rmsnorm_step0()：
        prod = matmul_raw * step0_m
        rounded = prod + (1 << (M_FRAC-1))
        shifted = rounded >> M_FRAC
        clip到24-bit
    不是float版本的requantize_m_k()（那個用float s_eff直接算，不是
    C++實際會用的STEP0_M整數表+round-shift），這裡要跟C++逐bit一致，
    testbench才有意義。"""
    prod = matmul_raw.astype(np.int64) * step0_m[None, :].astype(np.int64)
    rounded = prod + (1 << (M_FRAC - 1))
    shifted = rounded >> M_FRAC
    qmax = 2 ** (24 - 1) - 1
    qmin = -2 ** (24 - 1)
    return np.clip(shifted, qmin, qmax)

SAFETENSORS_PATH = (
    "C:/Users/eva huang/.cache/huggingface/hub/"
    "models--google--gemma-4-E2B-it-qat-mobile-transformers/"
    "snapshots/dd693ff40353f057ca5f07e945ad867f4afbf2ec/model.safetensors"
)
OUTPUTS_DIR = os.path.join(_THIS_DIR, "..", "outputs")
NUM_LAYERS = 35
HIDDEN_SIZE = 1536
N_TOKENS_PER_LAYER = 3  # 每層取幾個token當測試向量，求testbench檔案不要太肥

RECALIBRATED_GATE_UP_SCALE = get_all_recalibrated_gate_up_scales()


def emit_c_array_2d(name: str, arr: np.ndarray, dtype: str) -> str:
    lines = [f"static const {dtype} {name}[{arr.shape[0]}][{arr.shape[1]}] = {{"]
    for i in range(arr.shape[0]):
        row = ", ".join(str(int(v)) for v in arr[i])
        lines.append(f"    {{ {row} }},")
    lines.append("};")
    return "\n".join(lines)


def main():
    all_layers_inv_scale = np.array([1.0 / RECALIBRATED_GATE_UP_SCALE[i] for i in range(NUM_LAYERS)])
    inv_scale_frac_all, inv_scale_int_bits_all = derive_inv_scale_frac_per_layer(all_layers_inv_scale)

    pre_input_rows, pre_expected_rows, pre_layer_idx_rows = [], [], []
    post_input_rows, post_expected_rows, post_layer_idx_rows = [], [], []

    # ---- POST路徑的post_mid_frac是全域常數（不是per-layer！），一定要
    # 跟generate_rmsnorm_const_tables.py用完全一樣的算法（掃全部35層的
    # gamma_post_ln取全域最大值），不能在下面主迴圈裡各層各自算——
    # 這正是第一輪c-sim發現的bug：per-layer算出來的frac跟實際燒進
    # rmsnorm_layer_scale_table.h的全域frac對不上，導致POST路徑87個
    # case fail。POST本來就設計成全域共用一個值（S_RES是全域常數，
    # 不像PRE的gate_up_input_scale跨層差異大，見stage3_bitwidth_
    # review.md第4.1節），這裡也要維持全域算法。
    gamma_post_raw_max = 0.0
    for layer_idx in range(NUM_LAYERS):
        c = read_layer_constants(layer_idx, SAFETENSORS_PATH, manifest_path=None)
        gamma_post_raw_max = max(gamma_post_raw_max, np.abs(c["gamma_post_ln"]).max())

    X_UNIFORM_BITS = 24
    gamma_post_int_bits = int(np.ceil(np.log2(gamma_post_raw_max))) + 1
    gamma_post_total_bits = gamma_post_int_bits + GAMMA_POST_FRAC
    s_res_exp = int(round(-np.log2(S_RES)))
    post_inv_rms_scaled_int_bits = RSQ_T_IWIDTH + s_res_exp
    post_available = MULTIPLY_CHAIN_BUDGET_BITS - 1 - X_UNIFORM_BITS - gamma_post_total_bits
    post_mid_frac = post_available - post_inv_rms_scaled_int_bits

    for layer_idx in range(NUM_LAYERS):
        npz_path = os.path.join(OUTPUTS_DIR, f"real_activations_layer{layer_idx}.npz")
        if not os.path.exists(npz_path):
            print(f"[layer {layer_idx}] 找不到{npz_path}，跳過")
            continue
        data = np.load(npz_path)
        consts = read_layer_constants(layer_idx, SAFETENSORS_PATH, manifest_path=None)
        gate_up_scale = RECALIBRATED_GATE_UP_SCALE[layer_idx]

        n_tok = min(N_TOKENS_PER_LAYER, data["residual_pre"].shape[0])

        # ---- PRE路徑測試向量 ----
        residual_pre = data["residual_pre"][:n_tok]
        residual_int16 = np.clip(np.round(residual_pre / S_RES), -32768, 32767).astype(np.int64)

        gamma_pre_int = np.clip(np.round(consts["gamma_pre_ln"] * (2 ** GAMMA_PRE_FRAC)), -32768, 32767).astype(np.int64)
        inv_scale_frac = int(inv_scale_frac_all[layer_idx])
        inv_scale_int_bits = int(inv_scale_int_bits_all[layer_idx])
        inv_scale_val = 1.0 / gate_up_scale
        inv_scale_int = int(np.clip(np.round(inv_scale_val * (2 ** inv_scale_frac)), -32768, 32767))
        mid_frac = INV_RMS_SCALED_CONTAINER_BITS - (RSQ_T_IWIDTH + inv_scale_int_bits)

        pre_expected = simulate_pre_stage3(
            residual_int16, gamma_pre_int, inv_scale_int, inv_scale_frac, mid_frac,
            GAMMA_PRE_FRAC, apply_mid_round=True)

        for t in range(n_tok):
            pre_input_rows.append(residual_int16[t])
            pre_expected_rows.append(pre_expected[t])
            pre_layer_idx_rows.append(layer_idx)

        # ---- POST路徑測試向量 ----
        down_proj_out = data["down_proj_out"][:n_tok]
        S_d_in = consts["down_proj_input_scale"]
        S_d_w = consts["down_proj_weight_scale"]
        down_mac_recon = np.round(down_proj_out / (S_d_in * S_d_w))
        down_mac_recon = np.clip(down_mac_recon, -(2 ** 31), 2 ** 31 - 1)  # acc32_t範圍

        # STEP0_M整數表：跟generate_rmsnorm_const_tables.py算RMSNORM_
        # STEP0_M_POST時完全一樣的公式，不能用float s_eff直接乘，要用
        # 這張表 + bit_accurate_step0() 才會跟C++的rmsnorm_step0()逐bit一致
        s_eff = (S_d_in * S_d_w) / S_RES
        step0_m_int = np.clip(np.round(s_eff * (2 ** M_FRAC)), -(2**31), 2**31 - 1).astype(np.int64)
        x_uniform = bit_accurate_step0(down_mac_recon, step0_m_int)

        gamma_post_int = np.clip(np.round(consts["gamma_post_ln"] * (2 ** GAMMA_POST_FRAC)), -32768, 32767).astype(np.int64)
        # s_res_exp/post_mid_frac已經在迴圈外用全域算法算好，這裡不再
        # 各層各自算（那正是c-sim第一輪抓到的bug）

        post_expected = simulate_post_stage3(
            x_uniform, gamma_post_int, s_res_exp, post_mid_frac,
            GAMMA_POST_FRAC, apply_mid_round=True)

        # down_proj_mac_out是acc32_t（Step0的輸入，不是x_uniform本身），
        # testbench要餵給post_feedforward_layernorm_hls()的是down_mac_recon
        # （matmul MAC後的原始整數），不是requantize後的x_uniform——
        # rmsnorm_step0()會在C++內部自己算Step0。
        for t in range(n_tok):
            post_input_rows.append(down_mac_recon[t].astype(np.int64))
            post_expected_rows.append(post_expected[t])
            post_layer_idx_rows.append(layer_idx)

        print(f"[layer {layer_idx}] 加入 {n_tok} 筆測試向量（PRE+POST）")

    pre_input_arr = np.stack(pre_input_rows)
    pre_expected_arr = np.stack(pre_expected_rows)
    pre_layer_idx_arr = np.array(pre_layer_idx_rows)

    post_input_arr = np.stack(post_input_rows)
    post_expected_arr = np.stack(post_expected_rows)
    post_layer_idx_arr = np.array(post_layer_idx_rows)

    header = f"""\
// 自動產生檔案，請勿手動修改。由 generate_stage3_testbench_vectors.py 產生。
// 全35層 x {N_TOKENS_PER_LAYER}個token的真實QAT activation測試向量，
// 期望輸出是用跟rmsnorm_verify_stage3_bitwidth.py完全相同的
// bit-accurate Python模擬算出來的整數值——tb_rmsnorm.cpp應該要能
// 逐bit對上這些期望值，差1都算fail（這是驗證C++實作跟設計的整數
// 運算是否一致，不是驗證量化精度，量化精度已經在golden model那邊
// 驗證過，見hls/stage3_bitwidth_review.md）。
#ifndef TB_RMSNORM_VECTORS_H
#define TB_RMSNORM_VECTORS_H

#define TB_NUM_PRE_CASES  {pre_input_arr.shape[0]}
#define TB_NUM_POST_CASES {post_input_arr.shape[0]}
#define TB_HIDDEN_SIZE    {HIDDEN_SIZE}

static const int TB_PRE_LAYER_IDX[TB_NUM_PRE_CASES] = {{
    {", ".join(str(int(v)) for v in pre_layer_idx_arr)}
}};

{emit_c_array_2d("TB_PRE_INPUT", pre_input_arr, "int32_t")}

{emit_c_array_2d("TB_PRE_EXPECTED", pre_expected_arr, "int32_t")}

static const int TB_POST_LAYER_IDX[TB_NUM_POST_CASES] = {{
    {", ".join(str(int(v)) for v in post_layer_idx_arr)}
}};

{emit_c_array_2d("TB_POST_INPUT", post_input_arr, "int64_t")}

{emit_c_array_2d("TB_POST_EXPECTED", post_expected_arr, "int32_t")}

#endif // TB_RMSNORM_VECTORS_H
"""

    out_path = os.path.join(_THIS_DIR, "tb_rmsnorm_vectors.h")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(header)
    print(f"\n已產生 {out_path}")
    print(f"PRE測試案例數={pre_input_arr.shape[0]}, POST測試案例數={post_input_arr.shape[0]}")


if __name__ == "__main__":
    main()
