#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rmsnorm_verify_batch.py
=====================================================================
讀 extract_real_rmsnorm_activation_batch.py 產出的 .npz，對每一層、
每一個token分別跑golden model，輸出一張誤差分布表，而不是只看單一
數字。同時特別標注Layer 0（附錄C離群值層）的結果，這是最可能出問題
的地方。
"""

import os
import sys
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# rmsnorm_golden_model.py在../golden_model/，rmsnorm_verify_pre_post_mlp.py
# 跟本檔案同一個資料夾(qat_verification/)，兩個都要能import到。
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_GOLDEN_MODEL_DIR = os.path.join(_THIS_DIR, "..", "golden_model")
sys.path.insert(0, _GOLDEN_MODEL_DIR)
sys.path.insert(0, _THIS_DIR)

from rmsnorm_golden_model import (
    call_pre_feedforward_layernorm, call_post_feedforward_layernorm,
    LayerConstants, S_RES,
)
from rmsnorm_verify_pre_post_mlp import read_layer_constants
from layer_scale_calibration import get_all_recalibrated_gate_up_scales

# 實際本機快取的safetensors路徑（2026-09-16確認，HF_HOME=~/.cache/huggingface）。
SAFETENSORS_PATH = (
    "C:/Users/eva huang/.cache/huggingface/hub/"
    "models--google--gemma-4-E2B-it-qat-mobile-transformers/"
    "snapshots/dd693ff40353f057ca5f07e945ad867f4afbf2ec/model.safetensors"
)

OUTPUTS_DIR = os.path.join(_THIS_DIR, "..", "outputs")
LAYER_LIST = [0, 9, 14, 15, 31, 34]

# 2026-09-19：原本只修Layer 14（75個token樣本），現在改成套用WikiText
# 大樣本（40*512=20480個token，calibration/wikitext_calibrate_layer_
# scales.py）校準出來的全部35層gate_up_input_scale，不再只修一層。
# 校準方法（max/127）跟Layer 14當初的做法一致，只是樣本數大很多、
# 套用範圍擴大到全部層。
RECALIBRATED_GATE_UP_SCALE = get_all_recalibrated_gate_up_scales()


def cos_sim_per_token(ref: np.ndarray, test: np.ndarray) -> np.ndarray:
    """對每一個token（每一row）分別算cos_sim，回傳shape=(n_tokens,)的陣列，
    而不是把整批token攤平成一個向量算單一cos_sim——那樣會被token數量
    稀釋掉個別outlier token的影響，看不出「哪個token特別差」。"""
    dot = np.sum(ref * test, axis=-1)
    n1 = np.linalg.norm(ref, axis=-1)
    n2 = np.linalg.norm(test, axis=-1)
    return dot / (n1 * n2 + 1e-12)


def main():
    print(f"{'Layer':>6} {'#tok':>5} {'pre_ln cos_sim':>28} {'post_ln cos_sim':>28}")
    print(f"{'':>6} {'':>5} {'min':>9}{'mean':>9}{'max':>9}   {'min':>9}{'mean':>9}{'max':>9}")
    print("-" * 90)

    worst_layer = None
    worst_cos = 1.0

    for layer_idx in LAYER_LIST:
        npz_path = os.path.join(OUTPUTS_DIR, f"real_activations_layer{layer_idx}.npz")
        if not os.path.exists(npz_path):
            print(f"[layer {layer_idx}] 找不到 {npz_path}，跳過"
                  f"（先跑extract_real_rmsnorm_activation_batch.py）")
            continue

        data = np.load(npz_path)
        residual_pre = data["residual_pre"]
        real_pre_ln_out = data["pre_ln_out"]
        real_down_proj_out = data["down_proj_out"]
        real_post_ln_out = data["post_ln_out"]

        consts = read_layer_constants(layer_idx, SAFETENSORS_PATH, manifest_path=None)
        if layer_idx in RECALIBRATED_GATE_UP_SCALE:
            old_scale = consts["gate_up_input_scale"]
            consts["gate_up_input_scale"] = RECALIBRATED_GATE_UP_SCALE[layer_idx]
            print(f"[layer {layer_idx}] gate_up_input_scale重新校準："
                  f"{old_scale:.6f} -> {consts['gate_up_input_scale']:.6f}"
                  f"（outlier channel修正，見RECALIBRATED_GATE_UP_SCALE）")
        hidden_size = 1536

        lc_stub = LayerConstants(
            layer_idx=layer_idx, head_dim=256,
            qkv_input_scale=0.0, gate_up_input_scale=consts["gate_up_input_scale"],
            q_proj_weight_scale=np.zeros(1), k_proj_weight_scale=None, v_proj_weight_scale=None,
            o_proj_weight_scale=np.zeros(1),
            down_proj_weight_scale=consts["down_proj_weight_scale"],
            per_layer_proj_weight_scale=np.zeros(1),
            o_proj_input_scale=0.0, down_proj_input_scale=consts["down_proj_input_scale"],
            per_layer_proj_input_scale=0.0,
            gamma_input_layernorm=np.zeros(hidden_size),
            gamma_q_norm=np.zeros(1), gamma_k_norm=None, gamma_v_norm=None,
            gamma_post_attention_layernorm=np.zeros(hidden_size),
            gamma_pre_feedforward_layernorm=consts["gamma_pre_ln"],
            gamma_post_feedforward_layernorm=consts["gamma_post_ln"],
            gamma_post_per_layer_input_norm=np.zeros(hidden_size),
        )

        # ---- pre_feedforward_layernorm：逐token跑 ----
        residual_int16 = np.clip(np.round(residual_pre / S_RES), -32768, 32767)
        pre_int_out = np.stack([
            call_pre_feedforward_layernorm(residual_int16[i], lc_stub)
            for i in range(residual_int16.shape[0])
        ])
        pre_int_dq = pre_int_out.astype(np.float64) * consts["gate_up_input_scale"]
        pre_cos = cos_sim_per_token(real_pre_ln_out, pre_int_dq)

        # ---- post_feedforward_layernorm：逐token跑（含Step0反推）----
        S_down_in = consts["down_proj_input_scale"]
        S_down = consts["down_proj_weight_scale"]
        down_mac_recon = np.round(real_down_proj_out / (S_down_in * S_down))
        post_int_out = np.stack([
            call_post_feedforward_layernorm(down_mac_recon[i], lc_stub, S_down_input=S_down_in)
            for i in range(down_mac_recon.shape[0])
        ])
        post_int_dq = post_int_out.astype(np.float64) * S_RES
        post_cos = cos_sim_per_token(real_post_ln_out, post_int_dq)

        n = residual_pre.shape[0]
        print(f"{layer_idx:>6} {n:>5} "
              f"{pre_cos.min():>9.6f}{pre_cos.mean():>9.6f}{pre_cos.max():>9.6f}   "
              f"{post_cos.min():>9.6f}{post_cos.mean():>9.6f}{post_cos.max():>9.6f}")

        layer_worst = min(pre_cos.min(), post_cos.min())
        if layer_worst < worst_cos:
            worst_cos = layer_worst
            worst_layer = layer_idx

    print("-" * 90)
    if worst_layer is not None:
        print(f"最差的cos_sim出現在 Layer {worst_layer}：{worst_cos:.6f}")
        if worst_layer == 0:
            print("  -> Layer 0是附錄C的離群值層，如果這裡誤差明顯比其他層大，"
                  "代表目前的Q-format/位元寬度沒有涵蓋到離群值範圍，"
                  "需要考慮Layer 0獨立常數格式（文件7節open item 0(a)）")
        if worst_cos < 0.999:
            print("  -> cos_sim掉到0.999以下，建議在開始寫HLS之前先查清楚原因，"
                  "不要帶著已知的精度風險進硬體實作階段")


if __name__ == "__main__":
    main()
