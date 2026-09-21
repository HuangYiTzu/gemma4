#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rmsnorm_verify_gamma_quant_ablation.py
=====================================================================
到目前為止所有batch驗證（rmsnorm_verify_batch.py）算出來的cos_sim，
golden model裡用的gamma都還是全精度float64——golden model的
rmsnorm_core()本來就是直接用浮點算gamma/S_target，從沒真的模擬過
generate_rmsnorm_const_tables.py把gamma量化成INT16
（GAMMA_PRE_FRAC=5、GAMMA_POST_FRAC=8）這一步造成的誤差。

這支腳本做兩組對照，隔離出「gamma量化」這一步單獨造成多少誤差：
    (A) gamma保持全精度float，其餘量化步驟（Step0/rsqrt/residual INT16/
        輸出INT8-16）都跟現在一樣
    (B) gamma也套用跟HLS常數表相同的INT16量化（round(gamma*2^FRAC)/2^FRAC，
        FRAC值直接從generate_rmsnorm_const_tables.py算出來的
        GAMMA_PRE_FRAC/GAMMA_POST_FRAC讀，不是另外猜的）

如果(A)跟(B)的cos_sim幾乎沒差（例如都在0.9999以上互相接近），代表gamma
量化不是誤差主要來源，可以放心；如果(B)明顯比(A)差，代表FRAC選太小，
需要加寬GAMMA_BITS或降低犧牲的精度換位元寬。

同時針對Layer 14目前最差的token，把(A)/(B)兩種gamma處理方式的結果都
印出來，看看gamma量化是否是造成該token cos_sim=0.7997的原因。
"""

import os
import sys
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_GOLDEN_MODEL_DIR = os.path.join(_THIS_DIR, "..", "golden_model")
sys.path.insert(0, _GOLDEN_MODEL_DIR)
sys.path.insert(0, _THIS_DIR)

from rmsnorm_golden_model import (
    call_pre_feedforward_layernorm, call_post_feedforward_layernorm,
    LayerConstants, S_RES,
)
from rmsnorm_verify_pre_post_mlp import read_layer_constants

SAFETENSORS_PATH = (
    "C:/Users/eva huang/.cache/huggingface/hub/"
    "models--google--gemma-4-E2B-it-qat-mobile-transformers/"
    "snapshots/dd693ff40353f057ca5f07e945ad867f4afbf2ec/model.safetensors"
)
OUTPUTS_DIR = os.path.join(_THIS_DIR, "..", "outputs")
LAYER_LIST = [0, 9, 14, 15, 31, 34]

# 跟generate_rmsnorm_const_tables.py算出來的FRAC值保持一致，不是另外猜的。
# 如果之後重跑generate_rmsnorm_const_tables.py算出不同FRAC（例如checkpoint
# 更新），這裡也要跟著更新，或改成直接import該腳本的計算邏輯。
GAMMA_PRE_FRAC = 5
GAMMA_POST_FRAC = 8


def quantize_gamma(gamma: np.ndarray, frac: int, bits: int = 16) -> np.ndarray:
    """模擬generate_rmsnorm_const_tables.py實際會做的事：
    round(gamma*2^frac)後clip進bits位元，再除回2^frac還原成浮點值。
    這個「量化再dequant」的結果，才是HLS真正會用到的gamma數值。"""
    qmax = 2 ** (bits - 1) - 1
    qmin = -2 ** (bits - 1)
    q = np.clip(np.round(gamma * (2 ** frac)), qmin, qmax)
    return q / (2 ** frac)


def cos_sim_per_token(ref: np.ndarray, test: np.ndarray) -> np.ndarray:
    dot = np.sum(ref * test, axis=-1)
    n1 = np.linalg.norm(ref, axis=-1)
    n2 = np.linalg.norm(test, axis=-1)
    return dot / (n1 * n2 + 1e-12)


def build_lc(layer_idx, consts, gamma_pre, gamma_post):
    hidden_size = 1536
    return LayerConstants(
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
        gamma_pre_feedforward_layernorm=gamma_pre,
        gamma_post_feedforward_layernorm=gamma_post,
        gamma_post_per_layer_input_norm=np.zeros(hidden_size),
    )


def run_layer(layer_idx):
    npz_path = os.path.join(OUTPUTS_DIR, f"real_activations_layer{layer_idx}.npz")
    if not os.path.exists(npz_path):
        print(f"[layer {layer_idx}] 找不到 {npz_path}，跳過")
        return None

    data = np.load(npz_path)
    residual_pre = data["residual_pre"]
    real_pre_ln_out = data["pre_ln_out"]
    real_down_proj_out = data["down_proj_out"]
    real_post_ln_out = data["post_ln_out"]

    consts = read_layer_constants(layer_idx, SAFETENSORS_PATH, manifest_path=None)

    gamma_pre_float = consts["gamma_pre_ln"]
    gamma_post_float = consts["gamma_post_ln"]
    gamma_pre_q = quantize_gamma(gamma_pre_float, GAMMA_PRE_FRAC)
    gamma_post_q = quantize_gamma(gamma_post_float, GAMMA_POST_FRAC)

    lc_float = build_lc(layer_idx, consts, gamma_pre_float, gamma_post_float)
    lc_quant = build_lc(layer_idx, consts, gamma_pre_q, gamma_post_q)

    # ---- pre_feedforward_layernorm ----
    residual_int16 = np.clip(np.round(residual_pre / S_RES), -32768, 32767)
    pre_out_float = np.stack([call_pre_feedforward_layernorm(residual_int16[i], lc_float)
                               for i in range(residual_int16.shape[0])])
    pre_out_quant = np.stack([call_pre_feedforward_layernorm(residual_int16[i], lc_quant)
                               for i in range(residual_int16.shape[0])])
    pre_dq_float = pre_out_float.astype(np.float64) * consts["gate_up_input_scale"]
    pre_dq_quant = pre_out_quant.astype(np.float64) * consts["gate_up_input_scale"]
    pre_cos_float = cos_sim_per_token(real_pre_ln_out, pre_dq_float)
    pre_cos_quant = cos_sim_per_token(real_pre_ln_out, pre_dq_quant)

    # ---- post_feedforward_layernorm ----
    S_down_in = consts["down_proj_input_scale"]
    S_down = consts["down_proj_weight_scale"]
    down_mac_recon = np.round(real_down_proj_out / (S_down_in * S_down))
    post_out_float = np.stack([call_post_feedforward_layernorm(down_mac_recon[i], lc_float, S_down_input=S_down_in)
                                for i in range(down_mac_recon.shape[0])])
    post_out_quant = np.stack([call_post_feedforward_layernorm(down_mac_recon[i], lc_quant, S_down_input=S_down_in)
                                for i in range(down_mac_recon.shape[0])])
    post_dq_float = post_out_float.astype(np.float64) * S_RES
    post_dq_quant = post_out_quant.astype(np.float64) * S_RES
    post_cos_float = cos_sim_per_token(real_post_ln_out, post_dq_float)
    post_cos_quant = cos_sim_per_token(real_post_ln_out, post_dq_quant)

    return {
        "n": residual_pre.shape[0],
        "pre_cos_float": pre_cos_float, "pre_cos_quant": pre_cos_quant,
        "post_cos_float": post_cos_float, "post_cos_quant": post_cos_quant,
    }


def main():
    print(f"{'Layer':>6} {'#tok':>5}  {'pre(float)':>11} {'pre(INT16-γ)':>13} {'Δ':>9}   "
          f"{'post(float)':>11} {'post(INT16-γ)':>13} {'Δ':>9}")
    print("-" * 100)

    for layer_idx in LAYER_LIST:
        r = run_layer(layer_idx)
        if r is None:
            continue
        pre_f, pre_q = r["pre_cos_float"].mean(), r["pre_cos_quant"].mean()
        post_f, post_q = r["post_cos_float"].mean(), r["post_cos_quant"].mean()
        print(f"{layer_idx:>6} {r['n']:>5}  {pre_f:>11.6f} {pre_q:>13.6f} {pre_q-pre_f:>+9.6f}   "
              f"{post_f:>11.6f} {post_q:>13.6f} {post_q-post_f:>+9.6f}")

        if layer_idx == 14:
            worst_idx = np.argmin(r["pre_cos_float"])
            print(f"\n[Layer 14 診斷] 最差token idx={worst_idx}:")
            print(f"  gamma全精度float    cos_sim = {r['pre_cos_float'][worst_idx]:.6f}")
            print(f"  gamma INT16量化後   cos_sim = {r['pre_cos_quant'][worst_idx]:.6f}")
            diff = r["pre_cos_quant"][worst_idx] - r["pre_cos_float"][worst_idx]
            print(f"  差異 = {diff:+.6f}")
            if abs(diff) < 0.001:
                print("  -> gamma量化幾乎不影響這個token，不是這裡誤差的主因，"
                      "維持原本『小訊號token INT8精細度不足』的診斷")
            else:
                print("  -> gamma量化對這個token有明顯影響，需要重新檢視"
                      "GAMMA_PRE_FRAC是否選太小")

    print("-" * 100)


if __name__ == "__main__":
    main()
