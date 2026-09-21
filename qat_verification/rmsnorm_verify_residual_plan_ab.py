#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rmsnorm_verify_residual_plan_ab.py
=====================================================================
兩件事：
    (1) 驗證Attention分支（post_attention_layernorm）跟PLE分支
        （post_per_layer_input_norm）的golden model——這兩個呼叫點
        之前完全沒有真實checkpoint驗證過，residual的完整正確性判斷
        不能只靠MLP這一條分支（pre/post_feedforward_layernorm）。
    (2) 用(1)驗證過的三個分支delta（attn/mlp/ple）當輸入，做residual
        golden model的方案A vs 方案B對照實驗：
            方案A：3個delta+residual_in先用整數加總，最後才乘
                   layer_scalar、捨入一次（run_decoder_layer_residual）
            方案B：4項（residual_in + 3個delta）各自先乘layer_scalar、
                   獨立捨入，再加總（run_decoder_layer_residual_planB）
        比較兩者跟真實layer_output（= 官方forward()這一層的實際輸出）
        的cos_sim/MSE，看哪個方案的量化誤差更小、差距大不大。

前置需求：先跑extract_real_residual_activation.py，取得
outputs/real_residual_layer{L}.npz。
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
    call_post_attention_layernorm, call_post_feedforward_layernorm,
    call_post_per_layer_input_norm, LayerConstants, S_RES,
)
from residual_golden_model import (
    DecoderLayerDeltas, run_decoder_layer_residual,
    run_decoder_layer_residual_planB, LAYER_SCALAR,
)
from rmsnorm_verify_pre_post_mlp import read_layer_constants, rmsnorm_float_reference
from read_attn_ple_constants import read_attn_constants, read_ple_constants
from layer_scale_calibration import get_all_recalibrated_gate_up_scales

SAFETENSORS_PATH = (
    "C:/Users/eva huang/.cache/huggingface/hub/"
    "models--google--gemma-4-E2B-it-qat-mobile-transformers/"
    "snapshots/dd693ff40353f057ca5f07e945ad867f4afbf2ec/model.safetensors"
)
OUTPUTS_DIR = os.path.join(_THIS_DIR, "..", "outputs")
LAYER_LIST = [0, 9, 14, 15, 31, 34]
HIDDEN_SIZE = 1536

RECALIBRATED_GATE_UP_SCALE = get_all_recalibrated_gate_up_scales()


def cos_sim_per_token(ref, test):
    dot = np.sum(ref * test, axis=-1)
    n1 = np.linalg.norm(ref, axis=-1)
    n2 = np.linalg.norm(test, axis=-1)
    return dot / (n1 * n2 + 1e-12)


def evaluate(ref, test, label=""):
    cos = cos_sim_per_token(ref, test)
    mse = np.mean((ref - test) ** 2)
    mae = np.mean(np.abs(ref - test))
    print(f"  [{label}] cos_sim min/mean/max = {cos.min():.6f}/{cos.mean():.6f}/{cos.max():.6f}  "
          f"MSE={mse:.4f}  MAE={mae:.4f}")
    return cos, mse, mae


def build_lc(layer_idx, mlp_consts, attn_consts, ple_consts):
    return LayerConstants(
        layer_idx=layer_idx, head_dim=256,
        qkv_input_scale=0.0, gate_up_input_scale=mlp_consts["gate_up_input_scale"],
        q_proj_weight_scale=np.zeros(1), k_proj_weight_scale=None, v_proj_weight_scale=None,
        o_proj_weight_scale=attn_consts["o_proj_weight_scale"],
        down_proj_weight_scale=mlp_consts["down_proj_weight_scale"],
        per_layer_proj_weight_scale=ple_consts["per_layer_proj_weight_scale"],
        o_proj_input_scale=attn_consts["o_proj_input_scale"],
        down_proj_input_scale=mlp_consts["down_proj_input_scale"],
        per_layer_proj_input_scale=ple_consts["per_layer_proj_input_scale"],
        gamma_input_layernorm=np.zeros(HIDDEN_SIZE),
        gamma_q_norm=np.zeros(1), gamma_k_norm=None, gamma_v_norm=None,
        gamma_post_attention_layernorm=attn_consts["gamma_post_attention_layernorm"],
        gamma_pre_feedforward_layernorm=mlp_consts["gamma_pre_ln"],
        gamma_post_feedforward_layernorm=mlp_consts["gamma_post_ln"],
        gamma_post_per_layer_input_norm=ple_consts["gamma_post_per_layer_input_norm"],
    )


def main():
    planA_final_cos_all = []
    planB_final_cos_all = []

    for layer_idx in LAYER_LIST:
        print("\n" + "=" * 80)
        print(f"Layer {layer_idx}")
        print("=" * 80)

        npz_path = os.path.join(OUTPUTS_DIR, f"real_residual_layer{layer_idx}.npz")
        if not os.path.exists(npz_path):
            print(f"  找不到{npz_path}，跳過（先跑extract_real_residual_activation.py）")
            continue
        data = np.load(npz_path)

        mlp_consts = read_layer_constants(layer_idx, SAFETENSORS_PATH, manifest_path=None)
        mlp_consts["gate_up_input_scale"] = RECALIBRATED_GATE_UP_SCALE[layer_idx]
        attn_consts = read_attn_constants(layer_idx, SAFETENSORS_PATH)
        ple_consts = read_ple_constants(layer_idx, SAFETENSORS_PATH)
        lc = build_lc(layer_idx, mlp_consts, attn_consts, ple_consts)

        # =================================================================
        # (1) Attention分支 golden model驗證（浮點參考 + 整數golden model）
        # =================================================================
        print("\n-- (1a) Attention分支 (post_attention_layernorm) --")
        o_proj_out = data["o_proj_out"]
        real_attn_delta = data["attn_delta"]

        fp_attn = rmsnorm_float_reference(o_proj_out, attn_consts["gamma_post_attention_layernorm"])
        evaluate(real_attn_delta, fp_attn, "浮點參考 vs 真實輸出")

        S_o_in = attn_consts["o_proj_input_scale"]
        S_o_w = attn_consts["o_proj_weight_scale"]
        o_mac_recon = np.round(o_proj_out / (S_o_in * S_o_w))
        attn_delta_int = call_post_attention_layernorm(o_mac_recon, lc, S_o_proj_input=S_o_in)
        attn_delta_dq = attn_delta_int.astype(np.float64) * S_RES
        evaluate(real_attn_delta, attn_delta_dq, "整數golden model(dequant) vs 真實輸出")

        # =================================================================
        # (1b) PLE分支 golden model驗證
        # =================================================================
        print("\n-- (1b) PLE分支 (post_per_layer_input_norm) --")
        per_layer_proj_out = data["per_layer_proj_out"]
        real_ple_delta = data["ple_delta"]

        fp_ple = rmsnorm_float_reference(per_layer_proj_out, ple_consts["gamma_post_per_layer_input_norm"])
        evaluate(real_ple_delta, fp_ple, "浮點參考 vs 真實輸出")

        S_p_in = ple_consts["per_layer_proj_input_scale"]
        S_p_w = ple_consts["per_layer_proj_weight_scale"]
        p_mac_recon = np.round(per_layer_proj_out / (S_p_in * S_p_w))
        ple_delta_int = call_post_per_layer_input_norm(p_mac_recon, lc, S_ple_input=S_p_in)
        ple_delta_dq = ple_delta_int.astype(np.float64) * S_RES
        evaluate(real_ple_delta, ple_delta_dq, "整數golden model(dequant) vs 真實輸出")

        # =================================================================
        # (1c) MLP分支（複用已驗證過的邏輯，這裡重算一次確保跟同一批
        #      token對齊，方便residual組裝）
        # =================================================================
        print("\n-- (1c) MLP分支 (pre/post_feedforward_layernorm，重用已驗證邏輯) --")
        down_proj_out = data["down_proj_out"]
        real_mlp_delta = data["mlp_delta"]
        S_d_in = mlp_consts["down_proj_input_scale"]
        S_d_w = mlp_consts["down_proj_weight_scale"]
        d_mac_recon = np.round(down_proj_out / (S_d_in * S_d_w))
        mlp_delta_int = call_post_feedforward_layernorm(d_mac_recon, lc, S_down_input=S_d_in)
        mlp_delta_dq = mlp_delta_int.astype(np.float64) * S_RES
        evaluate(real_mlp_delta, mlp_delta_dq, "整數golden model(dequant) vs 真實輸出")

        # =================================================================
        # (2) Residual 方案A vs 方案B 對照實驗
        # =================================================================
        print("\n-- (2) Residual 方案A vs 方案B --")

        # residual_in（進入這一層之前的值）用真實浮點反推：
        # residual_after_add1 = residual_in + attn_delta_real
        # => residual_in = residual_after_add1 - attn_delta_real（精確減法，無量化誤差）
        residual_after_add1 = data["residual_after_add1"]
        residual_in_real = residual_after_add1 - real_attn_delta
        residual_in_int = np.clip(np.round(residual_in_real / S_RES), -32768, 32767)

        real_layer_output = data["layer_output"]  # 官方forward()這一層的真實輸出（乘完layer_scalar後）

        deltas = DecoderLayerDeltas(
            attn_delta=attn_delta_int, mlp_delta=mlp_delta_int, ple_delta=ple_delta_int)

        planA_out = run_decoder_layer_residual(residual_in_int, deltas, layer_idx)
        planA_dq = planA_out.astype(np.float64) * S_RES

        planB_out = run_decoder_layer_residual_planB(residual_in_int, deltas, layer_idx)
        planB_dq = planB_out.astype(np.float64) * S_RES

        cosA, mseA, maeA = evaluate(real_layer_output, planA_dq, "方案A vs 真實layer輸出")
        cosB, mseB, maeB = evaluate(real_layer_output, planB_dq, "方案B vs 真實layer輸出")

        planA_final_cos_all.append(cosA)
        planB_final_cos_all.append(cosB)

        diff_AB = np.mean(np.abs(planA_dq - planB_dq))
        print(f"  方案A vs 方案B 彼此的差異 MAE = {diff_AB:.6f}")
        winner = "A" if mseA < mseB else ("B" if mseB < mseA else "平手")
        print(f"  這一層誰的MSE較小: 方案{winner}")

    print("\n" + "=" * 80)
    print("總結：方案A vs 方案B 全部抽樣層彙整")
    print("=" * 80)
    if planA_final_cos_all:
        allA = np.concatenate(planA_final_cos_all)
        allB = np.concatenate(planB_final_cos_all)
        print(f"方案A 全部token cos_sim: min={allA.min():.6f} mean={allA.mean():.6f} max={allA.max():.6f}")
        print(f"方案B 全部token cos_sim: min={allB.min():.6f} mean={allB.mean():.6f} max={allB.max():.6f}")
        print(f"平均cos_sim差距 (A-B) = {allA.mean()-allB.mean():+.8f}")


if __name__ == "__main__":
    main()
