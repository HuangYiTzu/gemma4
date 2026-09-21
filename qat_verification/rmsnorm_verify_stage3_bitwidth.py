#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rmsnorm_verify_stage3_bitwidth.py
=====================================================================
位元預算過得了關，不代表精度過得了關——這支腳本驗證
generate_rmsnorm_const_tables.py（方案1+方案2混合）新增的Stage3中途
捨入（PRE路徑的RMSNORM_INV_RMS_SCALED_FRAC、POST路徑的
RMSNORM_POST_INV_RMS_SCALED_FRAC），到底讓cos_sim掉多少。

跟residual那邊Plan A/B實驗的關係：性質相反但道理一致。residual那次
是「捨入次數越少越好」（方案A單一捨入贏方案B四次捨入）；這次是
「被迫要多插一次中途捨入才裝得下64-bit乘法鏈」——不是「捨入次數
越少/越多越好」的通則，是每一步的動態範圍是否真的撐得住，要實測
才知道。

比對三種版本：
    (1) 真實QAT模型輸出（ground truth）
    (2) golden model理想版（rmsnorm_core的float backend，完全不受
        位元寬度限制，等於「這條pipeline理論上的數學上限」）
    (3) bit-accurate硬體模擬版，兩種：
        (3a) 有中途捨入（實際會燒進硬體的版本，用
             generate_rmsnorm_const_tables.py算出來的真實FRAC值）
        (3b) 無中途捨入（假設乘法鏈可以無限寬，只是為了量化「中途
             捨入」這一步單獨造成多少額外誤差，不是真的可實作方案）
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
    fast_inv_unit_python, requantize_m_k,
    LayerConstants, S_RES,
)
from rmsnorm_verify_pre_post_mlp import read_layer_constants
from layer_scale_calibration import get_all_recalibrated_gate_up_scales
from generate_rmsnorm_const_tables import (
    derive_inv_scale_frac_per_layer, RSQ_T_IWIDTH, GAMMA_PRE_FRAC, GAMMA_POST_FRAC,
    INV_RMS_SCALED_CONTAINER_BITS, MULTIPLY_CHAIN_BUDGET_BITS,
)

SAFETENSORS_PATH = (
    "C:/Users/eva huang/.cache/huggingface/hub/"
    "models--google--gemma-4-E2B-it-qat-mobile-transformers/"
    "snapshots/dd693ff40353f057ca5f07e945ad867f4afbf2ec/model.safetensors"
)
OUTPUTS_DIR = os.path.join(_THIS_DIR, "..", "outputs")
# 2026-09-20：從6層抽樣擴大到全35層，理由見hls/stage3_bitwidth_review.md
# 第7節——Layer 0 underflow是離散、特定層才會踩到的failure mode，
# 6層抽樣不保證其他29層裡沒有另一個同樣偏離全域值的層。
LAYER_LIST = list(range(35))
HIDDEN_SIZE = 1536
RSQ_T_FRAC = 20  # 跟rmsnorm_kernel.h的RSQ_T_FRAC/kernel.h的rsq_t一致

RECALIBRATED_GATE_UP_SCALE = get_all_recalibrated_gate_up_scales()


def cos_sim_per_token(ref, test):
    dot = np.sum(ref * test, axis=-1)
    n1 = np.linalg.norm(ref, axis=-1)
    n2 = np.linalg.norm(test, axis=-1)
    return dot / (n1 * n2 + 1e-12)


def evaluate(ref, test, label):
    cos = cos_sim_per_token(ref, test)
    mse = np.mean((ref - test) ** 2)
    print(f"  [{label}] cos_sim min/mean/max = {cos.min():.6f}/{cos.mean():.6f}/{cos.max():.6f}  MSE={mse:.4f}")
    return cos, mse


def round_shift(x_int: np.ndarray, shift_bits: int) -> np.ndarray:
    """round-then-shift，逐位元對應C++的 (x + (1<<(shift-1))) >> shift。"""
    if shift_bits <= 0:
        return x_int
    half = 1 << (shift_bits - 1)
    return (x_int + half) >> shift_bits


MS_T_FRAC = 24  # kernel_stub.h的ms_t = ap_ufixed<56,32,AP_RND>，frac=56-32=24


def bit_accurate_mean_sq(x_int: np.ndarray, D: int, ms_frac: int = MS_T_FRAC) -> np.ndarray:
    """對應rmsnorm_kernel.h的rmsnorm_stage1()：
        ssq = sum(x[i]^2)                       (ssq_t累加，exact，D=1536遠
                                                  小於float64能精確表示的範圍)
        inv_d = (ms_t)(1.0/D)                    (把1/D量化成ms_t的24-bit
                                                  frac，不是精確除法！)
        mean_sq = (ms_t)(ssq * inv_d)             (乘完再round回ms_t格式)

    2026-09-20 c-simulation發現：一開始這裡直接用np.mean()做精確除法，
    跟C++實際做的「1/D先量化成24-bit定點常數、乘完再round」不是同一個
    數字——雖然量化誤差極小（~1/2^24），但fast_inv_unit()的leading-one
    -detector對輸入的bit pattern很敏感，這個微小誤差在某些token上剛好
    卡在detector的邊界，導致inv_rms算出來的值明顯偏移（實測某個token
    整個1536維輸出都systematically偏差2~4%）。testbench比對出這個問題
    後才發現：golden model的「bit-accurate」模擬其實漏了這一步的定點
    量化，不是真的逐bit對應C++。"""
    ssq = np.sum(x_int.astype(np.float64) ** 2, axis=-1, keepdims=True)  # exact
    inv_d_raw = round((1.0 / D) * (2 ** ms_frac))
    inv_d = inv_d_raw / (2 ** ms_frac)
    mean_sq_real = ssq * inv_d
    mean_sq_int = np.round(mean_sq_real * (2 ** ms_frac)) / (2 ** ms_frac)  # round回ms_t frac
    return mean_sq_int


def simulate_pre_stage3(residual_int16, gamma_pre_int, inv_scale_int,
                         inv_scale_frac, mid_frac, gamma_frac,
                         apply_mid_round: bool):
    """bit-accurate（或無中途捨入的對照版）模擬Pre-Norm的Stage3。
    residual_int16: (n_tok, D) int64
    gamma_pre_int : (D,) int64，已經是round(gamma*2^gamma_frac)後的整數
    """
    D = residual_int16.shape[-1]
    mean_sq_int = bit_accurate_mean_sq(residual_int16, D)
    mean_sq_int = np.maximum(mean_sq_int, 1.0)
    inv_rms_true = fast_inv_unit_python(mean_sq_int, mode_recip=False, rt_frac=RSQ_T_FRAC)
    inv_rms_raw = np.round(inv_rms_true * (2 ** RSQ_T_FRAC)).astype(np.int64)  # (n_tok,1)

    # Step A
    inv_rms_scaled_wide = inv_rms_raw * inv_scale_int  # (n_tok,1)

    if apply_mid_round:
        drop_bits = (RSQ_T_FRAC + inv_scale_frac) - mid_frac
        inv_rms_scaled = round_shift(inv_rms_scaled_wide, drop_bits)
        final_shift = mid_frac + gamma_frac
    else:
        inv_rms_scaled = inv_rms_scaled_wide
        final_shift = (RSQ_T_FRAC + inv_scale_frac) + gamma_frac

    prod = residual_int16.astype(np.int64) * inv_rms_scaled  # broadcast (n_tok,D)
    prod = prod * gamma_pre_int[None, :]
    y_int = round_shift(prod, final_shift)
    y_int = np.clip(y_int, -128, 127)  # act8_t
    return y_int


def simulate_post_stage3(x_uniform, gamma_post_int, s_res_exp, post_mid_frac,
                          gamma_frac, apply_mid_round: bool):
    D = x_uniform.shape[-1]
    mean_sq_int = bit_accurate_mean_sq(x_uniform, D)
    mean_sq_int = np.maximum(mean_sq_int, 1.0)
    inv_rms_true = fast_inv_unit_python(mean_sq_int, mode_recip=False, rt_frac=RSQ_T_FRAC)
    inv_rms_raw = np.round(inv_rms_true * (2 ** RSQ_T_FRAC)).astype(np.int64)

    inv_rms_shifted = inv_rms_raw << s_res_exp

    if apply_mid_round:
        drop_bits = RSQ_T_FRAC - post_mid_frac
        inv_rms_scaled = round_shift(inv_rms_shifted, drop_bits)
        final_shift = post_mid_frac + gamma_frac
    else:
        inv_rms_scaled = inv_rms_shifted
        final_shift = RSQ_T_FRAC + gamma_frac

    prod = x_uniform.astype(np.int64) * inv_rms_scaled
    prod = prod * gamma_post_int[None, :]
    y_int = round_shift(prod, final_shift)
    y_int = np.clip(y_int, -32768, 32767)  # hid16_t
    return y_int


def main():
    print("=" * 90)
    print("Stage3中途捨入精度驗證：有中途捨入(bit-accurate硬體版) vs 無中途捨入(理論上限) vs 真實輸出")
    print("=" * 90)

    all_pre_with_cos, all_pre_without_cos = [], []
    all_post_with_cos, all_post_without_cos = [], []

    for layer_idx in LAYER_LIST:
        npz_path = os.path.join(OUTPUTS_DIR, f"real_activations_layer{layer_idx}.npz")
        if not os.path.exists(npz_path):
            continue
        data = np.load(npz_path)
        consts = read_layer_constants(layer_idx, SAFETENSORS_PATH, manifest_path=None)
        gate_up_scale = RECALIBRATED_GATE_UP_SCALE[layer_idx]

        print(f"\n{'='*90}\nLayer {layer_idx}\n{'='*90}")

        # ---- PRE路徑 ----
        residual_pre = data["residual_pre"]
        real_pre_ln_out = data["pre_ln_out"]
        residual_int16 = np.clip(np.round(residual_pre / S_RES), -32768, 32767)

        gamma_pre_int = np.clip(np.round(consts["gamma_pre_ln"] * (2 ** GAMMA_PRE_FRAC)), -32768, 32767).astype(np.int64)
        inv_scale_val = 1.0 / gate_up_scale
        # 2026-09-20修正：改用per-layer的frac推導（derive_inv_scale_frac_per_layer），
        # 不再是全35層共用一個frac——這正是Layer 0之前underflow的根因，
        # 全域共用frac套用到inv_scale遠小於全域最大值的層會直接捨到0。
        all_layers_inv_scale = np.array([
            1.0 / RECALIBRATED_GATE_UP_SCALE[li] for li in range(35)
        ])
        inv_scale_frac_all, inv_scale_int_bits_all = derive_inv_scale_frac_per_layer(all_layers_inv_scale)
        inv_scale_frac = int(inv_scale_frac_all[layer_idx])
        inv_scale_int_bits = int(inv_scale_int_bits_all[layer_idx])
        inv_scale_int = int(np.clip(np.round(inv_scale_val * (2 ** inv_scale_frac)), -32768, 32767))

        mid_frac = INV_RMS_SCALED_CONTAINER_BITS - (RSQ_T_IWIDTH + inv_scale_int_bits)

        y_with = simulate_pre_stage3(residual_int16, gamma_pre_int, inv_scale_int,
                                       inv_scale_frac, mid_frac, GAMMA_PRE_FRAC, apply_mid_round=True)
        y_without = simulate_pre_stage3(residual_int16, gamma_pre_int, inv_scale_int,
                                          inv_scale_frac, mid_frac, GAMMA_PRE_FRAC, apply_mid_round=False)

        y_with_dq = y_with.astype(np.float64) * gate_up_scale
        y_without_dq = y_without.astype(np.float64) * gate_up_scale

        print("-- Pre-Norm (pre_feedforward_layernorm) --")
        cos_with, _ = evaluate(real_pre_ln_out, y_with_dq, f"有中途捨入(frac={mid_frac}) vs 真實輸出")
        cos_without, _ = evaluate(real_pre_ln_out, y_without_dq, "無中途捨入(理論上限) vs 真實輸出")
        idealized_out = call_pre_feedforward_layernorm(
            residual_int16,
            LayerConstants(layer_idx=layer_idx, head_dim=256, qkv_input_scale=0.0,
                            gate_up_input_scale=gate_up_scale,
                            q_proj_weight_scale=np.zeros(1), k_proj_weight_scale=None, v_proj_weight_scale=None,
                            o_proj_weight_scale=np.zeros(1), down_proj_weight_scale=consts["down_proj_weight_scale"],
                            per_layer_proj_weight_scale=np.zeros(1),
                            o_proj_input_scale=0.0, down_proj_input_scale=consts["down_proj_input_scale"],
                            per_layer_proj_input_scale=0.0,
                            gamma_input_layernorm=np.zeros(HIDDEN_SIZE), gamma_q_norm=np.zeros(1),
                            gamma_k_norm=None, gamma_v_norm=None,
                            gamma_post_attention_layernorm=np.zeros(HIDDEN_SIZE),
                            gamma_pre_feedforward_layernorm=consts["gamma_pre_ln"],
                            gamma_post_feedforward_layernorm=np.zeros(HIDDEN_SIZE),
                            gamma_post_per_layer_input_norm=np.zeros(HIDDEN_SIZE)))
        idealized_dq = idealized_out.astype(np.float64) * gate_up_scale
        evaluate(real_pre_ln_out, idealized_dq, "golden model理想版(float rsqrt) vs 真實輸出")
        delta = cos_with.mean() - cos_without.mean()
        print(f"  中途捨入造成的cos_sim變化（有 - 無）= {delta:+.8f}")

        all_pre_with_cos.append(cos_with)
        all_pre_without_cos.append(cos_without)

        # ---- POST路徑 ----
        down_proj_out = data["down_proj_out"]
        real_post_ln_out = data["post_ln_out"]
        S_d_in = consts["down_proj_input_scale"]
        S_d_w = consts["down_proj_weight_scale"]
        down_mac_recon = np.round(down_proj_out / (S_d_in * S_d_w))
        s_eff = (S_d_in * S_d_w) / S_RES
        x_uniform = requantize_m_k(down_mac_recon, s_eff, out_bits=24)

        gamma_post_int = np.clip(np.round(consts["gamma_post_ln"] * (2 ** GAMMA_POST_FRAC)), -32768, 32767).astype(np.int64)
        s_res_exp = int(round(-np.log2(S_RES)))

        gamma_post_int_bits = int(np.ceil(np.log2(np.abs(consts["gamma_post_ln"]).max()))) + 1
        x_uniform_bits = 24
        gamma_post_total_bits = gamma_post_int_bits + GAMMA_POST_FRAC
        post_inv_rms_scaled_int_bits = RSQ_T_IWIDTH + s_res_exp
        post_available = MULTIPLY_CHAIN_BUDGET_BITS - 1 - x_uniform_bits - gamma_post_total_bits
        post_mid_frac = post_available - post_inv_rms_scaled_int_bits

        y_post_with = simulate_post_stage3(x_uniform, gamma_post_int, s_res_exp,
                                             post_mid_frac, GAMMA_POST_FRAC, apply_mid_round=True)
        y_post_without = simulate_post_stage3(x_uniform, gamma_post_int, s_res_exp,
                                                post_mid_frac, GAMMA_POST_FRAC, apply_mid_round=False)
        y_post_with_dq = y_post_with.astype(np.float64) * S_RES
        y_post_without_dq = y_post_without.astype(np.float64) * S_RES

        print("-- Post-Norm (post_feedforward_layernorm) --")
        cos_post_with, _ = evaluate(real_post_ln_out, y_post_with_dq, f"有中途捨入(frac={post_mid_frac}) vs 真實輸出")
        cos_post_without, _ = evaluate(real_post_ln_out, y_post_without_dq, "無中途捨入(理論上限) vs 真實輸出")
        delta_post = cos_post_with.mean() - cos_post_without.mean()
        print(f"  中途捨入造成的cos_sim變化（有 - 無）= {delta_post:+.8f}")

        all_post_with_cos.append(cos_post_with)
        all_post_without_cos.append(cos_post_without)

    print("\n" + "=" * 90)
    print("總結：全部抽樣層彙整")
    print("=" * 90)
    if all_pre_with_cos:
        w = np.concatenate(all_pre_with_cos)
        wo = np.concatenate(all_pre_without_cos)
        print(f"PRE  有中途捨入: mean={w.mean():.6f}  無中途捨入: mean={wo.mean():.6f}  差距={w.mean()-wo.mean():+.8f}")
    if all_post_with_cos:
        w = np.concatenate(all_post_with_cos)
        wo = np.concatenate(all_post_without_cos)
        print(f"POST 有中途捨入: mean={w.mean():.6f}  無中途捨入: mean={wo.mean():.6f}  差距={w.mean()-wo.mean():+.8f}")


if __name__ == "__main__":
    main()
