#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rmsnorm_verify_pre_post_mlp.py
=====================================================================
針對目前規格已定案的兩個呼叫點：
    pre_feedforward_layernorm   (Pre-Norm)
    post_feedforward_layernorm  (Post-Norm)
做三方比對，比照同學 mlp_golden_layer15.py 的驗證結構：

    (A) 浮點參考（no quantization）  vs  真實QAT模型輸出
        —— 回答「我對RMSNorm算法的理解，是不是Gemma真正在做的計算？」
           這一步如果就對不上，代表問題出在演算法本身（例如ln_gamma
           要不要+1、mean(x^2)算的維度範圍），不在量化。

    (B) 整數golden model（bit-accurate）  vs  浮點參考
        —— 回答「量化本身（INT8/INT16/per-channel requantize/rsqrt
           定點化）引入了多少誤差？」

    (C) 整數golden model（dequant回真實值）  vs  真實QAT模型輸出
        —— (A)+(B)疊加後的end-to-end總誤差，這是最終真正關心的數字。

跟同學MLP golden model的關係：
    這支腳本的整數golden model部分直接import並重用
    rmsnorm_golden_model.py的call_pre_feedforward_layernorm()/
    call_post_feedforward_layernorm()，不重新實作一份RMSNorm數學，
    確保「規格文件」「golden model骨架」「這次的驗證」三份東西用的
    是同一套邏輯，不會各自實作出兜不起來的版本。

前置需求：
    1. 先跑 extract_real_rmsnorm_activation.py，取得：
         real_residual_pre_layer{L}.npy
         real_pre_ln_out_layer{L}.npy
         real_down_proj_out_layer{L}.npy   (跟同學real_mlp_out_layer15.npy同一張量)
         real_post_ln_out_layer{L}.npy
    2. 需要safetensors讀取以下checkpoint內容：
         pre_feedforward_layernorm.weight   (gamma，v1.7已確認不加1)
         post_feedforward_layernorm.weight  (gamma)
         mlp.down_proj.weight_scale         (per-channel，附錄C已驗證)
       以及以下activation scale（來源同附錄C，這裡假設已經整理成
       manifest.json，格式仿同學的manifest；如果貴team是直接從
       safetensors的quantization_config讀，改read_layer_scales()
       裡的邏輯即可，其餘不用動）：
         mlp.gate_proj.input_activation_scale   (= pre_feedforward_layernorm的S_target)
         mlp.down_proj.input_activation_scale   (= S_down_in，down_proj matmul輸入端scale)
"""

import json
import os
import sys
from typing import Optional
import numpy as np

# Windows終端機預設用cp950(Big5)，印中文字（尤其簡體字誤植/罕見字）容易
# UnicodeEncodeError，這裡強制stdout/stderr用utf-8輸出。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------
# 重用rmsnorm_golden_model.py，不重新實作RMSNorm數學
# 若兩份檔案不在同一目錄，改成實際路徑或用sys.path.append()
# ---------------------------------------------------------------------
_GOLDEN_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "golden_model")
sys.path.insert(0, _GOLDEN_MODEL_DIR)
from rmsnorm_golden_model import (
    call_pre_feedforward_layernorm,
    call_post_feedforward_layernorm,
    LayerConstants,
    S_RES,
)

LAYER_IDX = 15
MANIFEST_PATH = "/content/manifest.json"          # 依實際路徑調整
SAFETENSORS_PATH = "C:/Users/eva huang/.cache/huggingface/hub/models--google--gemma-4-E2B-it-qat-mobile-transformers/snapshots/dd693ff40353f057ca5f07e945ad867f4afbf2ec/model.safetensors"   # 依實際路徑調整


# =======================================================================
# 0. 浮點參考：直接照HuggingFace官方Gemma4RMSNorm.forward()實作
#    （v1.7已用官方原始碼確認：normed_output * self.weight.float()，
#     沒有+1；weight初始化為ones(dim)，不是zeros）
# =======================================================================

def rmsnorm_float_reference(x: np.ndarray, gamma: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    全浮點、不做任何量化的RMSNorm參考實作。這是用來檢驗「演算法理解
    對不對」的基準，不是拿來檢驗量化誤差的（那是golden model的工作）。

    對應HuggingFace Gemma4RMSNorm.forward():
        output = x / sqrt(mean(x^2, dim=-1, keepdim=True) + eps)
        output = output * weight   (沒有+1)
    """
    x = x.astype(np.float64)
    mean_sq = np.mean(x ** 2, axis=-1, keepdims=True)
    normed = x / np.sqrt(mean_sq + eps)
    return normed * gamma.astype(np.float64)


# =======================================================================
# 1. 讀checkpoint常數：gamma、weight_scale、activation_scale
# =======================================================================

def read_layer_constants(layer_idx: int, safetensors_path: str,
                          manifest_path: Optional[str] = None) -> dict:
    """
    讀出驗證這兩個呼叫點所需的全部常數。不依賴manifest.json——
    weight_scale/gamma/activation_scale全部優先直接從safetensors讀，
    跟第一輪驗證q/k/v/o_proj per-channel時用的safe_open手法一致。

    讀取順序（每個值都嘗試多個候選key，因為不同checkpoint打包習慣
    可能把activation_scale存成不同的key名稱）：
      1. 優先：safetensors裡直接找同名的activation_scale key
         （例如 mlp.gate_proj.input_activation_scale 這種跟
          weight_scale同一層級存放的tensor，checkpoint若有存這個
          key，直接讀，不用等manifest）
      2. 次要：如果有manifest_path且檔案存在，從那裡讀（給之後同學
         把manifest排好、你想直接切換過去用時用）
      3. 最後fallback：附錄C硬編碼表（目前只有layer 9）

    這樣設計的理由：manifest.json本身也只是「有人先把safetensors/
    quantization_config的activation_scale整理成一份json」，如果
    checkpoint本身就把這些scale存成safetensors的tensor（用跟
    weight_scale一樣的key命名習慣），根本不需要等manifest就能驗證。
    """
    import torch
    from safetensors import safe_open

    prefix = f"model.language_model.layers.{layer_idx}"
    out = {}

    # 用framework="pt"（torch）讀取，而不是"numpy"：checkpoint裡的tensor
    # 是bfloat16，numpy不支援bfloat16 dtype，用safetensors的numpy backend
    # 讀會直接丟TypeError。改用torch讀進來後再.float()轉成fp32/numpy。
    with safe_open(safetensors_path, framework="pt") as f:
        keys = set(f.keys())

        def get(key, required=True):
            if key not in keys:
                if required:
                    raise KeyError(f"safetensors裡找不到key: {key}，"
                                    f"請用f.keys()核對實際命名")
                return None
            return f.get_tensor(key).to(torch.float32).numpy()

        def get_scalar(candidates):
            """依序嘗試多個候選key名稱，回傳第一個找到的純量值。"""
            for key in candidates:
                if key in keys:
                    t = f.get_tensor(key).to(torch.float32).numpy()
                    return float(np.asarray(t).flatten()[0]), key
            return None, None

        out["gamma_pre_ln"] = get(f"{prefix}.pre_feedforward_layernorm.weight").astype(np.float64)
        out["gamma_post_ln"] = get(f"{prefix}.post_feedforward_layernorm.weight").astype(np.float64)
        out["down_proj_weight_scale"] = get(f"{prefix}.mlp.down_proj.weight_scale").flatten().astype(np.float64)

        # ---- activation scale：先試safetensors直接讀 ----
        gate_up_candidates = [
            f"{prefix}.mlp.gate_proj.input_activation_scale",
            f"{prefix}.mlp.up_proj.input_activation_scale",
        ]
        down_in_candidates = [
            f"{prefix}.mlp.down_proj.input_activation_scale",
        ]
        gate_up_val, gate_up_key = get_scalar(gate_up_candidates)
        down_in_val, down_in_key = get_scalar(down_in_candidates)

    if gate_up_val is not None:
        print(f"[read_layer_constants] gate_up_input_scale 直接從safetensors讀到: "
              f"{gate_up_key} = {gate_up_val:.6g}")
        out["gate_up_input_scale"] = gate_up_val
    if down_in_val is not None:
        print(f"[read_layer_constants] down_proj_input_scale 直接從safetensors讀到: "
              f"{down_in_key} = {down_in_val:.6g}")
        out["down_proj_input_scale"] = down_in_val

    missing = [k for k in ("gate_up_input_scale", "down_proj_input_scale") if k not in out]
    if missing:
        print(f"[read_layer_constants] safetensors裡沒找到{missing}對應的"
              f"activation_scale key，改嘗試manifest.json")
        if manifest_path and os.path.exists(manifest_path):
            with open(manifest_path) as fh:
                manifest = json.load(fh)
            layer_cfg = manifest["layers"][str(layer_idx)]
            scales = layer_cfg["activation_scales"]
            out.setdefault("gate_up_input_scale", scales["gate_proj_input"])
            out.setdefault("down_proj_input_scale", scales["down_proj_input"])
        else:
            print(f"[read_layer_constants] 也沒有manifest.json，改用附錄C"
                  f"硬編碼fallback（僅layer 9有現成數值）")
            if layer_idx != 9:
                raise RuntimeError(
                    f"layer {layer_idx}：safetensors裡找不到activation_scale，"
                    f"也沒有manifest.json，fallback只有layer 9的硬編碼值。"
                    f"請先用f.keys()確認safetensors裡activation_scale的實際"
                    f"key命名（可能跟這裡猜測的不同），或補上manifest.json")
            out.setdefault("gate_up_input_scale", 0.028379)
            out.setdefault("down_proj_input_scale", 0.011442)  # 見文件第4節MLP資料鏈圖

    return out


# =======================================================================
# 2. 誤差指標（照抄同學evaluate()，保持兩份腳本輸出格式一致，方便並排看）
# =======================================================================

def evaluate(ref: np.ndarray, test: np.ndarray, label: str = ""):
    dot = np.dot(ref, test)
    n1 = np.linalg.norm(ref)
    n2 = np.linalg.norm(test)
    cos_sim = dot / (n1 * n2 + 1e-12)
    mae = np.mean(np.abs(ref - test))
    rmse = np.sqrt(np.mean((ref - test) ** 2))
    rel_rmse = rmse / (n1 / np.sqrt(len(ref)) + 1e-12)

    title = f"RMSNorm golden-model 對齊結果 {label}".strip()
    print(f"\n========= {title} =========")
    print(f"Cosine Similarity : {cos_sim:.6f}")
    print(f"MAE               : {mae:.6f}")
    print(f"Relative RMSE     : {rel_rmse:.6f}")
    return cos_sim, mae, rel_rmse


# =======================================================================
# 3. main
# =======================================================================

def main():
    hidden_size = 1536

    consts = read_layer_constants(LAYER_IDX, SAFETENSORS_PATH, MANIFEST_PATH)
    print(f"[layer {LAYER_IDX}] gate_up_input_scale={consts['gate_up_input_scale']:.6g}  "
          f"down_proj_input_scale={consts['down_proj_input_scale']:.6g}  "
          f"s_res={S_RES:.6g}")

    # ---- 讀真實活動值（extract_real_rmsnorm_activation.py產生，存在
    #      ../outputs/底下，見資料夾結構README）----
    OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs")

    def load_real(name):
        path = os.path.join(OUTPUTS_DIR, f"real_{name}_layer{LAYER_IDX}.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"找不到{path}，請先跑extract_real_rmsnorm_activation.py"
                f"並確認輸出的.npy檔在outputs/資料夾底下")
        return np.load(path).astype(np.float64)

    real_residual_pre = load_real("residual_pre")
    real_pre_ln_out = load_real("pre_ln_out")
    real_down_proj_out = load_real("down_proj_out")
    real_post_ln_out = load_real("post_ln_out")

    # =====================================================================
    # (A) 浮點參考 vs 真實QAT模型輸出 —— 檢驗演算法理解是否正確
    # =====================================================================
    print("\n" + "=" * 70)
    print("(A) 浮點參考 vs 真實QAT模型輸出 —— 檢驗RMSNorm演算法理解")
    print("=" * 70)

    fp_pre_ln_out = rmsnorm_float_reference(real_residual_pre, consts["gamma_pre_ln"])
    evaluate(real_pre_ln_out, fp_pre_ln_out,
              label="[pre_feedforward_layernorm] 浮點參考 vs 真實輸出")

    fp_post_ln_out = rmsnorm_float_reference(real_down_proj_out, consts["gamma_post_ln"])
    evaluate(real_post_ln_out, fp_post_ln_out,
              label="[post_feedforward_layernorm] 浮點參考 vs 真實輸出")

    print("\n如果這裡cos_sim已經接近1（例如>0.999），代表RMSNorm算法本身"
          "（mean(x^2)的維度、有沒有+1、epsilon位置）理解正確，可以放心往下"
          "看量化誤差。如果這裡就掉到很低，先別看後面的整數版本，問題在這裡。")

    # =====================================================================
    # (B)+(C) 整數golden model：分別看「量化本身的誤差」跟「跟真實模型的
    #          總誤差」。這裡刻意用「真實residual/真實down_proj輸出」去
    #          模擬硬體會收到的整數輸入，而不是用golden model自己合成的
    #          假資料，這樣量出來的誤差才是有意義的數字。
    # =====================================================================
    print("\n" + "=" * 70)
    print("(B)+(C) 整數golden model：量化誤差 + 跟真實模型的總誤差")
    print("=" * 70)

    # ---- (B1) pre_feedforward_layernorm：Pre-Norm，直接量化residual ----
    # 用s_res把真實residual量化成INT16，這一步本身的誤差，就是「residual
    # 進RMSNorm前先被s_res量化」這件事造成的損失，不含後面任何其他步驟。
    residual_int16 = np.clip(np.round(real_residual_pre / S_RES), -32768, 32767)

    lc_stub = LayerConstants(
        layer_idx=LAYER_IDX, head_dim=256,  # head_dim在這兩個呼叫點用不到，隨意填
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

    pre_ln_int_out = call_pre_feedforward_layernorm(residual_int16, lc_stub)
    pre_ln_int_dq = pre_ln_int_out.astype(np.float64) * consts["gate_up_input_scale"]

    evaluate(fp_pre_ln_out, pre_ln_int_dq,
              label="[pre_feedforward_layernorm] 整數golden model(dequant) vs 浮點參考"
                    "  <- 純量化誤差")
    evaluate(real_pre_ln_out, pre_ln_int_dq,
              label="[pre_feedforward_layernorm] 整數golden model(dequant) vs 真實輸出"
                    "  <- end-to-end總誤差")

    # ---- (B2) post_feedforward_layernorm：Post-Norm，需要先模擬出
    #           「down_proj MAC後的原始整數」。真實QAT模型跑的是fake-quant
    #           forward，down_proj真實輸出(real_down_proj_out)本身就已經是
    #           反量化後的物理真實值（等於Step0應該要算出來的x_uniform乘上
    #           S_target之前的樣子），所以這裡反推：假設down_proj的MAC整數
    #           輸出ps[k]，滿足 real_down_proj_out[k] ≈ ps[k] * S_down_in *
    #           S_down[k]，反解ps[k] = round(real_down_proj_out[k] /
    #           (S_down_in * S_down[k]))，再餵進golden model重新走一次
    #           Step0+Stage1~3，驗證Step0本身有沒有引入額外誤差。
    S_down_in = consts["down_proj_input_scale"]
    S_down = consts["down_proj_weight_scale"]
    down_mac_reconstructed = np.round(real_down_proj_out / (S_down_in * S_down))
    down_mac_reconstructed = np.clip(down_mac_reconstructed, -2 ** 31, 2 ** 31 - 1)

    post_ln_int_out = call_post_feedforward_layernorm(
        down_mac_reconstructed, lc_stub, S_down_input=S_down_in)
    post_ln_int_dq = post_ln_int_out.astype(np.float64) * S_RES

    evaluate(fp_post_ln_out, post_ln_int_dq,
              label="[post_feedforward_layernorm] 整數golden model(dequant) vs 浮點參考"
                    "  <- 純量化誤差(含Step0反推的重建誤差)")
    evaluate(real_post_ln_out, post_ln_int_dq,
              label="[post_feedforward_layernorm] 整數golden model(dequant) vs 真實輸出"
                    "  <- end-to-end總誤差")

    print("\n[注意] post_feedforward_layernorm這裡的down_mac_reconstructed是"
          "從真實浮點值『反推』出來的整數，這一步反推本身會引入一點誤差"
          "（因為real_down_proj_out已經是QAT fake-quant後的值，理論上"
          "round-trip應該很小，但如果這裡誤差偏大，要先確認是反推的問題"
          "還是Step0/Stage1~3本身的問題——可以把down_mac_reconstructed直接"
          "乘回S_down_in*S_down，看是否約等於real_down_proj_out來排除。")

    # =====================================================================
    # (D) Step0反推正確性的sanity check（對應(B2)注意事項的具體驗證）
    # =====================================================================
    print("\n" + "=" * 70)
    print("(D) Sanity check：down_mac_reconstructed反推是否合理")
    print("=" * 70)
    roundtrip = down_mac_reconstructed * S_down_in * S_down
    evaluate(real_down_proj_out, roundtrip,
              label="[down_proj] 反推整數*scale還原 vs 真實浮點輸出"
                    "  <- 應該非常接近1，若不是，反推邏輯本身有問題")

    # =====================================================================
    # (E) rsqrt backend比較："float"（快速驗證邏輯）vs "fast_inv_unit"
    #     （bit-accurate，比對HLS用）。
    #     ✅ FASTINV_MAGIC_RSQRT/RECIP已從論文推導出真實值並填入
    #     rmsnorm_golden_model.py（RSQRT常數與公開Quake magic number
    #     逐位元相同，交叉驗證過），不再是佔位0，這裡算的cos_sim是有
    #     意義的數字：反映Newton-Raphson兩次迭代近似rsqrt跟直接用
    #     float算1/sqrt(x)之間的真實硬體近似誤差。
    # =====================================================================
    print("\n" + "=" * 70)
    print("(E) rsqrt backend比較（float vs fast_inv_unit，真實magic constant）")
    print("=" * 70)

    post_ln_int_out_float = call_post_feedforward_layernorm(
        down_mac_reconstructed, lc_stub, S_down_input=S_down_in,
        rsqrt_backend="float")
    post_ln_int_out_fastinv = call_post_feedforward_layernorm(
        down_mac_reconstructed, lc_stub, S_down_input=S_down_in,
        rsqrt_backend="fast_inv_unit")

    print(f"[float]         非零輸出比例: {np.mean(post_ln_int_out_float != 0):.4f}")
    print(f"[fast_inv_unit] 非零輸出比例: {np.mean(post_ln_int_out_fastinv != 0):.4f}")

    evaluate(post_ln_int_out_float.astype(np.float64), post_ln_int_out_fastinv.astype(np.float64),
              label="[post_feedforward_layernorm] float backend vs fast_inv_unit backend"
                    "  <- rsqrt定點近似(2次Newton-Raphson)本身的誤差")

    fastinv_dq = post_ln_int_out_fastinv.astype(np.float64) * S_RES
    evaluate(real_post_ln_out, fastinv_dq,
              label="[post_feedforward_layernorm] fast_inv_unit backend(dequant) vs 真實輸出"
                    "  <- 換成bit-accurate rsqrt後的end-to-end總誤差")

    print("如果上面兩組cos_sim都還是接近1（預期在0.01%量級的差異），代表"
          "fast_inv_unit的magic constant與Newton-Raphson位元寬度設定"
          "（FASTINV_DT_WIDTH/FRAC/RT_FRAC，見kernel.h對應typedef）沒有"
          "明顯問題，可以放心拿fast_inv_unit backend的結果去跟HLS bit-"
          "accurate版本比對。如果cos_sim掉很多、或post_ln_int_out_fastinv"
          "出現大量inf/nan/飽和值，先檢查FASTINV_DT_WIDTH/FRAC是否跟"
          "kernel.h實際typedef一致，而不是先懷疑magic constant本身。")

    print("\n" + "=" * 70)
    print("全部完成。建議依cos_sim由(A)->(D)往回排查：")
    print("  (A)差 -> RMSNorm演算法理解錯（gamma+1、eps、mean維度）")
    print("  (D)差 -> down_mac反推邏輯錯（不是RMSNorm的問題，是這支腳本")
    print("           重建整數輸入的方式有問題）")
    print("  (B)差但(A)(D)都好 -> 量化pipeline本身（Step0/Stage1~3/rsqrt")
    print("           簡化eps）的誤差偏大，需要檢視是否要加寬中繼位元寬度")
    print("=" * 70)


if __name__ == "__main__":
    main()