#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_real_rmsnorm_activation_batch.py
=====================================================================
把單層單token版本擴大成「多層 x 多token」，目的是避免「剛好這層/
這個token誤差特別小」的偽陽性。

選層邏輯：
    LAYER_LIST 刻意涵蓋三種情況，而不是隨便選幾層：
      - Layer 0    : 附錄C的離群值層（down_proj.output_activation_scale
                     比其他層大50~8000倍），最容易暴露overflow/精度問題
      - Layer 9    : 你們前面幾輪一直拿來當範例的layer，跟文件對照方便
      - Layer 14   : full_attention層的最後一層（head_dim=512邊界，
                     雖然這裡只測MLP前後，但順便保留這層的activation
                     供之後Attention驗證重用）
      - Layer 15   : 你已經跑過、有既有結果可以對照
      - Layer 31   : s_res校準時的全模型max所在層（附錄A: 83.6358）
      - Layer 34   : 最後一層，檢查「靠近LM Head」這端有沒有特殊行為

每層存整個序列（不再只取最後一個token），驗證腳本那邊會對每個token
分別算cos_sim，而不是只看一個點。

效能備註（相對於原始草稿的修正）：
    原始寫法是「對每一層都重跑一次全模型forward」（外層迴圈layer_idx，
    內層跑4個prompt），等於6層*4句=24次完整forward。但hook只是攔截
    forward過程中的中繼輸出，跟「這次forward要不要順便也記錄其他層」
    無關——同一次forward，模型本來就會經過全部35層。所以改成「外層跑
    prompt（4次forward），內層在跑之前就把LAYER_LIST全部6層的hook都
    掛上」，一次forward同時拿到6層的activation，總共只需要4次forward，
    結果（每層.npz的內容）完全相同，只是省掉20次多餘的完整推論。
"""

import os
import sys
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Windows終端機預設cp950(Big5)，之前跑extract/verify腳本時因為中文字
# （尤其誤植的簡體字）觸發過UnicodeEncodeError，這裡統一先修正。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# 2026-09-20：從6層抽樣擴大到全35層。Stage3中途捨入的underflow是
# 「離散、特定層才會踩到」的failure mode（Layer 0是因為gate_up_input_
# scale偏離全域值33倍），不是連續漸變的精度劣化——6層抽樣剛好抓到
# Layer 0，不代表其他29層裡沒有另一個同樣偏離全域值、但沒被抽到的層。
# 全35層都跑一次同一次forward就能同時勾住（下面的hook設計本來就是
# 「每個prompt跑一次forward、同時勾住LAYER_LIST全部層」，加到35層
# 不會增加forward次數，只是多存一點資料）。
LAYER_LIST = list(range(35))
MODEL_ID = "google/gemma-4-E2B-it-qat-mobile-transformers"

# outputs/ 資料夾：跟extract_real_rmsnorm_activation.py（單層版）一致，
# 中間產物統一放這裡，不要散落在qat_verification/底下。
OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# 建議混用幾種不同性質的句子，而不是只用一句——短句/長句/重複token/
# 極端字元，盡量讓residual動態範圍多樣化，比較容易撞到outlier
PROMPTS = [
    "The quick brown fox jumps over the lazy dog.",
    "In 2024, the company reported revenue growth of 15.3% year over year, "
    "driven primarily by strong demand in the cloud computing segment.",
    "A A A A A A A A A A A A A A A A",  # 重複token，測試退化情況
    "請用繁體中文簡短說明什麼是矩陣乘法。",  # 非英文輸入
]


def find_submodule(model, layer_idx, subname):
    target = f"layers.{layer_idx}.{subname}"
    candidates = [(name, mod) for name, mod in model.named_modules()
                  if name.endswith(target) and "language_model" in name]
    if not candidates:
        raise RuntimeError(f"找不到 {target}（language_model底下）")
    return candidates[0][1]


def make_pre_hook(bucket, name):
    def hook(module, args, kwargs):
        x = args[0] if len(args) > 0 else kwargs.get("hidden_states")
        bucket[f"{name}_in"] = x.detach().to(torch.float64).cpu().numpy()
    return hook


def make_post_hook(bucket, name):
    def hook(module, args, kwargs, output):
        bucket[f"{name}_out"] = output.detach().to(torch.float64).cpu().numpy()
    return hook


def main():
    print(f"載入模型 {MODEL_ID} ...（已存在本機快取，不會重新下載）")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    # 注意：不要傳device_map="auto"——這台機器是CPU-only單一裝置，之前
    # 已實測過device_map="auto"搭配這個checkpoint的自訂quant_method
    # ("gemma") 在目前transformers dev版本會踩到device_map解析bug
    # （device_map被錯誤處理成字串"none"，拋ValueError），拿掉這個參數
    # 反而簡單且已驗證可行。
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True, dtype=torch.float32)
    model = model.to(torch.float32)
    model.eval()

    # ---- 每個prompt只跑一次forward，同時勾住全部LAYER_LIST的層 ----
    per_layer_data = {layer_idx: {"residual_pre": [], "pre_ln_out": [],
                                    "down_proj_out": [], "post_ln_out": []}
                       for layer_idx in LAYER_LIST}

    for prompt in PROMPTS:
        captured = {}
        handles = []
        for layer_idx in LAYER_LIST:
            pre_ln = find_submodule(model, layer_idx, "pre_feedforward_layernorm")
            post_ln = find_submodule(model, layer_idx, "post_feedforward_layernorm")
            pre_name = f"L{layer_idx}_pre_ln"
            post_name = f"L{layer_idx}_post_ln"
            handles.append(pre_ln.register_forward_pre_hook(
                make_pre_hook(captured, pre_name), with_kwargs=True))
            handles.append(pre_ln.register_forward_hook(
                make_post_hook(captured, pre_name), with_kwargs=True))
            handles.append(post_ln.register_forward_pre_hook(
                make_pre_hook(captured, post_name), with_kwargs=True))
            handles.append(post_ln.register_forward_hook(
                make_post_hook(captured, post_name), with_kwargs=True))

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            model(**inputs)
        for h in handles:
            h.remove()

        def all_tok(name):
            arr = captured[name]
            return arr.reshape(-1, arr.shape[-1]).astype(np.float64)

        for layer_idx in LAYER_LIST:
            pre_name = f"L{layer_idx}_pre_ln"
            post_name = f"L{layer_idx}_post_ln"
            per_layer_data[layer_idx]["residual_pre"].append(all_tok(f"{pre_name}_in"))
            per_layer_data[layer_idx]["pre_ln_out"].append(all_tok(f"{pre_name}_out"))
            per_layer_data[layer_idx]["down_proj_out"].append(all_tok(f"{post_name}_in"))
            per_layer_data[layer_idx]["post_ln_out"].append(all_tok(f"{post_name}_out"))

        print(f"[prompt] 已跑完一次forward：{prompt[:30]}...")

    # ---- 把每一層、所有prompt、所有token疊在一起存成一個.npz ----
    for layer_idx in LAYER_LIST:
        d = per_layer_data[layer_idx]
        residual_pre = np.concatenate(d["residual_pre"], axis=0)
        pre_ln_out = np.concatenate(d["pre_ln_out"], axis=0)
        down_proj_out = np.concatenate(d["down_proj_out"], axis=0)
        post_ln_out = np.concatenate(d["post_ln_out"], axis=0)

        out_path = os.path.join(OUTPUTS_DIR, f"real_activations_layer{layer_idx}.npz")
        np.savez(out_path,
                  residual_pre=residual_pre, pre_ln_out=pre_ln_out,
                  down_proj_out=down_proj_out, post_ln_out=post_ln_out)

        maxabs = np.abs(residual_pre).max()
        print(f"[layer {layer_idx}] 存檔 {out_path}，共 {residual_pre.shape[0]} 個token，"
              f"residual maxabs={maxabs:.4f}")


if __name__ == "__main__":
    main()
