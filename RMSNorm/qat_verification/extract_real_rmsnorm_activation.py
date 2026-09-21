#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_real_rmsnorm_activation.py
=====================================================================
比照同學 extract_real_activation.py 的做法，但這次 hook 的對象是
pre_feedforward_layernorm 跟 post_feedforward_layernorm 這兩個
RMSNorm 模組本身（而不是 mlp 整個 block），用來驗證 MLP 前後這兩次
RMSNorm 呼叫的 golden model 是否正確。

抓的東西：
  1) real_residual_pre_layer{L}.npy
       進入pre_feedforward_layernorm之前的residual（真實浮點值）。
       對應文件「Pre-Norm」場合的輸入。

  2) real_pre_ln_out_layer{L}.npy
       pre_feedforward_layernorm的真實輸出（也就是真正餵進
       gate_proj/up_proj的浮點hidden state）。
       這是要拿來跟我們自己的golden model輸出比對的ground truth。

  3) real_down_proj_out_layer{L}.npy
       down_proj的真實輸出（浮點，還沒加residual、還沒過
       post_feedforward_layernorm）。
       跟同學MLP golden model抓的real_mlp_out_layer15.npy是同一個
       張量，如果同學已經抓過，可以直接重複使用那份.npy，不需要
       重新跑一次forward。

  4) real_post_ln_out_layer{L}.npy
       post_feedforward_layernorm的真實輸出（RMSNorm算完、還沒
       加residual的那個delta）。
       這是要拿來跟我們自己的golden model輸出比對的ground truth。

  5) real_residual_after_add_layer{L}.npy
       post_feedforward_layernorm輸出加回residual之後的結果，
       也就是這個MLP分支真正結束時、要交給下一個模組的residual。
       用來做「residual add」這最後一步的sanity check（雖然加法
       本身沒有量化誤差可言，但可以確認我們對「加在哪個節點」的
       理解跟真實模型一致）。

用法（比照同學的方式，在Colab上跑）：
    !pip install -U transformers accelerate
    python extract_real_rmsnorm_activation.py
"""
import os
os.environ["HF_HOME"] = "C:/Users/eva huang/.cache/huggingface"
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


LAYER_IDX = 15
MODEL_ID = "google/gemma-4-E2B-it-qat-mobile-transformers"
PROMPT = "The quick brown fox jumps over the lazy dog."  # 換成你想測的真實句子

captured = {}


def make_pre_hook(name):
    """抓 module.forward 的輸入（呼叫前的浮點值）。"""
    def hook(module, args, kwargs):
        x = args[0] if len(args) > 0 else kwargs.get("hidden_states")
        captured[f"{name}_in"] = x.detach().to(torch.float64).cpu().numpy()
    return hook


def make_post_hook(name):
    """抓 module.forward 的輸出（呼叫後的浮點值）。"""
    def hook(module, args, kwargs, output):
        captured[f"{name}_out"] = output.detach().to(torch.float64).cpu().numpy()
    return hook


def find_submodule(model, layer_idx, subname):
    """
    比照同學find_mlp_module()的邏輯：這個模型是多模態的，
    vision_tower/audio_tower底下也可能有同名模組，必須指定
    "language_model"字串排除其他modality的同名層。
    """
    target = f"layers.{layer_idx}.{subname}"
    candidates = [(name, mod) for name, mod in model.named_modules()
                  if name.endswith(target) and "language_model" in name]
    if not candidates:
        print(f"[find_submodule] 找不到 {target}（在language_model底下），"
              f"以下是目前找到的所有候選（可能命名方式不同，自己核對）：")
        for name, _ in model.named_modules():
            if subname in name or f".{layer_idx}." in name:
                print("   ", name)
        raise RuntimeError(f"adjust subname for '{subname}' and retry")
    if len(candidates) > 1:
        print(f"[warning] 找到 {len(candidates)} 個符合的模組，取第一個: {candidates[0][0]}")
    else:
        print(f"[find_submodule] 鎖定模組: {candidates[0][0]}")
    return candidates[0][1]


def main():
    print(f"載入模型 {MODEL_ID} ...（第一次跑會下載，稍等）")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True, dtype=torch.float32)
    model = model.to(torch.float32)
    model.eval()

    pre_ln = find_submodule(model, LAYER_IDX, "pre_feedforward_layernorm")
    post_ln = find_submodule(model, LAYER_IDX, "post_feedforward_layernorm")

    handles = []
    handles.append(pre_ln.register_forward_pre_hook(make_pre_hook("pre_ln"), with_kwargs=True))
    handles.append(pre_ln.register_forward_hook(make_post_hook("pre_ln"), with_kwargs=True))
    handles.append(post_ln.register_forward_pre_hook(make_pre_hook("post_ln"), with_kwargs=True))
    handles.append(post_ln.register_forward_hook(make_post_hook("post_ln"), with_kwargs=True))

    inputs = tokenizer(PROMPT, return_tensors="pt").to(model.device)
    with torch.no_grad():
        model(**inputs)
    for h in handles:
        h.remove()

    # ---- 取序列最後一個token，跟同學MLP golden model的取法一致 ----
    def last_tok(name):
        arr = captured[name]
        return arr.reshape(-1, arr.shape[-1])[-1].astype(np.float64)

    residual_pre = last_tok("pre_ln_in")           # 進pre_feedforward_layernorm前的residual
    pre_ln_out = last_tok("pre_ln_out")             # pre_feedforward_layernorm真實輸出
    down_proj_out = last_tok("post_ln_in")          # post_feedforward_layernorm的輸入
                                                     # = down_proj真實輸出
                                                     # （跟同學real_mlp_out_layer15.npy是同一張量）
    post_ln_out = last_tok("post_ln_out")           # post_feedforward_layernorm真實輸出（delta）
    residual_after_add = residual_pre + post_ln_out  # 手動算一次加回residual後的結果，
                                                      # 作為之後sanity check用

    np.save(f"real_residual_pre_layer{LAYER_IDX}.npy", residual_pre)
    np.save(f"real_pre_ln_out_layer{LAYER_IDX}.npy", pre_ln_out)
    np.save(f"real_down_proj_out_layer{LAYER_IDX}.npy", down_proj_out)
    np.save(f"real_post_ln_out_layer{LAYER_IDX}.npy", post_ln_out)
    np.save(f"real_residual_after_add_layer{LAYER_IDX}.npy", residual_after_add)

    print(f"\n[main] 已存檔（layer {LAYER_IDX}）：")
    print(f"  real_residual_pre_layer{LAYER_IDX}.npy       shape={residual_pre.shape}")
    print(f"  real_pre_ln_out_layer{LAYER_IDX}.npy          shape={pre_ln_out.shape}")
    print(f"  real_down_proj_out_layer{LAYER_IDX}.npy       shape={down_proj_out.shape}")
    print(f"  real_post_ln_out_layer{LAYER_IDX}.npy         shape={post_ln_out.shape}")
    print(f"  real_residual_after_add_layer{LAYER_IDX}.npy  shape={residual_after_add.shape}")

    print(f"\n[sanity] residual_pre 動態範圍: "
          f"min={residual_pre.min():.4f} max={residual_pre.max():.4f} "
          f"maxabs={np.abs(residual_pre).max():.4f}")
    print(f"[sanity] 這個maxabs可以拿去跟附錄A第{LAYER_IDX}層的校準值互相對照，"
          f"確認同一個prompt/token下數值量級是否一致（附錄A是100筆平均後的統計，"
          f"單一token的值不會完全一樣，但量級應該接近）。")


if __name__ == "__main__":
    main()
