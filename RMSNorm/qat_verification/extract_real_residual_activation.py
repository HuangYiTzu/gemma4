#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_real_residual_activation.py
=====================================================================
補齊residual_golden_model.py __main__自己列的checklist第1、2項：

    1. 用hook抓真實QAT模型在Add①(Attention)/Add②(MLP)/Add③(PLE)之後、
       layer_scalar乘完之後，各自的真實浮點值，作為residual golden
       model的ground truth。
    2. 同時抓Attention分支（post_attention_layernorm）跟PLE分支
       （post_per_layer_input_norm）的RMSNorm輸入/輸出——這兩個呼叫點
       之前完全沒有真實checkpoint驗證過（只驗證過pre/post_feedforward_
       layernorm），residual的完整正確性判斷不能只靠MLP這一條分支。

對應HuggingFace官方forward()（modeling_gemma4.py:1399~1445）的資料流，
每個decoder layer要抓的點：

    residual_after_add1  = pre_feedforward_layernorm的輸入
                            （= Attention分支加完後的residual）
    o_proj_out            = post_attention_layernorm的輸入
                            （= Attention的o_proj原始輸出，浮點已反量化）
    attn_delta             = post_attention_layernorm的輸出
    down_proj_out          = post_feedforward_layernorm的輸入（MLP的
                            down_proj輸出，之前extract_real_rmsnorm_
                            activation_batch.py已經抓過，這裡重抓一次
                            確保跟其他欄位對齊在同一次forward）
    mlp_delta               = post_feedforward_layernorm的輸出
    residual_after_add2  = per_layer_input_gate的輸入
                            （= MLP分支加完後的residual，PLE分支的
                            輸入residual）
    per_layer_proj_out    = post_per_layer_input_norm的輸入
                            （= per_layer_projection原始輸出）
    ple_delta               = post_per_layer_input_norm的輸出
    layer_output            = 整個decoder layer的輸出（= Add③後再乘
                            layer_scalar，會變成下一層的residual）

有了這些點，就能重建：
    residual_after_add3（Add③後、乘layer_scalar前）
        = residual_after_add2 + ple_delta（理論上應該約等於
          layer_output / layer_scalar，可以互相驗證sanity check）
"""

import os
import sys
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

MODEL_ID = "google/gemma-4-E2B-it-qat-mobile-transformers"
LAYER_LIST = [0, 9, 14, 15, 31, 34]  # 跟之前MLP分支驗證用的6層一致，方便互相對照

OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

PROMPTS = [
    "The quick brown fox jumps over the lazy dog.",
    "In 2024, the company reported revenue growth of 15.3% year over year, "
    "driven primarily by strong demand in the cloud computing segment.",
    "A A A A A A A A A A A A A A A A",
    "請用繁體中文簡短說明什麼是矩陣乘法。",
]


def find_submodule(model, layer_idx, subname):
    target = f"layers.{layer_idx}.{subname}"
    candidates = [(name, mod) for name, mod in model.named_modules()
                  if name.endswith(target) and "language_model" in name]
    if not candidates:
        raise RuntimeError(f"找不到 {target}（language_model底下）")
    return candidates[0][1]


def find_decoder_layer(model, layer_idx):
    """找decoder layer本身（不是它底下的子模組），用來抓layer_scalar
    乘完之後的最終輸出。"""
    target = f"layers.{layer_idx}"
    candidates = [(name, mod) for name, mod in model.named_modules()
                  if name.endswith(target) and "language_model" in name]
    if not candidates:
        raise RuntimeError(f"找不到decoder layer {target}")
    return candidates[0][1]


def make_pre_hook(bucket, name):
    def hook(module, args, kwargs):
        x = args[0] if len(args) > 0 else kwargs.get("hidden_states")
        bucket[f"{name}_in"] = x.detach().to(torch.float64).cpu().numpy()
    return hook


def make_post_hook(bucket, name):
    def hook(module, args, kwargs, output):
        out = output[0] if isinstance(output, tuple) else output
        bucket[f"{name}_out"] = out.detach().to(torch.float64).cpu().numpy()
    return hook


def main():
    print(f"載入模型 {MODEL_ID} ...（已存在本機快取，不會重新下載）")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True, dtype=torch.float32)
    model = model.to(torch.float32)
    model.eval()

    per_layer_data = {layer_idx: {
        "residual_after_add1": [], "o_proj_out": [], "attn_delta": [],
        "down_proj_out": [], "mlp_delta": [],
        "residual_after_add2": [], "per_layer_proj_out": [], "ple_delta": [],
        "layer_output": [],
    } for layer_idx in LAYER_LIST}

    for prompt in PROMPTS:
        captured = {}
        handles = []
        for layer_idx in LAYER_LIST:
            pre_ln = find_submodule(model, layer_idx, "pre_feedforward_layernorm")
            post_attn_ln = find_submodule(model, layer_idx, "post_attention_layernorm")
            post_ln = find_submodule(model, layer_idx, "post_feedforward_layernorm")
            ple_gate = find_submodule(model, layer_idx, "per_layer_input_gate")
            post_ple_ln = find_submodule(model, layer_idx, "post_per_layer_input_norm")
            decoder_layer = find_decoder_layer(model, layer_idx)

            p = f"L{layer_idx}"
            handles.append(pre_ln.register_forward_pre_hook(
                make_pre_hook(captured, f"{p}_pre_ln"), with_kwargs=True))
            handles.append(post_attn_ln.register_forward_pre_hook(
                make_pre_hook(captured, f"{p}_post_attn_ln"), with_kwargs=True))
            handles.append(post_attn_ln.register_forward_hook(
                make_post_hook(captured, f"{p}_post_attn_ln"), with_kwargs=True))
            handles.append(post_ln.register_forward_pre_hook(
                make_pre_hook(captured, f"{p}_post_ln"), with_kwargs=True))
            handles.append(post_ln.register_forward_hook(
                make_post_hook(captured, f"{p}_post_ln"), with_kwargs=True))
            handles.append(ple_gate.register_forward_pre_hook(
                make_pre_hook(captured, f"{p}_ple_gate"), with_kwargs=True))
            handles.append(post_ple_ln.register_forward_pre_hook(
                make_pre_hook(captured, f"{p}_post_ple_ln"), with_kwargs=True))
            handles.append(post_ple_ln.register_forward_hook(
                make_post_hook(captured, f"{p}_post_ple_ln"), with_kwargs=True))
            handles.append(decoder_layer.register_forward_hook(
                make_post_hook(captured, f"{p}_layer"), with_kwargs=True))

        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            model(**inputs)
        for h in handles:
            h.remove()

        def all_tok(name):
            arr = captured[name]
            return arr.reshape(-1, arr.shape[-1]).astype(np.float64)

        for layer_idx in LAYER_LIST:
            p = f"L{layer_idx}"
            d = per_layer_data[layer_idx]
            d["residual_after_add1"].append(all_tok(f"{p}_pre_ln_in"))
            d["o_proj_out"].append(all_tok(f"{p}_post_attn_ln_in"))
            d["attn_delta"].append(all_tok(f"{p}_post_attn_ln_out"))
            d["down_proj_out"].append(all_tok(f"{p}_post_ln_in"))
            d["mlp_delta"].append(all_tok(f"{p}_post_ln_out"))
            d["residual_after_add2"].append(all_tok(f"{p}_ple_gate_in"))
            d["per_layer_proj_out"].append(all_tok(f"{p}_post_ple_ln_in"))
            d["ple_delta"].append(all_tok(f"{p}_post_ple_ln_out"))
            d["layer_output"].append(all_tok(f"{p}_layer_out"))

        print(f"[prompt] 已跑完一次forward：{prompt[:30]}...")

    for layer_idx in LAYER_LIST:
        d = per_layer_data[layer_idx]
        out_dict = {k: np.concatenate(v, axis=0) for k, v in d.items()}

        # ---- sanity check：residual_after_add2 + ple_delta 應該約等於
        #      layer_output / layer_scalar（驗證我們抓的點彼此一致）----
        recon_add3 = out_dict["residual_after_add2"] + out_dict["ple_delta"]
        # 這裡先不除layer_scalar，留給驗證腳本做（golden model那邊已有
        # LAYER_SCALAR表），這裡只存原始抓到的浮點值。

        out_path = os.path.join(OUTPUTS_DIR, f"real_residual_layer{layer_idx}.npz")
        np.savez(out_path, **out_dict)
        n = out_dict["residual_after_add1"].shape[0]
        print(f"[layer {layer_idx}] 存檔 {out_path}，共 {n} 個token")


if __name__ == "__main__":
    main()
