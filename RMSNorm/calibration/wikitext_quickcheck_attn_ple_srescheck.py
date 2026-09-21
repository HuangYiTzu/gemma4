#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
wikitext_quickcheck_attn_ple_srescheck.py
=====================================================================
快速檢查：post_attention_layernorm跟post_per_layer_input_norm這兩個
也共用S_RES的RMSNorm呼叫點，實際輸出量級大概多大、S_RES=2^-5夠不夠。

這兩個呼叫點目前完全沒有golden model/常數表實作（見checklist），這裡
只是先抓real activation的量級，快速判斷風險，不是完整驗證，樣本數比
wikitext_calibrate_layer_scales.py小（10個chunk而不是40個），求快。
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

MODEL_ID = "google/gemma-4-E2B-it-qat-mobile-transformers"
WIKITEXT_TRAIN_PARQUET = (
    "C:/vitisHLS2022_workspace/Softmax/wikitext_local/wikitext-2-raw-v1/"
    "train-00000-of-00001.parquet"
)
NUM_LAYERS = 35
HIDDEN_SIZE = 1536
SEQ_LEN = 512
NUM_CHUNKS = 10  # 求快，先看量級，不是正式校準


class LayerStats:
    def __init__(self, hidden_size):
        self.per_channel_max_abs = np.zeros(hidden_size, dtype=np.float64)
        self.per_token_max_abs = []
        self.n_tokens = 0

    def update(self, arr_2d):
        abs_arr = np.abs(arr_2d)
        self.per_channel_max_abs = np.maximum(self.per_channel_max_abs, abs_arr.max(axis=0))
        self.per_token_max_abs.extend(abs_arr.max(axis=1).tolist())
        self.n_tokens += arr_2d.shape[0]

    def summary(self):
        arr = np.array(self.per_token_max_abs)
        return {
            "n_tokens": self.n_tokens,
            "global_max": float(arr.max()),
            "p999": float(np.percentile(arr, 99.9)),
            "argmax_channel": int(np.argmax(self.per_channel_max_abs)),
        }


def load_wikitext_chunks(tokenizer, seq_len, num_chunks):
    df = pd.read_parquet(WIKITEXT_TRAIN_PARQUET)
    text = "\n\n".join(line for line in df["text"].tolist() if line.strip())
    ids = tokenizer(text, return_tensors="pt")["input_ids"][0]
    chunks = []
    for i in range(num_chunks):
        start = i * seq_len
        end = start + seq_len
        if end > ids.shape[0]:
            break
        chunks.append(ids[start:end])
    print(f"[wikitext] 取{len(chunks)}個長度{seq_len}的chunk，共{len(chunks) * seq_len}個token")
    return chunks


def find_submodule(model, layer_idx, subname):
    target = f"layers.{layer_idx}.{subname}"
    candidates = [(name, mod) for name, mod in model.named_modules()
                  if name.endswith(target) and "language_model" in name]
    if not candidates:
        raise RuntimeError(f"找不到 {target}（language_model底下）")
    return candidates[0][1]


def make_stat_hook(stats: LayerStats):
    def hook(module, args, kwargs, output):
        arr = output.detach().to(torch.float64).cpu().numpy()
        flat = arr.reshape(-1, arr.shape[-1])
        stats.update(flat)
    return hook


def main():
    print(f"載入模型 {MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True, dtype=torch.float32)
    model = model.to(torch.float32)
    model.eval()

    chunks = load_wikitext_chunks(tokenizer, SEQ_LEN, NUM_CHUNKS)

    attn_stats = {layer_idx: LayerStats(HIDDEN_SIZE) for layer_idx in range(NUM_LAYERS)}
    ple_stats = {layer_idx: LayerStats(HIDDEN_SIZE) for layer_idx in range(NUM_LAYERS)}

    handles = []
    for layer_idx in range(NUM_LAYERS):
        attn_ln = find_submodule(model, layer_idx, "post_attention_layernorm")
        ple_ln = find_submodule(model, layer_idx, "post_per_layer_input_norm")
        handles.append(attn_ln.register_forward_hook(
            make_stat_hook(attn_stats[layer_idx]), with_kwargs=True))
        handles.append(ple_ln.register_forward_hook(
            make_stat_hook(ple_stats[layer_idx]), with_kwargs=True))
    print(f"已掛上{len(handles)}個統計hook（35層 x post_attention + post_per_layer_input_norm）")

    t0 = time.time()
    for i, ids in enumerate(chunks):
        input_ids = ids.unsqueeze(0)
        with torch.no_grad():
            model(input_ids=input_ids)
        print(f"  chunk {i+1}/{len(chunks)} 完成 (累計耗時{time.time()-t0:.1f}s)")
    for h in handles:
        h.remove()

    cap_f5 = (2 ** 15 - 1) * (2.0 ** -5)
    cap_f8 = (2 ** 15 - 1) * (2.0 ** -8)
    print(f"\nF=5範圍: ±{cap_f5:.2f}   F=8範圍: ±{cap_f8:.2f}\n")

    print("=" * 90)
    print("post_attention_layernorm 全35層量級（S_RES用）")
    print("=" * 90)
    print(f"{'Layer':>6} {'#tok':>6} {'global_max':>12} {'p99.9':>10} {'argmax_ch':>10} {'F=5是否夠':>10}")
    global_max_attn = 0.0
    for layer_idx in range(NUM_LAYERS):
        s = attn_stats[layer_idx].summary()
        ok = "OK" if s["global_max"] <= cap_f5 else "飽和!"
        print(f"{layer_idx:>6} {s['n_tokens']:>6} {s['global_max']:>12.4f} "
              f"{s['p999']:>10.4f} {s['argmax_channel']:>10} {ok:>10}")
        global_max_attn = max(global_max_attn, s["global_max"])

    print("\n" + "=" * 90)
    print("post_per_layer_input_norm 全35層量級（S_RES用）")
    print("=" * 90)
    print(f"{'Layer':>6} {'#tok':>6} {'global_max':>12} {'p99.9':>10} {'argmax_ch':>10} {'F=5是否夠':>10}")
    global_max_ple = 0.0
    for layer_idx in range(NUM_LAYERS):
        s = ple_stats[layer_idx].summary()
        ok = "OK" if s["global_max"] <= cap_f5 else "飽和!"
        print(f"{layer_idx:>6} {s['n_tokens']:>6} {s['global_max']:>12.4f} "
              f"{s['p999']:>10.4f} {s['argmax_channel']:>10} {ok:>10}")
        global_max_ple = max(global_max_ple, s["global_max"])

    print(f"\n總結：")
    print(f"  post_attention_layernorm     全35層global max = {global_max_attn:.4f}")
    print(f"  post_per_layer_input_norm    全35層global max = {global_max_ple:.4f}")
    print(f"  （對照：post_feedforward_layernorm之前測到的global max = 595.7565）")
    overall_max = max(global_max_attn, global_max_ple, 595.7565)
    print(f"  三者合計的overall global max = {overall_max:.4f}")
    for F in [8, 7, 6, 5, 4, 3]:
        cap = (2 ** 15 - 1) * (2.0 ** -F)
        print(f"  F={F}: 範圍=±{cap:.2f}, {'仍會飽和!' if overall_max > cap else '不會飽和'}")


if __name__ == "__main__":
    main()
