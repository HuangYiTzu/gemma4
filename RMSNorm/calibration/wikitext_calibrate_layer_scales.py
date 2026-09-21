#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
wikitext_calibrate_layer_scales.py
=====================================================================
用WikiText-2 train split的大樣本（遠大於之前qat_verification/那邊
只用4個prompt/75個token），重新校準：
    1. 全部35層的 gate_up_input_scale（pre_feedforward_layernorm用，
       比照12.3節Layer 14的outlier-channel校準法，這次套用到全部層）
    2. S_RES的溢位風險（post_feedforward_layernorm的真實輸出分布，
       全部35層一起看，用來決定S_RES的F值要不要從8下修）

跟qat_verification/rmsnorm_verify_batch.py的關係：
    這裡不重新實作RMSNorm驗證邏輯，只負責「收集大樣本的real activation
    統計量」，算出來的校準值會被qat_verification/那邊的腳本讀取使用
    （見RECALIBRATED_GATE_UP_SCALE的載入方式）。

記憶體考量（這是跟extract_real_rmsnorm_activation_batch.py最大的
差異）：
    35層 x 數萬token x 1536維 x float64，如果像之前那樣把完整
    activation張量全部存下來會爆記憶體（35*20480*1536*8*2 bytes
    ≈ 17GB，不可行）。這裡改成streaming統計：forward hook只更新
    「目前為止看過的統計量」（per-channel running max、per-token
    max的清單），不保留完整張量，記憶體用量只跟「35層 x 1536
    channel」或「35層 x token數」成正比，跟channel數平方無關，
    可以跑遠大於75個token的樣本。
"""

import json
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
NUM_CHUNKS = 40   # 40*512=20480個token，是先前75個token樣本的約273倍

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(_THIS_DIR, "..", "outputs")


class LayerStats:
    """單一層、單一RMSNorm呼叫點的streaming統計量。不保留完整張量，
    只維護per-channel running max跟per-token max的清單（後者用來算
    percentile，記憶體只跟token數成正比，不含channel維度）。"""

    def __init__(self, hidden_size):
        self.per_channel_max_abs = np.zeros(hidden_size, dtype=np.float64)
        self.per_token_max_abs = []  # 每個token的max(|全部channel|)，list增量append
        self.n_tokens = 0

    def update(self, arr_2d):
        """arr_2d: shape=(n_tok_in_chunk, hidden_size)，float64"""
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
            "p9999": float(np.percentile(arr, 99.99)),
            "mean_of_token_max": float(arr.mean()),
            "argmax_channel": int(np.argmax(self.per_channel_max_abs)),
            "per_channel_max_abs_top5": [
                (int(idx), float(self.per_channel_max_abs[idx]))
                for idx in np.argsort(-self.per_channel_max_abs)[:5]
            ],
        }


def load_wikitext_chunks(tokenizer, seq_len, num_chunks):
    df = pd.read_parquet(WIKITEXT_TRAIN_PARQUET)
    text = "\n\n".join(line for line in df["text"].tolist() if line.strip())
    print(f"[wikitext] train set共{len(df)}行，非空文字總長度{len(text)}字元")

    ids = tokenizer(text, return_tensors="pt")["input_ids"][0]
    print(f"[wikitext] tokenize後共{ids.shape[0]}個token")

    chunks = []
    for i in range(num_chunks):
        start = i * seq_len
        end = start + seq_len
        if end > ids.shape[0]:
            break
        chunks.append(ids[start:end])
    print(f"[wikitext] 取{len(chunks)}個長度{seq_len}的不重疊chunk，"
          f"共{len(chunks) * seq_len}個token")
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

    pre_stats = {layer_idx: LayerStats(HIDDEN_SIZE) for layer_idx in range(NUM_LAYERS)}
    post_stats = {layer_idx: LayerStats(HIDDEN_SIZE) for layer_idx in range(NUM_LAYERS)}

    handles = []
    for layer_idx in range(NUM_LAYERS):
        pre_ln = find_submodule(model, layer_idx, "pre_feedforward_layernorm")
        post_ln = find_submodule(model, layer_idx, "post_feedforward_layernorm")
        handles.append(pre_ln.register_forward_hook(
            make_stat_hook(pre_stats[layer_idx]), with_kwargs=True))
        handles.append(post_ln.register_forward_hook(
            make_stat_hook(post_stats[layer_idx]), with_kwargs=True))
    print(f"已掛上{len(handles)}個streaming統計hook（35層 x pre+post）")

    t0 = time.time()
    for i, ids in enumerate(chunks):
        input_ids = ids.unsqueeze(0)
        with torch.no_grad():
            model(input_ids=input_ids)
        print(f"  chunk {i+1}/{len(chunks)} 完成 (累計耗時{time.time()-t0:.1f}s)")

    for h in handles:
        h.remove()

    # ---- 彙整結果 ----
    pre_summary = {layer_idx: pre_stats[layer_idx].summary() for layer_idx in range(NUM_LAYERS)}
    post_summary = {layer_idx: post_stats[layer_idx].summary() for layer_idx in range(NUM_LAYERS)}

    print("\n" + "=" * 100)
    print("pre_feedforward_layernorm 全35層統計（用來重新校準gate_up_input_scale）")
    print("=" * 100)
    print(f"{'Layer':>6} {'#tok':>6} {'global_max':>12} {'p99.9':>10} {'p99.99':>10} "
          f"{'argmax_ch':>10}")
    for layer_idx in range(NUM_LAYERS):
        s = pre_summary[layer_idx]
        print(f"{layer_idx:>6} {s['n_tokens']:>6} {s['global_max']:>12.4f} "
              f"{s['p999']:>10.4f} {s['p9999']:>10.4f} {s['argmax_channel']:>10}")

    print("\n" + "=" * 100)
    print("post_feedforward_layernorm 全35層統計（用來評估S_RES的F值）")
    print("=" * 100)
    print(f"{'Layer':>6} {'#tok':>6} {'global_max':>12} {'p99.9':>10} {'p99.99':>10} "
          f"{'argmax_ch':>10}")
    global_max_post = 0.0
    for layer_idx in range(NUM_LAYERS):
        s = post_summary[layer_idx]
        print(f"{layer_idx:>6} {s['n_tokens']:>6} {s['global_max']:>12.4f} "
              f"{s['p999']:>10.4f} {s['p9999']:>10.4f} {s['argmax_channel']:>10}")
        global_max_post = max(global_max_post, s["global_max"])

    print(f"\n全35層post_ln的global max（所有layer中的最大值）= {global_max_post:.4f}")
    for F in [8, 7, 6, 5, 4]:
        cap = (2 ** 15 - 1) * (2.0 ** -F)
        print(f"  F={F}: S_RES=2^-{F}, INT16可表示範圍=±{cap:.2f}, "
              f"{'仍會飽和!' if global_max_post > cap else '不會飽和，有餘裕'}")

    # ---- 存檔，供verify/perplexity腳本讀取 ----
    out_path = os.path.join(OUT_DIR, "wikitext_layer_scale_calibration.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "num_chunks": len(chunks), "seq_len": SEQ_LEN,
            "total_tokens": len(chunks) * SEQ_LEN,
            "pre_feedforward_layernorm": pre_summary,
            "post_feedforward_layernorm": post_summary,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n已存檔: {out_path}")


if __name__ == "__main__":
    main()
