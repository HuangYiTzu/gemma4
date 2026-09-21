#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rmsnorm_verify_perplexity.py
=====================================================================
cos_sim是component-level指標（只看單一RMSNorm呼叫點的輸出跟真實值
差多少），不代表整個模型的end-to-end品質——component誤差經過後面
20幾層的Attention/MLP，可能被放大、也可能被稀釋掉，光看cos_sim無法
回答「這樣量化下去，模型實際的語言建模能力掉多少」。

這支腳本做的事：
    1. 跑一次baseline：原始float模型在WikiText-2 test set上的perplexity。
    2. 跑一次quantized：用forward hook把全部35層的pre/post_feedforward_
       layernorm替換成golden model的量化模擬結果（INT8/INT16 dequant
       回float），其餘Attention/MLP的其他部分維持原始float（因為目前
       只有這兩個RMSNorm呼叫點的golden model跟常數表做完），在同一批
       WikiText文字上算perplexity。
    3. 比較兩者：perplexity差距、以及最後一層hidden state的MSE/MAE
       （比cos_sim更直接的end-to-end數值誤差指標）。

跟component-level驗證（rmsnorm_verify_batch.py）的關係：
    這裡故意沿用同一套read_layer_constants()跟call_pre/post_
    feedforward_layernorm()，包含Layer 14的RECALIBRATED_GATE_UP_SCALE
    修正——確保這裡量的是「目前golden model+已知修正」實際部署後的
    end-to-end影響，跟component-level驗證是同一套邏輯，不是另外兜一份。

    S_RES溢位（A5，Layer 0/14/34發現的問題）故意不在這裡修正，維持
    現狀的S_RES=2^-8——這樣算出來的perplexity差距，才反映「如果現在
    就照這個s_res部署，對end-to-end品質的真實衝擊有多大」，這個數字
    本身就是A5該不該優先處理的證據之一。
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

MODEL_ID = "google/gemma-4-E2B-it-qat-mobile-transformers"
SAFETENSORS_PATH = (
    "C:/Users/eva huang/.cache/huggingface/hub/"
    "models--google--gemma-4-E2B-it-qat-mobile-transformers/"
    "snapshots/dd693ff40353f057ca5f07e945ad867f4afbf2ec/model.safetensors"
)
WIKITEXT_TEST_PARQUET = (
    "C:/vitisHLS2022_workspace/Softmax/wikitext_local/wikitext-2-raw-v1/"
    "test-00000-of-00001.parquet"
)
NUM_LAYERS = 35
SEQ_LEN = 512          # 每個chunk的token數
NUM_CHUNKS = 12         # 總共取幾個chunk（CPU-only環境，先用適度樣本數
                         # 拿到穩定的perplexity估計，不追求跟論文等級的
                         # 全量測試集）

# 2026-09-19：跟rmsnorm_verify_batch.py同步，改用WikiText大樣本
# （40*512=20480個token）校準出來的全部35層gate_up_input_scale，
# 不再只修Layer 14。
RECALIBRATED_GATE_UP_SCALE = get_all_recalibrated_gate_up_scales()


def load_wikitext_chunks(tokenizer, seq_len, num_chunks):
    """把WikiText-2 test set的所有非空行接起來，切成固定長度的token chunk。"""
    df = pd.read_parquet(WIKITEXT_TEST_PARQUET)
    text = "\n\n".join(line for line in df["text"].tolist() if line.strip())
    print(f"[wikitext] test set共{len(df)}行，非空文字總長度{len(text)}字元")

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


def build_layer_constants():
    """預先讀好全部35層的常數，forward hook裡不要每次都重新讀safetensors。"""
    hidden_size = 1536
    layer_lc = {}
    for layer_idx in range(NUM_LAYERS):
        consts = read_layer_constants(layer_idx, SAFETENSORS_PATH, manifest_path=None)
        if layer_idx in RECALIBRATED_GATE_UP_SCALE:
            consts["gate_up_input_scale"] = RECALIBRATED_GATE_UP_SCALE[layer_idx]

        lc = LayerConstants(
            layer_idx=layer_idx, head_dim=256,
            qkv_input_scale=0.0, gate_up_input_scale=consts["gate_up_input_scale"],
            q_proj_weight_scale=np.zeros(1), k_proj_weight_scale=None, v_proj_weight_scale=None,
            o_proj_weight_scale=np.zeros(1), down_proj_weight_scale=consts["down_proj_weight_scale"],
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
        layer_lc[layer_idx] = {
            "lc": lc,
            "gate_up_input_scale": consts["gate_up_input_scale"],
            "down_proj_input_scale": consts["down_proj_input_scale"],
            "down_proj_weight_scale": consts["down_proj_weight_scale"],
        }
    return layer_lc


def find_submodule(model, layer_idx, subname):
    target = f"layers.{layer_idx}.{subname}"
    candidates = [(name, mod) for name, mod in model.named_modules()
                  if name.endswith(target) and "language_model" in name]
    if not candidates:
        raise RuntimeError(f"找不到 {target}（language_model底下）")
    return candidates[0][1]


def make_pre_ln_replace_hook(layer_idx, layer_lc):
    """替換pre_feedforward_layernorm的輸出：用golden model的INT8量化模擬
    結果（dequant回float）取代原本的float32運算結果。"""
    info = layer_lc[layer_idx]
    lc = info["lc"]
    S_target = info["gate_up_input_scale"]

    def hook(module, args, kwargs, output):
        x = args[0] if len(args) > 0 else kwargs.get("hidden_states")
        residual_np = x.detach().to(torch.float64).cpu().numpy()
        orig_shape = residual_np.shape
        flat = residual_np.reshape(-1, orig_shape[-1])

        residual_int16 = np.clip(np.round(flat / S_RES), -32768, 32767)
        pre_out = call_pre_feedforward_layernorm(residual_int16, lc)
        pre_dq = pre_out.astype(np.float64) * S_target
        pre_dq = pre_dq.reshape(orig_shape)

        return torch.from_numpy(pre_dq).to(dtype=output.dtype, device=output.device)

    return hook


def make_post_ln_replace_hook(layer_idx, layer_lc):
    """替換post_feedforward_layernorm的輸出：用golden model的Step0+INT16
    量化模擬結果（dequant回float）取代原本的float32運算結果。"""
    info = layer_lc[layer_idx]
    lc = info["lc"]
    S_down_in = info["down_proj_input_scale"]
    S_down = info["down_proj_weight_scale"]

    def hook(module, args, kwargs, output):
        x = args[0] if len(args) > 0 else kwargs.get("hidden_states")
        down_proj_out_np = x.detach().to(torch.float64).cpu().numpy()
        orig_shape = down_proj_out_np.shape
        flat = down_proj_out_np.reshape(-1, orig_shape[-1])

        down_mac_recon = np.round(flat / (S_down_in * S_down))
        down_mac_recon = np.clip(down_mac_recon, -2 ** 31, 2 ** 31 - 1)
        post_out = call_post_feedforward_layernorm(down_mac_recon, lc, S_down_input=S_down_in)
        post_dq = post_out.astype(np.float64) * S_RES
        post_dq = post_dq.reshape(orig_shape)

        return torch.from_numpy(post_dq).to(dtype=output.dtype, device=output.device)

    return hook


def compute_perplexity(model, chunks, label):
    """標準causal LM perplexity：對每個chunk算cross-entropy loss，取exp。"""
    total_loss = 0.0
    total_tokens = 0
    t0 = time.time()
    last_hidden_by_chunk = []
    for i, ids in enumerate(chunks):
        input_ids = ids.unsqueeze(0)
        with torch.no_grad():
            out = model(input_ids=input_ids, labels=input_ids, output_hidden_states=True)
        n_tok = input_ids.shape[1] - 1  # labels shift-by-1，少算一個token
        total_loss += out.loss.item() * n_tok
        total_tokens += n_tok
        last_hidden_by_chunk.append(out.hidden_states[-1].detach().to(torch.float64).cpu().numpy())
        print(f"  [{label}] chunk {i+1}/{len(chunks)} loss={out.loss.item():.4f}  "
              f"(累計耗時{time.time()-t0:.1f}s)")

    avg_loss = total_loss / total_tokens
    ppl = float(np.exp(avg_loss))
    return ppl, avg_loss, last_hidden_by_chunk


def main():
    print(f"載入模型 {MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True, dtype=torch.float32)
    model = model.to(torch.float32)
    model.eval()

    chunks = load_wikitext_chunks(tokenizer, SEQ_LEN, NUM_CHUNKS)
    if len(chunks) == 0:
        raise RuntimeError("WikiText token數不足以取出任何chunk，檢查SEQ_LEN/NUM_CHUNKS設定")

    print("\n" + "=" * 70)
    print("(1) Baseline：原始float模型 perplexity")
    print("=" * 70)
    ppl_base, loss_base, hidden_base = compute_perplexity(model, chunks, "baseline")
    print(f"\nBaseline perplexity = {ppl_base:.4f}  (avg loss={loss_base:.4f})")

    print("\n" + "=" * 70)
    print("(2) Quantized：全部35層pre/post_feedforward_layernorm替換成golden model模擬")
    print("=" * 70)
    print("預先讀取全部35層常數...")
    layer_lc = build_layer_constants()

    handles = []
    for layer_idx in range(NUM_LAYERS):
        pre_ln = find_submodule(model, layer_idx, "pre_feedforward_layernorm")
        post_ln = find_submodule(model, layer_idx, "post_feedforward_layernorm")
        handles.append(pre_ln.register_forward_hook(
            make_pre_ln_replace_hook(layer_idx, layer_lc), with_kwargs=True))
        handles.append(post_ln.register_forward_hook(
            make_post_ln_replace_hook(layer_idx, layer_lc), with_kwargs=True))
    print(f"已掛上{len(handles)}個replace hook（35層 x pre+post）")

    ppl_quant, loss_quant, hidden_quant = compute_perplexity(model, chunks, "quantized")
    for h in handles:
        h.remove()

    print(f"\nQuantized perplexity = {ppl_quant:.4f}  (avg loss={loss_quant:.4f})")

    print("\n" + "=" * 70)
    print("(3) 比較結果")
    print("=" * 70)
    delta_ppl = ppl_quant - ppl_base
    delta_pct = delta_ppl / ppl_base * 100
    print(f"Baseline  perplexity : {ppl_base:.4f}")
    print(f"Quantized perplexity : {ppl_quant:.4f}")
    print(f"差距                 : {delta_ppl:+.4f}  ({delta_pct:+.2f}%)")

    all_mse, all_mae, all_cos = [], [], []
    for hb, hq in zip(hidden_base, hidden_quant):
        diff = hb - hq
        all_mse.append(np.mean(diff ** 2))
        all_mae.append(np.mean(np.abs(diff)))
        hb_flat = hb.reshape(-1, hb.shape[-1])
        hq_flat = hq.reshape(-1, hq.shape[-1])
        cos = np.sum(hb_flat * hq_flat, axis=-1) / (
            np.linalg.norm(hb_flat, axis=-1) * np.linalg.norm(hq_flat, axis=-1) + 1e-12)
        all_cos.append(cos.mean())

    print(f"\n最後一層hidden state（output_hidden_states[-1]）逐chunk平均：")
    print(f"  MSE     : {np.mean(all_mse):.6f}")
    print(f"  MAE     : {np.mean(all_mae):.6f}")
    print(f"  cos_sim : {np.mean(all_cos):.6f}")

    print("\n" + "=" * 70)
    print("解讀：")
    print("  - 如果perplexity差距很小（例如<5%）、最後一層cos_sim接近1，")
    print("    代表component-level的cos_sim掉到0.9x等級，經過剩下20幾層")
    print("    Attention/MLP後大部分被稀釋掉，對整體語言建模能力影響有限。")
    print("  - 如果差距明顯，代表component誤差有被放大，即使component")
    print("    cos_sim看起來還行，也不能掉以輕心，需要更保守的位元寬度。")
    print("  - 這裡故意保留S_RES=2^-8現狀（A5已知風險，Layer 0/14/34會飽和），")
    print("    這個perplexity差距本身就是A5該不該優先處理的量化證據。")
    print("=" * 70)


if __name__ == "__main__":
    main()
