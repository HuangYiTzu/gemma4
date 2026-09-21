#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
read_attn_ple_constants.py
=====================================================================
比照rmsnorm_verify_pre_post_mlp.py的read_layer_constants()，但這次讀
Attention分支（post_attention_layernorm）跟PLE分支（post_per_layer_
input_norm）需要的常數：gamma、weight_scale（per-channel）、
input_activation_scale（per-tensor純量）。

跟MLP分支的read_layer_constants()保持獨立、不合併，是因為這兩個
函式讀的safetensors key前綴不同（self_attn.o_proj vs mlp.down_proj、
per_layer_projection），合併成一個大函式反而不好維護。
"""

import numpy as np


def read_attn_constants(layer_idx: int, safetensors_path: str) -> dict:
    from safetensors import safe_open
    prefix = f"model.language_model.layers.{layer_idx}"
    with safe_open(safetensors_path, framework="pt") as f:
        import torch
        gamma = f.get_tensor(f"{prefix}.post_attention_layernorm.weight").to(torch.float32).numpy().astype(np.float64)
        o_proj_weight_scale = f.get_tensor(f"{prefix}.self_attn.o_proj.weight_scale").to(torch.float32).numpy().flatten().astype(np.float64)
        o_proj_input_scale = float(f.get_tensor(f"{prefix}.self_attn.o_proj.input_activation_scale").to(torch.float32).numpy())
    return {
        "gamma_post_attention_layernorm": gamma,
        "o_proj_weight_scale": o_proj_weight_scale,
        "o_proj_input_scale": o_proj_input_scale,
    }


def read_ple_constants(layer_idx: int, safetensors_path: str) -> dict:
    from safetensors import safe_open
    prefix = f"model.language_model.layers.{layer_idx}"
    with safe_open(safetensors_path, framework="pt") as f:
        import torch
        gamma = f.get_tensor(f"{prefix}.post_per_layer_input_norm.weight").to(torch.float32).numpy().astype(np.float64)
        weight_scale = f.get_tensor(f"{prefix}.per_layer_projection.weight_scale").to(torch.float32).numpy().flatten().astype(np.float64)
        input_scale = float(f.get_tensor(f"{prefix}.per_layer_projection.input_activation_scale").to(torch.float32).numpy())
    return {
        "gamma_post_per_layer_input_norm": gamma,
        "per_layer_proj_weight_scale": weight_scale,
        "per_layer_proj_input_scale": input_scale,
    }
