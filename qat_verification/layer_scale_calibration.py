#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
layer_scale_calibration.py
=====================================================================
2026-09-19：把「用實測activation重新校準gate_up_input_scale」這個
邏輯（12.3節Layer 14的outlier-channel校準法）套用到全部35層，不再
只修Layer 14一層。

讀calibration/wikitext_calibrate_layer_scales.py產生的
outputs/wikitext_layer_scale_calibration.json（WikiText-2 train
split，40*512=20480個token，比之前75個token的樣本大約273倍），
對每一層用 global_max/127 重新算gate_up_input_scale——跟Layer 14
當初的做法（max/127）一致，只是這次樣本數大很多、套用到全部層。

用法：
    from layer_scale_calibration import get_recalibrated_gate_up_scale
    consts["gate_up_input_scale"] = get_recalibrated_gate_up_scale(layer_idx)
"""

import json
import os

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CALIB_JSON = os.path.join(_THIS_DIR, "..", "outputs", "wikitext_layer_scale_calibration.json")

_cache = None


def _load():
    global _cache
    if _cache is None:
        if not os.path.exists(_CALIB_JSON):
            raise FileNotFoundError(
                f"找不到{_CALIB_JSON}，請先跑"
                f"calibration/wikitext_calibrate_layer_scales.py")
        with open(_CALIB_JSON, encoding="utf-8") as f:
            _cache = json.load(f)
    return _cache


def get_recalibrated_gate_up_scale(layer_idx: int) -> float:
    """回傳用WikiText大樣本重新校準的gate_up_input_scale
    （= global_max / 127，INT8 max-based calibration，跟Layer 14
    當初的校準方法一致）。"""
    data = _load()
    s = data["pre_feedforward_layernorm"][str(layer_idx)]
    return s["global_max"] / 127.0


def get_all_recalibrated_gate_up_scales() -> dict:
    data = _load()
    return {int(k): v["global_max"] / 127.0
            for k, v in data["pre_feedforward_layernorm"].items()}


if __name__ == "__main__":
    scales = get_all_recalibrated_gate_up_scales()
    for layer_idx in sorted(scales):
        print(f"Layer {layer_idx:>2}: {scales[layer_idx]:.6f}")
