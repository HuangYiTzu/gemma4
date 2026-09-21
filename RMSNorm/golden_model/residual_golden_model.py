"""
Residual Golden Model — 參數化骨架
=====================================================================
適用模型：Gemma-4-E2B-it-qat-mobile-transformers
對應：Gemma4TextDecoderLayer.forward()（modeling_gemma4.py:1388~1445）
跟RMSNorm golden model（rmsnorm_golden_model.py）的關係：本檔案不重新
實作RMSNorm數學，residual加法的兩個運算元都是「已經算好的RMSNorm輸出」
（呼叫`rmsnorm_golden_model.py`的call_post_attention_layernorm()/
call_post_feedforward_layernorm()/call_post_per_layer_input_norm()
拿到），這裡只負責「怎麼把這些delta正確地加回residual」這件事本身。

============================================================================
背景：這裡在解決什麼問題
============================================================================
2026-09-19從HuggingFace官方原始碼（modeling_gemma4.py，見下方引用）確認
了完整的residual資料流，發現兩件事之前golden model/文件都沒處理過：

    1. residual累加是**序列式**的，不是三個分支各自獨立加回同一個
       原始residual：MLP分支讀到的residual是Attention分支加完之後的
       值，不是最初進入這一層的原始residual；PLE分支又是MLP分支加完
       之後的值。這在rmsnorm_golden_model.py的__main__示範程式碼裡
       曾經寫錯（三個分支都用同一個`residual`變數），那裡只是示範
       每個call_xxx()能不能跑，不代表正確的資料流順序，這裡才是
       正確版本。

    2. 每個decoder layer的**最後**，三個分支全部加完之後，會乘一個
       `layer_scalar`純量（原始碼`hidden_states *= self.layer_scalar`）。
       這個值在checkpoint裡是per-layer的真實訓練值，不是恆等於1的
       佔位buffer——實測全35層範圍0.027~0.89，全部小於1。

官方原始碼（modeling_gemma4.py:1399~1445，節錄，已標註對應本檔案的
call_xxx()）：

    residual = hidden_states
    hidden_states = post_attention_layernorm(self_attn(input_layernorm(hidden_states)))
    hidden_states = residual + hidden_states          # <- residual_add_attn()

    residual = hidden_states
    hidden_states = post_feedforward_layernorm(mlp(pre_feedforward_layernorm(hidden_states)))
    hidden_states = residual + hidden_states          # <- residual_add_mlp()

    if hidden_size_per_layer_input:
        residual = hidden_states
        hidden_states = post_per_layer_input_norm(...)
        hidden_states = residual + hidden_states      # <- residual_add_ple()

    hidden_states *= self.layer_scalar                # <- apply_layer_scalar()
    return hidden_states  # 這個值會變成下一層的residual

============================================================================
關鍵開放問題：layer_scalar該在哪個精度/哪個時機套用（見checklist B0-2）
============================================================================
官方float原始碼是「三個分支的delta先用完整float精度加總，最後才乘一次
layer_scalar」。这在定點硬體上有兩種可能實作方式，數學上（無限精度下）
等價，但定點量化下的誤差特性不同：

    方案A（本檔案目前採用，忠於官方forward順序）：
        residual buffer中途需要能裝下「三個分支加總、但還沒乘
        layer_scalar」的未縮放大值。以Layer 14為例，post_feedforward_
        layernorm的delta量級可到595.76，residual_pre量級約65，
        兩者相加後、乘layer_scalar(0.0493)之前，中繼值可達~660。
        這正是rmsnorm_golden_model.py的checklist A5選用S_RES=2^-5
        （±1024）的依據——如果residual buffer全程都要用S_RES表示，
        必須扛得住這種未縮放的中繼值。

    方案B（硬體優化，本檔案未實作，留待team決定）：
        利用「純量乘法對加法可分配」（layer_scalar*(a+b+c) =
        layer_scalar*a + layer_scalar*b + layer_scalar*c）這個數學
        性質，把layer_scalar拆開套用到每一個分支的delta上（在delta
        要加進residual之前，先乘上layer_scalar），這樣residual buffer
        全程維持「已縮放」的小值（不會超過~65~140量級，見WikiText
        校準的residual_pre實測範圍），S_RES可能F=7甚至F=8就夠，不需要
        犧牲8倍精細度。
        ⚠️ 但這只在「無限精度／浮點」下跟方案A完全等價。定點量化下，
        方案A是「大值先加總、最後才量化一次」；方案B是「三個小值分別
        量化、各自舍入，再加總」——三次獨立舍入 vs 一次舍入，累積的
        rounding誤差路徑不同，不能假設兩者的量化誤差一樣小，需要各自
        用golden model實測驗證（比照rmsnorm_golden_model.py驗證Step0/
        WB折疊時的做法）才能下結論。
        如果要走方案B，這裡的residual_add_xxx()系列函式需要改成接收
        「尚未乘layer_scalar的delta_int」跟「layer_scalar」兩個參數，
        內部先做`delta_scaled = delta_int * layer_scalar`（一次額外的
        per-tensor純量乘法，跟rmsnorm的INV_GATE_UP_SCALE同一類操作，
        硬體成本低），再加進residual，而不是在三個分支都加完之後才乘。

本檔案暫時只實作方案A，架構上把「residual累加」跟「layer_scalar套用」
拆成兩個獨立函式（residual_add_xxx() vs apply_layer_scalar()），方便
之後如果決定要走方案B，只需要改resiudal_add_xxx()內部的實作，呼叫端
（跑完整一層的組裝邏輯）不用大改。
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

import sys
import os

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rmsnorm_golden_model import S_RES


# =====================================================================
# 基礎定點工具
# =====================================================================

def residual_add(residual_int: np.ndarray, delta_int: np.ndarray,
                  out_bits: int = 16) -> np.ndarray:
    """
    residual累加的核心運算：純整數加法 + 飽和clip。

    重要前提：residual_int跟delta_int必須已經是**同一個scale**（都是
    @ S_RES的INT16整數）——這是文件0-3決策C「全程維持同一目標scale，
    不需重複轉換」的直接體現。因為兩者scale相同，這裡不需要像
    rmsnorm_golden_model.py的Step0那樣做per-channel requantize，
    單純整數加法即可，這是residual加法跟RMSNorm輸入端requantize
    最大的不同（RMSNorm輸入來自matmul、per-channel weight_scale
    不同；residual加法的兩個運算元都已經是RMSNorm輸出，scale一致）。

    參數：
        residual_int : 目前的residual狀態，INT16 @ S_RES
        delta_int    : 某個分支的RMSNorm輸出（already @ S_RES），
                       shape需跟residual_int一致
        out_bits     : 輸出位寬，預設16

    回傳：
        加總後的residual，int64儲存，已clip在out_bits範圍內。

    備註（硬體對應）：
        文件6節提到「Stage3輸出不做飽和，飽和留給residual add之後」，
        這裡的clip就是文件講的「留給residual add之後」的那個飽和點——
        這是整個pipeline裡唯一「故意允許/需要處理飽和」的地方，跟
        rmsnorm_core()裡的clip（golden model debug用、之後應拿掉）
        性質不同。
    """
    qmax = 2 ** (out_bits - 1) - 1
    qmin = -2 ** (out_bits - 1)
    y = residual_int.astype(np.int64) + delta_int.astype(np.int64)
    return np.clip(y, qmin, qmax).astype(np.int64)


def apply_layer_scalar(residual_int: np.ndarray, layer_scalar: float,
                        out_bits: int = 16) -> np.ndarray:
    """
    每個decoder layer最後的`hidden_states *= self.layer_scalar`。

    方案A（本檔案採用）：三個分支residual_add()做完之後，對最終結果
    呼叫這個函式一次。輸入輸出都是@S_RES的整數，用浮點乘法模擬（golden
    model階段先求數學正確，bit-accurate版本再套上真正的定點乘法/位移）。

    參數：
        residual_int : 三個分支residual_add()做完後的整數，INT16 @ S_RES
        layer_scalar : 該層的layer_scalar值（來自checkpoint，非2的冪次，
                       每層不同，範圍實測0.027~0.89，見LAYER_SCALAR表）
        out_bits     : 輸出位寬，預設16

    回傳：
        乘上layer_scalar後的residual，int64儲存，已clip在out_bits範圍內，
        這個值會變成下一層的residual輸入。
    """
    qmax = 2 ** (out_bits - 1) - 1
    qmin = -2 ** (out_bits - 1)
    y = np.round(residual_int.astype(np.float64) * layer_scalar)
    return np.clip(y, qmin, qmax).astype(np.int64)


def scale_and_round(x_int: np.ndarray, layer_scalar: float,
                     out_bits: int = 16) -> np.ndarray:
    """單一項目乘layer_scalar並獨立捨入一次，clip到out_bits範圍。
    方案B的基礎積木：每一項（包含residual_in本身）各自呼叫一次。"""
    qmax = 2 ** (out_bits - 1) - 1
    qmin = -2 ** (out_bits - 1)
    y = np.round(x_int.astype(np.float64) * layer_scalar)
    return np.clip(y, qmin, qmax).astype(np.int64)


def run_decoder_layer_residual_planB(residual_in_int: np.ndarray,
                                       deltas: "DecoderLayerDeltas",
                                       layer_idx: int,
                                       out_bits: int = 16) -> np.ndarray:
    """
    方案B（未採用，僅供之後比較實驗用）：把layer_scalar拆開套用到
    **每一項**（包含residual_in本身，不是只有delta！），在加總之前
    先各自獨立捨入一次，再相加。

    ⚠️ 重要更正（2026-09-19，寫第一版時的疏漏）：
    數學展開 layer_scalar*(residual_in + attn + mlp + ple) =
        layer_scalar*residual_in + layer_scalar*attn +
        layer_scalar*mlp + layer_scalar*ple
    這四項**全部**都要乘layer_scalar才跟方案A（先加總、最後乘一次）
    數學上等價，不是只乘三個分支的delta——residual_in是進入這一層
    之前、上一層已經算完的值，這一層的layer_scalar一樣會作用在它
    身上（官方forward()的`residual = hidden_states`只是變數重新
    賦值，不是重新定義新的無關變數，最後`hidden_states *= layer_
    scalar`是對整個累加鏈——包含最原始的residual_in——一次乘完）。
    第一版的residual_add_with_layer_scalar_folded()只對delta做
    scale、residual_in維持不變，數學上不等價於方案A，已刪除，改用
    這個正確版本。

    跟方案A的差異點（這是這次要驗證的重點）：
        方案A：4項先用整數加總（無精度損失，整數加法exact），
               最後才乘layer_scalar、捨入**一次**。
        方案B：4項先各自乘layer_scalar、捨入**四次**，再加總。
        無限精度下兩者結果相同，定點量化下「捨入四次」通常會累積
        比「捨入一次」更多的rounding誤差（每次捨入引入最多0.5 LSB
        的誤差，四次獨立捨入的誤差在統計上會疊加，除非誤差剛好正負
        抵消），這是這次驗證要實測確認、不是靠猜的地方。
    """
    r = scale_and_round(residual_in_int, LAYER_SCALAR[layer_idx], out_bits=out_bits)
    a = scale_and_round(deltas.attn_delta, LAYER_SCALAR[layer_idx], out_bits=out_bits)
    m = scale_and_round(deltas.mlp_delta, LAYER_SCALAR[layer_idx], out_bits=out_bits)
    y = r.astype(np.int64) + a.astype(np.int64) + m.astype(np.int64)
    if deltas.ple_delta is not None:
        p = scale_and_round(deltas.ple_delta, LAYER_SCALAR[layer_idx], out_bits=out_bits)
        y = y + p.astype(np.int64)
    qmax = 2 ** (out_bits - 1) - 1
    qmin = -2 ** (out_bits - 1)
    return np.clip(y, qmin, qmax).astype(np.int64)


# =====================================================================
# 全35層layer_scalar常數（2026-09-19從checkpoint safetensors實測）
# =====================================================================
# model.language_model.layers.{i}.layer_scalar，shape=(1,)，真實訓練值，
# 不是恆等於1的佔位buffer。之後若要從safetensors直接讀取，用
# safe_open(SAFETENSORS_PATH, framework="pt")取
# f"model.language_model.layers.{layer_idx}.layer_scalar"即可，
# 這裡先硬編碼方便golden model骨架單獨跑測試。
LAYER_SCALAR = {
    0: 0.027222, 1: 0.179688, 2: 0.761719, 3: 0.291016, 4: 0.527344,
    5: 0.605469, 6: 0.515625, 7: 0.628906, 8: 0.453125, 9: 0.460938,
    10: 0.443359, 11: 0.343750, 12: 0.314453, 13: 0.089355, 14: 0.049316,
    15: 0.308594, 16: 0.574219, 17: 0.609375, 18: 0.582031, 19: 0.500000,
    20: 0.546875, 21: 0.656250, 22: 0.632812, 23: 0.445312, 24: 0.462891,
    25: 0.765625, 26: 0.832031, 27: 0.835938, 28: 0.839844, 29: 0.835938,
    30: 0.890625, 31: 0.875000, 32: 0.894531, 33: 0.671875, 34: 0.261719,
}


# =====================================================================
# 整層residual組裝：把三個分支串起來，順序跟HF官方forward()一致
# =====================================================================

@dataclass
class DecoderLayerDeltas:
    """一次decoder layer forward需要的三個分支delta，每個都已經是
    rmsnorm_golden_model.py對應call_xxx()算出來的INT16 @ S_RES輸出。
    PLE分支不是每個模型都有（取決於hidden_size_per_layer_input是否
    設定），attn_delta/mlp_delta則一定存在。"""
    attn_delta: np.ndarray   # call_post_attention_layernorm()的輸出
    mlp_delta: np.ndarray    # call_post_feedforward_layernorm()的輸出
    ple_delta: Optional[np.ndarray] = None  # call_post_per_layer_input_norm()的輸出


def run_decoder_layer_residual(residual_in_int: np.ndarray,
                                 deltas: DecoderLayerDeltas,
                                 layer_idx: int,
                                 out_bits: int = 16) -> np.ndarray:
    """
    完整跑一次decoder layer的residual組裝，順序忠於HF官方forward()
    （modeling_gemma4.py:1399~1445）：
        residual = residual_in
        residual = residual_add(residual, attn_delta)      # Add①
        residual = residual_add(residual, mlp_delta)        # Add②
        if ple_delta is not None:
            residual = residual_add(residual, ple_delta)    # Add③
        residual = apply_layer_scalar(residual, layer_scalar[layer_idx])
        return residual   # 這個值會變成下一層的residual_in

    這是方案A（見檔案開頭說明）：三個分支的delta都是已經量化好的
    INT16整數，依序加總，最後才乘layer_scalar，跟官方float forward
    的運算順序一一對應，方便驗證階段跟真實QAT模型的residual逐步
    比對（而不是只比對最終輸出，中間任何一步錯了都能定位到）。

    參數：
        residual_in_int : 進入這一層之前的residual，INT16 @ S_RES
        deltas          : DecoderLayerDeltas，三個分支的delta
        layer_idx       : 用來查LAYER_SCALAR表
        out_bits        : 位寬，預設16

    回傳：
        這一層結束後的residual，INT16 @ S_RES，會變成下一層的輸入。
    """
    residual = residual_add(residual_in_int, deltas.attn_delta, out_bits=out_bits)
    residual = residual_add(residual, deltas.mlp_delta, out_bits=out_bits)
    if deltas.ple_delta is not None:
        residual = residual_add(residual, deltas.ple_delta, out_bits=out_bits)

    layer_scalar = LAYER_SCALAR[layer_idx]
    residual = apply_layer_scalar(residual, layer_scalar, out_bits=out_bits)
    return residual


# =====================================================================
# 骨架自我測試（假資料，驗證shape/流程，不是正確性驗證）
# =====================================================================

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    hidden_size = 1536

    residual_in = rng.integers(-2 ** 10, 2 ** 10, size=(hidden_size,))
    deltas = DecoderLayerDeltas(
        attn_delta=rng.integers(-2 ** 12, 2 ** 12, size=(hidden_size,)),
        mlp_delta=rng.integers(-2 ** 12, 2 ** 12, size=(hidden_size,)),
        ple_delta=rng.integers(-2 ** 10, 2 ** 10, size=(hidden_size,)),
    )

    for layer_idx in [0, 9, 14, 34]:
        out = run_decoder_layer_residual(residual_in, deltas, layer_idx)
        print(f"Layer {layer_idx:>2} (layer_scalar={LAYER_SCALAR[layer_idx]:.4f}): "
              f"輸出residual maxabs={np.abs(out).max()}, "
              f"對照輸入residual maxabs={np.abs(residual_in).max()}")

    print("\n[checklist] 這個骨架還缺的東西：")
    print("  1. 用extract_real_rmsnorm_activation_batch.py風格的hook，抓真實")
    print("     QAT模型的residual逐步演進（Add①/②/③後、layer_scalar乘完後")
    print("     各自的真實浮點值），跟這裡的golden model輸出做cos_sim比對")
    print("     （比照rmsnorm_verify_pre_post_mlp.py的驗證結構）。")
    print("  2. Attention分支（call_post_attention_layernorm）跟PLE分支")
    print("     （call_post_per_layer_input_norm）目前都還沒有真實checkpoint")
    print("     驗證過，只有pre/post_feedforward_layernorm做過（見")
    print("     rmsnorm_const_table_redesign.md）。")
    print("  3. 方案A vs 方案B（layer_scalar該在哪個時機套用）的cos_sim")
    print("     對照實驗還沒做，見run_decoder_layer_residual_planB()。")
    print("  4. layer_scalar目前硬編碼在LAYER_SCALAR，之後應該跟")
    print("     read_layer_constants()一樣改成直接從safetensors讀。")
