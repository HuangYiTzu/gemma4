"""
RMSNorm Golden Model — 參數化骨架
=====================================================================
適用模型：Gemma-4-E2B-it-qat-mobile-transformers
對應文件：RMSNorm 硬體介面規格文件 v2.0

設計目標
---------------------------------------------------------------------
把 RMSNorm 的「核心數學」與「呼叫情境（Pre-Norm/Post-Norm）」拆成兩層：

    rmsnorm_core  : 只認「已經對齊到單一 scale 的整數輸入」，
                    不管這個 scale 是從哪裡來的。這是唯一真正做
                    Stage1(平方均值) + Stage2(rsqrt) + Stage3(乘γ、
                    對齊輸出scale) 數學運算的地方。

    rmsnorm_call  : 外層 wrapper。依 `is_post_norm` 決定要不要先做
                    Step0 per-channel requantize，再把結果丟給
                    rmsnorm_core。

    call_xxx()    : 每個呼叫點（input_layernorm, q_norm, ...）各自
                    一個薄 wrapper，只負責「把規格書寫的參數填進
                    rmsnorm_call」，不含任何額外邏輯。

這樣分層的好處：
  1. 之後 Stage2 要換成 fast_inv_unit 的 bit-accurate C model，
     只需替換 rsqrt_fixed() 這一個函式，其餘全部不用動。
  2. q/k/v_norm 原本規格書誤判成「可以跳過反量化」，實際上因為
     q/k/v_proj 的 weight_scale 全部是 per-channel（已於本對話中
     用 safetensors 實測驗證），所以正確分類是 Post-Norm。這個
     分類差異在本架構下只是呼叫時傳 is_post_norm=True/False，
     不需要另外為它們開特殊分支。
  3. 目前還沒拍板的常數（S_q/S_k/S_v 的 INT8 vs INT16、
     per_layer_projection_norm 的 S_target 等）只是「先用暫定值
     當參數傳入」，定案後只需要改呼叫處傳的參數值，不需要動
     rmsnorm_core / rmsnorm_call 的程式碼結構。

版本對應
---------------------------------------------------------------------
本檔案對應規格文件 v2.0，已串起以下呼叫點：
    - input_layernorm            (Pre-Norm)
    - self_attn.q_norm           (Post-Norm，v1.7修正後的正確分類)
    - self_attn.k_norm           (Post-Norm，僅 Layer 0~14)
    - self_attn.v_norm           (Post-Norm，無 γ，僅 Layer 0~14)
    - post_attention_layernorm   (Post-Norm)
    - pre_feedforward_layernorm  (Pre-Norm)
    - post_feedforward_layernorm (Post-Norm)
    - post_per_layer_input_norm  (Post-Norm)

尚未串起（文件第7節仍為 open item，待數值/資料流拍板後比照現有
call_xxx() 的 pattern 各自新增一個 wrapper 即可）：
    - model.language_model.norm             （S_lmhead 待確認）
    - model.language_model.per_layer_projection_norm
      （全域單一模組，decoder 迴圈前執行；per_layer_model_projection
       未被官方量化，S_target 待團隊決策，見文件 v2.0 第7節第3項）
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


# =====================================================================
# 基礎定點工具
# =====================================================================

def requantize_m_k(x_int: np.ndarray, s_eff: np.ndarray, out_bits: int = 16) -> np.ndarray:
    """
    per-channel requantize（對應文件 2.1 節「Step0」）。

    用途：
        matmul MAC 後的原始整數，每個 channel 背後隱含的 scale
        不同（因為 weight_scale 是 per-channel），不能直接拿去
        算 mean(X^2)。這個函式把它們對齊到「單一目標 scale」，
        使輸出向量所有元素才真正共用同一個 scale，可以安全進
        Stage1。

    參數：
        x_int    : matmul MAC 後的原始整數向量，shape=(..., N)
        s_eff    : per-channel 有效乘數，定義為
                       S_eff[k] = (S_matmul_in * weight_scale[k]) / S_target
                   也就是「乘上 s_eff 之後就等於把 x_int[k] 對齊到
                   S_target 這把尺」的比例係數，不是 S_target 本身。
                   shape 需與 x_int 最後一維一致（可 broadcast）。
        out_bits : 輸出整數位寬，預設 16（Post-Norm 場合輸出通常
                   接 residual 或後續 RMSNorm 的整數input，需要
                   INT16 的動態範圍）。

    回傳：
        對齊到 S_target 的整數向量（int64 儲存，但數值已 clip 在
        out_bits 對應的有號整數範圍內）。

    備註：
        golden model 階段用 round + clip 模擬定點乘法/位移，
        之後接 bit-accurate 版本時，這裡對應到
        `mlp_golden_model.py` 裡 requantize_m_k 的硬體版本
        （M-multiply-shift），介面（輸入/輸出都是整數）維持一致。
    """
    y = np.round(x_int.astype(np.float64) * s_eff)
    qmax = 2 ** (out_bits - 1) - 1
    qmin = -2 ** (out_bits - 1)
    return np.clip(y, qmin, qmax).astype(np.int64)


def rsqrt_fixed(mean_sq_int: np.ndarray, backend: str = "float") -> np.ndarray:
    """
    Stage2：rsqrt。現在有兩種backend可切換，同一份pipeline可以同時
    驗證「數學邏輯對不對」跟「跟HLS用同一顆核心時數值上差多少」。

    backend="float"（預設）：
        直接用float算1/sqrt(x)，數學上跟目標定點行為等價，用來驗證
        整體pipeline邏輯正確性，不含fast_inv_unit本身的近似誤差。

    backend="fast_inv_unit"：
        呼叫fast_inv_unit_python()，是MLP.cpp裡fast_inv_unit<DT,RT>
        的bit-accurate Python翻譯版（leading-one detector + magic
        constant seed + 兩次Newton-Raphson），數值上會帶有這顆硬體
        核心本身的近似誤差，之後要跟HLS testbench比對時用這個。

        ⚠️ 目前狀態：演算法邏輯已照MLP.cpp逐行翻譯，但兩個magic
        constant（FASTINV_MAGIC_RECIP/FASTINV_MAGIC_RSQRT）與精確
        位元寬度（ms_t/rsq_t等）定義在kernel.h，目前拿不到這份檔案，
        fast_inv_unit_python()裡先用ap_fixed風格的參數化寫法留空、
        搭配合理預設值，等拿到kernel.h的實際數值後，直接把
        FASTINV_MAGIC_RSQRT等常數換掉即可，呼叫方式與其餘pipeline
        完全不用改。在拿到之前，這個backend算出來的數值還不是
        bit-accurate，只能先驗證「兩次Newton-Raphson後的近似行為
        大致正確」，不能拿來跟HLS結果做逐位元比對。

    Epsilon 處理：
        不追求精確複現原始epsilon（config裡的1e-6），改用
        mean_sq_int的固定下限max(mean_sq_int,1)避免除以零，對應
        文件2.1節「Epsilon處理原則」的簡化建議。這個clamp在兩種
        backend都會做，是Stage1/2共通的前處理，不算進fast_inv_unit
        本身的邏輯。

    參數：
        mean_sq_int : Stage1算出的均方值，shape=(..., 1)
        backend     : "float" 或 "fast_inv_unit"

    回傳：
        1/sqrt(mean_sq_int)，同shape。
    """
    ms = np.maximum(mean_sq_int.astype(np.float64), 1.0)
    if backend == "float":
        return 1.0 / np.sqrt(ms)
    elif backend == "fast_inv_unit":
        return fast_inv_unit_python(ms, mode_recip=False)
    else:
        raise ValueError(f"未知的rsqrt backend: {backend}，只接受'float'或'fast_inv_unit'")


# =====================================================================
# fast_inv_unit 的 bit-accurate Python 翻譯
# 對應 MLP.cpp 第64~130行的 fast_inv_unit<DT,RT>()
# =====================================================================

# ✅ 已從論文（An Accurate and Compact Design Integrating Seven Common
# Nonlinear Functions in Deep Learning, ISCAS'25）推導出精確數值。
#
# 論文Section II-A/B給出：
#   B（exponent bias）= 127
#   L（scaling factor）= 2^23 = 8388608（對應23-bit mantissa）
#   σ = 0.0450465（用log(1+h)≈h+σ這個近似式的修正常數，論文原文明確
#       印出這個數字）
#   rsqrt magic（論文式7前段，PDF文字抽取遺失了公式裡的3/2係數，這是
#       PDF對分數/上標常見的抽取瑕疵）：
#       a1 = floor(1.5 * (B - σ) * L)
#   reciprocal magic（論文式10，此處係數"2"在PDF裡有完整印出，不受
#       抽取瑕疵影響）：
#       a2 = floor(2.0 * (B - σ) * L)
#
# ✅ 交叉驗證：算出的a1 = floor(1.5*(127-0.0450465)*8388608)
#   = 1597463007 = 0x5f3759df，與業界公開、行之有年的Quake fast
#   inverse square root magic number逐位元相同。這不是巧合——本論文
#   的推導本身就是同一套IEEE754-style技巧的一般化延伸（用同一個σ
#   常數），這個逐位元吻合可視為對這裡推導過程正確性的獨立驗證。
FASTINV_MAGIC_RSQRT = 0x5f3759df   # = floor(1.5*(127-0.0450465)*2^23)，
                                    # 與公開Quake magic number逐位元相同
FASTINV_MAGIC_RECIP = 0x7ef477d5   # = floor(2.0*(127-0.0450465)*2^23)，
                                    # 依論文式10直接推導，未對照到獨立
                                    # 公開來源核對，但推導方式與已驗證
                                    # 的RSQRT常數同一套公式結構，數學上
                                    # 可信度高；RMSNorm只用到RSQRT這個
                                    # mode，RECIP暫時用不到，供之後
                                    # softmax/sigmoid等模組共用fast_inv_unit
                                    # 時使用

# ✅ 已從kernel.h確認的精確Q-format（原本是依註解推測的猜測值，
# 現在是kernel.h裡的實際typedef）：
#   typedef ap_ufixed<56, 32, AP_RND>  ms_t;   // DT：輸入mean_sq+eps
#   typedef ap_ufixed<32, 12>          rsq_t;  // RT：1/sqrt(mean_sq)輸出
# ap_ufixed<W, I>的W是總位元數、I是整數部分位元數（含符號位，此處
# 皆為unsigned故無符號位），frac = W - I：
#   ms_t  : W=56, I=32 -> F=24 fractional bits  （fast_inv_unit的DT::width/iwidth用得到）
#   rsq_t : W=32, I=12 -> F=20 fractional bits
FASTINV_DT_WIDTH = 56     # ms_t總位元數（kernel.h實際值）
FASTINV_DT_FRAC = 24      # ms_t小數位元數（kernel.h實際值：56-32）
FASTINV_RT_FRAC = 20      # rsq_t小數位元數（kernel.h實際值：32-12）

# 其餘epilogue相關Q-format（kernel.h一併提供，golden model其他部分
# 若之後要做bit-accurate版本會用到，先記錄在這裡）：
#   norm_t : ap_fixed<32, 16, AP_RND, AP_SAT>  -> down_proj dequant後的
#            物理真值，frac=16，對應文件2.1節場合B的Step0輸出
#   ssq_t  : ap_ufixed<64, 42>                 -> Stage1平方和累加器，frac=22
#   lnw_t  : ap_fixed<16, 3>                   -> (1+gamma)，frac=13，
#            ⚠️ 注意這是MLP.cpp原本假設(1+gamma)的舊介面，已於文件
#            0-2節拍板改用純gamma，這個型別命名/含義待HLS同學同步修正
#            （見文件0-2節，跟這裡的fast_inv_unit無關，僅一併記錄避免
#            之後對照kernel.h時搞混）
#   hid_t  : ap_fixed<16, 8, AP_RND, AP_SAT>   -> residual/輸出，Q7.8，
#            frac=8，對應文件0-3決策C目前的s_res=2^-8約定


def fast_inv_unit_python(x: np.ndarray, mode_recip: bool = False,
                          dt_width: int = FASTINV_DT_WIDTH,
                          dt_frac: int = FASTINV_DT_FRAC,
                          rt_frac: int = FASTINV_RT_FRAC,
                          magic_recip: int = FASTINV_MAGIC_RECIP,
                          magic_rsqrt: int = FASTINV_MAGIC_RSQRT) -> np.ndarray:
    """
    fast_inv_unit<DT,RT>的bit-accurate Python翻譯（對應MLP.cpp:77-130）。
    逐元素處理（不是vectorized，因為leading-one detector跟位移量
    每個元素不同，之後效能不夠可以再優化，golden model階段先求正確）。

    演算法四步驟（跟MLP.cpp逐行對應）：
        1) leading-one detector：找出x的定點表示中最高位的1，
           換算出指數e_x跟23-bit假想mantissa M_x（模仿IEEE754格式）
        2) magic seed：用查表常數在INT32域做一次減法，這是Quake
           fast-inverse-sqrt那類技巧的通用化版本
        3) FP-to-INT：把step2的結果解讀回「指數+mantissa」，重建
           成定點數y0，當作Newton-Raphson的初始猜測值
        4) 兩次Newton-Raphson修正：
             mode_recip=True  (1/x)     : y = y*(2 - x*y)
             mode_recip=False (1/sqrt(x)): y = y*(1.5 - 0.5*x*y^2)

    參數：
        x           : 輸入，shape任意，內部依dt_frac轉成定點整數處理
        mode_recip  : False=rsqrt模式（RMSNorm用這個），True=reciprocal模式
        dt_width/dt_frac : DT（輸入x）的ap_fixed位元寬度/小數位元數
        rt_frac     : RT（輸出y）的小數位元數
        magic_recip/magic_rsqrt : 兩個magic constant，待kernel.h補上

    回傳：
        1/sqrt(x)（或1/x）的定點近似值，以float64陣列回傳（golden
        model階段用float64儲存定點數的數值，方便後續運算，實際硬體
        上這是整數定點表示）
    """
    x_flat = np.atleast_1d(x).astype(np.float64).flatten()
    y_flat = np.empty_like(x_flat)

    W = dt_width
    F = dt_frac

    for idx, xv in enumerate(x_flat):
        # ---- 把x轉成W-bit定點整數表示（模擬ap_fixed<W, W-F>的bit pattern）----
        x_int = int(round(xv * (1 << F)))
        if x_int <= 0:
            # x<=0在rsqrt/reciprocal下沒有數學意義，理論上Stage1的
            # clamp(mean_sq_int,1)已經避免這個情況，這裡防禦性處理
            y_flat[idx] = 0.0
            continue

        # ---- 1) leading-one detector ----
        pos = x_int.bit_length() - 1  # 最高位1的位置（對應MLP.cpp的pos）
        e_x = pos - F

        mb = x_int & ~(1 << pos)  # 清掉最高位的隱含1
        if pos >= 23:
            M_x = mb >> (pos - 23)
        else:
            M_x = mb << (23 - pos)
        M_x &= (1 << 23) - 1

        # ---- 2) magic-constant seed（INT32域）----
        I_x = ((e_x + 127) & 0x1FF) << 23 | M_x
        if mode_recip:
            I_y = (magic_recip - I_x) & 0xFFFFFFFF
        else:
            I_y = (magic_rsqrt - (I_x >> 1)) & 0xFFFFFFFF

        # ---- 3) FP-to-INT：重建y0 ----
        e_y = ((I_y >> 23) & 0x1FF) - 127
        M_y = I_y & ((1 << 23) - 1)

        # yb是Q25.31風格的寬定點暫存器（對應MLP.cpp的ap_ufixed<56,25> yb），
        # bit31放隱含的leading 1（2^0），bit[30:8]放M_y
        yb_int = (1 << 31) | (M_y << 8)
        # y = yb << e_y（e_y可正可負，對應MLP.cpp e_y>=0時左移、否則右移）
        if e_y >= 0:
            y0_scaled = yb_int << e_y
        else:
            y0_scaled = yb_int >> (-e_y)
        # yb的基準是2^-31（bit31代表2^0），換算成RT的Q-format（rt_frac位小數）
        y = y0_scaled / (1 << (31 + (31 - rt_frac)))
        # 上一行對齊：yb_int本身以2^-31為單位，y0_scaled同理；
        # 除以2^31拿到「真實浮點值」，再乘上2^rt_frac、取整、除回，
        # 等效於量化到rt_frac小數位元，這裡直接留實數，Newton-Raphson
        # 在golden model階段一樣用float64算，不逐位元模擬中間舍入
        y = y0_scaled / float(1 << 31)

        # ---- 4) 兩次Newton-Raphson ----
        # ⚠️ 2026-09-20 c-simulation發現：原本這裡整段用float64算，
        # docstring也承認「不逐位元模擬中間舍入」——這是刻意的簡化，
        # 但實測發現對某些mean_sq值，這個簡化會讓算出來的inv_rms跟
        # 真正的C++版本差1個ULP（fast_inv_unit.h裡xy跟t這兩個中繼值
        # 分別是ap_ufixed<32,18,AP_RND>跟ap_ufixed<26,2,AP_RND>，
        # 是有限位元寬+round-to-nearest，不是float64精度）。這個
        # 1-ULP差異在大部分情況下無關痛癢，但如果下游剛好有除法/
        # 捨入卡在整數邊界（例如Stage3的inv_rms_scaled捨入），1-ULP
        # 會被放大成一個明顯的整數階躍，實測造成整個1536維輸出
        # systematically偏差~3%（見qat_verification/
        # rmsnorm_verify_stage3_bitwidth.py的testbench發現）。
        # 修正：xy跟t都比照ap_ufixed<W,I,AP_RND>的語意——round到
        # 2^(W-I)分之一、clip到[0, 2^I)範圍。
        XY_FRAC = 32 - 18  # ap_ufixed<32,18,AP_RND>
        T_FRAC = 26 - 2    # ap_ufixed<26,2,AP_RND>

        def quantize_ufixed(v, frac, int_bits):
            scaled = round(v * (1 << frac))
            qmax = (1 << (int_bits + frac)) - 1
            scaled = max(0, min(scaled, qmax))  # ap_ufixed飽和/截斷（無號）
            return scaled / float(1 << frac)

        x_real = x_int / float(1 << F)
        for _ in range(2):
            xy = quantize_ufixed(x_real * y, XY_FRAC, 18)
            if mode_recip:
                t = quantize_ufixed(2.0 - xy, T_FRAC, 2)
            else:
                t = quantize_ufixed(1.5 - 0.5 * xy * y, T_FRAC, 2)
            # C++: y = (RT)(y * t)——每次迭代後都捨入回RT(rt_frac)，
            # 不是只有最後回傳時才捨入，這裡跟著補上，否則第二次
            # 迭代用的y就不是C++實際會用的那個已捨入值
            y_raw = round(y * t * (1 << rt_frac))
            y = y_raw / float(1 << rt_frac)

        y_flat[idx] = y

    return y_flat.reshape(np.shape(x)) if hasattr(x, "shape") else y_flat[0]


# =====================================================================
# Stage1+2+3 核心
# =====================================================================

def rmsnorm_core(x_int: np.ndarray,
                  has_gamma: bool,
                  gamma: Optional[np.ndarray],
                  S_target: float,
                  out_bits: int = 16,
                  rsqrt_backend: str = "float") -> np.ndarray:
    """
    RMSNorm 的核心數學（對應文件「通用流程模板」）：

        Stage1: mean_sq_int = mean(X_int^2)
        Stage2: inv_rms_int = rsqrt_fixed(mean_sq_int)
        Stage3: WB[i] = round(gamma_i / S_target * 2^F)   （硬體上離線折疊進ROM）
                y_i    = round(X_i_int * inv_rms_int * WB[i] >> shift)

    golden model 用浮點直接算 WB 的等價運算（x * inv_rms * gamma /
    S_target），不真的做 2^F 位移，數學上等價，方便先驗證正確性；
    bit-accurate 版本再套上真正的定點 WB 常數表與位移。

    重要前提（呼叫此函式前必須確保）：
        x_int 的所有元素已經共用「同一個」純量 scale。
        如果 x_int 來自 matmul 輸出且 weight_scale 是 per-channel，
        必須先經過 requantize_m_k() 對齊，才能丟進這裡 —— 這件事
        由外層 rmsnorm_call() 負責判斷與執行，本函式不做任何
        「這個輸入到底有沒有對齊」的檢查。

    參數：
        x_int     : Stage1輸入的整數向量，shape=(..., d)，
                    所有元素已共用單一scale
        has_gamma : 是否要乘γ。唯一的 False 案例是 v_norm
                    （with_scale=False，文件場合C特例）。
        gamma     : γ權重，shape=(d,) 或可broadcast至(..., d)。
                    has_gamma=False時可傳None。
        S_target  : 輸出要對齊的目標scale（浮點值表示，即使
                    S_target本身是2的冪次也一樣傳浮點）。
        out_bits  : 輸出整數位寬（INT8=8 或 INT16=16）。
        rsqrt_backend : "float"（預設，快速驗證邏輯）或
                    "fast_inv_unit"（bit-accurate，比對HLS用）。
                    見rsqrt_fixed()的說明。

    回傳：
        對齊S_target的整數向量，int64儲存，數值已clip在out_bits
        對應範圍內。

    備註（硬體對應）：
        文件6節提到「Stage3輸出不做飽和，飽和留給residual add之
        後」；這裡的clip只是golden model為了避免int64運算時中間
        值溢位、方便debug，bit-accurate版本應依文件規格拿掉這裡
        的飽和邏輯，改為單純的位元截斷或依實際硬體規格處理。
    """
    d = x_int.shape[-1]

    # ---- Stage1: 平方均值 ----
    mean_sq_int = np.mean(x_int.astype(np.float64) ** 2, axis=-1, keepdims=True)

    # ---- Stage2: rsqrt ----
    inv_rms_int = rsqrt_fixed(mean_sq_int, backend=rsqrt_backend)

    # ---- Stage3: 乘γ（若有）、對齊輸出scale ----
    if has_gamma:
        assert gamma is not None and gamma.shape[-1] == d, \
            f"has_gamma=True 但 gamma shape 不符：gamma={None if gamma is None else gamma.shape}, d={d}"
        y_real = x_int.astype(np.float64) * inv_rms_int * gamma
    else:
        # v_norm 特例：跳過γ相乘（場合C）
        y_real = x_int.astype(np.float64) * inv_rms_int

    y_int = y_real / S_target
    qmax = 2 ** (out_bits - 1) - 1
    qmin = -2 ** (out_bits - 1)
    return np.clip(np.round(y_int), qmin, qmax).astype(np.int64)


# =====================================================================
# 外層 wrapper：依 Pre-Norm / Post-Norm 決定是否做 Step0
# =====================================================================

def rmsnorm_call(x_int: np.ndarray,
                  d: int,
                  is_post_norm: bool,
                  S_target: float,
                  has_gamma: bool,
                  gamma: Optional[np.ndarray] = None,
                  S_matmul_in: Optional[float] = None,
                  weight_scale: Optional[np.ndarray] = None,
                  step0_out_bits: int = 16,
                  out_bits: int = 16,
                  rsqrt_backend: str = "float") -> np.ndarray:
    """
    所有呼叫點最終都會呼叫的同一個通用入口。

    這個函式體現文件2.1節的核心判準：
        「輸入來源的上一步驟，其weight_scale是per-tensor還是
        per-channel」，而不是「這一層是不是residual」。

    is_post_norm=False（Pre-Norm，如 input_layernorm）：
        x_int 直接是residual原始整數，本身已是單一scale，不需要
        Step0，S_matmul_in/weight_scale可以不傳（傳了也不會用到）。
        此時x_int.shape[-1]必須直接等於d。

    is_post_norm=True（Post-Norm，如 post_attention/feedforward_
    layernorm、q/k/v_norm）：
        x_int 是matmul MAC後的原始整數，weight_scale是per-channel，
        必須先做Step0 requantize對齊到S_target，才能進Stage1。
        注意：q/k/v_norm在本文件先前版本被誤判為可以跳過此步驟，
        已於v1.7/v1.8修正——它們的輸入來源（q/k/v_proj）weight_
        scale同樣是per-channel（已實測驗證），故必須用
        is_post_norm=True呼叫。

        ⚠️ 重要架構細節（q/k/v_norm特有，post_attention/feedforward_
        layernorm不會遇到）：
        Step0的作用範圍跟Stage1~3的作用範圍不一定相同！
        以q_norm為例：weight_scale是對整個q_proj輸出per-channel
        （例如512維head_dim時，q_proj總輸出是4096=8heads*512，
        weight_scale.shape=(4096,)），Step0必須對這4096維全部做
        per-channel requantize；但RMSNorm的mean(X^2)是**逐head**
        算的（d=head_dim=512，每個head獨立算自己的RMS，不會跨head
        平均）。所以這裡x_int.shape[-1]（做Step0時的channel數）
        跟d（做Stage1~3時的normalize維度）可能不同。

        本函式的處理方式：
          1. weight_scale.shape[-1]決定Step0在哪個維度上做
             per-channel對齊（不一定等於d）。
          2. Step0做完後，若x_uniform.shape[-1] != d，自動reshape
             成(..., num_groups, d)，對每個group獨立呼叫
             rmsnorm_core（對應「逐head做RMSNorm」），再reshape
             回原本的channel數。
          3. gamma的shape必須是(d,)（每個head共用同一份γ，這是
             HuggingFace官方q_norm/k_norm的實際做法：一份head_dim
             大小的γ，對每個head重複套用），不是(總channel數,)。

        post_attention_layernorm/post_feedforward_layernorm/
        post_per_layer_input_norm沒有這個問題，因為它們的
        weight_scale.shape[-1]本來就等於d=1536，Step0與Stage1~3
        作用在同一個維度上，不需要reshape。

    參數：
        x_int          : 上游輸出的原始整數，shape[-1]應等於
                          weight_scale.shape[-1]（Post-Norm）或
                          直接等於d（Pre-Norm）
        d              : RMSNorm實際normalize的維度（256/512/1536）
        is_post_norm   : 決定是否執行Step0的唯一開關
        S_target       : Step0對齊目標 = Stage3輸出目標（Post-Norm
                          場合下兩者是同一個值，對應文件場合B
                          「全程維持同一目標scale，不需重複轉換」）
        has_gamma      : 是否乘γ
        gamma          : γ權重，shape必須是(d,)
        S_matmul_in    : 【僅Post-Norm需要】matmul輸入端的activation
                          scale
        weight_scale   : 【僅Post-Norm需要】該matmul的per-channel
                          權重scale，shape=(該matmul實際輸出channel
                          數,)，不一定等於d（見上方q_norm說明）
        step0_out_bits : Step0 requantize後的中繼整數位寬，預設16
        out_bits       : 最終Stage3輸出位寬

    回傳：
        對齊S_target的整數向量，shape與x_int相同（若有做reshape，
        回傳前已reshape回原本的channel數）。
    """
    if is_post_norm:
        assert S_matmul_in is not None and weight_scale is not None, \
            "Post-Norm場合必須提供S_matmul_in與weight_scale才能做Step0"
        assert x_int.shape[-1] == weight_scale.shape[-1], \
            f"x_int維度({x_int.shape[-1]})必須與weight_scale維度" \
            f"({weight_scale.shape[-1]})一致，Step0是對matmul實際輸出" \
            f"channel數做per-channel requantize"

        s_eff = (S_matmul_in * weight_scale) / S_target
        x_uniform = requantize_m_k(x_int, s_eff, out_bits=step0_out_bits)

        total_channels = x_uniform.shape[-1]
        if total_channels != d:
            # q_norm/k_norm/v_norm的情況：Step0作用在總channel數上，
            # 但Stage1~3要逐head獨立算，這裡reshape成(num_heads, d)，
            # 對每個head呼叫rmsnorm_core，最後攤平回總channel數。
            assert total_channels % d == 0, \
                f"總channel數({total_channels})必須是d({d})的整數倍才能reshape成逐head"
            num_groups = total_channels // d
            x_grouped = x_uniform.reshape(*x_uniform.shape[:-1], num_groups, d)
            y_grouped = rmsnorm_core(x_grouped, has_gamma, gamma, S_target,
                                      out_bits=out_bits, rsqrt_backend=rsqrt_backend)
            return y_grouped.reshape(*x_uniform.shape[:-1], total_channels)
        else:
            # post_attention/feedforward/per_layer_input_norm的情況：
            # weight_scale維度本來就等於d，Step0與Stage1~3同一維度，
            # 不需要reshape。
            return rmsnorm_core(x_uniform, has_gamma, gamma, S_target,
                                 out_bits=out_bits, rsqrt_backend=rsqrt_backend)
    else:
        assert x_int.shape[-1] == d, \
            f"Pre-Norm場合輸入維度不符：x_int.shape={x_int.shape}, 預期d={d}"
        return rmsnorm_core(x_int, has_gamma, gamma, S_target,
                             out_bits=out_bits, rsqrt_backend=rsqrt_backend)


# =====================================================================
# Layer 常數容器
# =====================================================================

@dataclass
class LayerConstants:
    """
    一層份的常數集合。真實情況下這些值應該從safetensors checkpoint
    讀出（對應附錄C / C-2的驗證方法），本骨架先用
    make_dummy_layer_constants() 產生假資料讓架構能跑起來。

    之後接真實資料時，只需要重寫一個新的 factory function
    （例如 load_layer_constants_from_checkpoint(layer_idx, safetensors_path)），
    其餘 call_xxx() / rmsnorm_call / rmsnorm_core 完全不用動。

    欄位對照文件出處：
        qkv_input_scale              -> 附錄C `self_attn.q_proj.input_activation_scale`
        gate_up_input_scale          -> 附錄C `mlp.gate_proj.input_activation_scale`
        q/k/v/o_proj_weight_scale    -> per-channel，已於對話中實測驗證shape=(N,1)
        down_proj_weight_scale       -> 附錄C，per-channel
        per_layer_proj_weight_scale  -> 附錄C，per-channel（v1.9驗證完成）
        gamma_*                      -> checkpoint的RMSNorm權重，v1.7已確認
                                         直接用γ本身，不加1（見文件0-2）
    """
    layer_idx: int
    head_dim: int  # 256 或 512，見附錄B

    # --- Pre-Norm用：checkpoint固定值，per-layer查表 ---
    qkv_input_scale: float
    gate_up_input_scale: float

    # --- Post-Norm用：per-channel weight_scale ---
    q_proj_weight_scale: np.ndarray
    k_proj_weight_scale: Optional[np.ndarray]  # 僅Layer 0~14存在
    v_proj_weight_scale: Optional[np.ndarray]
    o_proj_weight_scale: np.ndarray
    down_proj_weight_scale: np.ndarray
    per_layer_proj_weight_scale: np.ndarray

    # --- Post-Norm用：matmul輸入端scale ---
    o_proj_input_scale: float
    down_proj_input_scale: float
    per_layer_proj_input_scale: float

    # --- γ（各RMSNorm呼叫點各自獨立，per-channel/per-dim）---
    gamma_input_layernorm: np.ndarray
    gamma_q_norm: np.ndarray
    gamma_k_norm: Optional[np.ndarray]
    gamma_v_norm: Optional[np.ndarray]     # v_norm無γ，僅佔位不使用
    gamma_post_attention_layernorm: np.ndarray
    gamma_pre_feedforward_layernorm: np.ndarray
    gamma_post_feedforward_layernorm: np.ndarray
    gamma_post_per_layer_input_norm: np.ndarray


# =====================================================================
# 全域常數（不隨layer變動）
# =====================================================================
#
# 以下三組值皆為「暫定值」，狀態見文件第7節：
#   S_RES        : 已定案 v1.6（決策C），但仍待fake-quantization驗證、
#                  擴大樣本數後確認F=8溢位餘裕是否足夠（文件第1節）
#   S_Q/S_K/S_V  : 已於v1.8校準完成，但INT8/INT16最終位寬待Attention
#                  模組負責人拍板（文件0-4節）；目前先採INT16版本
#
# 這三個值之後若改變（例如F=8降為F=7，或INT16改INT8），只需要改這裡
# 的常數定義，所有call_xxx()都會自動套用新值，不需要逐一修改。

S_RES = 2.0 ** -5   # 2026-09-19更新（原F=8，見checklist A5）：
                     # 用WikiText-2 train split 40*512=20480個token、
                     # 全35層streaming統計實測（calibration/
                     # wikitext_calibrate_layer_scales.py），
                     # post_feedforward_layernorm真實輸出全35層的
                     # global max = 595.76（Layer 14, channel 850），
                     # F=8/7/6在這個樣本下全部飽和，F=5（±1024）才能
                     # 完全涵蓋。若排除Layer 0/14/34這3個outlier層，
                     # 剩下32層的最大值只有138.97（F=7就夠），這代表
                     # 用單一全域F=5會讓32個層都被迫犧牲精細度去遷就
                     # 3個層——分組/per-layer S_RES是有數據支持的更優
                     # 架構，但需要在每次跨層交接處插入residual
                     # rescale（per-tensor純量乘法，理論上便宜，但
                     # 目前完全沒有實作/驗證過，屬於未來優化方向，
                     # 見rmsnorm_const_table_redesign.md第15節），
                     # 這次先用保證不溢位的全域F=5。
S_Q = 2.0 ** -10    # 附錄C-2：S_q，暫定INT16
S_K = 2.0 ** -14    # 附錄C-2：S_k，暫定INT16（k_norm量級明顯小於q/v，獨立校準）
S_V = 2.0 ** -10    # 附錄C-2：S_v，暫定INT16

KV_SHARED_START = 15  # layer_idx >= 15為KV共享層，無k/v_proj與k_norm/v_norm


# =====================================================================
# 各呼叫點 wrapper
# =====================================================================
# 每個函式只做一件事：把規格書對這個呼叫點的敘述，翻譯成一次
# rmsnorm_call()的參數。不含額外邏輯，方便對照文件逐條檢查正確性。

def call_input_layernorm(residual_int: np.ndarray, lc: LayerConstants,
                 rsqrt_backend: str = "float") -> np.ndarray:
    """
    Pre-Norm。輸入：residual原始INT16整數（Stage1/2不理會s_res數值）。
    輸出：INT8，對齊q/k/v_proj共用的input_activation_scale，
    直接可送入q/k/v_proj matmul。
    """
    return rmsnorm_call(
        x_int=residual_int, d=1536,
        is_post_norm=False,
        S_target=lc.qkv_input_scale,
        has_gamma=True, gamma=lc.gamma_input_layernorm,
        out_bits=8, rsqrt_backend=rsqrt_backend,
    )


def call_q_norm(q_proj_mac_out: np.ndarray, lc: LayerConstants,
                 S_q_input: float,
                 rsqrt_backend: str = "float") -> np.ndarray:
    """
    Post-Norm（v1.7修正後的正確分類）。
    S_q_input：q_proj matmul的輸入端scale，即input_layernorm輸出時
    對齊的S_target（= lc.qkv_input_scale，假設q/k/v_proj共用同一
    輸入，若之後Attention側確認並非如此，改這裡傳入的值即可）。
    輸出INT16 @ S_q，直接可接RoPE。
    """
    return rmsnorm_call(
        x_int=q_proj_mac_out, d=lc.head_dim,
        is_post_norm=True,
        S_target=S_Q,
        has_gamma=True, gamma=lc.gamma_q_norm,
        S_matmul_in=S_q_input, weight_scale=lc.q_proj_weight_scale,
        out_bits=16, rsqrt_backend=rsqrt_backend,
    )


def call_k_norm(k_proj_mac_out: np.ndarray, lc: LayerConstants,
                 S_k_input: float,
                 rsqrt_backend: str = "float") -> Optional[np.ndarray]:
    """
    Post-Norm。僅Layer 0~14存在（KV共享層layer_idx>=15無此模組，
    回傳None）。
    """
    if lc.layer_idx >= KV_SHARED_START:
        return None
    return rmsnorm_call(
        x_int=k_proj_mac_out, d=lc.head_dim,
        is_post_norm=True,
        S_target=S_K,
        has_gamma=True, gamma=lc.gamma_k_norm,
        S_matmul_in=S_k_input, weight_scale=lc.k_proj_weight_scale,
        out_bits=16, rsqrt_backend=rsqrt_backend,
    )


def call_v_norm(v_proj_mac_out: np.ndarray, lc: LayerConstants,
                 S_v_input: float,
                 rsqrt_backend: str = "float") -> Optional[np.ndarray]:
    """
    Post-Norm，文件場合C特例：無γ（with_scale=False）。
    僅Layer 0~14存在。
    """
    if lc.layer_idx >= KV_SHARED_START:
        return None
    return rmsnorm_call(
        x_int=v_proj_mac_out, d=lc.head_dim,
        is_post_norm=True,
        S_target=S_V,
        has_gamma=False, gamma=None,
        S_matmul_in=S_v_input, weight_scale=lc.v_proj_weight_scale,
        out_bits=16, rsqrt_backend=rsqrt_backend,
    )


def call_post_attention_layernorm(o_proj_mac_out: np.ndarray, lc: LayerConstants,
                                   S_o_proj_input: float,
                 rsqrt_backend: str = "float") -> np.ndarray:
    """
    Post-Norm。輸入：o_proj MAC原始整數（per-channel weight_scale）。
    輸出：INT16 @ s_res，準備與Attention前的residual相加。
    """
    return rmsnorm_call(
        x_int=o_proj_mac_out, d=1536,
        is_post_norm=True,
        S_target=S_RES,
        has_gamma=True, gamma=lc.gamma_post_attention_layernorm,
        S_matmul_in=S_o_proj_input, weight_scale=lc.o_proj_weight_scale,
        out_bits=16, rsqrt_backend=rsqrt_backend,
    )


def call_pre_feedforward_layernorm(residual_int: np.ndarray, lc: LayerConstants,
                 rsqrt_backend: str = "float") -> np.ndarray:
    """
    Pre-Norm。輸出INT8，對齊gate/up_proj共用input_activation_scale。
    """
    return rmsnorm_call(
        x_int=residual_int, d=1536,
        is_post_norm=False,
        S_target=lc.gate_up_input_scale,
        has_gamma=True, gamma=lc.gamma_pre_feedforward_layernorm,
        out_bits=8, rsqrt_backend=rsqrt_backend,
    )


def call_post_feedforward_layernorm(down_proj_mac_out: np.ndarray, lc: LayerConstants,
                                     S_down_input: float,
                 rsqrt_backend: str = "float",
                 step0_out_bits: int = 24) -> np.ndarray:
    """
    Post-Norm。對應MLP.cpp交棒點：本函式接收的是down_proj MAC後、
    尚未做per-channel requantize的原始整數（即文件0-3決策B/C要求
    HLS同學的MLP.cpp只算到ep_dequant為止、算出ypre[j]的物理真值後，
    這裡再依S_RES重新量化）。輸出INT16 @ s_res。

    step0_out_bits預設從16改成24（2026-09-18修正，見批次驗證發現）：
        Layer 0是附錄C的離群值層，down_proj_input_scale=27.8（其他層
        僅0.01~0.1量級，差近1000倍），導致Step0的s_eff（= S_down_in *
        weight_scale / S_RES）落在28~108，跟down_mac_reconstructed
        （量級±5000~5500）相乘後輕鬆超過INT16(±32767)，實測有
        11692/115200（約10%）元素在Step0這一步就飽和截斷，是造成
        Layer 0 post_feedforward_layernorm cos_sim掉到0.77的直接原因
        （不是rsqrt/gamma/eps等RMSNorm數學本身的問題，純粹是Step0
        中繼緩衝位寬對這一層的scale比例不夠）。
        24-bit（±8,388,607）對目前觀察到的所有35層都有足夠餘裕，
        改成全層統一24-bit（而非只給Layer 0特例），理由同文件2.1節
        對WB/M常數表的取捨：硬體上與其為每一層動態決定中繼位寬，
        不如全部拉齊到能涵蓋最壞情況的統一寬度，犧牲一點點電路面積
        換取控制邏輯簡單、且不用擔心之後校準資料變動又要重新分層
        判斷哪些層需要加寬。
    """
    return rmsnorm_call(
        x_int=down_proj_mac_out, d=1536,
        is_post_norm=True,
        S_target=S_RES,
        has_gamma=True, gamma=lc.gamma_post_feedforward_layernorm,
        S_matmul_in=S_down_input, weight_scale=lc.down_proj_weight_scale,
        out_bits=16, rsqrt_backend=rsqrt_backend,
        step0_out_bits=step0_out_bits,
    )


def call_post_per_layer_input_norm(per_layer_proj_mac_out: np.ndarray, lc: LayerConstants,
                                    S_ple_input: float,
                 rsqrt_backend: str = "float") -> np.ndarray:
    """
    Post-Norm（PLE分支）。輸入：per_layer_projection MAC原始整數。
    輸出：INT16 @ s_res，加回主幹1536維residual（非獨立分支，
    見文件第5節重要修正1）。
    """
    return rmsnorm_call(
        x_int=per_layer_proj_mac_out, d=1536,
        is_post_norm=True,
        S_target=S_RES,
        has_gamma=True, gamma=lc.gamma_post_per_layer_input_norm,
        S_matmul_in=S_ple_input, weight_scale=lc.per_layer_proj_weight_scale,
        out_bits=16, rsqrt_backend=rsqrt_backend,
    )


# =====================================================================
# TODO（尚未串起，待規格拍板後比照上面call_xxx()的pattern新增）
# =====================================================================
#
# def call_final_norm(...):
#     """model.language_model.norm，文件場合D。
#     待辦：S_lmhead（LM Head輸入所需格式）尚未確認，見文件第7節第2項。
#     一旦確認，這裡直接呼叫：
#         rmsnorm_call(x_int=residual_int, d=1536, is_post_norm=False,
#                      S_target=S_LMHEAD, has_gamma=True, gamma=...)
#     """
#
# def call_per_layer_projection_norm(...):
#     """model.language_model.per_layer_projection_norm。
#     待辦（文件v2.0第7節第3項）：
#       (a) per_layer_model_projection未被官方量化(modules_to_not_
#           convert排除)，硬體上是否仍要量化這個Linear層需團隊決策
#       (b) 若量化，weight_scale需自行離線校準（無官方數值，類似
#           s_res處境）
#       (c) per_layer_model_projection_scale=1/sqrt(1536)與
#           per_layer_input_scale=1/sqrt(2)這兩個已知數學常數需摺
#           入pipeline（非新增校準負擔，只是待接入）
#     這是全域單一模組（非逐層），在decoder layer迴圈之前執行一次，
#     256維，同一組γ廣播套用至全部35個層切片，呼叫方式上與現有
#     call_xxx()不同之处：不屬於某個layer_idx的LayerConstants，
#     需要獨立的全域常數容器。
#     """


# =====================================================================
# 假資料產生（骨架驗證用，之後換成真實checkpoint讀取）
# =====================================================================

def make_dummy_layer_constants(layer_idx: int, head_dim: int,
                                rng: np.random.Generator) -> LayerConstants:
    """
    產生單一層份的假資料，讓整體pipeline先能跑起來、驗證shape/流程
    是否正確。

    ⚠️ 待辦（正式接checkpoint前必須修正）：
        q_proj_weight_scale的N目前用head_dim*4或*8隨意估計，
        必須改成附錄C實測的真實值：
            head_dim=256層 -> q_proj weight_scale shape=(2048,1)
            head_dim=512層 -> q_proj weight_scale shape=(4096,1)
        （即num_attention_heads*head_dim，非head_dim本身乘的估計值）

    之後要做的事：
        寫一個load_layer_constants_from_checkpoint(layer_idx, path)，
        直接用safetensors讀取真實weight_scale/activation_scale/
        gamma，取代這個函式。呼叫端（call_xxx系列）完全不用改。
    """
    has_kv = layer_idx < KV_SHARED_START
    q_channels = 4096 if head_dim == 512 else 2048  # 對應附錄C實測shape

    return LayerConstants(
        layer_idx=layer_idx, head_dim=head_dim,
        qkv_input_scale=float(rng.uniform(0.01, 0.5)),
        gate_up_input_scale=float(rng.uniform(0.01, 0.05)),
        q_proj_weight_scale=rng.uniform(0.002, 0.01, size=q_channels),
        k_proj_weight_scale=(rng.uniform(0.001, 0.01, size=head_dim) if has_kv else None),
        v_proj_weight_scale=(rng.uniform(0.005, 0.03, size=head_dim) if has_kv else None),
        o_proj_weight_scale=rng.uniform(0.005, 0.015, size=1536),
        down_proj_weight_scale=rng.uniform(0.001, 0.02, size=1536),
        per_layer_proj_weight_scale=rng.uniform(0.005, 0.02, size=1536),
        o_proj_input_scale=float(rng.uniform(0.01, 0.05)),
        down_proj_input_scale=float(rng.uniform(0.005, 0.02)),
        per_layer_proj_input_scale=float(rng.uniform(0.05, 0.2)),
        gamma_input_layernorm=rng.normal(1.0, 0.05, size=1536),
        gamma_q_norm=rng.normal(1.0, 0.05, size=head_dim),
        gamma_k_norm=(rng.normal(1.0, 0.05, size=head_dim) if has_kv else None),
        gamma_v_norm=None,
        gamma_post_attention_layernorm=rng.normal(1.0, 0.05, size=1536),
        gamma_pre_feedforward_layernorm=rng.normal(1.0, 0.05, size=1536),
        gamma_post_feedforward_layernorm=rng.normal(1.0, 0.05, size=1536),
        gamma_post_per_layer_input_norm=rng.normal(1.0, 0.05, size=1536),
    )


# =====================================================================
# 串接範例：單一token跑過Attention/MLP/PLE三條分支
# =====================================================================

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    head_dim = 512  # 假設是full_attention層，如Layer 9
    lc = make_dummy_layer_constants(layer_idx=9, head_dim=head_dim, rng=rng)

    residual = rng.integers(-2 ** 14, 2 ** 14, size=(1536,))

    # --- Attention分支 ---
    x_in_attn = call_input_layernorm(residual, lc)

    q_mac = rng.integers(-127, 127, size=lc.q_proj_weight_scale.shape[0])
    k_mac = rng.integers(-127, 127, size=head_dim) if lc.k_proj_weight_scale is not None else None
    v_mac = rng.integers(-127, 127, size=head_dim) if lc.v_proj_weight_scale is not None else None

    q_out = call_q_norm(q_mac, lc, S_q_input=lc.qkv_input_scale)
    k_out = call_k_norm(k_mac, lc, S_k_input=lc.qkv_input_scale) if k_mac is not None else None
    v_out = call_v_norm(v_mac, lc, S_v_input=lc.qkv_input_scale) if v_mac is not None else None

    o_mac = rng.integers(-127, 127, size=1536)  # 假設Attention運算後o_proj MAC輸出
    attn_residual_delta = call_post_attention_layernorm(o_mac, lc, S_o_proj_input=lc.o_proj_input_scale)

    # --- MLP分支 ---
    x_in_mlp = call_pre_feedforward_layernorm(residual, lc)
    down_mac = rng.integers(-127, 127, size=1536)  # 假設gate/up/down運算後down_proj MAC輸出
    mlp_residual_delta = call_post_feedforward_layernorm(down_mac, lc, S_down_input=lc.down_proj_input_scale)

    # --- PLE分支 ---
    ple_mac = rng.integers(-127, 127, size=1536)
    ple_residual_delta = call_post_per_layer_input_norm(ple_mac, lc, S_ple_input=lc.per_layer_proj_input_scale)

    print("q_norm out:", q_out.shape, q_out[:8])
    print("k_norm out:", None if k_out is None else (k_out.shape, k_out[:8]))
    print("v_norm out:", None if v_out is None else (v_out.shape, v_out[:8]))
    print("attn residual delta:", attn_residual_delta.shape, attn_residual_delta[:8])
    print("mlp residual delta :", mlp_residual_delta.shape, mlp_residual_delta[:8])
    print("ple residual delta :", ple_residual_delta.shape, ple_residual_delta[:8])


# =====================================================================
# 通往HLS之前，還需要補上的東西（Checklist）
# =====================================================================
"""
以下按優先順序列出，從golden model走到可以開始寫HLS，還缺的部分：

【A. 規格拍板類（不寫code，但沒定案code沒法定型）】
  A1. q/k/v_norm 的 S_q/S_k/S_v 最終INT8/INT16位寬
      -> 待Attention模組負責人依QKᵀ/weighted-sum engine需求決定
      -> 影響：out_bits參數、附錄C-2對應F值（INT8: F_q=2,F_k=6,F_v=2）
  A2. ln_gamma命名/介面的(1+γ)問題
      -> 數學上已定案不加1（v1.7），但MLP.cpp/mlp_model.h的舊註解
         （lnw_t等）需要HLS同學實際同步修正，目前golden model已
         採「直接用γ」，若checkpoint reader寫錯仍會跟這裡對不上
  A3. model.language_model.norm的S_lmhead
      -> 需要LM Head的輸入量化需求（文件第7節第2項）
  A4. per_layer_projection_norm的S_target
      -> per_layer_model_projection未被官方量化，需團隊決策是否
         要量化、weight_scale如何校準（文件v2.0第7節第3項）
  A5. 【2026-09-19已修正，F=8 -> F=5】s_res的溢位餘裕驗證
      -> 原估計：僅100筆WikiText樣本，83.6對128只有1.53倍餘裕。
      -> 中間證據（6層*4prompt*75token實測）：抽樣的6層裡有3層
         （Layer 0、14、34）超出±128範圍，最壞3.96倍。
      -> 最終證據（WikiText-2 train split，40*512=20480 token，
         全35層streaming統計，calibration/
         wikitext_calibrate_layer_scales.py）：
         post_feedforward_layernorm全35層的global max=595.76
         （Layer 14, channel 850）。F=8/7/6在這個樣本下全部飽和，
         F=5（±1024）才完全涵蓋。
      -> 若排除Layer 0(498.5)/14(595.8)/34(208.3)這3個outlier層，
         剩下32層的最大值只有138.97（Layer 13），F=7就綽綽有餘——
         代表分組/per-layer S_RES是有堅實數據支持的更優架構（32層
         不用被迫犧牲精細度），但S_RES是residual stream本身的
         表示格式，35層共用同一個累加buffer，要分組必須在每次跨層
         交接處插入rescale（per-tensor純量乘法，理論上硬體成本低，
         但完全沒實作/驗證過），評估後決定這次先不做，記錄為
         有數據支持的後續優化方向，見rmsnorm_const_table_redesign.md
         第15節。
      -> 已修正：S_RES從2^-8改成2^-5（本檔案global常數定義處）。
      -> 已用batch verify + end-to-end perplexity驗證修正後的效果，
         見rmsnorm_const_table_redesign.md第16節。
      -> S_q/S_k/S_v同樣只有100筆樣本，需一併比照這次的方法擴充驗證
         （屬未來工作，這次只處理了S_RES）。
  A6. 【2026-09-19更新：pre_ln部分已修正，post_ln部分併入A5】
      Layer 14 pre_feedforward_layernorm在部分token上cos_sim掉到
      0.7997（batch驗證）。
      -> 原始診斷（2026-09-18，已更正）誤判為「小訊號token INT8
         精細度不足」——當時只看了輸出向量的前5個元素恰好量級小，
         誤導了判斷。
      -> 正確根因（2026-09-19重新診斷，逐channel誤差分析）：
         channel 1133是這一層的activation outlier channel（跟
         Layer 9的channel 1295、Layer 31/34的channel 438同一類
         現象——固定少數channel系統性遠大於其他channel，是
         transformer常見的outlier channel現象）。checkpoint提供的
         gate_up_input_scale=0.0176353只夠涵蓋±2.24，channel 1133
         實測衝到13.66（超出6倍），INT8直接飽和，單一channel誤差
         +11.4，主導了整個1536維向量的cos_sim。golden model沒有
         算錯——這是checkpoint本身的activation_scale校準覆蓋範圍
         不足，若真的部署，硬體用同一個scale也會在這個token上飽和。
      -> 已修正（qat_verification/rmsnorm_verify_batch.py的
         RECALIBRATED_GATE_UP_SCALE）：用目前75個token重新校準
         gate_up_input_scale（0.0176353 -> 0.107579，6.1倍），
         pre_ln cos_sim從min=0.7997/mean=0.9162改善到
         min=0.9952/mean=0.9968。這個校準值樣本數小（僅75個
         token），還不夠穩健，需要wikitext大規模校準才能定案，
         暫時當作「已驗證修正方向正確」的驗證，不是最終數值。
      -> post_feedforward_layernorm的部分（同一個Layer 14，
         cos_sim min=0.9118）經診斷是S_RES溢位（channel 850實測
         507 vs ±128範圍），跟pre_ln是不同問題、不同根因，已併入
         上面A5一起處理（S_RES是全域常數，不是Layer 14獨有問題）。

【B0. 新發現，建議回饋進規格文件】
  B0-1. q/k/v_norm的Step0作用範圍(weight_scale.shape[-1]，即
        num_heads*head_dim)跟Stage1~3的normalize維度(d=head_dim)
        不同，RMSNorm是逐head獨立算RMS，不是跨整個q_proj輸出算。
        文件目前完全沒提到這個reshape細節，post_attention/
        feedforward/per_layer_input_norm則不會遇到（它們weight_
        scale維度本來就等於normalize維度）。建議在文件2.1節「場合
        B」補充這個「Step0與Stage1~3作用維度可能不同」的但書，
        避免其他人實作時漏掉逐head這件事。
  B0-2. 【2026-09-19新發現，高優先度，影響residual/S_RES整體設計】
        每個decoder layer的最後有一個`layer_scalar`純量乘法
        （HuggingFace官方原始碼modeling_gemma4.py:1444
        `hidden_states *= self.layer_scalar`），在Attention/MLP/PLE
        三個分支的residual加法**全部做完之後才乘一次**，之前golden
        model跟這份文件完全沒有提到這個乘法、也沒有處理過。
        實測全35層的layer_scalar值，全部<1.0（範圍0.027~0.89，
        見residual_golden_model.py），不是恆等於1的無效buffer。
        這直接解釋了一個之前令人困惑的現象：post_feedforward_
        layernorm的delta量級可以到595.76（Layer 14），但下一層
        實測的residual卻只有~65量級——因為Layer 14的layer_scalar=
        0.0493，三個分支加總後乘上這個係數，大delta被壓縮回合理
        範圍才交給下一層。
        關鍵開放問題（跟S_RES的F值選擇直接相關，見A5）：
          方案A（忠於官方float forward）：residual buffer中途需要
          能裝下三個分支加總、但還沒乘layer_scalar的未縮放大值
          （Layer 14最壞可到~660），這正是目前A5用F=5的依據。
          方案B（硬體優化）：利用純量乘法對加法可分配的數學性質，
          把layer_scalar拆開套用到每個分支的delta上再加總，
          residual buffer全程維持「已縮放」的小值，可能F=7甚至F=8
          就夠——但這只在exact數學下跟方案A等價，定點量化下兩者的
          rounding誤差路徑不同，需要各自驗證，不能假設等價。
        這是residual模組（不是RMSNorm模組）的核心設計決策，
        residual_golden_model.py先以方案A（忠於官方forward順序）
        搭骨架，方案B是否要做留給team決定，見該檔案內的說明。

【B. Golden model本身需要補的正確性驗證】
  B1. 拿真實checkpoint取代make_dummy_layer_constants()，跑過全部
      35層，確認q_proj_weight_scale等shape與附錄C實測一致
  B2. 加上與HuggingFace官方float forward的cosine similarity驗證
      （比照MLP golden model check_v6.ipynb Cell 13/14的作法），
      确认golden model輸出（INT16結果乘回S_target還原真實值後）
      跟float reference的誤差在可接受範圍
  B3. 補齊k/v_norm在KV共享層(layer_idx>=15)的None處理是否會在
      下游（Attention分支組裝、residual accumulate）正確跳過，
      目前只在本檔案內測試了單層，尚未跑過跨層/完整35層的串接
  B4. Step0 requantize的s_eff計算目前完全用float精度算，之後要
      驗證golden model算出的S_eff，換成bit-accurate的M-multiply-
      shift（對應requantize_m_k的硬體版本，M=round(S_eff*2^k_bits)
      這個離線常數）之後誤差是否可接受

【C. 從golden model轉HLS前，需要額外產生的東西】
  C1. ✅ 【已完成】rsqrt_fixed()已支援backend="fast_inv_unit"切換。
      magic constant已從論文（An Accurate and Compact Design...,
      ISCAS'25）Section II-A/B推導出精確數值：
        FASTINV_MAGIC_RSQRT = 0x5f3759df
          （= floor(1.5*(127-0.0450465)*2^23)，且此值與業界公開的
           Quake fast inverse square root magic number逐位元相同，
           可視為推導過程的獨立交叉驗證，信心度高）
        FASTINV_MAGIC_RECIP = 0x7ef477d5
          （= floor(2.0*(127-0.0450465)*2^23)，依論文式10直接推導，
           RMSNorm用不到這個mode，供之後softmax/sigmoid等模組共用
           fast_inv_unit時使用）
      DT/RT的精確Q-format（ms_t=ap_ufixed<56,32,AP_RND>、
      rsq_t=ap_ufixed<32,12>）已從kernel.h取得並填入。
      剩餘待辦：
        (a) 用真實magic constant重新跑一次完整的(A)~(E)驗證，取代
            先前用Quake常數demo的結果（技術上兩者算出的magic RSQRT
            數值相同，但RECIP不同，如果之後有用到reciprocal模式的
            模組要注意這點）
        (b) 依決策A確認DT/RT在Attention側(256/512維)的動態範圍是否
            需要跟MLP側(1536維)不同的模板參數（見文件0-3決策A待辦），
            目前fast_inv_unit_python()已把dt_width/dt_frac/rt_frac
            做成參數，一旦確認不同呼叫點需要不同Q-format，直接在
            呼叫時傳不同參數即可，不需要改函式本身
        (c) 建議請同學/硬體那邊確認一下這兩個常數是否與他們燒進ROM
            的實際數值一致（尤其RECIP沒有公開來源可以交叉驗證），
            避免golden model跟硬體實作用不同常數而長期對不齊
  C2. WB[k]常數表產生腳本：把γ/S_target折算成
      WB[i]=round(γ_i/S_target*2^F)，per-layer per-channel，
      離線算好、輸出成HLS可以#include的常數陣列/ROM初始化檔
      （目前golden model是直接用浮點算等價結果，還沒有真的產生
      這張表）
  C3. Step0的S_eff/M常數表：比照C2，把
      S_eff[k]=(S_matmul_in*weight_scale[k])/S_target
      為每個Post-Norm呼叫點、每一層、每個channel算出對應的
      M(乘數)與shift常數，輸出成常數表
  C4. 針對三種維度(256/512/1536)產生對應的HLS template參數與
      testbench輸入向量（目前golden model是用numpy跑浮點/整數
      混合運算，還沒有production成HLS tb_input.bin這類格式）
  C5. corner case測試向量：比照MLP.cpp的gen_stimulus()裡
      ZERO_X/SMALL_MS/LARGE_MS等案例，針對Attention側維度
      (256/512)也產生對應的极端值測試，驗證eps簡化
      (max(mean_sq_int,1))在小維度下是否仍然安全
  C6. AXI/buffer相關的常數定義（STAGE3_PAR等並行度參數）目前完
      全沒有反映在golden model裡，屬於純硬體設計參數，golden
      model不需要模擬，但需要在HLS端另外定義

【D. 跨模組介接需要的東西】
  D1. 與HLS同學確認MLP.cpp修改後的實際輸出介面（ypre[j]的具體
      型別/shape），golden model目前假設down_proj_mac_out直接是
      「down_proj MAC後的原始整數」，需要跟同學修改後的真實交棒
      點對齊
  D2. 與Attention模組負責人確認q/k/v_proj是否真的共用同一個輸入
      scale（目前golden model在call_q_norm/k_norm/v_norm裡都傳
      S_q_input=lc.qkv_input_scale這個假設，需要實際核對）
"""