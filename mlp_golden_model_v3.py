import numpy as np

# ============================================================
# 1. 拆包 —— 不需要改,任何dequant策略都要用到這個
# ============================================================
def unpack_bits(packed_bytes, num_bits, output_shape):
    """把打包成uint8的低位元權重,拆解成獨立的整數陣列"""
    values_per_byte = 8 // num_bits
    unpacked = []
    for byte in packed_bytes.flatten():
        for i in range(values_per_byte):
            shift = i * num_bits
            mask = (1 << num_bits) - 1
            val = (byte >> shift) & mask
            unpacked.append(val)
    return np.array(unpacked).reshape(output_shape)


# ============================================================
# 2. 純整數MAC累加 —— 不需要改,這是硬體PE array實際在做的事
# ============================================================
def mac_accumulate(x_int, W_int):
    """
    x_int: 量化整數輸入,形狀 [input_dim]
    W_int: 拆包後的量化整數權重,形狀 [output_dim, input_dim]
    回傳:Q_acc,純整數累加結果,形狀 [output_dim]
    """
    acc = np.zeros(W_int.shape[0], dtype=np.int32)
    for k in range(len(x_int)):
        acc += W_int[:, k].astype(np.int32) * x_int[k].astype(np.int32)
    return acc

# 待釐清：acc用int64? -> 解決了，把它全部改成int32，到dequant_to_next_int的時候才改成int64


# ============================================================
# 3. 明確dequant —— 用在「下一步是非線性運算」的地方(這裡是GELU)
# ============================================================
def dequant_to_float(Q_acc, S_x, S_w):
    """
    Q_acc: mac_accumulate算出來的純整數累加結果
    S_x:   輸入的activation scale(純量)
    S_w:   weight的scale(逐output channel)
    回傳:真實浮點數值
    """
    return Q_acc.astype(np.float32) * S_w.flatten() * S_x

# 待釐清：是因為scale是fp32，所以acc才是轉乘fp32嗎？-> 不是，是因為乘上scale之後，整數acc會自然而然變成非整數
# 待釐清：到底是要dequant回多少精度？下面這段式子是把它dequant回bf16的版本
# 下面這段先不用用，先求邏輯對不對,再求精度貼近硬體（有餘力要驗證「模擬BF16精度損失後,誤差會不會變得不能接受」再用）
"""
import ml_dtypes  # pip install ml_dtypes --break-system-packages

def dequant_to_float(Q_acc, S_x, S_w, simulate_bf16=True):
    y = Q_acc.astype(np.float32) * S_w.flatten() * S_x
    if simulate_bf16:
        y = y.astype(ml_dtypes.bfloat16).astype(np.float32)  # 先捨入成bf16精度,再轉回fp32方便後續運算
    return y
"""

# ============================================================
# 4. S_eff融合 —— 用在「下一步直接是相乘/另一個量化層」的地方(這裡是up_proj)
# ============================================================
def dequant_to_next_int(Q_acc, S_x, S_w, S_next_in, k_bits=16):
    """
    S_next_in: 對齊的目標scale(這裡要跟GELU重新量化用的scale一致)
    回傳:Q_next_in,可以直接跟其他同尺度整數相乘的量化整數
    """
    S_eff = (S_x * S_w.flatten()) / S_next_in # S_eff是新的scale
    M = np.round(S_eff * (2 ** k_bits)).astype(np.int64) 
    # 把S_eff這個浮點數,轉換成「整數M + 位移量k」的形式。
    # S_eff × 2^k,是把S_eff放大2^k倍(讓小數點後的精度,被搬到整數位數裡保留下來),np.round(...)則是四捨五入取整數
    # 這一步是「離線」用軟體算好的,算出來的M,是一個固定常數,會被燒錄進硬體,不是硬體runtime要做的事。
    Temp = Q_acc.astype(np.int64) * M
    return (Temp >> k_bits).astype(np.int32)

# 待釐清：確認這個M, k公式是我們最終的想法。另外要確認這個精度是否正確合理

# ============================================================
# 5. GELU-sigmoid近似 —— 你們之前決定的近似公式,新的函式
# ============================================================
def gelu_sigmoid_approx(x):
    """
    用sigmoid去近似GELU(對照你們選定的公式,這裡先用標準數學sigmoid當作golden基準,
    之後如果要更精確模擬硬體log2/exp2近似,可以替換這個函式內部實作)
    """
    return x * (1 / (1 + np.exp(-1.702 * x)))   # 這是GELU-sigmoid近似常見的一種寫法,係數可依你們實際採用的公式調整

# 待釐清：最後採用的gelu的sigmoid近似公式是這個嗎？還有可能改成查表嗎？


# ============================================================
# 6. Requantize —— 新的函式,把GELU算完的浮點結果,重新量化回整數
# ============================================================
def requantize(x_float, S_target):
    """
    x_float:  要重新量化的浮點數值(這裡是GELU算完的結果)
    S_target: 目標scale(靜態、事先訂好的常數,要跟up_proj那邊S_eff的S_next_in一致)
    回傳:量化後的整數
    """
    x_int = np.round(x_float / S_target)
    x_int = np.clip(x_int, -128, 127)   # np.clip是為了處理「範圍溢出」的防呆,不是為了處理zero_point。
    # 
    return x_int.astype(np.int8)


# ============================================================
# 7. 完整MLP組裝 —— 把以上全部串起來
# ============================================================
def mlp_golden(x_int, S_x,
                W_gate_packed, S_gate, num_bits_gate,
                W_up_packed, S_up, num_bits_up,
                W_down_packed, S_down, num_bits_down,
                S_gelu_out,           # GELU重新量化用的靜態scale
                gate_shape, up_shape, down_shape,
                k_bits=16):
    
    # ---- gate_proj:MAC → dequant成浮點數 → GELU → requantize回整數 ----
    W_gate_int = unpack_bits(W_gate_packed, num_bits_gate, gate_shape)
    gate_acc = mac_accumulate(x_int, W_gate_int)
    gate_float = dequant_to_float(gate_acc, S_x, S_gate)
    gate_activated_float = gelu_sigmoid_approx(gate_float)
    gate_activated_int = requantize(gate_activated_float, S_gelu_out)

    # ---- up_proj:MAC → 直接用S_eff對齊到S_gelu_out,不dequant成浮點數 ----
    W_up_int = unpack_bits(W_up_packed, num_bits_up, up_shape)
    up_acc = mac_accumulate(x_int, W_up_int)
    up_next_int = dequant_to_next_int(up_acc, S_x, S_up, S_next_in=S_gelu_out, k_bits=k_bits)

    # ---- 逐元素相乘(兩邊都是整數,且共用S_gelu_out這個尺度)----
    activated_int = gate_activated_int.astype(np.int32) * up_next_int.astype(np.int32)
    # 這裡要注意:兩個int8×int8相乘完的結果,數值範圍會變大,可能需要再一次requantize才送進down_proj
    # (取決於你們down_proj預期吃到的輸入精度是多少bit,這是你們要另外確認的規格)
    S_activated = S_gelu_out * S_gelu_out   # 相乘完的等效scale(兩邊scale相乘)
    activated_int_final = requantize(
        activated_int.astype(np.float32) * S_activated, S_down_input_scale
    )  # S_down_input_scale 需要你們補上,是down_proj預期吃到的輸入scale

    # ---- down_proj:MAC → dequant成浮點數(因為後面接RMSNorm)----
    W_down_int = unpack_bits(W_down_packed, num_bits_down, down_shape)
    down_acc = mac_accumulate(activated_int_final, W_down_int)
    down_float = dequant_to_float(down_acc, S_down_input_scale, S_down)

    return down_float

