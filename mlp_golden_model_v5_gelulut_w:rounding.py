import numpy as np

# ============================================================
# 1. 共用基礎模組 (硬體實際在做的事)
# ============================================================
def unpack_bits(packed_bytes, num_bits, output_shape):
    values_per_byte = 8 // num_bits
    unpacked = []
    for byte in packed_bytes.flatten():
        for i in range(values_per_byte):
            shift = i * num_bits
            mask = (1 << num_bits) - 1
            val = (byte >> shift) & mask
            unpacked.append(val)
    return np.array(unpacked).reshape(output_shape)

def mac_accumulate(x_int, W_int):
    acc = np.zeros(W_int.shape[0], dtype=np.int32)
    for k in range(len(x_int)):
        acc += W_int[:, k].astype(np.int32) * x_int[k].astype(np.int32)
    return acc

def requantize_m_k(Q_acc, S_eff, k_bits=16, out_bits=8):
    """
    統一的 M >> k 重縮放模組。
    取代了舊版的 dequant_to_next_int 與 dequant_to_float。
    """
    M = np.round(S_eff * (2 ** k_bits)).astype(np.int64) 
    Temp = Q_acc.astype(np.int64) * M
    shifted = Temp >> k_bits
    
    if out_bits == 8:
        return np.clip(shifted, -128, 127).astype(np.int8)
    elif out_bits == 16:
        return np.clip(shifted, -32768, 32767).astype(np.int16)
    return shifted.astype(np.int32)

# ============================================================
# 2. [Offline] 離線產生 INT8 LUT
# ============================================================
def generate_gelu_lut(S_gate_out, S_gelu_out):
    lut = np.zeros(256, dtype=np.int8)
    for i in range(256):
        x_int8 = i - 256 if i >= 128 else i # 將 0-255 映射到 -128 到 127 (在標準的硬體與 C/Python 底層中，INT8是用二補數來儲存)
        x_float = x_int8 * S_gate_out
        gelu_float = x_float * (1 / (1 + np.exp(-1.702 * x_float)))
        y_int8 = np.clip(np.round(gelu_float / S_gelu_out), -128, 127)
        lut[i] = np.int8(y_int8)
    return lut

# ============================================================
# 3. 完整 MLP (INT8 LUT 版)
# ============================================================
def mlp_golden_lut(x_int, S_x,
                   W_gate_packed, S_gate, num_bits_gate,
                   W_up_packed, S_up, num_bits_up,
                   W_down_packed, S_down, num_bits_down,
                   S_gate_out, S_up_out, S_gelu_out, S_down_in,
                   gate_shape, up_shape, down_shape,
                   k_bits=16):
    
    # 離線準備 LUT
    gelu_lut = generate_gelu_lut(S_gate_out, S_gelu_out)

    # ---- gate_proj: MAC → M>>k (壓回 INT8) → LUT查表 ----
    W_gate_int = unpack_bits(W_gate_packed, num_bits_gate, gate_shape)
    gate_acc = mac_accumulate(x_int, W_gate_int)
    S_eff_gate = (S_x * S_gate.flatten()) / S_gate_out
    gate_int8 = requantize_m_k(gate_acc, S_eff_gate, k_bits, out_bits=8)
    
    # 直接利用 bitwise AND 把 INT8 轉成 0~255 的合法記憶體位址
    # lut_indices = gate_int8 & 0xFF 
    lut_indices = gate_int8.astype(np.int32) & 0xFF   # 修正
    gate_activated_int8 = gelu_lut[lut_indices]

    # ---- up_proj: MAC → M>>k (壓回 INT8) ----
    W_up_int = unpack_bits(W_up_packed, num_bits_up, up_shape)
    up_acc = mac_accumulate(x_int, W_up_int)
    S_eff_up = (S_x * S_up.flatten()) / S_up_out
    up_int8 = requantize_m_k(up_acc, S_eff_up, k_bits, out_bits=8)

    # ---- 逐元素相乘 (INT8 * INT8 = INT16) ----
    activated_int16 = gate_activated_int8.astype(np.int16) * up_int8.astype(np.int16)

    # ---- 對齊 down_proj: INT16 → M>>k (壓回 INT8) ----
    S_eff_mult = (S_gelu_out * S_up_out) / S_down_in
    down_in_int8 = requantize_m_k(activated_int16, S_eff_mult, k_bits, out_bits=8)

    # ---- down_proj: MAC → 離開加速器前轉回 Float ----
    W_down_int = unpack_bits(W_down_packed, num_bits_down, down_shape)
    down_acc = mac_accumulate(down_in_int8, W_down_int)
    down_float = down_acc.astype(np.float32) * S_down_in * S_down.flatten()

    #待釐清：後面的RMSNorm是整數輸入or浮點輸入？如果是整數輸入，這裡就不需要轉回浮點數了。 
    return down_float