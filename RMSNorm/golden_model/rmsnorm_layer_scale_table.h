// 自動產生檔案，請勿手動修改。由 generate_rmsnorm_const_tables.py 產生。
// 2026-09-20第三版（per-layer drop_bits/shift，取代全域共用版本）：
// 全域共用一組frac/drop_bits時，Layer 0這種inv_scale遠小於全域最大值
// 的層，Stage3中途捨入會直接underflow到cos_sim=0.000000（不是精度
// 下降，是資訊整個被捨光，見qat_verification/
// rmsnorm_verify_stage3_bitwidth.py的實測結果）。改成每一層各自的
// int_bits/frac/drop_bits/最終shift，不再共用單一#define。
//
// RMSNORM_INV_GATE_UP_SCALE_FRAC[layer]：每層自己的frac（不是共用值）
// RMSNORM_PRE_DROP_BITS[layer]         ：Step A算完後要捨去的bit數
// RMSNORM_STAGE3_SHIFT_PRE[layer]      ：Stage3最終shift量（取代原本的
//                                        單一#define RMSNORM_STAGE3_SHIFT_PRE）
#ifndef RMSNORM_LAYER_SCALE_TABLE_H
#define RMSNORM_LAYER_SCALE_TABLE_H

#define RMSNORM_INV_S_RES_SHIFT        5
#define RMSNORM_POST_INV_RMS_SCALED_FRAC 7   // Post-Norm中途捨入用（全域共用，POST路徑沒有跨層量級差30倍的問題，見bit-width check）

static const int16_t RMSNORM_INV_GATE_UP_SCALE[35] = {
    23324, 21906, 27354, 27031, 19745, 18683, 28778, 18472, 17481, 25664, 28354, 17510, 17732, 23824, 26341, 21411, 17246, 22329, 29622, 18058, 31877, 28123, 30600, 24560, 16756, 21339, 31018, 25107, 25117, 26585, 26239, 32544, 21431, 21319, 25485
};

// 每層自己的frac，用來interpret上面那張表的每一個值（不是共用單一frac）
static const int8_t RMSNORM_INV_GATE_UP_SCALE_FRAC[35] = {
    14, 13, 11, 10, 11, 10, 10, 9, 10, 10, 10, 10, 10, 9, 12, 9, 9, 11, 11, 9, 10, 11, 10, 11, 11, 9, 10, 10, 10, 11, 10, 10, 9, 9, 10
};

// Step A（inv_rms x inv_scale）算完後，per-layer要右移多少bit才落進
// 24-bit中繼容器
static const int8_t RMSNORM_PRE_DROP_BITS[35] = {
    24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24
};

// Stage3最終shift量（乘完gamma之後要右移多少bit），per-layer
static const int8_t RMSNORM_STAGE3_SHIFT_PRE[35] = {
    15, 14, 12, 11, 12, 11, 11, 10, 11, 11, 11, 11, 11, 10, 13, 10, 10, 12, 12, 10, 11, 12, 11, 12, 12, 10, 11, 11, 11, 12, 11, 11, 10, 10, 11
};

#endif // RMSNORM_LAYER_SCALE_TABLE_H
