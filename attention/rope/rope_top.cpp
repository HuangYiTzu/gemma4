#include "rope_top.h"

// 暫定:介面一律用 ap_memory(陣列)埠;表也是記憶體埠(表在外面)。
// 之後若改成 hls::stream 或 AXI,只需改這個檔,rope.h 的運算不變。
// pragma 用舊式寫法(`INTERFACE ap_memory port=`),新舊版 Vitis HLS 都接受。

void rope_sliding_q(const rope_io_t x[8][256], int p, const rope_tab_t cos_tab[rope::MAX_POS][128],
                    const rope_tab_t sin_tab[rope::MAX_POS][128], rope_io_t y[8][256]) {
#pragma HLS INTERFACE ap_memory port = x
#pragma HLS INTERFACE ap_memory port = y
#pragma HLS INTERFACE ap_memory port = cos_tab
#pragma HLS INTERFACE ap_memory port = sin_tab
#pragma HLS INTERFACE ap_none port = p
    rope::Rope<256, 128, 10, 10>::apply_heads<8>(x, p, cos_tab, sin_tab, y);
}

void rope_sliding_k(const rope_io_t x[1][256], int p, const rope_tab_t cos_tab[rope::MAX_POS][128],
                    const rope_tab_t sin_tab[rope::MAX_POS][128], rope_io_t y[1][256]) {
#pragma HLS INTERFACE ap_memory port = x
#pragma HLS INTERFACE ap_memory port = y
#pragma HLS INTERFACE ap_memory port = cos_tab
#pragma HLS INTERFACE ap_memory port = sin_tab
#pragma HLS INTERFACE ap_none port = p
    rope::Rope<256, 128, 14, 14>::apply_heads<1>(x, p, cos_tab, sin_tab, y);
}

void rope_global_q(const rope_io_t x[8][512], int p, const rope_tab_t cos_tab[rope::MAX_POS][64],
                   const rope_tab_t sin_tab[rope::MAX_POS][64], rope_io_t y[8][512]) {
#pragma HLS INTERFACE ap_memory port = x
#pragma HLS INTERFACE ap_memory port = y
#pragma HLS INTERFACE ap_memory port = cos_tab
#pragma HLS INTERFACE ap_memory port = sin_tab
#pragma HLS INTERFACE ap_none port = p
    rope::Rope<512, 64, 10, 10>::apply_heads<8>(x, p, cos_tab, sin_tab, y);
}

void rope_global_k(const rope_io_t x[1][512], int p, const rope_tab_t cos_tab[rope::MAX_POS][64],
                   const rope_tab_t sin_tab[rope::MAX_POS][64], rope_io_t y[1][512]) {
#pragma HLS INTERFACE ap_memory port = x
#pragma HLS INTERFACE ap_memory port = y
#pragma HLS INTERFACE ap_memory port = cos_tab
#pragma HLS INTERFACE ap_memory port = sin_tab
#pragma HLS INTERFACE ap_none port = p
    rope::Rope<512, 64, 14, 14>::apply_heads<1>(x, p, cos_tab, sin_tab, y);
}
