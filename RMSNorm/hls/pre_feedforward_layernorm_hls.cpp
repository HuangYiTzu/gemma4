#include "rmsnorm_kernel.h"

// 2026-10-14修正：原本pre_/post_兩個top function本體共用一份
// rmsnorm_hls.cpp，但kernel專案裡每個Hardware Function各自獨立
// v++ -c --kernel編譯，IDE預設把src/底下所有檔案都餵給每個Hardware
// Function編譯，於是這份.cpp被編進pre_...跟post_...兩個.xo，
// --link階段兩邊都看到兩個符號各自定義一次，爆multiple definition。
// 拆成一個top function一個.cpp檔，各自只給對應的Hardware Function
// Source Files勾選，才不會重複。
//
// extern "C"：跟rmsnorm_kernel.h裡的宣告linkage對齊（v++/XRT用符號
// 名稱找kernel，避免C++ name mangling），跟MLP.cpp的top function
// 慣例一致。
extern "C" {

void pre_feedforward_layernorm_hls(
    const hid16_t residual_in[1536],
    int           layer_idx,
    act8_t        y_out[1536])
{
#pragma HLS INTERFACE m_axi port=residual_in offset=slave bundle=gmemS depth=1536
#pragma HLS INTERFACE m_axi port=y_out       offset=slave bundle=gmemS depth=1536
#pragma HLS INLINE off
    rmsnorm_core<1536, /*IS_POST_NORM=*/false, /*HAS_GAMMA=*/true,
                 hid16_t, act8_t>(
        residual_in, /*step0_m=*/nullptr, layer_idx,
        RMSNORM_GAMMA_PRE, y_out);
}

}  // extern "C"
