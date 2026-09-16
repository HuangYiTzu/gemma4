// embed_lut.h -- 主要 embedding (embed_tokens) 的 host 端查表模組
//
// 對應的 golden model：embed_lut_golden.py，已用真實模型逐值驗證
//   (float 版本 vs 真實輸出，最大絕對誤差 1.37e-07)
//
// 這支程式跑在 host CPU，不是 HLS kernel，不需要任何 synthesis pragma。
#ifndef EMBED_LUT_H
#define EMBED_LUT_H

#include <cstdint>
#include <cstddef>

// ==== 模型參數（跟 manifest.json / config.json 對齊）====
#define EMBED_VOCAB_SIZE   262144
#define EMBED_HIDDEN_SIZE  1536
#define EMBED_NUM_BITS     2                                       // embedding_quantized 是 INT2
#define EMBED_BPC          (EMBED_HIDDEN_SIZE / (8 / EMBED_NUM_BITS))  // 384 bytes / token

// sqrt(hidden_size)：這是模型架構本身寫死的常數（Gemma 的
// ScaledWordEmbedding 設計），不是量化 scale，不會因為換 checkpoint 而變。
// = sqrt(1536)
static const double EMBED_SQRT_HIDDEN = 39.191835884530846;

// s_res：bf16 → INT16 的 scale，由負責 RMSNorm 的同學確認、目前定案的
// 數字，公式來自他的回覆：
//   y_int16 = clamp16(round( embed_lookup_int * embedding_scale * sqrt(1536)
//                             / s_res ))
// = 2^-8
static const double EMBED_S_RES = 0.00390625;  // 2^-8

// ==== 嵌入表本體：host 端直接吃平坦陣列，不像 kernel 端的權重需要
//      HBM channel tiling（因為這裡是 CPU 隨機存取單一 token_id，
//      不需要像 FPGA 那樣為平行 port 做特殊排列）====
struct EmbedTable {
    uint8_t *quantized;   // [EMBED_VOCAB_SIZE * EMBED_BPC]，INT2 打包
    float   *scale;       // [EMBED_VOCAB_SIZE]，每個 token 一個 scale
};

// 從 bin 檔載入嵌入表到 host 記憶體。成功回傳 0，失敗回傳負值。
int embed_table_load(EmbedTable *table,
                      const char *quantized_path,
                      const char *scale_path);

void embed_table_free(EmbedTable *table);

// 查一個 token_id 的 embedding：查表 -> offset-binary 解碼 -> dequant
// -> 乘 sqrt(hidden_size) -> 量化成 int16，寫進 out[EMBED_HIDDEN_SIZE]。
//
// S_embed_out 是外部傳入的參數，不寫死在函式內——目前呼叫端應該傳入
// EMBED_S_RES（同學確認過的正式值 2^-8），保留參數化寫法是為了以後
// 如果這個 scale 又需要調整，只要改呼叫端傳的值，不用動這支函式或
// 重新編譯這個模組。
void embed_lookup_int16(const EmbedTable *table,
                         int token_id,
                         double S_embed_out,
                         int16_t *out /* 呼叫端配置好 EMBED_HIDDEN_SIZE 大小 */);

#endif // EMBED_LUT_H
