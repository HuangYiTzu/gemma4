// embed_lut.cpp -- 實作，對應 embed_lut_golden.py 裡的
//   decode_offset_binary() + embed_lookup_float() + embed_lookup_int16()
#include "embed_lut.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>

// ============================================================
// 載入 / 釋放
// ============================================================
int embed_table_load(EmbedTable *table,
                      const char *quantized_path,
                      const char *scale_path) {
    // ---- embedding_quantized ----
    FILE *fq = fopen(quantized_path, "rb");
    if (!fq) {
        fprintf(stderr, "HOST-Error: cannot open %s\n", quantized_path);
        return -1;
    }
    size_t n_bytes_q = (size_t)EMBED_VOCAB_SIZE * EMBED_BPC;
    table->quantized = (uint8_t *)malloc(n_bytes_q);
    if (!table->quantized) {
        fprintf(stderr, "HOST-Error: out of memory for embedding_quantized\n");
        fclose(fq);
        return -2;
    }
    size_t read_q = fread(table->quantized, 1, n_bytes_q, fq);
    fclose(fq);
    if (read_q != n_bytes_q) {
        fprintf(stderr, "HOST-Error: %s size mismatch (got %zu, expected %zu)\n",
                quantized_path, read_q, n_bytes_q);
        free(table->quantized);
        table->quantized = nullptr;
        return -3;
    }

    // ---- embedding_scale ----
    FILE *fs = fopen(scale_path, "rb");
    if (!fs) {
        fprintf(stderr, "HOST-Error: cannot open %s\n", scale_path);
        free(table->quantized);
        table->quantized = nullptr;
        return -4;
    }
    size_t n_bytes_s = (size_t)EMBED_VOCAB_SIZE * sizeof(float);
    table->scale = (float *)malloc(n_bytes_s);
    if (!table->scale) {
        fprintf(stderr, "HOST-Error: out of memory for embedding_scale\n");
        fclose(fs);
        free(table->quantized);
        table->quantized = nullptr;
        return -5;
    }
    size_t read_s = fread(table->scale, 1, n_bytes_s, fs);
    fclose(fs);
    if (read_s != n_bytes_s) {
        fprintf(stderr, "HOST-Error: %s size mismatch (got %zu, expected %zu)\n",
                scale_path, read_s, n_bytes_s);
        free(table->quantized);
        free(table->scale);
        table->quantized = nullptr;
        table->scale = nullptr;
        return -6;
    }

    return 0;
}

void embed_table_free(EmbedTable *table) {
    free(table->quantized);
    free(table->scale);
    table->quantized = nullptr;
    table->scale = nullptr;
}

// ============================================================
// 位元解碼 / 捨入 / 飽和 -- 跟 Python golden model 的邏輯逐行對應
// ============================================================

// offset-binary 解碼：value = code - 2^(num_bits-1)。
// 對應 embed_lut_golden.py 的 decode_offset_binary()，已用真實模型驗證過
// （不是 decode_already_flipped() 那種 two's-complement 判斷，兩者不能
//  混用——這是 MLP 那邊踩過的坑，這裡直接用驗證過的版本，不要重蹈覆轍）。
//
// lane 對應到 byte 裡的第幾組 2-bit：lane 0 是最低位那組。
static inline int8_t decode_2bit_code(uint8_t packed_byte, int lane) {
    const int offset = 1 << (EMBED_NUM_BITS - 1);   // = 2
    const int mask = (1 << EMBED_NUM_BITS) - 1;       // = 3
    int raw = (packed_byte >> (lane * EMBED_NUM_BITS)) & mask;
    return (int8_t)(raw - offset);
}

// round-half-to-even：用 nearbyint() 而不是 llround()，因為 llround() 是
// round-half-away-from-zero，跟 Python 的 np.round()／MLP 那邊 rne() 用的
// 捨入規則不一樣。nearbyint() 沿用當前浮點捨入模式，預設(FE_TONEAREST)
// 就是 round-half-to-even，才跟 golden model 對齊。
static inline int16_t clamp16_round(double v) {
    double r = nearbyint(v);
    if (r > 32767.0) return 32767;
    if (r < -32768.0) return -32768;
    return (int16_t)r;
}

// ============================================================
// 主要函式
// ============================================================
void embed_lookup_int16(const EmbedTable *table,
                         int token_id,
                         double S_embed_out,
                         int16_t *out) {
    const uint8_t *row = table->quantized + (size_t)token_id * EMBED_BPC;
    const double row_scale = (double)table->scale[token_id];
    // 兩個乘數先合併好，每個維度少算一次乘法
    const double combined_scale = row_scale * EMBED_SQRT_HIDDEN;

    const int factor = 8 / EMBED_NUM_BITS;   // = 4，每個 byte 打包幾個值

    for (int byte_idx = 0; byte_idx < EMBED_BPC; byte_idx++) {
        uint8_t packed = row[byte_idx];
        for (int lane = 0; lane < factor; lane++) {
            int d = byte_idx * factor + lane;          // 對應 hidden_size 的第幾維
            int8_t code = decode_2bit_code(packed, lane);
            double real_val = (double)code * combined_scale;  // 查表 -> dequant -> 乘 sqrt(hidden_size)
            out[d] = clamp16_round(real_val / S_embed_out);   // 量化成 int16
        }
    }
}
