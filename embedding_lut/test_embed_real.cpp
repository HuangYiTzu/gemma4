// test_embed_real.cpp -- 用真實的 embed_quantized.bin / embed_scale.bin
// 測試 embed_lookup_int16()，拿印出來的結果跟 Python golden model
// (embed_lut_golden.py) 印出來的數字逐一比對。
//
// 對應 Python 那邊這四個 token 的正式結果 (s_res = 2^-8)：
//   token 1    -> [-270, -541, -270, ...]
//   token 2    -> [-333, -333,    0, ...]
//   token 100  -> [ 280,  280, -280, ...]
//   token 1000 -> [   0, -265,    0, ...]
#include "embed_lut.h"
#include <cstdio>

int main() {
    EmbedTable table;
    int rc = embed_table_load(&table, "embed_quantized.bin", "embed_scale.bin");
    if (rc != 0) {
        printf("embed_table_load 失敗，rc=%d，先確認 bin 檔路徑對不對\n", rc);
        return 1;
    }

    int test_token_ids[4] = {1, 2, 100, 1000};

    for (int i = 0; i < 4; i++) {
        int16_t out[EMBED_HIDDEN_SIZE];
        embed_lookup_int16(&table, test_token_ids[i], EMBED_S_RES, out);

        printf("token_id=%-6d  前3個 int16 = [%d, %d, %d]\n",
               test_token_ids[i], out[0], out[1], out[2]);
    }

    embed_table_free(&table);
    return 0;
}
