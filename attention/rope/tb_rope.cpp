// RoPE C 模擬 testbench:讀 golden/export_rope_vectors.py 匯出的向量,逐位元比對。
// 用法:tb_rope [向量資料夾,預設 vectors]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "rope.h"
#include "rope_top.h"

using rope::MAX_POS;

static std::string g_dir = "vectors";

static std::vector<int> read_ints(const std::string& path) {
    FILE* f = fopen(path.c_str(), "r");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path.c_str()); exit(2); }
    std::vector<int> v;
    int x;
    while (fscanf(f, "%d", &x) == 1) v.push_back(x);
    fclose(f);
    return v;
}

// 表:[MAX_POS][N] 靜態儲存
static ap_int<14> cos_sl[MAX_POS][128], sin_sl[MAX_POS][128];
static ap_int<14> cos_gl[MAX_POS][64], sin_gl[MAX_POS][64];

template <int N>
static void load_table(const char* nm, const char* kind, ap_int<14> (*tab)[N]) {
    std::vector<int> v = read_ints(g_dir + "/rope_" + nm + "_" + kind + ".txt");
    if (v.size() != (size_t)MAX_POS * N) { fprintf(stderr, "Table size error %s_%s\n", nm, kind); exit(2); }
    for (int p = 0; p < MAX_POS; p++)
        for (int j = 0; j < N; j++) tab[p][j] = v[(size_t)p * N + j];
}

template <int D, int N_ROT, int IN_FRAC, int OUT_FRAC, int H>
static int run_case(const std::string& name, int T, const ap_int<14> (*ct)[N_ROT], const ap_int<14> (*st)[N_ROT],
                    void (*top)(const ap_int<16>[H][D], int, const ap_int<14>[MAX_POS][N_ROT], const ap_int<14>[MAX_POS][N_ROT], ap_int<16>[H][D])) {
    typedef rope::Rope<D, N_ROT, IN_FRAC, OUT_FRAC> R;
    std::vector<int> x = read_ints(g_dir + "/" + name + "_x.txt");
    std::vector<int> y = read_ints(g_dir + "/" + name + "_y.txt");
    std::vector<int> pos = read_ints(g_dir + "/" + name + "_pos.txt");
    if (x.size() != (size_t)T * H * D || y.size() != x.size() || pos.size() != (size_t)T) {
        fprintf(stderr, "[%s] vector sizes do not match cases.txt\n", name.c_str());
        return 1;
    }
    long mism = 0;
    static typename R::in_t xin[H][D];
    static typename R::out_t yout[H][D];
    for (int t = 0; t < T; t++) {
        for (int h = 0; h < H; h++)
            for (int d = 0; d < D; d++) xin[h][d] = x[((size_t)t * H + h) * D + d];
        if (top) top(xin, pos[t], ct, st, yout);                 // 綜合用 top(預設組態)
        else R::template apply_heads<H>(xin, pos[t], ct, st, yout);
        for (int h = 0; h < H; h++)
            for (int d = 0; d < D; d++) {
                int got = (int)yout[h][d], exp = y[((size_t)t * H + h) * D + d];
                if (got != exp) {
                    if (mism < 5)
                        printf("  MISMATCH %s t=%d h=%d d=%d pos=%d x=%d got=%d expected=%d\n", name.c_str(), t, h, d,
                               pos[t], x[((size_t)t * H + h) * D + d], got, exp);
                    mism++;
                }
            }
    }
    printf("[%s] %s%s  (%d tokens x %d heads x %d dims, %ld mismatches)\n", name.c_str(), mism ? "FAIL" : "PASS", top ? " via top" : "", T, H, D, mism);
    return mism ? 1 : 0;
}

#define RUN(D, NR, IN, OUT, H, CT, ST, TOP)                                                             \
    if (n == #D "_" #NR "_" #IN "_" #OUT "_" #H) { fails += run_case<D, NR, IN, OUT, H>(name, T, CT, ST, TOP); handled = true; }

int main(int argc, char** argv) {
    if (argc > 1) g_dir = argv[1];
    load_table<128>("cos", "sliding", cos_sl);
    load_table<128>("sin", "sliding", sin_sl);
    load_table<64>("cos", "global", cos_gl);
    load_table<64>("sin", "global", sin_gl);

    FILE* f = fopen((g_dir + "/cases.txt").c_str(), "r");
    if (!f) { fprintf(stderr, "cases.txt not found\n"); return 2; }
    char nm[128], tk[32];
    int D, NR, IN, OUT, T, H, fails = 0, ncase = 0;
    while (fscanf(f, "%127s %d %d %d %d %d %d %31s", nm, &D, &NR, &IN, &OUT, &T, &H, tk) == 8) {
        std::string name = nm;
        std::string n = std::to_string(D) + "_" + std::to_string(NR) + "_" + std::to_string(IN) + "_" +
                        std::to_string(OUT) + "_" + std::to_string(H);
        bool handled = false;
        RUN(256, 128, 10, 10, 8, cos_sl, sin_sl, rope_sliding_q)
        RUN(256, 128, 14, 14, 1, cos_sl, sin_sl, rope_sliding_k)
        RUN(256, 128, 12, 10, 8, cos_sl, sin_sl, 0)
        RUN(512, 64, 10, 10, 8, cos_gl, sin_gl, rope_global_q)
        RUN(512, 64, 14, 14, 1, cos_gl, sin_gl, rope_global_k)
        RUN(512, 64, 12, 10, 8, cos_gl, sin_gl, 0)
        if (!handled) { printf("[%s] unsupported configuration %s (add a RUN line in tb_rope.cpp)\n", nm, n.c_str()); fails++; }
        ncase++;
    }
    fclose(f);
    printf("%s: %d cases, %d failures\n", fails ? "FAIL" : "ALL PASS", ncase, fails);
    return fails ? 1 : 0;
}
