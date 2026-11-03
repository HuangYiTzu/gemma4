// exp2 與 1/ℓ 查表的 C 模擬 testbench:讀 golden/export_softmax_aux_vectors.py 匯出的向量,逐位元比對。
// 用法:tb_aux [向量資料夾,預設 vectors]
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "aux_top.h"

static std::string g_dir = "vectors";

static FILE* open_or_die(const std::string& name) {
    FILE* f = fopen((g_dir + "/" + name).c_str(), "r");
    if (!f) { fprintf(stderr, "Cannot open %s/%s\n", g_dir.c_str(), name.c_str()); exit(2); }
    return f;
}

static int test_exp2() {
    FILE* f = open_or_die("exp2_vec.txt");
    std::vector<int> d, y;
    int a, b;
    while (fscanf(f, "%d %d", &a, &b) == 2) { d.push_back(a); y.push_back(b); }
    fclose(f);
    long mism = 0;
    static ap_int<16> din[AUX_N];
    static ap_uint<17> dout[AUX_N];
    for (size_t base = 0; base < d.size(); base += AUX_N) {
        int n = (int)((d.size() - base < (size_t)AUX_N) ? (d.size() - base) : AUX_N);
        for (int i = 0; i < AUX_N; i++) din[i] = (i < n) ? d[base + i] : 0;   // 補齊的元素 d = 0 不比對
        exp2_top(din, dout);
        for (int i = 0; i < n; i++) {
            int got = (int)dout[i];
            if (got != y[base + i]) {
                if (mism < 5) printf("  MISMATCH exp2 d=%d got=%d expected=%d\n", d[base + i], got, y[base + i]);
                mism++;
            }
        }
    }
    printf("[exp2]  %s  (%u values of d, exhaustive over -32767..0, %ld mismatches)\n", mism ? "FAIL" : "PASS", (unsigned)d.size(), mism);
    return mism ? 1 : 0;
}

static int test_recip() {
    FILE* f = open_or_die("recip_vec.txt");
    std::vector<long> l;
    std::vector<int> R, w;
    long a;
    int b, c;
    while (fscanf(f, "%ld %d %d", &a, &b, &c) == 3) { l.push_back(a); R.push_back(b); w.push_back(c); }
    fclose(f);
    long mism = 0;
    static ap_uint<25> lin[AUX_N];
    static ap_uint<13> rout[AUX_N];
    static ap_int<6> wout[AUX_N];
    for (size_t base = 0; base < l.size(); base += AUX_N) {
        int n = (int)((l.size() - base < (size_t)AUX_N) ? (l.size() - base) : AUX_N);
        for (int i = 0; i < AUX_N; i++) lin[i] = (i < n) ? l[base + i] : 1;
        recip_top(lin, rout, wout);
        for (int i = 0; i < n; i++) {
            int gr = (int)rout[i], gw = (int)wout[i];
            if (gr != R[base + i] || gw != w[base + i]) {
                if (mism < 5)
                    printf("  MISMATCH recip l=%ld got R=%d w=%d expected R=%d w=%d\n", l[base + i], gr, gw, R[base + i], w[base + i]);
                mism++;
            }
        }
    }
    printf("[recip] %s  (%u values of l, %ld mismatches)\n", mism ? "FAIL" : "PASS", (unsigned)l.size(), mism);
    return mism ? 1 : 0;
}

int main(int argc, char** argv) {
    if (argc > 1) g_dir = argv[1];
    int fails = test_exp2() + test_recip();
    printf("%s: %d failures\n", fails ? "FAIL" : "ALL PASS", fails);
    return fails ? 1 : 0;
}
