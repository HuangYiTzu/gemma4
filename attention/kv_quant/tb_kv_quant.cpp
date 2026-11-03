// kv_quant C 模擬 testbench:讀 golden/export_kv_quant_vectors.py 匯出的向量,逐位元比對。
//   (a) 窮舉:15 個 producer 層 × (K、V) = 30 組,INT16 全部 65536 個輸入,與 golden quantize 逐位元一致;
//       同時核對 kvq_consts.h 的常數表與匯出檔的 (M, B) 一致。
//   (b) 端對端:隨機 INT16 列 → kv_quant top → KvCache::write_row → 讀回整列,與 golden 預期 code 逐元素一致。
// 用法:tb_kv_quant [向量資料夾,預設 vectors]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "../kv_cache/kv_cache.h"
#include "kv_quant_top.h"

static std::string g_dir = "vectors";

static FILE* open_or_die(const std::string& name) {
    FILE* f = fopen((g_dir + "/" + name).c_str(), "r");
    if (!f) { fprintf(stderr, "Cannot open %s/%s\n", g_dir.c_str(), name.c_str()); exit(2); }
    return f;
}

static std::vector<int> read_ints(const std::string& name) {
    FILE* f = open_or_die(name);
    std::vector<int> v;
    int x;
    while (fscanf(f, "%d", &x) == 1) v.push_back(x);
    fclose(f);
    return v;
}

static int test_exhaustive() {
    FILE* f = open_or_die("exh_cases.txt");
    char name[64], type[16], kind[4];
    int idx;
    unsigned long long M, B;
    int groups = 0, fails = 0;
    long long total_mism = 0;
    while (fscanf(f, "%63s %15s %d %3s %llu %llu", name, type, &idx, kind, &M, &B) == 6) {
        const bool sliding = std::strcmp(type, "sliding") == 0, isK = kind[0] == 'K';
        unsigned int hM = sliding ? (isK ? KVQ_SLIDING_K_M[idx] : KVQ_SLIDING_V_M[idx]) : (isK ? KVQ_GLOBAL_K_M[idx] : KVQ_GLOBAL_V_M[idx]);
        unsigned int hB = sliding ? (isK ? KVQ_SLIDING_K_B[idx] : KVQ_SLIDING_V_B[idx]) : (isK ? KVQ_GLOBAL_K_B[idx] : KVQ_GLOBAL_V_B[idx]);
        if (hM != M || hB != B) {
            printf("  constant table mismatch %s: header M=%u B=%u, exported M=%llu B=%llu\n", name, hM, hB, M, B);
            fails++;
        }
        std::vector<int> exp = read_ints(std::string("exh_") + name + ".txt");
        if (exp.size() != 65536) { fprintf(stderr, "[%s] expected file size %u != 65536\n", name, (unsigned)exp.size()); return 1; }
        long long mism = 0;
        for (int x = -32768; x <= 32767; x++) {
            const int got = (int)kvq::KvQuant<256>::quant_one(ap_int<16>(x), kvq::KvQuant<256>::m_t(hM), kvq::KvQuant<256>::b_t(hB));
            if (got != exp[x + 32768]) {
                if (mism < 3) printf("  MISMATCH %s x=%d got=%d expected=%d\n", name, x, got, exp[x + 32768]);
                mism++;
            }
        }
        total_mism += mism;
        if (mism) fails++;
        groups++;
    }
    fclose(f);
    printf("[exhaustive] %s  (%d groups x 65536 inputs, %lld mismatches, %d failures)\n", fails ? "FAIL" : "PASS", groups, total_mism, fails);
    return fails ? 1 : 0;
}

template <int D, int CL, int NL, int RING, int LANES, int NB, class TopFn>
static int e2e_case(const std::string& name, int layer, int n, TopFn top, const char* label) {
    typedef kv::KvCache<D, CL, NL, RING, LANES, NB> C;
    std::vector<int> kin = read_ints("e2e_" + name + "_kin.txt"), vin = read_ints("e2e_" + name + "_vin.txt");
    std::vector<int> kexp = read_ints("e2e_" + name + "_kexp.txt"), vexp = read_ints("e2e_" + name + "_vexp.txt");
    if (kin.size() != (size_t)n * D || vin.size() != kin.size() || kexp.size() != kin.size() || vexp.size() != kin.size()) {
        fprintf(stderr, "[e2e %s] vector size mismatch\n", name.c_str());
        return 1;
    }
    typename C::word_t (*k_mem)[NB][C::DEPTH] = new typename C::word_t[NL][NB][C::DEPTH]();
    typename C::word_t (*v_mem)[NB][C::DEPTH] = new typename C::word_t[NL][NB][C::DEPTH]();
    long long mq = 0, mr = 0;
    ap_int<16> ki[D], vi[D];
    kv::elem_t ko[D], vo[D];
    for (int r = 0; r < n; r++) {
        for (int d = 0; d < D; d++) { ki[d] = kin[(size_t)r * D + d]; vi[d] = vin[(size_t)r * D + d]; }
        top(ki, vi, layer, ko, vo);                                       // 量化(經過綜合用 top)
        for (int d = 0; d < D; d++) {
            if ((int)ko[d] != kexp[(size_t)r * D + d] || (int)vo[d] != vexp[(size_t)r * D + d]) {
                if (mq < 3) printf("  QUANT MISMATCH %s row=%d d=%d K got=%d exp=%d, V got=%d exp=%d\n", name.c_str(), r, d, (int)ko[d], kexp[(size_t)r * D + d], (int)vo[d], vexp[(size_t)r * D + d]);
                mq++;
            }
        }
        C::write_row(k_mem, v_mem, layer, r, ko, vo);                     // 寫入 KV cache,位置 = 列號
    }
    for (int r = 0; r < n; r++) {                                         // 讀回整列(平行讀取),與 golden 預期 code 比對
        const int g = r / NB, b = r % NB;
        typename C::word_t kw[NB], vw[NB];
        for (int c = 0; c < C::CHUNKS; c++) {
            C::read_group(k_mem, layer, g, c, kw);
            C::read_group(v_mem, layer, g, c, vw);
            for (int l = 0; l < LANES; l++) {
                const int d = c * LANES + l;
                if ((int)C::unpack(kw[b], l) != kexp[(size_t)r * D + d] || (int)C::unpack(vw[b], l) != vexp[(size_t)r * D + d]) {
                    if (mr < 3) printf("  READBACK MISMATCH %s row=%d d=%d\n", name.c_str(), r, d);
                    mr++;
                }
            }
        }
    }
    delete[] k_mem;
    delete[] v_mem;
    printf("[e2e %s] %-12s %s  (%d rows; quantization mismatches %lld, readback mismatches %lld)\n", name.c_str(), label, (mq || mr) ? "FAIL" : "PASS", n, mq, mr);
    return (mq || mr) ? 1 : 0;
}

int main(int argc, char** argv) {
    if (argc > 1) g_dir = argv[1];
    int fails = test_exhaustive();
    FILE* f = open_or_die("e2e_cases.txt");
    char nm[32];
    int D, cl, nl, ring, layer, n;
    while (fscanf(f, "%31s %d %d %d %d %d %d", nm, &D, &cl, &nl, &ring, &layer, &n) == 7) {
        if (D == 256 && cl == 512 && nl == 12 && ring == 1) {
            fails += e2e_case<256, 512, 12, 1, 8, 8>(nm, layer, n, kv_quant_sliding, "cache 8x8");
            fails += e2e_case<256, 512, 12, 1, 8, 1>(nm, layer, n, kv_quant_sliding, "cache 8x1");
        } else if (D == 512 && cl == 4096 && nl == 3 && ring == 0) {
            fails += e2e_case<512, 4096, 3, 0, 8, 8>(nm, layer, n, kv_quant_global, "cache 8x8");
            fails += e2e_case<512, 4096, 3, 0, 8, 1>(nm, layer, n, kv_quant_global, "cache 8x1");
        } else {
            printf("[e2e %s] unsupported configuration\n", nm);
            fails++;
        }
    }
    fclose(f);
    printf("%s: %d failures\n", fails ? "FAIL" : "ALL PASS", fails);
    return fails ? 1 : 0;
}
