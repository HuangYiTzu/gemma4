// KV cache C 模擬 testbench:讀 golden/export_kv_cache_vectors.py 匯出的向量(獨立的環形緩衝模型),逐元素比對。
// 每組案例用多組 (LANES, NB) 跑:預設 (8, 8) 同時經過綜合用 top;另有 (8,1)、(16,4)、(32,8) 驗證位元打包與 bank 對應。
// 讀取以 (layer, pos) 讀整列:以平行讀取(位置群組 g = pos / NB,取 bank pos % NB)逐 chunk 組出整列,再與預期比對。
// 用法:tb_kv_cache [向量資料夾,預設 vectors]
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "kv_cache_top.h"

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

struct CaseInfo {
    std::string name;
    int D, cache_len, nlayers, ring, nops, nwrites, nreads;
};

// 實作 1:直接用 template(任意 LANES、NB)
template <int D, int CL, int NL, int RING, int LANES, int NB>
struct ImplTemplate {
    typedef kv::KvCache<D, CL, NL, RING, LANES, NB> C;
    typename C::word_t (*k_mem)[NB][C::DEPTH];
    typename C::word_t (*v_mem)[NB][C::DEPTH];
    ImplTemplate() {
        k_mem = new typename C::word_t[NL][NB][C::DEPTH]();      // 值初始化:全 0(從未寫入的位址讀到 0)
        v_mem = new typename C::word_t[NL][NB][C::DEPTH]();
    }
    ~ImplTemplate() { delete[] k_mem; delete[] v_mem; }
    void write(int layer, int pos, const int* k, const int* v) {
        kv::elem_t ke[D], ve[D];
        for (int d = 0; d < D; d++) { ke[d] = k[d]; ve[d] = v[d]; }
        C::write_row(k_mem, v_mem, layer, pos, ke, ve);
    }
    void read(int layer, int pos, int* k, int* v) {
        const int g = pos / NB, b = pos % NB;
        typename C::word_t ko[NB], vo[NB];
        for (int c = 0; c < C::CHUNKS; c++) {
            C::read_group(k_mem, layer, g, c, ko);
            C::read_group(v_mem, layer, g, c, vo);
            for (int l = 0; l < LANES; l++) {
                k[c * LANES + l] = (int)C::unpack(ko[b], l);
                v[c * LANES + l] = (int)C::unpack(vo[b], l);
            }
        }
    }
};

// 實作 2:經過綜合用 top(預設 8 × 8)
template <int D, int NB, int LANES>
struct ImplTop {
    typedef void (*TopFn)(ap_uint<1>, int, int, const kv::elem_t[D], const kv::elem_t[D], int, int, kv_word_t[NB], kv_word_t[NB]);
    TopFn fn;
    explicit ImplTop(TopFn f) : fn(f) {}
    void write(int layer, int pos, const int* k, const int* v) {
        kv::elem_t ke[D], ve[D];
        kv_word_t dummy_k[NB], dummy_v[NB];
        for (int d = 0; d < D; d++) { ke[d] = k[d]; ve[d] = v[d]; }
        fn(1, layer, pos, ke, ve, 0, 0, dummy_k, dummy_v);
    }
    void read(int layer, int pos, int* k, int* v) {
        kv::elem_t ke[D], ve[D];
        kv_word_t ko[NB], vo[NB];
        const int g = pos / NB, b = pos % NB;
        for (int d = 0; d < D; d++) { ke[d] = 0; ve[d] = 0; }
        for (int c = 0; c < D / LANES; c++) {
            fn(0, layer, 0, ke, ve, g, c, ko, vo);
            for (int l = 0; l < LANES; l++) {
                kv::elem_t a, bb;
                ap_uint<8> uk = ko[b].range(8 * l + 7, 8 * l), uv = vo[b].range(8 * l + 7, 8 * l);
                a.range(7, 0) = uk;
                bb.range(7, 0) = uv;
                k[c * LANES + l] = (int)a;
                v[c * LANES + l] = (int)bb;
            }
        }
    }
};

template <class Impl>
static int run_ops(const CaseInfo& ci, Impl& impl, const char* label) {
    const int D = ci.D;
    std::vector<int> wk = read_ints(ci.name + "_wk.txt"), wv = read_ints(ci.name + "_wv.txt");
    std::vector<int> rk = read_ints(ci.name + "_rk.txt"), rv = read_ints(ci.name + "_rv.txt");
    if (wk.size() != (size_t)ci.nwrites * D || wv.size() != wk.size() || rk.size() != (size_t)ci.nreads * D || rv.size() != rk.size()) {
        fprintf(stderr, "[%s] vector sizes do not match cases.txt\n", ci.name.c_str());
        return 1;
    }
    FILE* f = open_or_die(ci.name + "_ops.txt");
    char op[8];
    int layer, pos, iw = 0, ir = 0;
    long mism = 0;
    std::vector<int> k(D), v(D);
    while (fscanf(f, "%7s %d %d", op, &layer, &pos) == 3) {
        if (op[0] == 'W') {
            impl.write(layer, pos, &wk[(size_t)iw * D], &wv[(size_t)iw * D]);
            iw++;
        } else {
            impl.read(layer, pos, &k[0], &v[0]);
            for (int d = 0; d < D; d++) {
                const int ek = rk[(size_t)ir * D + d], ev = rv[(size_t)ir * D + d];
                if (k[d] != ek || v[d] != ev) {
                    if (mism < 5)
                        printf("  MISMATCH %s (%s) read layer=%d pos=%d d=%d: K got=%d exp=%d, V got=%d exp=%d\n", ci.name.c_str(), label, layer, pos, d, k[d], ek, v[d], ev);
                    mism++;
                }
            }
            ir++;
        }
    }
    fclose(f);
    printf("[%s] %-14s %s  (%d rows written, %d rows read, %ld element mismatches)\n", ci.name.c_str(), label, mism ? "FAIL" : "PASS", iw, ir, mism);
    return mism ? 1 : 0;
}

#define VARIANT(D_, CL_, NL_, RING_, LANES_, NB_)                                                      \
    {                                                                                                  \
        ImplTemplate<D_, CL_, NL_, RING_, LANES_, NB_> impl;                                           \
        fails += run_ops(ci, impl, "LANES=" #LANES_ " NB=" #NB_);                                      \
    }

int main(int argc, char** argv) {
    if (argc > 1) g_dir = argv[1];
    FILE* f = open_or_die("cases.txt");
    char nm[64];
    CaseInfo ci;
    int fails = 0, n = 0;
    while (fscanf(f, "%63s %d %d %d %d %d %d %d", nm, &ci.D, &ci.cache_len, &ci.nlayers, &ci.ring, &ci.nops, &ci.nwrites, &ci.nreads) == 8) {
        ci.name = nm;
        if (ci.D == 256 && ci.cache_len == 512 && ci.nlayers == 12 && ci.ring == 1) {
            ImplTop<256, 8, 8> top(kv_cache_sliding);
            fails += run_ops(ci, top, "top (8x8)");
            VARIANT(256, 512, 12, 1, 8, 1)
            VARIANT(256, 512, 12, 1, 16, 4)
            VARIANT(256, 512, 12, 1, 32, 8)
        } else if (ci.D == 512 && ci.cache_len == 4096 && ci.nlayers == 3 && ci.ring == 0) {
            ImplTop<512, 8, 8> top(kv_cache_global);
            fails += run_ops(ci, top, "top (8x8)");
            VARIANT(512, 4096, 3, 0, 8, 1)
            VARIANT(512, 4096, 3, 0, 16, 4)
            VARIANT(512, 4096, 3, 0, 32, 8)
        } else {
            printf("[%s] unsupported configuration\n", nm);
            fails++;
        }
        n++;
    }
    fclose(f);
    printf("%s: %d cases, %d failures\n", fails ? "FAIL" : "ALL PASS", n, fails);
    return fails ? 1 : 0;
}
