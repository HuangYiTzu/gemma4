// attn_core C 模擬 testbench:讀 golden/export_attn_core_vectors.py 匯出的向量,逐位元比對。
//   K/V 先依邏輯位置以 KvCache::write_row 依序寫入 bank 陣列(sliding 以 pos mod 512 覆蓋,即環形緩衝),再逐位置跑:
//   (a) 抽樣位置:逐 tile 的狀態(m、l、O[D])與獨立參考一致(每個 head);
//   (b) 全部位置:最終 o_in code 與 golden(預填第 p 列)一致。
// 同一份向量用多組並行度 (HEAD_PAR, LANES, NB, EXP_PAR) 跑,結果必須完全相同(並行度不影響位元結果);
// 預設 (8, 8, 1, 8) 另外經過綜合用 top。
// 用法:tb_attn_core [向量資料夾,預設 vectors]
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "attn_top.h"

static std::string g_dir = "vectors";
static bool g_quick = false;   // 第二個參數 "quick":只跑綜合用 top 的預設組態(Vivado csim 為 debug 模式,很慢)

static FILE* open_or_die(const std::string& name) {
    FILE* f = fopen((g_dir + "/" + name).c_str(), "r");
    if (!f) { fprintf(stderr, "Cannot open %s/%s\n", g_dir.c_str(), name.c_str()); exit(2); }
    return f;
}

static std::vector<long long> read_all(const std::string& name) {
    FILE* f = open_or_die(name);
    std::vector<long long> v;
    long long x;
    while (fscanf(f, "%lld", &x) == 1) v.push_back(x);
    fclose(f);
    return v;
}

struct CaseInfo {
    std::string name;
    int D, cache_len, window, T, npos, nstate;
    long long m_s, c2;
};

struct Rec {   // 參考傾印的一個 tile 之後的狀態
    int j;
    long long m, l;
    std::vector<long long> O;
};

// 一組並行度的型別:Core(任意並行度)與綜合用 top 的函式指標型別(只有預設並行度有 top)
template <int D, int CL, int W, int NL, int HP, int LANES, int NB, int EXP_PAR>
struct Variant {
    typedef attn::AttnCore<D, CL, W, NL, 8, LANES, NB, EXP_PAR, HP> Core;
    typedef typename Core::Cache Cache;
    typedef typename Cache::word_t word_t;
    typedef void (*TopFn)(const attn::q_t[8][D], const word_t[NL][NB][Cache::DEPTH], const word_t[NL][NB][Cache::DEPTH], int, int,
                          attn::ms_t, attn::c2_t, attn::code_t[8][D]);
};

template <int D, int CL, int W, int NL, int HP, int LANES, int NB, int EXP_PAR>
static int run_case(const CaseInfo& ci, typename Variant<D, CL, W, NL, HP, LANES, NB, EXP_PAR>::TopFn top, const char* label) {
    typedef Variant<D, CL, W, NL, HP, LANES, NB, EXP_PAR> V;
    typedef typename V::Core Core;
    typedef typename V::Cache Cache;
    typedef typename V::word_t word_t;
    const int H = 8;
    const int layer = NL - 1;                      // 用最後一個 layer 索引,順便驗證 layer 位址
    std::vector<long long> pos = read_all(ci.name + "_pos.txt");
    std::vector<long long> qv = read_all(ci.name + "_q.txt");
    std::vector<long long> kv = read_all(ci.name + "_k.txt");
    std::vector<long long> vv = read_all(ci.name + "_v.txt");
    std::vector<long long> cv = read_all(ci.name + "_code.txt");
    std::vector<long long> sv = read_all(ci.name + "_state.txt");
    if ((int)pos.size() != ci.npos || qv.size() != (size_t)ci.npos * H * D || cv.size() != qv.size() ||
        kv.size() != (size_t)ci.T * D || vv.size() != kv.size()) {
        fprintf(stderr, "[%s] vector sizes do not match cases.txt\n", ci.name.c_str());
        return 1;
    }
    std::map<std::pair<int, int>, std::vector<Rec> > states;
    const size_t rl = 5 + D;
    if (sv.size() != (size_t)ci.nstate * rl) { fprintf(stderr, "[%s] state file size mismatch\n", ci.name.c_str()); return 1; }
    for (int i = 0; i < ci.nstate; i++) {
        const long long* r = &sv[(size_t)i * rl];
        Rec rec;
        rec.j = (int)r[2]; rec.m = r[3]; rec.l = r[4];
        rec.O.assign(r + 5, r + 5 + D);
        states[std::make_pair((int)r[0], (int)r[1])].push_back(rec);
    }

    word_t (*k_mem)[NB][Cache::DEPTH] = new word_t[NL][NB][Cache::DEPTH]();   // 全 0
    word_t (*v_mem)[NB][Cache::DEPTH] = new word_t[NL][NB][Cache::DEPTH]();
    static attn::q_t q[H][D];
    static attn::code_t code[H][D];

    long long bad_state = 0, bad_code = 0;
    int written = 0;                               // 邏輯位置 [0, written) 已寫入 cache
    for (int ip = 0; ip < ci.npos; ip++) {
        const int p = (int)pos[ip];
        for (; written <= p; written++) {          // 依序寫入;sliding 以 pos mod 512 覆蓋(環形緩衝)
            kv::elem_t ke[D], ve[D];
            for (int d = 0; d < D; d++) {
                ke[d] = (int)kv[(size_t)written * D + d];
                ve[d] = (int)vv[(size_t)written * D + d];
            }
            Cache::write_row(k_mem, v_mem, layer, written, ke, ve);
        }
        for (int h = 0; h < H; h++)
            for (int d = 0; d < D; d++) q[h][d] = (int)qv[((size_t)ip * H + h) * D + d];

        // (a) 逐 tile 狀態:以「head 組」為單位(HP 個 head 一起走完同一串 tile),比對組內每個 head
        for (int hg = 0; hg < H / HP; hg++) {
            bool any = false;
            for (int hp = 0; hp < HP; hp++) any = any || states.count(std::make_pair(p, hg * HP + hp));
            if (!any) continue;
            int j_lo, j_hi;
            Core::tile_range(p, j_lo, j_hi);
            typename Core::GroupState st;
            Core::init_state(st);
            attn::q_t q_loc[HP][Core::CHUNKS][LANES];
            Core::load_q(&q[hg * HP], q_loc);
            bool head_bad[HP];
            for (int hp = 0; hp < HP; hp++) {
                head_bad[hp] = false;
                const std::vector<Rec>& recs = states[std::make_pair(p, hg * HP + hp)];
                if ((int)recs.size() != j_hi - j_lo + 1) {
                    printf("  STATE %s p=%d h=%d: tile count %d != reference %d\n", ci.name.c_str(), p, hg * HP + hp, j_hi - j_lo + 1, (int)recs.size());
                    bad_state++;
                    head_bad[hp] = true;
                }
            }
            for (int j = j_lo; j <= j_hi; j++) {
                Core::tile_update(q_loc, k_mem, v_mem, layer, p, j, attn::ms_t((unsigned long long)ci.m_s), st);
                for (int hp = 0; hp < HP; hp++) {
                    if (head_bad[hp]) continue;
                    const int h = hg * HP + hp;
                    const Rec& r = states[std::make_pair(p, h)][j - j_lo];
                    bool ok = (r.j == j) && ((long long)st.m[hp] == r.m) && ((long long)st.l[hp] == r.l);
                    int bad_d = -1;
                    for (int d = 0; d < D && ok; d++)
                        if ((long long)st.O_at(hp, d) != r.O[d]) { ok = false; bad_d = d; }
                    if (!ok) {
                        if (bad_state < 5)
                            printf("  STATE MISMATCH %s p=%d h=%d tile=%d (ref j=%d): m=%d/%lld l=%lld/%lld O[%d]=%lld/%lld (got/ref)\n", ci.name.c_str(), p,
                                   h, j, r.j, (int)st.m[hp], r.m, (long long)st.l[hp], r.l, bad_d, bad_d >= 0 ? (long long)st.O_at(hp, bad_d) : 0LL,
                                   bad_d >= 0 ? r.O[bad_d] : 0LL);
                        bad_state++;
                        head_bad[hp] = true;
                    }
                }
            }
        }

        // (b) 最終 code
        if (top) top(q, k_mem, v_mem, layer, p, attn::ms_t((unsigned long long)ci.m_s), attn::c2_t((unsigned long long)ci.c2), code);
        else Core::run(q, k_mem, v_mem, layer, p, attn::ms_t((unsigned long long)ci.m_s), attn::c2_t((unsigned long long)ci.c2), code);
        for (int h = 0; h < H; h++)
            for (int d = 0; d < D; d++) {
                const long long exp = cv[((size_t)ip * H + h) * D + d];
                if ((long long)code[h][d] != exp) {
                    if (bad_code < 5)
                        printf("  CODE MISMATCH %s p=%d h=%d d=%d got=%d expected=%lld\n", ci.name.c_str(), p, h, d, (int)code[h][d], exp);
                    bad_code++;
                }
            }
    }
    delete[] k_mem;
    delete[] v_mem;
    printf("[%s] %-24s %s  (%d positions x %d heads x %d dims; state mismatches %lld, code mismatches %lld)\n", ci.name.c_str(), label,
           (bad_state || bad_code) ? "FAIL" : "PASS", ci.npos, H, D, bad_state, bad_code);
    return (bad_state || bad_code) ? 1 : 0;
}

int main(int argc, char** argv) {
    if (argc > 1) g_dir = argv[1];
    if (argc > 2 && std::string(argv[2]) == "quick") g_quick = true;
    FILE* f = open_or_die("cases.txt");
    char nm[128];
    CaseInfo ci;
    int fails = 0, n = 0;
    while (fscanf(f, "%127s %d %d %d %d %d %lld %lld %d", nm, &ci.D, &ci.cache_len, &ci.window, &ci.T, &ci.npos, &ci.m_s, &ci.c2, &ci.nstate) == 9) {
        ci.name = nm;
        // 變體標籤:HP = HEAD_PAR、L = LANES、NB、E = EXP_PAR
        if (ci.D == 256 && ci.cache_len == 512 && ci.window == 512) {
            fails += run_case<256, 512, 512, 12, 8, 8, 1, 8>(ci, attn_core_sliding, "top HP=8 L=8 NB=1 E=8");
            if (!g_quick) {
                fails += run_case<256, 512, 512, 12, 1, 8, 8, 8>(ci, 0, "HP=1 L=8 NB=8 E=8");
                fails += run_case<256, 512, 512, 12, 2, 8, 4, 4>(ci, 0, "HP=2 L=8 NB=4 E=4");
                fails += run_case<256, 512, 512, 12, 4, 16, 2, 4>(ci, 0, "HP=4 L=16 NB=2 E=4");
                fails += run_case<256, 512, 512, 12, 8, 32, 8, 8>(ci, 0, "HP=8 L=32 NB=8 E=8");
            }
        } else if (ci.D == 512 && ci.cache_len == 4096 && ci.window == 0) {
            fails += run_case<512, 4096, 0, 3, 8, 8, 1, 8>(ci, attn_core_global, "top HP=8 L=8 NB=1 E=8");
            if (!g_quick) {
                fails += run_case<512, 4096, 0, 3, 1, 8, 8, 8>(ci, 0, "HP=1 L=8 NB=8 E=8");
                fails += run_case<512, 4096, 0, 3, 2, 8, 4, 4>(ci, 0, "HP=2 L=8 NB=4 E=4");
                fails += run_case<512, 4096, 0, 3, 4, 16, 2, 4>(ci, 0, "HP=4 L=16 NB=2 E=4");
                fails += run_case<512, 4096, 0, 3, 8, 32, 8, 8>(ci, 0, "HP=8 L=32 NB=8 E=8");
            }
        } else {
            printf("[%s] unsupported configuration D=%d cache=%d window=%d\n", nm, ci.D, ci.cache_len, ci.window);
            fails++;
        }
        n++;
    }
    fclose(f);
    printf("%s: %d cases, %d failures\n", fails ? "FAIL" : "ALL PASS", n, fails);
    return fails ? 1 : 0;
}
