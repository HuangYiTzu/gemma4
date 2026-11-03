// softmax/PV tile 引擎 + 收尾(decode,Br = 1;spec §5、§6;golden:attention_golden.py 的 attention_fa2_int,參數 HW_D1)。位元精確。
//   每個 head:m = −2^14、l = 0、O[D] = 0;對 tile j(絕對位置 BC·j .. BC·j+BC−1)由小到大:
//     S_acc[c] = Σ_d Q[d]·K[addr(pos)][d];S = clip(round_half_up(S_acc·M_S / 2^24), −2^14, 2^14−1)
//     valid    = (pos ≤ p) 且 (sliding:p − pos < WINDOW);m_new = max(m, max_valid S)
//     P̃[c]    = valid ? min(round_half_up(exp2(S − m_new) / 2^4), 4096) : 0       遮罩處由邏輯強制為 0
//     α        = exp2(m − m_new)                                                 UQ1.16
//     l = round_half_up(α·l / 2^16) + Σ P̃;  O[d] = round_half_up(α·O[d] / 2^16) + Σ_c P̃[c]·V[addr(pos)][d]
//   收尾:(R, w) = Recip(l);y = round_half_up(O·R / 2^(14 + w));code = clip(round_half_up(y·C2 / 2^22), −127, 127)
//   M_S = round(k_scale·2^(8−10)·2^24)、C2 = round(v_scale/s_o_in·2^12),每層一個常數(執行期參數)。
//   位寬:最壞情況、不加飽和(使用者決定):S_acc 32-bit、l 25-bit 無號、O 32-bit 有號;乘法中間值用夠寬的 ap_int(見各處)。
//   所有右移為算術右移,round-half-up = 加 2^(s−1) 再右移(明確寫出)。C++98 相容(Vivado HLS 2019.2)。
//
//   K/V 從 KV cache(hls/kv_cache)的 bank 陣列平行讀取:一個 word = LANES 個 int8,NB 個 bank 一次讀 NB 個位置。
//   一個 tile 的 64 個位置分成 64/NB 個群組(位置 BC·j + NB·gi + b,b = bank),QK^T 與 PV 各以 (群組, chunk) 攤平成單一
//   pipelined 迴圈,每週期處理 NB·LANES 個元素。
//   HEAD_PAR(HP):Gemma 是 MQA,H 個 Q head 共用同一組 K/V。同一次 K/V 讀取同時餵給 HP 個 head 的乘法器:
//   乘加 / 週期 = HP·NB·LANES,cache 讀取 / 週期仍是 NB·LANES。各 head 的整數運算序列互相獨立,結果與 HP = 1 完全相同。
//
//   暫定(spec §8 ⛔):HEAD_PAR、LANES、NB、EXP_PAR(每個 head 每週期 exp2 個數);M_S、C2 的埠寬(20、16 bit)是暫定值
//   (實測 35 層 M_S ≤ 15 bit、C2 ≤ 14 bit)。
#pragma once
#include <ap_int.h>
#include <cassert>

#include "../kv_cache/kv_cache.h"
#include "../softmax_aux/exp2.h"
#include "../softmax_aux/recip.h"

namespace attn {

const int BC = 64;                  // tile 大小(✅ spec §6;Bc 會影響結果,必須與 golden 一致)
const int S_FRAC_MIN = -(1 << 14);  // S 的最小值,m 的初值(不是 −inf)
const int S_FRAC_MAX = (1 << 14) - 1;

typedef ap_int<16> q_t;       // RoPE 之後的 Q:INT16,小數 10 位
typedef ap_uint<20> ms_t;     // M_S(暫定位寬)
typedef ap_uint<16> c2_t;     // C2(暫定位寬)
typedef ap_int<8> code_t;     // o_in:int8,飽和 ±127

// D:HEAD_DIM(256/512);CACHE_LEN:512(sliding)或 4096(global);WINDOW:512(sliding)或 0(global,純 causal);
// NLAYERS:cache 內同類型 producer 層數(sliding 12、global 3);layer 為執行期索引;H % HEAD_PAR == 0
template <int D, int CACHE_LEN, int WINDOW, int NLAYERS, int H = 8, int LANES = 8, int NB = 8, int EXP_PAR = 8, int HEAD_PAR = 1>
struct AttnCore {
    typedef kv::KvCache<D, CACHE_LEN, NLAYERS, (WINDOW > 0) ? 1 : 0, LANES, NB> Cache;
    typedef typename Cache::word_t word_t;
    static const int CHUNKS = Cache::CHUNKS;     // 一列有幾個 word
    static const int DEPTH = Cache::DEPTH;
    static const int GROUPS = BC / NB;           // 一個 tile 有幾個位置群組
    static const int HP = HEAD_PAR;
    AUX_STATIC_ASSERT(WINDOW == 0 || CACHE_LEN == WINDOW, "attn_core: sliding 的環形緩衝長度必須等於 window");
    AUX_STATIC_ASSERT(BC % NB == 0 && BC % EXP_PAR == 0, "attn_core: NB、EXP_PAR 必須整除 BC = 64");
    AUX_STATIC_ASSERT(CHUNKS >= 4, "attn_core: D / LANES 至少要 4(pv 累加的讀寫相隔 CHUNKS 次迭代,需大於記憶體讀寫延遲)");
    AUX_STATIC_ASSERT(HEAD_PAR >= 1 && H % HEAD_PAR == 0, "attn_core: H 必須能被 HEAD_PAR 整除");

    typedef aux::Exp2<> exp2_t;
    typedef aux::Recip<> recip_t;

    // HEAD_PAR 個 head 的狀態(同一組一起走完同一串 tile)
    struct GroupState {
        ap_int<16> m[HEAD_PAR];                 // running max(S 的格式,小數 8 位)
        ap_uint<25> l[HEAD_PAR];                // ℓ:小數 12 位
        ap_int<32> O[HEAD_PAR][CHUNKS][LANES];  // O:小數 12 位;元素 d 在 [d / LANES][d % LANES](與 pv 相同的排列)
        ap_int<32> &O_at(int hp, int d) { return O[hp][d / LANES][d % LANES]; }
        const ap_int<32> &O_at(int hp, int d) const { return O[hp][d / LANES][d % LANES]; }
    };

    static void init_state(GroupState &st) {
        for (int hp = 0; hp < HEAD_PAR; hp++) {
#pragma HLS UNROLL
            st.m[hp] = S_FRAC_MIN;
            st.l[hp] = 0;
        }
    init_o:
        for (int c = 0; c < CHUNKS; c++) {
#pragma HLS PIPELINE II = 1   // 暫定:不加 pipeline 時 init_state 每個 head 要 641 個週期(綜合報告)
            for (int hp = 0; hp < HEAD_PAR; hp++)
                for (int l = 0; l < LANES; l++) st.O[hp][c][l] = 0;
        }
    }

    // 把 HEAD_PAR 個 head 的 Q[D] 載入成 [HEAD_PAR][CHUNKS][LANES](每組一次;每個 tile 重複使用),lane 與 head 維度完全分割,
    // 這樣 qk_loop 每次迭代能同時讀 HEAD_PAR·LANES 個 Q 元素。
    static void load_q(const q_t q[HEAD_PAR][D], q_t q_loc[HEAD_PAR][CHUNKS][LANES]) {
#pragma HLS ARRAY_PARTITION variable = q_loc complete dim = 1
#pragma HLS ARRAY_PARTITION variable = q_loc complete dim = 3
    load_q_loop:
        for (int c = 0; c < CHUNKS; c++) {
#pragma HLS PIPELINE II = 1   // 暫定:q 來源是記憶體埠,一週期只讀得到 1~2 個元素,實際 II 取決於來源(綜合報告 II = 4)
            for (int hp = 0; hp < HEAD_PAR; hp++)
                for (int l = 0; l < LANES; l++) q_loc[hp][c][l] = q[hp][c * LANES + l];
        }
    }

    // 要處理的 tile 範圍 [j_lo, j_hi]:由小到大;跳過整塊在未來(j > p>>6)或整塊在 window 外
    // (p − (BC·j + BC−1) ≥ WINDOW)的 tile,與 golden 的 skip 條件相同。所有 head 相同(只與 p 有關)。
    static void tile_range(int p, int &j_lo, int &j_hi) {
        j_hi = p / BC;
        if (WINDOW > 0) {
            const int t = p - WINDOW + 1;
            j_lo = (t > 0) ? (t / BC) : 0;
        } else {
            j_lo = 0;
        }
    }

    // HEAD_PAR 個 head、一個 tile 的更新。k_mem、v_mem 是 KV cache 的 bank 陣列;layer 是該類型層的索引。
    static void tile_update(const q_t q[HEAD_PAR][CHUNKS][LANES], const word_t k_mem[NLAYERS][NB][DEPTH],
                            const word_t v_mem[NLAYERS][NB][DEPTH], int layer, int p, int j, ms_t m_s, GroupState &st) {
#pragma HLS ARRAY_PARTITION variable = q complete dim = 1
#pragma HLS ARRAY_PARTITION variable = q complete dim = 3
        assert(p >= 0 && j >= 0);
        ap_int<16> S[HEAD_PAR][BC];
        ap_uint<1> valid[BC];                                                    // 與 head 無關(只與位置有關),共用
#pragma HLS ARRAY_PARTITION variable = S complete dim = 0
#pragma HLS ARRAY_PARTITION variable = valid complete dim = 1
    valid_loop:
        for (int c = 0; c < BC; c++) {
#pragma HLS UNROLL
            const int pos = BC * j + c;
            const bool v = (pos <= p) && (WINDOW == 0 || (p - pos) < WINDOW);   // 尚未寫入的位置(pos > p)視同被遮罩
            valid[c] = v ? 1 : 0;
            for (int hp = 0; hp < HEAD_PAR; hp++) {
#pragma HLS UNROLL
                S[hp][c] = 0;
            }
        }

        // --- QK^T → S,tile 內最大值(遮罩內):(群組, chunk) 攤平,每次迭代讀 NB 個 K word 一次,餵給 HEAD_PAR 個 head ---
        ap_int<32> acc[HEAD_PAR][NB];                                            // 最壞 32767·127·512 < 2^31
        ap_int<16> tmax[HEAD_PAR];
#pragma HLS ARRAY_PARTITION variable = acc complete dim = 0
#pragma HLS ARRAY_PARTITION variable = tmax complete dim = 1
        for (int hp = 0; hp < HEAD_PAR; hp++) {
#pragma HLS UNROLL   // 所有「對 HEAD_PAR 個 head」的小迴圈都要展開,否則 HLS 會依序執行(綜合報告:每個 tile 多花約 1136 個週期)
            tmax[hp] = S_FRAC_MIN;
        }
        const int g0 = (BC * j) / NB;                                            // 這個 tile 的第一個位置群組
    qk_loop:
        for (int it = 0; it < GROUPS * CHUNKS; it++) {
#pragma HLS PIPELINE II = 1   // 暫定
            const int gi = it / CHUNKS;
            const int chunk = it % CHUNKS;
            word_t kw[NB];
            Cache::read_group(k_mem, layer, g0 + gi, chunk, kw);
        qk_head:
            for (int hp = 0; hp < HEAD_PAR; hp++) {
                ap_int<16> cand[NB];
            qk_bank:
                for (int b = 0; b < NB; b++) {
                    ap_int<32> part = 0;
                qk_lane:
                    for (int l = 0; l < LANES; l++) part += q[hp][chunk][l] * Cache::unpack(kw[b], l);
                    const ap_int<32> a = ((chunk == 0) ? ap_int<32>(0) : acc[hp][b]) + part;
                    acc[hp][b] = a;
                    cand[b] = S_FRAC_MIN;
                    if (chunk == CHUNKS - 1) {
                        // round_half_up(a·M_S / 2^24),再 clip 到 [−2^14, 2^14−1];a 最壞 32 bit、M_S 20 bit → 乘積 < 2^53
                        const ap_int<33> a33 = a;
                        const ap_int<21> m21 = m_s;
                        const ap_int<54> prod = a33 * m21;
                        const ap_int<54> r = ap_int<54>(prod + (ap_int<54>(1) << 23)) >> 24;
                        ap_int<16> s;
                        if (r > S_FRAC_MAX) s = S_FRAC_MAX;
                        else if (r < S_FRAC_MIN) s = S_FRAC_MIN;
                        else s = r;
                        S[hp][gi * NB + b] = s;
                        if (valid[gi * NB + b]) cand[b] = s;
                    }
                }
                if (chunk == CHUNKS - 1) {
                    // NB 個候選值取最大(二元樹),再與 tile 內累計的最大值比較
                max_tree:
                    for (int stride = 1; stride < NB; stride *= 2)
                        for (int i = 0; i + stride < NB; i += 2 * stride)
                            if (cand[i + stride] > cand[i]) cand[i] = cand[i + stride];
                    if (cand[0] > tmax[hp]) tmax[hp] = cand[0];
                }
            }
        }
        ap_int<16> m_new[HEAD_PAR];
#pragma HLS ARRAY_PARTITION variable = m_new complete dim = 1
        for (int hp = 0; hp < HEAD_PAR; hp++) {
#pragma HLS UNROLL
            m_new[hp] = (st.m[hp] > tmax[hp]) ? st.m[hp] : tmax[hp];   // m 初值 −2^14,不是 −inf
        }

        // --- P̃(UQ1.12):每個 head EXP_PAR 個 exp2 實例平行 ---
        ap_uint<13> P[HEAD_PAR][BC];
        ap_uint<19> sumP[HEAD_PAR];                                              // ≤ 64·4096
#pragma HLS ARRAY_PARTITION variable = P complete dim = 0
#pragma HLS ARRAY_PARTITION variable = sumP complete dim = 1
        for (int hp = 0; hp < HEAD_PAR; hp++) {
#pragma HLS UNROLL
            sumP[hp] = 0;
        }
    p_loop:
        for (int c0 = 0; c0 < BC; c0 += EXP_PAR) {
#pragma HLS PIPELINE II = 1   // 暫定
            for (int hp = 0; hp < HEAD_PAR; hp++) {
                ap_uint<19> part = 0;
            p_inner:
                for (int e = 0; e < EXP_PAR; e++) {
                    const int c = c0 + e;
                    const ap_int<16> dd = valid[c] ? ap_int<16>(S[hp][c] - m_new[hp]) : ap_int<16>(0);
                    const ap_uint<17> pe = exp2_t::eval(dd);
                    ap_uint<18> pr = (ap_uint<18>(pe) + 8) >> 4;                 // round_half_up(pe / 2^4)
                    if (pr > 4096) pr = 4096;
                    P[hp][c] = valid[c] ? ap_uint<13>(pr) : ap_uint<13>(0);      // 被遮罩處由遮罩邏輯強制為 0
                    part += P[hp][c];
                }
                sumP[hp] += part;
            }
        }

        // --- α(UQ1.16)與 ℓ 更新 ---
        ap_uint<17> alpha[HEAD_PAR];
#pragma HLS ARRAY_PARTITION variable = alpha complete dim = 1
        for (int hp = 0; hp < HEAD_PAR; hp++) {
#pragma HLS UNROLL
            alpha[hp] = exp2_t::eval(ap_int<16>(st.m[hp] - m_new[hp]));
            const ap_uint<50> al = ap_uint<50>(alpha[hp]) * ap_uint<50>(st.l[hp]);
            st.l[hp] = ap_uint<25>(((al + (ap_uint<50>(1) << 15)) >> 16) + sumP[hp]);
        }

        // --- PV:(群組, chunk) 攤平,每次迭代讀 NB 個 V word 一次,餵給 HEAD_PAR 個 head;NB 個位置的乘積以加法樹合併 ---
        ap_int<28> pv[HEAD_PAR][CHUNKS][LANES];                                  // Σ_c P̃·V:13+8+6 bit
#pragma HLS ARRAY_PARTITION variable = pv complete dim = 1
#pragma HLS ARRAY_PARTITION variable = pv complete dim = 3
    pv_init:
        for (int c = 0; c < CHUNKS; c++) {
#pragma HLS PIPELINE II = 1
            for (int hp = 0; hp < HEAD_PAR; hp++)
                for (int l = 0; l < LANES; l++) pv[hp][c][l] = 0;
        }
    pv_loop:
        for (int it = 0; it < GROUPS * CHUNKS; it++) {
#pragma HLS PIPELINE II = 1   // 暫定
            // 同一個 chunk 的存取相隔 CHUNKS 次迭代(>= 4),沒有實際的迭代間相依
#pragma HLS DEPENDENCE variable = pv inter false
            const int gi = it / CHUNKS;
            const int chunk = it % CHUNKS;
            word_t vw[NB];
            Cache::read_group(v_mem, layer, g0 + gi, chunk, vw);
        pv_head:
            for (int hp = 0; hp < HEAD_PAR; hp++) {
            pv_lane:
                for (int l = 0; l < LANES; l++) {
                    ap_int<28> sum = 0;
                pv_bank:
                    for (int b = 0; b < NB; b++) sum += P[hp][gi * NB + b] * Cache::unpack(vw[b], l);
                    pv[hp][chunk][l] += sum;
                }
            }
        }

        // --- O 更新:每次迭代一個 chunk(HEAD_PAR × LANES 個元素)---
    o_loop:
        for (int c = 0; c < CHUNKS; c++) {
#pragma HLS PIPELINE II = 1   // 暫定
            for (int hp = 0; hp < HEAD_PAR; hp++) {
            o_lane:
                for (int l = 0; l < LANES; l++) {
                    const ap_int<50> ao = ap_int<50>(alpha[hp]) * ap_int<50>(st.O[hp][c][l]);
                    st.O[hp][c][l] = ap_int<32>(((ao + (ap_int<50>(1) << 15)) >> 16) + pv[hp][c][l]);   // 算術右移(負數也是 round-half-up)
                }
            }
        }
        for (int hp = 0; hp < HEAD_PAR; hp++) {
#pragma HLS UNROLL
            st.m[hp] = m_new[hp];
        }
    }

    // 收尾(HEAD_PAR 個 head 同時做):1/ℓ 查表、y、o_in 量化。
    static void finalize(const GroupState &st, c2_t c2, code_t code[HEAD_PAR][D]) {
        typename recip_t::r_t R[HEAD_PAR];
        int sh[HEAD_PAR];
#pragma HLS ARRAY_PARTITION variable = R complete dim = 1
#pragma HLS ARRAY_PARTITION variable = sh complete dim = 1
        for (int hp = 0; hp < HEAD_PAR; hp++) {
#pragma HLS UNROLL
            typename recip_t::w_t w;
            recip_t::eval(st.l[hp], R[hp], w);
            sh[hp] = 14 + (int)w;                                                // Fo + Fr − Fy + w;w ∈ [−12, 12] → 2..26
            assert(sh[hp] >= 1);
        }
    fin_loop:
        for (int d = 0; d < D; d++) {
#pragma HLS PIPELINE II = 1   // 暫定
            for (int hp = 0; hp < HEAD_PAR; hp++) {
                const ap_int<48> prod = ap_int<48>(st.O_at(hp, d)) * ap_int<48>(R[hp]);
                const ap_int<24> y = ap_int<24>(ap_int<48>(prod + (ap_int<48>(1) << (sh[hp] - 1))) >> sh[hp]);   // y:小數 10 位
                const ap_int<41> t = ap_int<41>(y) * ap_int<41>(c2);
                const ap_int<41> r = ap_int<41>(t + (ap_int<41>(1) << 21)) >> 22;                                // Fy + c_bits = 22
                if (r > 127) code[hp][d] = 127;
                else if (r < -127) code[hp][d] = -127;
                else code[hp][d] = r;
            }
        }
    }

    // 一個 token(位置 p):H / HEAD_PAR 組依序處理,每組 HEAD_PAR 個 head 同時處理。
    static void run(const q_t q[H][D], const word_t k_mem[NLAYERS][NB][DEPTH], const word_t v_mem[NLAYERS][NB][DEPTH], int layer,
                    int p, ms_t m_s, c2_t c2, code_t code[H][D]) {
        int j_lo, j_hi;
        tile_range(p, j_lo, j_hi);
    group_loop:
        for (int hg = 0; hg < H / HEAD_PAR; hg++) {
            GroupState st;
#pragma HLS ARRAY_PARTITION variable = st.m complete dim = 1
#pragma HLS ARRAY_PARTITION variable = st.l complete dim = 1
#pragma HLS ARRAY_PARTITION variable = st.O complete dim = 1
#pragma HLS ARRAY_PARTITION variable = st.O complete dim = 3
            q_t q_loc[HEAD_PAR][CHUNKS][LANES];
#pragma HLS ARRAY_PARTITION variable = q_loc complete dim = 1
#pragma HLS ARRAY_PARTITION variable = q_loc complete dim = 3
            init_state(st);
            load_q(&q[hg * HEAD_PAR], q_loc);
        tile_loop:
            for (int j = j_lo; j <= j_hi; j++) {
                // 只影響報告的 Latency 估計(tile 數由執行期的 p 決定):global 最多 CACHE_LEN/BC = 64 個 tile,
                // sliding 最多 WINDOW/BC + 1 = 9 個(pragma 需要常數,這裡取 64,所以 sliding 的上限估計偏大)
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 64
                tile_update(q_loc, k_mem, v_mem, layer, p, j, m_s, st);
            }
            finalize(st, c2, &code[hg * HEAD_PAR]);
        }
    }
};

}  // namespace attn
