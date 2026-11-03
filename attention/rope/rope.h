// RoPE(spec Â§4;golden:attention_golden.py ??? rope_fixed)?‚ä?å?ƒç²¾ç¢ºã??
//   split-half ??å??:ç¶­åº¦ j ??? j+HEAD_DIM/2??  y1 = x1Â·cos ??? x2Â·sin,  y2 = x2Â·cos + x1Â·sin
//   ä¹˜ç?ä?ç?™å?Œæ•´ä½å¯¬?å…©??…ç›¸?? å?Œæ?æ¨?…¥ä¸?æ¬?:y = sat16((acc + 2^(SHIFT-1)) >> SHIFT)
//   SHIFT = IN_FRAC + TAB_FRAC ??? OUT_FRAC;round-half-up(??? 2^(SHIFT-1) å¾Œç?—è?“å³ç§?);é£½å??,ä¸ç’°ç¹ã??
//   norm å·²æ?? Î³ ??˜é?²è¼¸?‡º,?™è£¡ä¸å?ä?˜ä»»ä½•å¸¸?•¸(ä½¿ç”¨?…æ±ºå®?)??
//   cos/sin ?ªè®?è¡?(golden/rope_tables.npz ?Œ¯?‡º),ä¸è‡ªå·±ç?—ã?‚V ä¸ç?“é?? RoPE??
#pragma once
#include <ap_int.h>
#include <cassert>

// Vivado HLS 2019.2 ??„å?ç«¯??è¨­ä¸æ˜¯ C++11,??ä»¥ä?ç”¨ constexpr / static_assert??
// ç·¨è­¯??Ÿæª¢?Ÿ¥:C++11 ä»¥ä?Šç”¨ static_assert;?¦??‡ç”¨?Œé™£??—å¤§å°ç‚ºè² æ•¸å°±ç·¨è­¯å¤±??—ã?ç?„å¯«æ³?(??ˆæ?œç›¸???,è¨Šæ¯è¼ƒä?ç›´è§?)??
#define ROPE_CAT2(a, b) a##b
#define ROPE_CAT(a, b) ROPE_CAT2(a, b)
#if __cplusplus >= 201103L
#define ROPE_STATIC_ASSERT(cond, msg) static_assert(cond, msg)
#else
#define ROPE_STATIC_ASSERT(cond, msg) typedef char ROPE_CAT(rope_static_assert_, __LINE__)[(cond) ? 1 : -1]
#endif

namespace rope {

const int MAX_POS = 4096;   // ç³»çµ±?•·åº¦ä?Šé??(??? spec Â§4);ä½ç½® 0..4095,è¶…å‡º?˜¯?Œ¯èª?

// ä¸æ?‹è?‰ç?„ç??(j >= N_ROT)ä»? cos = 2^TAB_FRAC?sin = 0 èµ°å?Œä?æ¢ç?—è??(??? golden ?›¸???);
// IN_FRAC == OUT_FRAC ??‚ç?æ?œè?‡è¼¸?…¥?ä?å?ƒç›¸???,ç·¨è­¯?™¨??ƒæ?Šå¸¸?•¸ä¹˜æ?•å?–ç°¡??‰ã??
//
// HEAD_DIM  256(sliding)??? 512(global);N_ROT  ??‹è?‰ç?„ç?„æ•¸:sliding 128?global 64(rot_idx ?‚º 0..N_ROT-1,
//           å°æ?? golden ??? rot_idx;global ??„ç¶­åº? 0~63 ??? 256~319)
// IN_FRAC / OUT_FRAC  Q ??è¨­ 10/10?K ??è¨­ 14/14(norm è¼¸å‡º? ¼å¼? ??? ä½¿ç”¨?…å·²æ±ºå??)
template <int HEAD_DIM, int N_ROT, int IN_FRAC, int OUT_FRAC, int TAB_FRAC = 12, int OUT_BITS = 16>
struct Rope {
    static const int HALF = HEAD_DIM / 2;
    static const int SHIFT = IN_FRAC + TAB_FRAC - OUT_FRAC;
    static const int TAB_W = TAB_FRAC + 2;                    // ??‰è?? 14 ä½å??(1 ç¬¦è?Ÿè?‡æ•´?•¸ä½? + 12 å°æ•¸)
    static const int ACC_W = 16 + TAB_W + 2;                  // ?…©?? 16?—TAB_W ä¹˜ç?ç›¸???,32 ä½å??(TAB_FRAC=12)
    ROPE_STATIC_ASSERT(IN_FRAC + TAB_FRAC - OUT_FRAC > 0, "IN_FRAC + TAB_FRAC - OUT_FRAC å¿…é?? > 0");
    ROPE_STATIC_ASSERT(HEAD_DIM % 2 == 0, "HEAD_DIM å¿…é?ˆç‚º?¶?•¸");
    ROPE_STATIC_ASSERT(N_ROT >= 1 && N_ROT <= HEAD_DIM / 2, "N_ROT å¿…é?ˆåœ¨ 1..HEAD_DIM/2");
    ROPE_STATIC_ASSERT(OUT_BITS >= 2 && OUT_BITS <= 16, "OUT_BITS ?š«å®šåª?”¯?´?ˆ° 16");

    typedef ap_int<16> in_t;
    typedef ap_int<TAB_W> tab_t;
    typedef ap_int<OUT_BITS> out_t;
    typedef ap_int<ACC_W> acc_t;

    // round-half-up ?³ç§»å?Œé£½??Œåˆ° OUT_BITS(ç®—è?“å³ç§»æ?ç¢ºå¯«å‡º)
    static out_t round_sat(acc_t v) {
#pragma HLS INLINE
        const acc_t rnd = acc_t(1) << (SHIFT - 1);
        const acc_t r = (acc_t(v + rnd)) >> SHIFT;            // ap_int ??? >> å°æ?‰è?Ÿæ•¸?˜¯ç®—è?“å³ç§?
        const acc_t hi = (acc_t(1) << (OUT_BITS - 1)) - 1;
        const acc_t lo = -(acc_t(1) << (OUT_BITS - 1));
        if (r > hi) return out_t(hi);
        if (r < lo) return out_t(lo);
        return out_t(r);
    }

    // ä¸??? D ç¶­å?‘é??(Q ??„ä??? head,??? K)?‚cos_row/sin_row:è©²ä?ç½®??„ä????,?•·åº? N_ROT??
    static void apply(const in_t x[HEAD_DIM], const tab_t cos_row[N_ROT], const tab_t sin_row[N_ROT],
                      out_t y[HEAD_DIM]) {
#pragma HLS INLINE off
        const tab_t ONE = tab_t(1) << TAB_FRAC;               // 1.0 = 4096(14 ä½å?ƒæ?‰è?Ÿå¯ç²¾ç¢ºè¡¨ç¤º)
    pair_loop:
        for (int j = 0; j < HALF; j++) {
#pragma HLS PIPELINE II = 1   // ?š«å®?:??å?é?è?? DSP ??ç?—æœªæ±?(spec Â§8 ???)
            const tab_t c = (j < N_ROT) ? cos_row[j] : ONE;
            const tab_t s = (j < N_ROT) ? sin_row[j] : tab_t(0);
            const ap_int<16> x1 = x[j], x2 = x[j + HALF];
            const acc_t a = acc_t(x1 * c) - acc_t(x2 * s);
            const acc_t b = acc_t(x2 * c) + acc_t(x1 * s);
            y[j] = round_sat(a);
            y[j + HALF] = round_sat(b);
        }
    }

    // ä¸??? token ??? H ?? head(Q:H=8?K:H=1),?…±?”¨ä½ç½® p ??„å?Œä???—è¡¨??
    // cos_tab/sin_tab:[MAX_POS][N_ROT]?‚p å¿…é?ˆåœ¨ 0..MAX_POS-1(C æ¨¡æ“¬ä»? assert æª¢æŸ¥;ç¡¬é?”ä?æª¢?Ÿ¥,?š«å®?)??
    template <int H>
    static void apply_heads(const in_t x[H][HEAD_DIM], int p, const tab_t cos_tab[MAX_POS][N_ROT],
                            const tab_t sin_tab[MAX_POS][N_ROT], out_t y[H][HEAD_DIM]) {
        assert(p >= 0 && p < MAX_POS);
        tab_t c_row[N_ROT], s_row[N_ROT];
    load_row:
        for (int j = 0; j < N_ROT; j++) {
#pragma HLS PIPELINE II = 1
            c_row[j] = cos_tab[p][j];
            s_row[j] = sin_tab[p][j];
        }
    head_loop:
        for (int h = 0; h < H; h++) apply(x[h], c_row, s_row, y[h]);
    }
};

}  // namespace rope
