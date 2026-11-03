// RoPE ??„ç?œå?ˆç”¨ top ?‡½å¼?(?š«å®?:ä»‹é¢??‹å?è?‡å?å?é?æœªæ±?,spec Â§8 ???)??
// è¡¨ç”±å¤–é¢??ä??(cos_tab / sin_tab ?˜¯ top ??„è?˜æ†¶é«”å??);è¦æ”¾ BRAM ??? URAM ?”±??æ?‰é?™ä»½è¡¨ç?„é‚£ä¸?å±¤æ±ºå®?,
// ?™è£¡ä¸å¯« BIND_STORAGE?‚æ?ç¨®(å±¤å?? ?? Q/K)ä¸??? top,?? ç‚º HEAD_DIM?N_ROT?IN/OUT_FRAC ?˜¯ç·¨è­¯??Ÿå¸¸?•¸??
#pragma once
#include "rope.h"

typedef ap_int<16> rope_io_t;
typedef ap_int<14> rope_tab_t;   // ??‰è?? 14 ä½å??,å°æ•¸ 12 ä½?(TAB_FRAC = 12 ???)

// sliding:D = 256,128 çµ„æ?‹è??(Î¸ = 1e4)?‚Q:8 ?? head,å°æ•¸ 10 ??? 10;K:1 ?? head,å°æ•¸ 14 ??? 14
void rope_sliding_q(const rope_io_t x[8][256], int p, const rope_tab_t cos_tab[rope::MAX_POS][128],
                    const rope_tab_t sin_tab[rope::MAX_POS][128], rope_io_t y[8][256]);
void rope_sliding_k(const rope_io_t x[1][256], int p, const rope_tab_t cos_tab[rope::MAX_POS][128],
                    const rope_tab_t sin_tab[rope::MAX_POS][128], rope_io_t y[1][256]);

// global:D = 512,?ª??? 64 çµ„æ?‹è??(Î¸ = 1e6,ç¶­åº¦ 0~63 ??? 256~319),?…¶é¤˜æ?è·¯
void rope_global_q(const rope_io_t x[8][512], int p, const rope_tab_t cos_tab[rope::MAX_POS][64],
                   const rope_tab_t sin_tab[rope::MAX_POS][64], rope_io_t y[8][512]);
void rope_global_k(const rope_io_t x[1][512], int p, const rope_tab_t cos_tab[rope::MAX_POS][64],
                   const rope_tab_t sin_tab[rope::MAX_POS][64], rope_io_t y[1][512]);
