/* Direct decode kernels for IQ2_XXS/XS/S and IQ3_XXS/S (SSSE3; mx_unpack_iq.c has the layouts and the
 * fallback). Their grid values are all positive, so the grid is the unsigned pmaddubsw operand and each
 * sign goes onto the activation byte instead (exact: Q8_K clamps to -127..127). Same integer per block
 * and the same float fold as the unpacked path, so a decode token is bit-identical to a prefill. */
#include "mx_common.h"
#include "mx_iq_grids.h"

#if defined(__SSSE3__)
/* sc * sum(grid * q8 with signs) over 16 values (signs: s0 for the first 8, s1 for the next 8). */
static inline __m128i mx_iq_dot16(__m128i grid, uint8_t s0, uint8_t s1, const int8_t *q8, int sc) {
    const __m128i bits = _mm_setr_epi8(1, 2, 4, 8, 16, 32, 64, -128, 1, 2, 4, 8, 16, 32, 64, -128);
    const __m128i rep = _mm_shuffle_epi8(_mm_cvtsi32_si128(s0 | (s1 << 8)),
                                         _mm_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1));
    const __m128i neg = _mm_cmpeq_epi8(_mm_and_si128(rep, bits), bits);
    const __m128i q = _mm_loadu_si128((const __m128i *)q8);
    const __m128i q_signed = _mm_sub_epi8(_mm_xor_si128(q, neg), neg);
    return _mm_madd_epi16(_mm_maddubs_epi16(grid, q_signed), _mm_set1_epi16((short)sc));
}

static inline __m128i mx_iq_grid8x2(const int8_t *g0, const int8_t *g1) {
    return _mm_unpacklo_epi64(_mm_loadl_epi64((const __m128i *)g0), _mm_loadl_epi64((const __m128i *)g1));
}

static inline __m128i mx_iq_grid4x4(const int8_t *g0, const int8_t *g1, const int8_t *g2, const int8_t *g3) {
    int32_t w[4];
    memcpy(&w[0], g0, 4);
    memcpy(&w[1], g1, 4);
    memcpy(&w[2], g2, 4);
    memcpy(&w[3], g3, 4);
    return _mm_loadu_si128((const __m128i *)w);
}

float mx_vec_dot_iq2_xxs(const uint8_t *row, const mx_block_q8_k *y, int nb) {
    float sumf = 0.0f;
    for (int i = 0; i < nb; ++i) {
        const uint8_t *blk = row + (size_t)i * MX_IQ2_XXS_BYTES;
        const float wd = mx_fp16(blk) * 0.125f;
        const uint8_t *qs = blk + 2;
        __m128i acc = _mm_setzero_si128();
        for (int ib32 = 0; ib32 < 8; ++ib32) {
            const uint8_t *word = qs + ib32 * 8;
            uint32_t word1;
            memcpy(&word1, word + 4, sizeof(word1));
            const int sc = 2 * (int)(word1 >> 28) + 1;
            for (int group = 0; group < 4; group += 2) {
                const __m128i grid = mx_iq_grid8x2(MX_IQ2XXS_GRID + (size_t)word[group] * 8,
                                                   MX_IQ2XXS_GRID + (size_t)word[group + 1] * 8);
                acc = _mm_add_epi32(acc, mx_iq_dot16(grid, MX_KSIGNS_IQ2XS[(word1 >> (7 * group)) & 127],
                                                     MX_KSIGNS_IQ2XS[(word1 >> (7 * group + 7)) & 127],
                                                     y[i].qs + ib32 * 32 + group * 8, sc));
            }
        }
        sumf += y[i].d * wd * (float)mx_hsum_epi32(acc);
    }
    return sumf;
}

float mx_vec_dot_iq2_xs(const uint8_t *row, const mx_block_q8_k *y, int nb) {
    float sumf = 0.0f;
    for (int i = 0; i < nb; ++i) {
        const uint8_t *blk = row + (size_t)i * MX_IQ2_XS_BYTES;
        const float wd = mx_fp16(blk) * 0.125f;
        const uint8_t *qs_bytes = blk + 2;
        const uint8_t *scales = blk + 66;
        __m128i acc = _mm_setzero_si128();
        for (int ib32 = 0; ib32 < 8; ++ib32) {
            for (int group = 0; group < 4; group += 2) {
                uint16_t qv[2];
                memcpy(qv, qs_bytes + (ib32 * 4 + group) * 2, sizeof(qv));
                const int ls = group == 0 ? (scales[ib32] & 0x0F) : (scales[ib32] >> 4);
                const __m128i grid = mx_iq_grid8x2(MX_IQ2XS_GRID + (size_t)(qv[0] & 511) * 8,
                                                   MX_IQ2XS_GRID + (size_t)(qv[1] & 511) * 8);
                acc = _mm_add_epi32(acc, mx_iq_dot16(grid, MX_KSIGNS_IQ2XS[qv[0] >> 9], MX_KSIGNS_IQ2XS[qv[1] >> 9],
                                                     y[i].qs + ib32 * 32 + group * 8, 2 * ls + 1));
            }
        }
        sumf += y[i].d * wd * (float)mx_hsum_epi32(acc);
    }
    return sumf;
}

float mx_vec_dot_iq2_s(const uint8_t *row, const mx_block_q8_k *y, int nb) {
    float sumf = 0.0f;
    for (int i = 0; i < nb; ++i) {
        const uint8_t *blk = row + (size_t)i * MX_IQ2_S_BYTES;
        const float wd = mx_fp16(blk) * 0.125f;
        const uint8_t *qs_lo = blk + 2;
        const uint8_t *sign_bytes = blk + 34;
        const uint8_t *qh = blk + 66;
        const uint8_t *scales = blk + 74;
        __m128i acc = _mm_setzero_si128();
        for (int ib32 = 0; ib32 < 8; ++ib32) {
            for (int group = 0; group < 4; group += 2) {
                const int idx0 = qs_lo[ib32 * 4 + group] | ((qh[ib32] << (8 - 2 * group)) & 0x300);
                const int idx1 = qs_lo[ib32 * 4 + group + 1] | ((qh[ib32] << (6 - 2 * group)) & 0x300);
                const int ls = group == 0 ? (scales[ib32] & 0x0F) : (scales[ib32] >> 4);
                const __m128i grid = mx_iq_grid8x2(MX_IQ2S_GRID + (size_t)idx0 * 8, MX_IQ2S_GRID + (size_t)idx1 * 8);
                acc = _mm_add_epi32(acc, mx_iq_dot16(grid, sign_bytes[ib32 * 4 + group],
                                                     sign_bytes[ib32 * 4 + group + 1],
                                                     y[i].qs + ib32 * 32 + group * 8, 2 * ls + 1));
            }
        }
        sumf += y[i].d * wd * (float)mx_hsum_epi32(acc);
    }
    return sumf;
}

float mx_vec_dot_iq3_xxs(const uint8_t *row, const mx_block_q8_k *y, int nb) {
    float sumf = 0.0f;
    for (int i = 0; i < nb; ++i) {
        const uint8_t *blk = row + (size_t)i * MX_IQ3_XXS_BYTES;
        const float wd = mx_fp16(blk) * 0.25f;
        const uint8_t *grid_idx = blk + 2;
        const uint8_t *aux_bytes = blk + 66;
        __m128i acc = _mm_setzero_si128();
        for (int ib32 = 0; ib32 < 8; ++ib32) {
            uint32_t aux;
            memcpy(&aux, aux_bytes + ib32 * 4, sizeof(aux));
            const int sc = 2 * (int)(aux >> 28) + 1;
            const uint8_t *idx = grid_idx + ib32 * 8;
            for (int group = 0; group < 4; group += 2) {
                const __m128i grid = mx_iq_grid4x4(
                    MX_IQ3XXS_GRID + (size_t)idx[2 * group] * 4, MX_IQ3XXS_GRID + (size_t)idx[2 * group + 1] * 4,
                    MX_IQ3XXS_GRID + (size_t)idx[2 * group + 2] * 4, MX_IQ3XXS_GRID + (size_t)idx[2 * group + 3] * 4);
                acc = _mm_add_epi32(acc, mx_iq_dot16(grid, MX_KSIGNS_IQ2XS[(aux >> (7 * group)) & 127],
                                                     MX_KSIGNS_IQ2XS[(aux >> (7 * group + 7)) & 127],
                                                     y[i].qs + ib32 * 32 + group * 8, sc));
            }
        }
        sumf += y[i].d * wd * (float)mx_hsum_epi32(acc);
    }
    return sumf;
}

float mx_vec_dot_iq3_s(const uint8_t *row, const mx_block_q8_k *y, int nb) {
    float sumf = 0.0f;
    for (int i = 0; i < nb; ++i) {
        const uint8_t *blk = row + (size_t)i * MX_IQ3_S_BYTES;
        const float wd = mx_fp16(blk);
        const uint8_t *qs = blk + 2;
        const uint8_t *qh = blk + 66;
        const uint8_t *signs = blk + 74;
        const uint8_t *scales = blk + 106;
        __m128i acc = _mm_setzero_si128();
        for (int i2 = 0; i2 < 4; ++i2) {
            for (int half = 0; half < 2; ++half) {
                const int ib32 = i2 * 2 + half;
                const int sc = 1 + 2 * (half == 0 ? (scales[i2] & 0x0F) : (scales[i2] >> 4));
                const int qh_byte = qh[ib32];
                const uint8_t *qs_off = qs + i2 * 16 + half * 8;
                const uint8_t *sign_off = signs + i2 * 8 + half * 4;
                for (int group = 0; group < 4; group += 2) {
                    const int8_t *g[4];
                    for (int k = 0; k < 4; ++k) {
                        const int pos = 2 * group + k;
                        g[k] = MX_IQ3S_GRID + (size_t)(qs_off[pos] | ((qh_byte << (8 - pos)) & 256)) * 4;
                    }
                    acc = _mm_add_epi32(acc, mx_iq_dot16(mx_iq_grid4x4(g[0], g[1], g[2], g[3]), sign_off[group],
                                                         sign_off[group + 1], y[i].qs + ib32 * 32 + group * 8, sc));
                }
            }
        }
        sumf += y[i].d * wd * (float)mx_hsum_epi32(acc);
    }
    return sumf;
}
#endif
