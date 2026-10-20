/* Shared helpers for the grid-codebook I-quants (IQ2_XXS/XS/S, IQ3_XXS/S, IQ1_S/M) on the K-quants' integer path: every weight
 * is base_d * sc * q, with sc a small odd integer per 16 values and q a signed grid value (sign or IQ1
 * delta applied). Stored as u = q + off and m = sc * off, so mx_dot_unpacked's integer sums give the
 * signed result exactly (has_min = 0), with SIMD byte dots instead of one float multiply per weight.
 *
 * Bounds (mx_block_unpacked): u <= 2 * 62 = 124 < 128, so a pmaddubsw pair stays below 32767 and NEON's
 * signed dot still holds; sc <= 31. Layouts and scales follow the float kernels these replace and the
 * app/gguf/dequant iq{1,2,3}_family strategies:
 * - IQ2_*: d * (0.5 + ls) * 0.25 = (d / 8) * (2 * ls + 1), grid values 8/25/43;
 * - IQ3_XXS: d * (0.5 + ls) * 0.5 = (d / 4) * (2 * ls + 1), grid values up to 62;
 * - IQ3_S: d * (1 + 2 * ls), grid values up to 15;
 * - IQ1_S/M: d * (2 * k + 1) * (g +- 1/8) = (d / 8) * (2 * k + 1) * (8 * g +- 1).
 *
 * Unpackers: mx_unpack_iq.c (IQ2/IQ3), mx_unpack_iq1.c. Decode uses fused per-row kernels (IQ2/IQ3 with
 * SSSE3: mx_dot_iq_ssse3.c; otherwise MX_IQ_FUSED_UNPACKED); each gives the same integer per block and the
 * same float fold as mx_dot_unpacked, so a decode token is bit-identical to the same token in a prefill. */
#ifndef MX_IQ_UNPACK_H
#define MX_IQ_UNPACK_H

#include "mx_common.h"
#include "mx_iq_grids.h"

#define MX_IQ2_OFF 43
#define MX_IQ3_XXS_OFF 62
#define MX_IQ3_S_OFF 15
#define MX_IQ1_OFF 9

/* One integer scale for sub-blocks s and s + 1 (32 values). */
static inline void mx_iq_scale32(mx_block_unpacked *b, int ib32, int sc, int off) {
    b->sc[2 * ib32] = b->sc[2 * ib32 + 1] = (int16_t)sc;
    b->m[2 * ib32] = b->m[2 * ib32 + 1] = (int16_t)(sc * off);
}

static inline void mx_iq_scale16(mx_block_unpacked *b, int s, int sc, int off) {
    b->sc[s] = (int16_t)sc;
    b->m[s] = (int16_t)(sc * off);
}

/* 16 signed grid values (two groups of 8: g0[0..8) then g1[0..8)), group k's signs in byte s_k (bit j
 * negates value j), offset to unsigned. `w` is the grid row width: 8 (IQ2) or 4 (IQ3, where each group is
 * two rows: g0, g0b and g1, g1b). */
static inline void mx_iq_put16(uint8_t *u, const int8_t *g0, const int8_t *g0b, const int8_t *g1,
                               const int8_t *g1b, int w, uint8_t s0, uint8_t s1, int off) {
    int8_t v[16];
    memcpy(v, g0, (size_t)w);
    memcpy(v + w, g0b, (size_t)(8 - w));
    memcpy(v + 8, g1, (size_t)w);
    memcpy(v + 8 + w, g1b, (size_t)(8 - w));
#if defined(__SSSE3__)
    const __m128i bits = _mm_setr_epi8(1, 2, 4, 8, 16, 32, 64, -128, 1, 2, 4, 8, 16, 32, 64, -128);
    const __m128i rep = _mm_shuffle_epi8(_mm_cvtsi32_si128(s0 | (s1 << 8)),
                                         _mm_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1));
    const __m128i neg = _mm_cmpeq_epi8(_mm_and_si128(rep, bits), bits);
    const __m128i grid = _mm_loadu_si128((const __m128i *)v);
    const __m128i signed_v = _mm_sub_epi8(_mm_xor_si128(grid, neg), neg);
    _mm_storeu_si128((__m128i *)u, _mm_add_epi8(signed_v, _mm_set1_epi8((char)off)));
#else
    for (int j = 0; j < 16; ++j) {
        const uint8_t signs = j < 8 ? s0 : s1;
        u[j] = (uint8_t)(((signs >> (j & 7)) & 1 ? -v[j] : v[j]) + off);
    }
#endif
}

/* 16 ternary IQ1 values (two groups of 8) as 8 * g + delta_k + off, delta_k = +-1 (the 1/8 shift x 8). */
static inline void mx_iq1_put16(uint8_t *u, const int8_t *g0, const int8_t *g1, int delta0, int delta1) {
#if defined(__SSSE3__)
    const __m128i grid = _mm_unpacklo_epi64(_mm_loadl_epi64((const __m128i *)g0),
                                            _mm_loadl_epi64((const __m128i *)g1));
    /* g + 1 in 0..2 picks -8, 0, +8 (plus the offset) from a 3-entry table. */
    const __m128i table = _mm_setr_epi8(MX_IQ1_OFF - 8, MX_IQ1_OFF, MX_IQ1_OFF + 8, 0, 0, 0, 0, 0, 0, 0, 0,
                                        0, 0, 0, 0, 0);
    const __m128i base = _mm_shuffle_epi8(table, _mm_add_epi8(grid, _mm_set1_epi8(1)));
    const __m128i delta = _mm_unpacklo_epi64(_mm_set1_epi8((char)delta0), _mm_set1_epi8((char)delta1));
    _mm_storeu_si128((__m128i *)u, _mm_add_epi8(base, delta));
#else
    for (int j = 0; j < 8; ++j) {
        u[j] = (uint8_t)(8 * g0[j] + delta0 + MX_IQ1_OFF);
        u[8 + j] = (uint8_t)(8 * g1[j] + delta1 + MX_IQ1_OFF);
    }
#endif
}

static inline void mx_iq_begin(mx_block_unpacked *b, float d) {
    b->d = d;
    b->dmin = 0.0f;
}

/* Row unpackers (prefill: once per row, reused for every token). */
#define MX_IQ_UNPACK(name, bytes)                                                                      \
    void mx_unpack_##name(const uint8_t *restrict row, mx_block_unpacked *restrict out, int nb) {      \
        for (int i = 0; i < nb; ++i) {                                                                 \
            mx_unpack_##name##_block(row + (size_t)i * (bytes), out + i);                              \
        }                                                                                              \
    }

/* Fused per-row dot (decode) through the unpacked block: used for IQ1 (its grid has -1, so the direct
 * kernels below don't apply) and without SSSE3. */
#define MX_IQ_FUSED_UNPACKED(name, bytes)                                                              \
    float mx_vec_dot_##name(const uint8_t *row, const mx_block_q8_k *y, int nb) {                      \
        mx_block_unpacked b;                                                                           \
        float sumf = 0.0f;                                                                             \
        for (int i = 0; i < nb; ++i) {                                                                 \
            mx_unpack_##name##_block(row + (size_t)i * (bytes), &b);                                   \
            sumf += mx_dot_unpacked(&b, y + i, 1, 0);                                                  \
        }                                                                                              \
        return sumf;                                                                                   \
    }

#endif
