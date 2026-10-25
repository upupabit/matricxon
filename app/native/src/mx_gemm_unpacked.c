/* Prefill path of mx_gemm for types that unpack rows (K-quants, grid I-quants), past the fused per-token kernel's
 * range: many tokens times unpacked rows. Rows go in tiles of mx_row_tile: a thread unpacks one tile, then runs every
 * token against all of its rows, so each token's Q8_K activations are read once per tile instead of once per row.
 * With 256 tokens of a 6144-wide layer (~1.8 MB of activations, past a 256 KB L2) the old one-row loop streamed
 * them from L3 for every output row. Same dot (mx_dot_unpacked), same results; only the loop order changed. */
#include <stdlib.h>

#include "mx_gemm.h"

#ifndef MX_GEMM_ROW_TILE
#define MX_GEMM_ROW_TILE 8
#endif
#define MX_GEMM_MAX_ROW_TILE 64

/* Runtime copy, so a benchmark can tune it without a rebuild (mx_set_gemm_row_tile, tools/benchmark_kernels.py). */
static int mx_row_tile = MX_GEMM_ROW_TILE;

void mx_set_gemm_row_tile(int rows) {
    mx_row_tile = rows < 1 ? 1 : (rows > MX_GEMM_MAX_ROW_TILE ? MX_GEMM_MAX_ROW_TILE : rows);
}

#if defined(__SSSE3__) && !defined(MX_WIDE_AVX2) && !defined(MX_WIDE_NEON_DOT)
#define MX_DOT_X4 1
/* Four tokens against one unpacked row (the SSSE3 build): each weight load and scale broadcast serves four tokens,
 * and the per-16 mins are two pmaddwd instead of a scalar loop. Integer sums are exact and each block's float fold
 * is mx_dot_unpacked's expression in the same order, so every result is bit-identical to it. `x` is token 0's
 * first block; token t's block i is x[t * stride + i].
 * Q4_K/Q5_K (has_min) share one scale per 32 values (two equal sc slots), so two pmaddubsw sums are added in int16
 * first (|sum| <= 2 * 2*31*127, no saturation) and scaled by one pmaddwd: a quarter fewer multiplies, same sums.
 * Measured on Sandy Bridge (Qwen3.5-4B, 64 tokens): Q4_K/Q5_K 26 -> 32 GMAC/s; 8 tokens per pass added nothing. */
static void mx_dot_unpacked_x4(const mx_block_unpacked *w, const mx_block_q8_k *x, size_t stride, int nb,
                               int has_min, float *out) {
    float sumf[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int i = 0; i < nb; ++i) {
        __m128i acc[4] = {_mm_setzero_si128(), _mm_setzero_si128(), _mm_setzero_si128(), _mm_setzero_si128()};
        if (has_min) {
            for (int s = 0; s < MX_QK_K / 16; s += 2) {
                const __m128i a0 = _mm_loadu_si128((const __m128i *)(w[i].u + 16 * s));
                const __m128i a1 = _mm_loadu_si128((const __m128i *)(w[i].u + 16 * s + 16));
                const __m128i scale = _mm_set1_epi16(w[i].sc[s]);
                for (int t = 0; t < 4; ++t) {
                    const int8_t *q = x[t * stride + i].qs + 16 * s;
                    const __m128i p = _mm_add_epi16(_mm_maddubs_epi16(a0, _mm_loadu_si128((const __m128i *)q)),
                                                    _mm_maddubs_epi16(a1, _mm_loadu_si128((const __m128i *)(q + 16))));
                    acc[t] = _mm_add_epi32(acc[t], _mm_madd_epi16(p, scale));
                }
            }
        } else {
            for (int s = 0; s < MX_QK_K / 16; ++s) {
                const __m128i a = _mm_loadu_si128((const __m128i *)(w[i].u + 16 * s));
                const __m128i scale = _mm_set1_epi16(w[i].sc[s]);
                for (int t = 0; t < 4; ++t) {
                    const __m128i b = _mm_loadu_si128((const __m128i *)(x[t * stride + i].qs + 16 * s));
                    acc[t] = _mm_add_epi32(acc[t], _mm_madd_epi16(_mm_maddubs_epi16(a, b), scale));
                }
            }
        }
        const __m128i m0 = _mm_loadu_si128((const __m128i *)w[i].m);
        const __m128i m1 = _mm_loadu_si128((const __m128i *)(w[i].m + 8));
        for (int t = 0; t < 4; ++t) {
            const mx_block_q8_k *y = &x[t * stride + i];
            const __m128i mins = _mm_add_epi32(_mm_madd_epi16(m0, _mm_loadu_si128((const __m128i *)y->bsums)),
                                               _mm_madd_epi16(m1, _mm_loadu_si128((const __m128i *)(y->bsums + 8))));
            const int32_t sumi = mx_hsum_epi32(acc[t]);
            const int32_t summ = mx_hsum_epi32(mins);
            if (has_min) {
                sumf[t] += y->d * (w[i].d * (float)sumi - w[i].dmin * (float)summ);
            } else {
                sumf[t] += y->d * w[i].d * (float)(sumi - summ);
            }
        }
    }
    for (int t = 0; t < 4; ++t) {
        out[t] = sumf[t];
    }
}
#endif

/* w: packed rows; xq: (n_tokens, nb) Q8_K activations; y: (n_tokens, out_features). 0, or -2 out of memory. */
int mx_gemm_unpacked(const mx_kernel *k, const uint8_t *w, const mx_block_q8_k *xq, float *y, int n_tokens,
                     int out_features, int nb, int n_threads) {
    const int tile_rows = mx_row_tile;
    const int n_tiles = (out_features + tile_rows - 1) / tile_rows;
    int status = 0;
#pragma omp parallel num_threads(n_threads)
    {
        /* One tile of unpacked rows per thread: tile_rows * nb * 328 bytes, e.g. 63 KB for 8 rows of 6144. */
        mx_block_unpacked *wu = malloc((size_t)tile_rows * nb * sizeof(mx_block_unpacked));
        if (wu == NULL) {
#pragma omp atomic write
            status = -2;
        }
#pragma omp for schedule(dynamic, 1)
        for (int tile = 0; tile < n_tiles; ++tile) {
            if (wu == NULL) {
                continue;
            }
            const int o0 = tile * tile_rows;
            const int rows = out_features - o0 < tile_rows ? out_features - o0 : tile_rows;
            for (int r = 0; r < rows; ++r) {
                k->unpack(w + (size_t)(o0 + r) * k->row_bytes, wu + (size_t)r * nb, nb);
            }
            int t = 0;
#if defined(MX_DOT_X4)
            for (; t + 4 <= n_tokens; t += 4) {
                for (int r = 0; r < rows; ++r) {
                    float out[4];
                    mx_dot_unpacked_x4(wu + (size_t)r * nb, xq + (size_t)t * nb, (size_t)nb, nb, k->has_min, out);
                    for (int j = 0; j < 4; ++j) {
                        y[(size_t)(t + j) * out_features + o0 + r] = out[j];
                    }
                }
            }
#endif
            for (; t < n_tokens; ++t) {
                const mx_block_q8_k *xt = xq + (size_t)t * nb;
                float *yt = y + (size_t)t * out_features + o0;
                for (int r = 0; r < rows; ++r) {
                    yt[r] = mx_dot_unpacked(wu + (size_t)r * nb, xt, nb, k->has_min);
                }
            }
        }
        free(wu);
    }
    return status;
}
