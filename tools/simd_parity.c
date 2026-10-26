/* SIMD parity harness: runs mx_gemm on fixed pseudo-random weights for every unpacking type and several token counts
 * (per-token kernels, the 4-token pass and its remainder) and prints each result's bits as hex. Built once per SIMD
 * variant by tools/check_simd_parity.py; equal output means bit-identical kernels. Each block's f16 scale is set to
 * a valid value, so no result is NaN. IQ1_M is left out: its scale is spread over packed bit fields. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../app/native/src/mx_gemm.h"

int mx_gemm(int ggml_type, const uint8_t *w, const float *x, float *y, int n_tokens, int out_features,
            int in_features, int n_threads);

static uint32_t rng_state = 12345u;

static uint32_t rng(void) {
    rng_state = rng_state * 1664525u + 1013904223u;
    return rng_state >> 8;
}

/* A small positive normal f16 (about 6e-5 .. 1). */
static void put_f16(uint8_t *p) {
    const uint16_t bits = (uint16_t)(0x0400 + rng() % 0x3800);
    memcpy(p, &bits, 2);
}

typedef struct {
    int type;
    int f16_offsets[2];
} type_case;

int main(void) {
    const type_case cases[] = {
        {MX_TYPE_Q3_K, {108, -1}}, {MX_TYPE_Q4_K, {0, 2}},     {MX_TYPE_Q5_K, {0, 2}},
        {MX_TYPE_Q6_K, {208, -1}}, {MX_TYPE_IQ2_XXS, {0, -1}}, {MX_TYPE_IQ2_XS, {0, -1}},
        {MX_TYPE_IQ2_S, {0, -1}},  {MX_TYPE_IQ3_XXS, {0, -1}}, {MX_TYPE_IQ3_S, {0, -1}},
        {MX_TYPE_IQ1_S, {0, -1}},
    };
    const int token_counts[] = {1, 3, 4, 5, 8, 9, 13, 37};
    const int out_f = 24, in_f = 512, n_threads = 2;
    for (size_t c = 0; c < sizeof cases / sizeof cases[0]; ++c) {
        mx_kernel k;
        if (!mx_supports(cases[c].type, in_f) || !mx_kernel_for(cases[c].type, in_f, &k)) {
            printf("type %d unsupported\n", cases[c].type);
            continue;
        }
        const size_t block_bytes = k.row_bytes / (size_t)(in_f / k.block_size);
        uint8_t *w = malloc(k.row_bytes * out_f);
        for (size_t b = 0; b < k.row_bytes * out_f; ++b) {
            w[b] = (uint8_t)rng();
        }
        for (size_t blk = 0; blk < k.row_bytes * out_f / block_bytes; ++blk) {
            for (int j = 0; j < 2; ++j) {
                if (cases[c].f16_offsets[j] >= 0) {
                    put_f16(w + blk * block_bytes + cases[c].f16_offsets[j]);
                }
            }
        }
        for (size_t n = 0; n < sizeof token_counts / sizeof token_counts[0]; ++n) {
            const int tokens = token_counts[n];
            float *x = malloc(sizeof(float) * tokens * in_f), *y = malloc(sizeof(float) * tokens * out_f);
            for (int i = 0; i < tokens * in_f; ++i) {
                x[i] = (float)((int)(rng() % 4001) - 2000) / 1000.0f;
            }
            const int status = mx_gemm(cases[c].type, w, x, y, tokens, out_f, in_f, n_threads);
            printf("type %d tokens %d status %d:", cases[c].type, tokens, status);
            for (int i = 0; i < tokens * out_f; ++i) {
                uint32_t bits;
                memcpy(&bits, &y[i], 4);
                printf(" %08x", bits);
            }
            printf("\n");
            free(x);
            free(y);
        }
        /* Decode vs prefill inside this build: each token alone must equal the same token in a 9-token call. */
        const int tokens = 9;
        float *x = malloc(sizeof(float) * tokens * in_f), *all = malloc(sizeof(float) * tokens * out_f);
        float *one = malloc(sizeof(float) * out_f);
        for (int i = 0; i < tokens * in_f; ++i) {
            x[i] = (float)((int)(rng() % 4001) - 2000) / 1000.0f;
        }
        mx_gemm(cases[c].type, w, x, all, tokens, out_f, in_f, n_threads);
        int same = 0;
        for (int t = 0; t < tokens; ++t) {
            mx_gemm(cases[c].type, w, x + (size_t)t * in_f, one, 1, out_f, in_f, n_threads);
            same += memcmp(one, all + (size_t)t * out_f, sizeof(float) * out_f) == 0;
        }
        printf("type %d decode-equals-prefill %d/%d\n", cases[c].type, same, tokens);
        free(x);
        free(all);
        free(one);
        free(w);
    }
    return 0;
}
