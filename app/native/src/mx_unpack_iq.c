/* IQ2_XXS/XS/S and IQ3_XXS/S unpacked to mx_block_unpacked (scales and bounds: mx_iq_unpack.h). */
#include "mx_iq_unpack.h"

static void mx_unpack_iq2_xxs_block(const uint8_t *blk, mx_block_unpacked *b) {
    mx_iq_begin(b, mx_fp16(blk) * 0.125f);
    const uint8_t *qs = blk + 2;
    for (int ib32 = 0; ib32 < 8; ++ib32) {
        const uint8_t *word = qs + ib32 * 8;
        uint32_t word1;
        memcpy(&word1, word + 4, sizeof(word1));
        mx_iq_scale32(b, ib32, 2 * (int)(word1 >> 28) + 1, MX_IQ2_OFF);
        for (int group = 0; group < 4; group += 2) {
            const int8_t *g0 = MX_IQ2XXS_GRID + (size_t)word[group] * 8;
            const int8_t *g1 = MX_IQ2XXS_GRID + (size_t)word[group + 1] * 8;
            mx_iq_put16(b->u + ib32 * 32 + group * 8, g0, g0, g1, g1, 8,
                        MX_KSIGNS_IQ2XS[(word1 >> (7 * group)) & 127],
                        MX_KSIGNS_IQ2XS[(word1 >> (7 * group + 7)) & 127], MX_IQ2_OFF);
        }
    }
}

static void mx_unpack_iq2_xs_block(const uint8_t *blk, mx_block_unpacked *b) {
    mx_iq_begin(b, mx_fp16(blk) * 0.125f);
    const uint8_t *qs_bytes = blk + 2;
    const uint8_t *scales = blk + 66;
    for (int ib32 = 0; ib32 < 8; ++ib32) {
        mx_iq_scale16(b, 2 * ib32, 2 * (scales[ib32] & 0x0F) + 1, MX_IQ2_OFF);
        mx_iq_scale16(b, 2 * ib32 + 1, 2 * (scales[ib32] >> 4) + 1, MX_IQ2_OFF);
        for (int group = 0; group < 4; group += 2) {
            uint16_t qv[2];
            memcpy(qv, qs_bytes + (ib32 * 4 + group) * 2, sizeof(qv));
            const int8_t *g0 = MX_IQ2XS_GRID + (size_t)(qv[0] & 511) * 8;
            const int8_t *g1 = MX_IQ2XS_GRID + (size_t)(qv[1] & 511) * 8;
            mx_iq_put16(b->u + ib32 * 32 + group * 8, g0, g0, g1, g1, 8, MX_KSIGNS_IQ2XS[qv[0] >> 9],
                        MX_KSIGNS_IQ2XS[qv[1] >> 9], MX_IQ2_OFF);
        }
    }
}

static void mx_unpack_iq2_s_block(const uint8_t *blk, mx_block_unpacked *b) {
    mx_iq_begin(b, mx_fp16(blk) * 0.125f);
    const uint8_t *qs_lo = blk + 2;
    const uint8_t *sign_bytes = blk + 34;
    const uint8_t *qh = blk + 66;
    const uint8_t *scales = blk + 74;
    for (int ib32 = 0; ib32 < 8; ++ib32) {
        mx_iq_scale16(b, 2 * ib32, 2 * (scales[ib32] & 0x0F) + 1, MX_IQ2_OFF);
        mx_iq_scale16(b, 2 * ib32 + 1, 2 * (scales[ib32] >> 4) + 1, MX_IQ2_OFF);
        for (int group = 0; group < 4; group += 2) {
            const int idx0 = qs_lo[ib32 * 4 + group] | ((qh[ib32] << (8 - 2 * group)) & 0x300);
            const int idx1 = qs_lo[ib32 * 4 + group + 1] | ((qh[ib32] << (6 - 2 * group)) & 0x300);
            const int8_t *g0 = MX_IQ2S_GRID + (size_t)idx0 * 8;
            const int8_t *g1 = MX_IQ2S_GRID + (size_t)idx1 * 8;
            mx_iq_put16(b->u + ib32 * 32 + group * 8, g0, g0, g1, g1, 8, sign_bytes[ib32 * 4 + group],
                        sign_bytes[ib32 * 4 + group + 1], MX_IQ2_OFF);
        }
    }
}

static void mx_unpack_iq3_xxs_block(const uint8_t *blk, mx_block_unpacked *b) {
    mx_iq_begin(b, mx_fp16(blk) * 0.25f);
    const uint8_t *grid_idx = blk + 2;
    const uint8_t *aux_bytes = blk + 66;
    for (int ib32 = 0; ib32 < 8; ++ib32) {
        uint32_t aux;
        memcpy(&aux, aux_bytes + ib32 * 4, sizeof(aux));
        mx_iq_scale32(b, ib32, 2 * (int)(aux >> 28) + 1, MX_IQ3_XXS_OFF);
        const uint8_t *idx = grid_idx + ib32 * 8;
        for (int group = 0; group < 4; group += 2) {
            mx_iq_put16(b->u + ib32 * 32 + group * 8, MX_IQ3XXS_GRID + (size_t)idx[2 * group] * 4,
                        MX_IQ3XXS_GRID + (size_t)idx[2 * group + 1] * 4,
                        MX_IQ3XXS_GRID + (size_t)idx[2 * group + 2] * 4,
                        MX_IQ3XXS_GRID + (size_t)idx[2 * group + 3] * 4, 4,
                        MX_KSIGNS_IQ2XS[(aux >> (7 * group)) & 127],
                        MX_KSIGNS_IQ2XS[(aux >> (7 * group + 7)) & 127], MX_IQ3_XXS_OFF);
        }
    }
}

static void mx_unpack_iq3_s_block(const uint8_t *blk, mx_block_unpacked *b) {
    mx_iq_begin(b, mx_fp16(blk));
    const uint8_t *qs = blk + 2;
    const uint8_t *qh = blk + 66;
    const uint8_t *signs = blk + 74;
    const uint8_t *scales = blk + 106;
    for (int i2 = 0; i2 < 4; ++i2) {
        for (int half = 0; half < 2; ++half) {
            const int ib32 = i2 * 2 + half;
            const int ls = half == 0 ? (scales[i2] & 0x0F) : (scales[i2] >> 4);
            mx_iq_scale32(b, ib32, 1 + 2 * ls, MX_IQ3_S_OFF);
            const int qh_byte = qh[ib32];
            const uint8_t *qs_off = qs + i2 * 16 + half * 8;
            const uint8_t *sign_off = signs + i2 * 8 + half * 4;
            for (int group = 0; group < 4; group += 2) {
                const int8_t *g[4];
                for (int k = 0; k < 4; ++k) {
                    const int pos = 2 * group + k;
                    g[k] = MX_IQ3S_GRID + (size_t)(qs_off[pos] | ((qh_byte << (8 - pos)) & 256)) * 4;
                }
                mx_iq_put16(b->u + ib32 * 32 + group * 8, g[0], g[1], g[2], g[3], 4, sign_off[group],
                            sign_off[group + 1], MX_IQ3_S_OFF);
            }
        }
    }
}

MX_IQ_UNPACK(iq2_xxs, MX_IQ2_XXS_BYTES)
MX_IQ_UNPACK(iq2_xs, MX_IQ2_XS_BYTES)
MX_IQ_UNPACK(iq2_s, MX_IQ2_S_BYTES)
MX_IQ_UNPACK(iq3_xxs, MX_IQ3_XXS_BYTES)
MX_IQ_UNPACK(iq3_s, MX_IQ3_S_BYTES)

#if !defined(__SSSE3__)
/* Without SSSE3 the IQ2/IQ3 decode goes through the unpacked block too (mx_dot_iq_ssse3.c has the direct ones). */
MX_IQ_FUSED_UNPACKED(iq2_xxs, MX_IQ2_XXS_BYTES)
MX_IQ_FUSED_UNPACKED(iq2_xs, MX_IQ2_XS_BYTES)
MX_IQ_FUSED_UNPACKED(iq2_s, MX_IQ2_S_BYTES)
MX_IQ_FUSED_UNPACKED(iq3_xxs, MX_IQ3_XXS_BYTES)
MX_IQ_FUSED_UNPACKED(iq3_s, MX_IQ3_S_BYTES)
#endif
