/* IQ1_S/IQ1_M unpacked to mx_block_unpacked. Their grid has -1, so decode also goes through the
 * unpacked block (no direct kernel as for IQ2/IQ3). Scales: mx_iq_unpack.h. */
#include "mx_iq_unpack.h"

static void mx_unpack_iq1_s_block(const uint8_t *blk, mx_block_unpacked *b) {
    mx_iq_begin(b, mx_fp16(blk) * 0.125f);
    const uint8_t *qs = blk + 2;
    const uint8_t *qh_bytes = blk + 34;
    for (int ib = 0; ib < 8; ++ib) {
        uint16_t qh;
        memcpy(&qh, qh_bytes + ib * 2, sizeof(qh));
        mx_iq_scale32(b, ib, 2 * ((qh >> 12) & 7) + 1, MX_IQ1_OFF);
        const int delta = (qh & 0x8000) ? -1 : 1;
        for (int group = 0; group < 4; group += 2) {
            const int idx0 = qs[ib * 4 + group] | (((qh >> (3 * group)) & 7) << 8);
            const int idx1 = qs[ib * 4 + group + 1] | (((qh >> (3 * group + 3)) & 7) << 8);
            mx_iq1_put16(b->u + ib * 32 + group * 8, MX_IQ1S_GRID + (size_t)idx0 * 8,
                         MX_IQ1S_GRID + (size_t)idx1 * 8, delta, delta);
        }
    }
}

static void mx_unpack_iq1_m_block(const uint8_t *blk, mx_block_unpacked *b) {
    const uint8_t *qs = blk;
    const uint8_t *qh = blk + 32;
    uint16_t sc[4];
    memcpy(sc, blk + 48, sizeof(sc));
    /* No `d` field: the fp16 scale is rebuilt from the top nibble of each `sc` word. */
    const uint16_t scale_u16 = (uint16_t)((sc[0] >> 12) | ((sc[1] >> 8) & 0xF0) | ((sc[2] >> 4) & 0xF00) |
                                          (sc[3] & 0xF000));
    uint8_t scale_bytes[2];
    memcpy(scale_bytes, &scale_u16, sizeof(scale_u16));
    mx_iq_begin(b, mx_fp16(scale_bytes) * 0.125f);
    for (int ib = 0; ib < 8; ++ib) {
        const uint16_t sc_i2 = sc[ib / 2];
        const int shift = 6 * (ib % 2);
        mx_iq_scale16(b, 2 * ib, 2 * ((sc_i2 >> shift) & 7) + 1, MX_IQ1_OFF);
        mx_iq_scale16(b, 2 * ib + 1, 2 * ((sc_i2 >> (shift + 3)) & 7) + 1, MX_IQ1_OFF);
        const int qh0 = qh[ib * 2];
        const int qh1 = qh[ib * 2 + 1];
        const int idx[4] = {
            qs[ib * 4 + 0] | ((qh0 << 8) & 0x700),
            qs[ib * 4 + 1] | ((qh0 << 4) & 0x700),
            qs[ib * 4 + 2] | ((qh1 << 8) & 0x700),
            qs[ib * 4 + 3] | ((qh1 << 4) & 0x700),
        };
        const int delta[4] = {
            (qh0 & 0x08) ? -1 : 1,
            (qh0 & 0x80) ? -1 : 1,
            (qh1 & 0x08) ? -1 : 1,
            (qh1 & 0x80) ? -1 : 1,
        };
        for (int g = 0; g < 4; g += 2) {
            mx_iq1_put16(b->u + ib * 32 + g * 8, MX_IQ1S_GRID + (size_t)idx[g] * 8,
                         MX_IQ1S_GRID + (size_t)idx[g + 1] * 8, delta[g], delta[g + 1]);
        }
    }
}

MX_IQ_UNPACK(iq1_s, MX_IQ1_S_BYTES)
MX_IQ_UNPACK(iq1_m, MX_IQ1_M_BYTES)
MX_IQ_FUSED_UNPACKED(iq1_s, MX_IQ1_S_BYTES)
MX_IQ_FUSED_UNPACKED(iq1_m, MX_IQ1_M_BYTES)
