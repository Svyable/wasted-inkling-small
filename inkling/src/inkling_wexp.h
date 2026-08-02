/* SPDX-License-Identifier: Apache-2.0
 * Converter-private WEXP/VQ reader for Inkling routed experts.
 */
#ifndef WASTE_INKLING_WEXP_H
#define WASTE_INKLING_WEXP_H

#include <stddef.h>
#include <stdint.h>

#include "inkling_layer.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int bank_fd;
    int layer, experts, hidden, intermediate;
    int stages, entries, vec_dim, index_block, fmt, codebook_base;
    uint64_t record_bytes;
    int verify_crc;
    float *codebooks; /* [3*stages][entries][vec_dim] */
    float *gate, *up, *down;
    unsigned char *record;
    size_t matrix_floats, record_capacity;
    int owns_workspace;
} waste_inkling_wexp_bank;

int waste_inkling_wexp_bank_open(
    waste_inkling_wexp_bank *bank,
    const char *bank_path,
    const char *codebook_path,
    int layer, int experts, int hidden, int intermediate,
    int stages, int entries, int vec_dim, int index_block,
    int codebook_base, int verify_crc);

/* Open a bank while sharing the large record and dequantization buffers
 * across layers. Codebooks remain per-bank and resident; the caller owns
 * all workspace passed here and must keep it alive until bank_close(). */
int waste_inkling_wexp_bank_open_with_workspace(
    waste_inkling_wexp_bank *bank,
    const char *bank_path,
    const char *codebook_path,
    int layer, int experts, int hidden, int intermediate,
    int stages, int entries, int vec_dim, int index_block,
    int codebook_base, int verify_crc,
    float *gate, float *up, float *down, size_t matrix_floats,
    unsigned char *record, size_t record_capacity);

int waste_inkling_wexp_expert_get(
    void *ctx, int layer, int expert,
    waste_inkling_expert_weights *out);

void waste_inkling_wexp_bank_close(waste_inkling_wexp_bank *bank);

#ifdef __cplusplus
}
#endif
#endif
