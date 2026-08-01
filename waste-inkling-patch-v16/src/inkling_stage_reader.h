/* SPDX-License-Identifier: Apache-2.0 */
#ifndef WASTE_INKLING_STAGE_READER_H
#define WASTE_INKLING_STAGE_READER_H
#include <stddef.h>
#include <stdint.h>
#include "inkling_layer.h"
#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int fd;
    uint64_t payload_off;
    uint64_t payload_bytes;
    uint64_t stored_bytes;
    uint32_t payload_crc32;
    uint32_t shape[4];
    uint32_t rows, cols;
    int ndim;
    int dtype; /* 1 BF16, 2 F16, 3 F32 */
    unsigned char *io;
    size_t io_bytes;
} waste_inkling_stage_tensor;

int waste_inkling_stage_tensor_open(waste_inkling_stage_tensor *t,
                                    const char *path, const char *canonical_name,
                                    const int *shape, int ndim, int verify_crc);
int waste_inkling_stage_tensor_row(void *ctx, int row, int cols, float *out);
int waste_inkling_stage_tensor_read_all(waste_inkling_stage_tensor *t,
                                        float *out, size_t out_count);
void waste_inkling_stage_tensor_close(waste_inkling_stage_tensor *t);

typedef struct {
    int fd;
    int layer, experts, hidden, intermediate;
    uint64_t record_bytes;
    int verify_crc;
    float *gate, *up, *down;
    unsigned char *raw;
    size_t matrix_floats, raw_bytes;
    int owns_workspace;
} waste_inkling_stage_bank;
int waste_inkling_stage_bank_open(waste_inkling_stage_bank *b, const char *path,
                                  int layer, int experts, int hidden,
                                  int intermediate, uint64_t record_bytes,
                                  int verify_crc);
int waste_inkling_stage_bank_open_with_workspace(
    waste_inkling_stage_bank *b, const char *path,
    int layer, int experts, int hidden, int intermediate,
    uint64_t record_bytes, int verify_crc,
    float *gate, float *up, float *down, size_t matrix_floats,
    unsigned char *raw, size_t raw_bytes);
int waste_inkling_stage_expert_get(void *ctx, int layer, int expert,
                                   waste_inkling_expert_weights *out);
void waste_inkling_stage_bank_close(waste_inkling_stage_bank *b);
#ifdef __cplusplus
}
#endif
#endif
