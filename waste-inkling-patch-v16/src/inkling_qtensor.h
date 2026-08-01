/* SPDX-License-Identifier: Apache-2.0 */
#ifndef WASTE_INKLING_QTENSOR_H
#define WASTE_INKLING_QTENSOR_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int fd, fmt, ndim, group;
    int shape[4];
    uint64_t rows, cols, qbytes, scale_bytes, scale_off;
    size_t rowbytes, groups;
    uint32_t q_crc32, scale_crc32;
    int verify_crc;
    unsigned char *qrow;
    uint16_t *scales;
    unsigned char *qdata;   /* optional resident payload */
    uint16_t *scale_data;   /* optional resident scales */
} waste_inkling_qtensor;

int waste_inkling_qtensor_open(waste_inkling_qtensor *t, const char *path,
                               const char *name, const int *shape, int ndim,
                               int verify_crc);
void waste_inkling_qtensor_close(waste_inkling_qtensor *t);
int waste_inkling_qtensor_load_resident(waste_inkling_qtensor *t);
size_t waste_inkling_qtensor_resident_bytes(const waste_inkling_qtensor *t);
int waste_inkling_qtensor_row(waste_inkling_qtensor *t, uint64_t row,
                              float *out, size_t cols);
int waste_inkling_qtensor_matvec_rows(waste_inkling_qtensor *t, const float *x,
                                      float *out, size_t row0, size_t rows,
                                      size_t cols);
int waste_inkling_qtensor_matvec(waste_inkling_qtensor *t, const float *x,
                                 float *out, size_t rows, size_t cols);
#ifdef __cplusplus
}
#endif
#endif
