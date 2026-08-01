/* SPDX-License-Identifier: Apache-2.0 */
#define _POSIX_C_SOURCE 200809L
#include "inkling_wexp.h"

#include <fcntl.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <io.h>
#include <sys/stat.h>
#include <windows.h>
#ifndef O_BINARY
#define O_BINARY _O_BINARY
#endif
#define OPEN_RO(path) _open((path), _O_RDONLY | O_BINARY)
#define CLOSE_FD(fd) _close(fd)
#else
#include <sys/stat.h>
#include <unistd.h>
#ifndef O_BINARY
#define O_BINARY 0
#endif
#define OPEN_RO(path) open((path), O_RDONLY | O_BINARY)
#define CLOSE_FD(fd) close(fd)
#endif

#define WEXP_MAGIC 0x50584557u
#define WCBK_MAGIC 0x4b424357u
#define WQ_VQ3R 4u
#define WQ_VQ2R 5u
#define WASTE_ALIGN 4096u
#define WEXP_HEADER 48u
#define WCBK_HEADER 16u

static uint16_t rd16(const unsigned char *p)
{
    return (uint16_t)(p[0] | ((uint16_t)p[1] << 8));
}

static uint32_t rd32(const unsigned char *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static float f16_to_f32(uint16_t h)
{
    unsigned s = h >> 15, e = (h >> 10) & 31u, m = h & 1023u;
    union { uint32_t u; float f; } v;
    if (!e) {
        if (!m) v.u = s << 31;
        else {
            int ee = 1;
            while (!(m & 1024u)) { m <<= 1; ee--; }
            m &= 1023u;
            v.u = (s << 31) | ((uint32_t)(ee + 112) << 23) | (m << 13);
        }
    } else if (e == 31) {
        v.u = (s << 31) | 0x7f800000u | (m << 13);
    } else {
        v.u = (s << 31) | ((e + 112u) << 23) | (m << 13);
    }
    return v.f;
}

static uint32_t crc32_update(uint32_t c, const unsigned char *p, size_t n)
{
    for (size_t i = 0; i < n; i++) {
        c ^= p[i];
        for (int k = 0; k < 8; k++)
            c = (c >> 1) ^ ((0u - (c & 1u)) & 0xedb88320u);
    }
    return c;
}

static uint32_t crc32u(const unsigned char *p, size_t n)
{
    return ~crc32_update(~0u, p, n);
}

static int preadn(int fd, void *buf, size_t n, uint64_t off)
{
    unsigned char *p = (unsigned char *)buf;
#ifdef _WIN32
    HANDLE h = (HANDLE)_get_osfhandle(fd);
    if (h == INVALID_HANDLE_VALUE) return -1;
    while (n) {
        DWORD ask = n > 0x7fffffffu ? 0x7fffffffu : (DWORD)n;
        DWORD got = 0;
        OVERLAPPED ov;
        memset(&ov, 0, sizeof ov);
        ov.Offset = (DWORD)off;
        ov.OffsetHigh = (DWORD)(off >> 32);
        if (!ReadFile(h, p, ask, &got, &ov) || got == 0) return -1;
        p += got; n -= got; off += got;
    }
#else
    while (n) {
        ssize_t got = pread(fd, p, n, (off_t)off);
        if (got <= 0) return -1;
        p += (size_t)got; n -= (size_t)got; off += (uint64_t)got;
    }
#endif
    return 0;
}

static int file_size(int fd, uint64_t *out)
{
#ifdef _WIN32
    struct _stat64 st;
    if (_fstat64(fd, &st) || st.st_size < 0) return -1;
    *out = (uint64_t)st.st_size;
#else
    struct stat st;
    if (fstat(fd, &st) || st.st_size < 0) return -1;
    *out = (uint64_t)st.st_size;
#endif
    return 0;
}

static int mul_size(size_t a, size_t b, size_t *out)
{
    if (!out || (a && b > SIZE_MAX / a)) return -1;
    *out = a * b;
    return 0;
}

static size_t padded_rows(int rows, int block)
{
    return ((size_t)rows + (size_t)block - 1u) / (size_t)block * (size_t)block;
}

static int index_bytes(int rows, int cols, int vec_dim, int stages,
                       int block, size_t *out)
{
    size_t value;
    if (rows <= 0 || cols <= 0 || vec_dim <= 0 || stages <= 0 || block <= 0 ||
        cols % vec_dim || mul_size(padded_rows(rows, block),
                                  (size_t)(cols / vec_dim), &value) ||
        mul_size(value, (size_t)stages, &value)) return -1;
    *out = value;
    return 0;
}

static int load_codebooks(waste_inkling_wexp_bank *b, const char *path)
{
    const int fd = OPEN_RO(path);
    const size_t count = (size_t)3 * (size_t)b->stages;
    size_t values, payload, record;
    uint64_t actual;
    unsigned char header[WCBK_HEADER];
    if (fd < 0 || mul_size((size_t)b->entries, (size_t)b->vec_dim, &values) ||
        mul_size(values, 2u, &payload) || payload > SIZE_MAX - WCBK_HEADER) {
        if (fd >= 0) CLOSE_FD(fd);
        return -1;
    }
    record = WCBK_HEADER + payload;
    if (file_size(fd, &actual) || actual != (uint64_t)record * count ||
        mul_size(count, values, &values) || values > SIZE_MAX / sizeof(float) ||
        !(b->codebooks = (float *)malloc(values * sizeof(float)))) {
        CLOSE_FD(fd); return -1;
    }
    unsigned char *raw = (unsigned char *)malloc(payload);
    if (!raw) { CLOSE_FD(fd); return -1; }
    for (size_t i = 0; i < count; i++) {
        const uint64_t off = (uint64_t)i * record;
        if (preadn(fd, header, sizeof header, off) ||
            rd32(header) != WCBK_MAGIC ||
            rd16(header + 4) != (uint16_t)(b->codebook_base + (int)i) ||
            header[6] != (unsigned)b->fmt || header[7] != (unsigned)b->vec_dim ||
            rd32(header + 8) != (uint32_t)b->entries || rd32(header + 12) != 0 ||
            preadn(fd, raw, payload, off + WCBK_HEADER)) {
            free(raw); CLOSE_FD(fd); return -1;
        }
        float *dst = b->codebooks + i * (size_t)b->entries * (size_t)b->vec_dim;
        for (size_t j = 0; j < payload / 2u; j++) dst[j] = f16_to_f32(rd16(raw + 2u * j));
    }
    free(raw);
    CLOSE_FD(fd);
    return 0;
}

static int record_identity(const waste_inkling_wexp_bank *b,
                           const unsigned char *h, int expert,
                           size_t *scale_end)
{
    size_t gate_idx, up_idx, down_idx;
    if (index_bytes(b->intermediate, b->hidden, b->vec_dim, b->stages,
                    b->index_block, &gate_idx) ||
        index_bytes(b->intermediate, b->hidden, b->vec_dim, b->stages,
                    b->index_block, &up_idx) ||
        index_bytes(b->hidden, b->intermediate, b->vec_dim, b->stages,
                    b->index_block, &down_idx)) return -1;
    const size_t gate_off = WEXP_HEADER;
    const size_t up_off = gate_off + gate_idx;
    const size_t down_off = up_off + up_idx;
    const size_t corr_off = down_off + down_idx;
    const size_t end = corr_off + (size_t)(2 * b->intermediate + b->hidden) * 2u;
    if (rd32(h) != WEXP_MAGIC || rd16(h + 4) != (uint16_t)b->layer ||
        rd16(h + 6) != (uint16_t)expert || h[8] != (unsigned)b->fmt || h[9] != 0 ||
        rd16(h + 10) != (uint16_t)b->codebook_base || rd16(h + 12) != 0 ||
        rd16(h + 14) != 0 || rd32(h + 16) * WASTE_ALIGN != b->record_bytes ||
        rd32(h + 20) != gate_off || rd32(h + 24) != up_off ||
        rd32(h + 28) != down_off || rd32(h + 32) != corr_off ||
        rd32(h + 40) != 0 || rd32(h + 44) != 0 || end > b->record_bytes)
        return -1;
    *scale_end = end;
    return 0;
}

static void decode_matrix(const waste_inkling_wexp_bank *b,
                          int kind, int rows, int cols,
                          size_t index_off, size_t scale_off, float *out)
{
    const int nvr = cols / b->vec_dim;
    const unsigned char *idx = b->record + index_off;
    const unsigned char *scales = b->record + scale_off;
    for (int row = 0; row < rows; row++) {
        const int block = row / b->index_block;
        const int in_block = row % b->index_block;
        const float scale = f16_to_f32(rd16(scales + (size_t)row * 2u));
        for (int vector = 0; vector < nvr; vector++) {
            float *dst = out + ((size_t)row * (size_t)nvr + (size_t)vector) * (size_t)b->vec_dim;
            for (int d = 0; d < b->vec_dim; d++) dst[d] = 0.0f;
            const size_t base = (((size_t)block * (size_t)nvr + (size_t)vector) *
                                 (size_t)b->index_block + (size_t)in_block) *
                                (size_t)b->stages;
            for (int stage = 0; stage < b->stages; stage++) {
                const unsigned code = idx[base + (size_t)stage];
                const float *book = b->codebooks +
                    ((size_t)kind * (size_t)b->stages + (size_t)stage) *
                    (size_t)b->entries * (size_t)b->vec_dim +
                    (size_t)code * (size_t)b->vec_dim;
                for (int d = 0; d < b->vec_dim; d++) dst[d] += book[d];
            }
            for (int d = 0; d < b->vec_dim; d++) dst[d] *= scale;
        }
    }
}

static int bank_open_common(
    waste_inkling_wexp_bank *b, const char *bank_path, const char *codebook_path,
    int layer, int experts, int hidden, int intermediate,
    int stages, int entries, int vec_dim, int index_block,
    int codebook_base, int verify_crc,
    float *gate, float *up, float *down, size_t matrix_floats,
    unsigned char *record, size_t record_capacity, int owns_workspace)
{
    unsigned char h[WEXP_HEADER];
    uint64_t actual;
    size_t matrix;
    if (!b || !bank_path || !codebook_path || layer < 0 || experts <= 0 ||
        hidden <= 0 || intermediate <= 0 || (stages != 2 && stages != 3) ||
        entries <= 1 || entries > 256 || vec_dim <= 0 || vec_dim > 64 ||
        index_block <= 0 || codebook_base < 0 || codebook_base > 65535 ||
        hidden % vec_dim || intermediate % vec_dim ||
        mul_size((size_t)hidden, (size_t)intermediate, &matrix)) return -1;
    if (!owns_workspace && (!gate || !up || !down || !record ||
                            matrix_floats < matrix)) return -1;
    memset(b, 0, sizeof *b);
    b->bank_fd = -1;
    b->layer = layer; b->experts = experts; b->hidden = hidden;
    b->intermediate = intermediate; b->stages = stages; b->entries = entries;
    b->vec_dim = vec_dim; b->index_block = index_block;
    b->fmt = stages == 3 ? WQ_VQ3R : WQ_VQ2R;
    b->codebook_base = codebook_base; b->verify_crc = verify_crc;
    b->matrix_floats = matrix;
    b->owns_workspace = owns_workspace;
    if (load_codebooks(b, codebook_path)) goto fail;
    b->bank_fd = OPEN_RO(bank_path);
    if (b->bank_fd < 0 || preadn(b->bank_fd, h, sizeof h, 0) ||
        rd32(h) != WEXP_MAGIC || rd32(h + 16) == 0) goto fail;
    b->record_bytes = (uint64_t)rd32(h + 16) * WASTE_ALIGN;
    if (b->record_bytes > SIZE_MAX ||
        file_size(b->bank_fd, &actual) ||
        actual != b->record_bytes * (uint64_t)experts) goto fail;
    size_t scale_end;
    if (record_identity(b, h, 0, &scale_end)) goto fail;
    (void)scale_end;
    if (owns_workspace) {
        b->record_capacity = (size_t)b->record_bytes;
        b->record = (unsigned char *)malloc(b->record_capacity);
        b->gate = (float *)malloc(matrix * sizeof(float));
        b->up = (float *)malloc(matrix * sizeof(float));
        b->down = (float *)malloc(matrix * sizeof(float));
        if (!b->record || !b->gate || !b->up || !b->down) goto fail;
    } else {
        if (record_capacity < (size_t)b->record_bytes) goto fail;
        b->record_capacity = record_capacity;
        b->record = record;
        b->gate = gate; b->up = up; b->down = down;
    }
    return 0;
fail:
    waste_inkling_wexp_bank_close(b);
    return -1;
}

int waste_inkling_wexp_bank_open(
    waste_inkling_wexp_bank *b, const char *bank_path, const char *codebook_path,
    int layer, int experts, int hidden, int intermediate,
    int stages, int entries, int vec_dim, int index_block,
    int codebook_base, int verify_crc)
{
    return bank_open_common(b, bank_path, codebook_path, layer, experts,
                            hidden, intermediate, stages, entries, vec_dim,
                            index_block, codebook_base, verify_crc,
                            NULL, NULL, NULL, 0, NULL, 0, 1);
}

int waste_inkling_wexp_bank_open_with_workspace(
    waste_inkling_wexp_bank *b, const char *bank_path, const char *codebook_path,
    int layer, int experts, int hidden, int intermediate,
    int stages, int entries, int vec_dim, int index_block,
    int codebook_base, int verify_crc,
    float *gate, float *up, float *down, size_t matrix_floats,
    unsigned char *record, size_t record_capacity)
{
    return bank_open_common(b, bank_path, codebook_path, layer, experts,
                            hidden, intermediate, stages, entries, vec_dim,
                            index_block, codebook_base, verify_crc,
                            gate, up, down, matrix_floats,
                            record, record_capacity, 0);
}

int waste_inkling_wexp_expert_get(void *ctx, int layer, int expert,
                                  waste_inkling_expert_weights *out)
{
    waste_inkling_wexp_bank *b = (waste_inkling_wexp_bank *)ctx;
    size_t scale_end, gate_idx, up_idx, down_idx;
    if (!b || !out || b->bank_fd < 0 || layer != b->layer || expert < 0 ||
        expert >= b->experts ||
        preadn(b->bank_fd, b->record, (size_t)b->record_bytes,
               (uint64_t)expert * b->record_bytes) ||
        record_identity(b, b->record, expert, &scale_end) ||
        (b->verify_crc && crc32u(b->record + WEXP_HEADER, scale_end - WEXP_HEADER) !=
                          rd32(b->record + 36)) ||
        index_bytes(b->intermediate, b->hidden, b->vec_dim, b->stages,
                    b->index_block, &gate_idx) ||
        index_bytes(b->intermediate, b->hidden, b->vec_dim, b->stages,
                    b->index_block, &up_idx) ||
        index_bytes(b->hidden, b->intermediate, b->vec_dim, b->stages,
                    b->index_block, &down_idx)) return -1;
    const size_t gate_off = WEXP_HEADER;
    const size_t up_off = gate_off + gate_idx;
    const size_t down_off = up_off + up_idx;
    const size_t corr_off = down_off + down_idx;
    decode_matrix(b, 0, b->intermediate, b->hidden, gate_off, corr_off, b->gate);
    decode_matrix(b, 1, b->intermediate, b->hidden, up_off,
                  corr_off + (size_t)b->intermediate * 2u, b->up);
    decode_matrix(b, 2, b->hidden, b->intermediate, down_off,
                  corr_off + (size_t)b->intermediate * 4u, b->down);
    out->gate = b->gate; out->up = b->up; out->down = b->down;
    return 0;
}

void waste_inkling_wexp_bank_close(waste_inkling_wexp_bank *b)
{
    if (!b) return;
    if (b->bank_fd >= 0) CLOSE_FD(b->bank_fd);
    free(b->codebooks);
    if (b->owns_workspace) {
        free(b->gate); free(b->up); free(b->down); free(b->record);
    }
    memset(b, 0, sizeof *b);
    b->bank_fd = -1;
}
