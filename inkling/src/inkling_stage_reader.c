/* SPDX-License-Identifier: Apache-2.0 */
#define _POSIX_C_SOURCE 200809L
#include "inkling_stage_reader.h"

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

#define IKTN_MAGIC 0x4e544b49u
#define IKBF_MAGIC 0x46424b49u
#define STAGE_VERSION 1u
#define STAGE_ALIGN 4096u

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

static uint16_t rd16(const unsigned char *p)
{
    return (uint16_t)(p[0] | ((uint16_t)p[1] << 8));
}

static uint32_t rd32(const unsigned char *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static uint64_t rd64(const unsigned char *p)
{
    return (uint64_t)rd32(p) | ((uint64_t)rd32(p + 4) << 32);
}

static float bf16_to_f32(uint16_t x)
{
    union { uint32_t u; float f; } v;
    v.u = (uint32_t)x << 16;
    return v.f;
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
        ssize_t r = pread(fd, p, n, (off_t)off);
        if (r <= 0) return -1;
        p += (size_t)r; n -= (size_t)r; off += (uint64_t)r;
    }
#endif
    return 0;
}

static int file_size(int fd, uint64_t *out)
{
    if (!out) return -1;
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

static int add64(uint64_t a, uint64_t b, uint64_t *out)
{
    if (!out || b > UINT64_MAX - a) return -1;
    *out = a + b;
    return 0;
}

static int mul64(uint64_t a, uint64_t b, uint64_t *out)
{
    if (!out || (a && b > UINT64_MAX / a)) return -1;
    *out = a * b;
    return 0;
}

static int align64(uint64_t value, uint64_t alignment, uint64_t *out)
{
    uint64_t x;
    if (!alignment || add64(value, alignment - 1, &x)) return -1;
    *out = x / alignment * alignment;
    return 0;
}

static int verify_crc_fd(int fd, uint64_t off, uint64_t bytes, uint32_t want)
{
    unsigned char buf[1u << 16];
    uint32_t c = ~0u;
    while (bytes) {
        size_t n = bytes > sizeof buf ? sizeof buf : (size_t)bytes;
        if (preadn(fd, buf, n, off)) return -1;
        c = crc32_update(c, buf, n);
        off += n;
        bytes -= n;
    }
    return ~c == want ? 0 : -1;
}

int waste_inkling_stage_tensor_open(waste_inkling_stage_tensor *t,
                                    const char *path, const char *name,
                                    const int *shape, int ndim, int verify)
{
    unsigned char h[64];
    uint64_t elems = 1, bytes, stored, actual;
    if (!t || !path || !name || !shape || ndim < 1 || ndim > 4) return -1;
    memset(t, 0, sizeof *t);
    t->fd = -1;
    const int fd = OPEN_RO(path);
    if (fd < 0 || preadn(fd, h, sizeof h, 0)) {
        if (fd >= 0) CLOSE_FD(fd);
        return -1;
    }
    if (rd32(h) != IKTN_MAGIC || rd16(h + 4) != STAGE_VERSION ||
        h[7] != (unsigned)ndim || rd16(h + 8) != 0 || rd16(h + 10) != 0) {
        CLOSE_FD(fd); return -1;
    }
    for (int i = 44; i < 64; i++) if (h[i]) { CLOSE_FD(fd); return -1; }
    const int dtype = h[6];
    const uint64_t elem = dtype == 3 ? 4u : (dtype == 1 || dtype == 2 ? 2u : 0u);
    if (!elem) { CLOSE_FD(fd); return -1; }
    for (int i = 0; i < 4; i++) {
        const uint32_t d = rd32(h + 12 + 4 * i);
        if (i < ndim) {
            if (shape[i] <= 0 || d != (uint32_t)shape[i] || mul64(elems, d, &elems)) {
                CLOSE_FD(fd); return -1;
            }
        } else if (d != 0) {
            CLOSE_FD(fd); return -1;
        }
    }
    if (mul64(elems, elem, &bytes) || bytes != rd64(h + 28) ||
        add64(64u, bytes, &stored) || align64(stored, STAGE_ALIGN, &stored) ||
        file_size(fd, &actual) || actual != stored ||
        rd32(h + 40) != crc32u((const unsigned char *)name, strlen(name))) {
        CLOSE_FD(fd); return -1;
    }
    if (verify && verify_crc_fd(fd, 64u, bytes, rd32(h + 36))) {
        CLOSE_FD(fd); return -1;
    }
    t->fd = fd;
    t->payload_off = 64u;
    t->payload_bytes = bytes;
    t->stored_bytes = stored;
    t->payload_crc32 = rd32(h + 36);
    t->ndim = ndim;
    t->dtype = dtype;
    for (int i = 0; i < ndim; i++) t->shape[i] = (uint32_t)shape[i];
    t->rows = t->shape[0];
    t->cols = ndim == 1 ? 1u : t->shape[1];
    if (ndim == 2 && t->cols > 0) {
        uint64_t rb;
        if (mul64(t->cols, elem, &rb) || rb > SIZE_MAX) {
            waste_inkling_stage_tensor_close(t); return -1;
        }
        t->io = (unsigned char *)malloc((size_t)rb);
        if (!t->io) { waste_inkling_stage_tensor_close(t); return -1; }
        t->io_bytes = (size_t)rb;
    }
    return 0;
}

static void convert_values(const unsigned char *src, int dtype,
                           float *out, size_t count)
{
    if (dtype == 3) {
        for (size_t i = 0; i < count; i++) {
            union { uint32_t u; float f; } v;
            v.u = rd32(src + 4 * i);
            out[i] = v.f;
        }
    } else {
        for (size_t i = 0; i < count; i++) {
            const uint16_t x = rd16(src + 2 * i);
            out[i] = dtype == 1 ? bf16_to_f32(x) : f16_to_f32(x);
        }
    }
}

int waste_inkling_stage_tensor_row(void *ctx, int row, int cols, float *out)
{
    waste_inkling_stage_tensor *t = (waste_inkling_stage_tensor *)ctx;
    if (!t || t->fd < 0 || !out || t->ndim != 2 || row < 0 ||
        row >= (int)t->rows || cols != (int)t->cols || !t->io) return -1;
    const size_t elem = t->dtype == 3 ? 4u : 2u;
    const size_t n = (size_t)cols * elem;
    if (n != t->io_bytes || preadn(t->fd, t->io, n,
        t->payload_off + (uint64_t)row * n)) return -1;
    convert_values(t->io, t->dtype, out, (size_t)cols);
    return 0;
}

int waste_inkling_stage_tensor_read_all(waste_inkling_stage_tensor *t,
                                        float *out, size_t out_count)
{
    if (!t || t->fd < 0 || !out) return -1;
    const size_t elem = t->dtype == 3 ? 4u : 2u;
    if (t->payload_bytes % elem || t->payload_bytes / elem != out_count) return -1;
    const size_t block_values = 1u << 15;
    unsigned char *buf = (unsigned char *)malloc(block_values * elem);
    if (!buf) return -1;
    uint64_t off = t->payload_off;
    size_t done = 0;
    while (done < out_count) {
        size_t n = out_count - done;
        if (n > block_values) n = block_values;
        if (preadn(t->fd, buf, n * elem, off)) { free(buf); return -1; }
        convert_values(buf, t->dtype, out + done, n);
        done += n;
        off += (uint64_t)n * elem;
    }
    free(buf);
    return 0;
}

void waste_inkling_stage_tensor_close(waste_inkling_stage_tensor *t)
{
    if (!t) return;
    if (t->fd >= 0) CLOSE_FD(t->fd);
    free(t->io);
    memset(t, 0, sizeof *t);
    t->fd = -1;
}

int waste_inkling_stage_bank_open_with_workspace(
    waste_inkling_stage_bank *b, const char *path,
    int layer, int experts, int hidden, int inter,
    uint64_t rec, int verify,
    float *gate, float *up, float *down, size_t matrix_floats,
    unsigned char *raw, size_t raw_bytes)
{
    uint64_t matrix, payload, expected, actual;
    if (!b || !path || layer < 0 || experts < 1 || hidden < 1 || inter < 1 ||
        rec < 64 || rec % STAGE_ALIGN || !gate || !up || !down || !raw ||
        mul64((uint64_t)hidden, (uint64_t)inter, &matrix) ||
        matrix > SIZE_MAX || matrix_floats < (size_t)matrix ||
        mul64(matrix, 6u, &payload) || payload > SIZE_MAX || raw_bytes < (size_t)payload ||
        mul64(rec, (uint64_t)experts, &expected)) return -1;
    memset(b, 0, sizeof *b);
    b->fd = -1;
    const int fd = OPEN_RO(path);
    if (fd < 0 || file_size(fd, &actual) || actual != expected) {
        if (fd >= 0) CLOSE_FD(fd);
        return -1;
    }
    b->fd = fd;
    b->layer = layer;
    b->experts = experts;
    b->hidden = hidden;
    b->intermediate = inter;
    b->record_bytes = rec;
    b->verify_crc = verify;
    b->gate = gate; b->up = up; b->down = down;
    b->raw = raw;
    b->matrix_floats = matrix_floats;
    b->raw_bytes = raw_bytes;
    return 0;
}

int waste_inkling_stage_bank_open(waste_inkling_stage_bank *b, const char *path,
                                  int layer, int experts, int hidden,
                                  int inter, uint64_t rec, int verify)
{
    uint64_t matrix, payload;
    if (!b || mul64((uint64_t)hidden, (uint64_t)inter, &matrix) ||
        matrix > SIZE_MAX / sizeof(float) || mul64(matrix, 6u, &payload) ||
        payload > SIZE_MAX) return -1;
    float *gate = (float *)malloc((size_t)matrix * sizeof(float));
    float *up = (float *)malloc((size_t)matrix * sizeof(float));
    float *down = (float *)malloc((size_t)matrix * sizeof(float));
    unsigned char *raw = (unsigned char *)malloc((size_t)payload);
    if (!gate || !up || !down || !raw ||
        waste_inkling_stage_bank_open_with_workspace(
            b, path, layer, experts, hidden, inter, rec, verify,
            gate, up, down, (size_t)matrix, raw, (size_t)payload)) {
        free(gate); free(up); free(down); free(raw);
        return -1;
    }
    b->owns_workspace = 1;
    return 0;
}

int waste_inkling_stage_expert_get(void *ctx, int layer, int expert,
                                   waste_inkling_expert_weights *out)
{
    waste_inkling_stage_bank *b = (waste_inkling_stage_bank *)ctx;
    unsigned char h[64];
    uint64_t base, one, payload;
    if (!b || !out || b->fd < 0 || layer != b->layer || expert < 0 ||
        expert >= b->experts || mul64((uint64_t)b->hidden,
        (uint64_t)b->intermediate, &one) || mul64(one, 2u, &one) ||
        mul64(one, 3u, &payload) || payload > b->raw_bytes ||
        mul64((uint64_t)expert, b->record_bytes, &base) ||
        preadn(b->fd, h, sizeof h, base)) return -1;
    if (rd32(h) != IKBF_MAGIC || rd16(h + 4) != STAGE_VERSION ||
        rd16(h + 6) != (uint16_t)layer || rd16(h + 8) != (uint16_t)expert ||
        h[10] != 1 || h[11] != 0 || rd32(h + 12) != (uint32_t)b->hidden ||
        rd32(h + 16) != (uint32_t)b->intermediate || rd64(h + 20) != 64u ||
        rd64(h + 28) != 64u + one || rd64(h + 36) != 64u + 2u * one ||
        rd64(h + 44) != payload || 64u + payload > b->record_bytes) return -1;
    for (int i = 56; i < 64; i++) if (h[i]) return -1;
    if (preadn(b->fd, b->raw, (size_t)payload, base + 64u) ||
        (b->verify_crc && crc32u(b->raw, (size_t)payload) != rd32(h + 52))) return -1;
    const size_t n = (size_t)(one / 2u);
    for (size_t i = 0; i < n; i++) {
        b->gate[i] = bf16_to_f32(rd16(b->raw + 2u * i));
        b->up[i] = bf16_to_f32(rd16(b->raw + one + 2u * i));
        b->down[i] = bf16_to_f32(rd16(b->raw + 2u * one + 2u * i));
    }
    out->gate = b->gate; out->up = b->up; out->down = b->down;
    return 0;
}

void waste_inkling_stage_bank_close(waste_inkling_stage_bank *b)
{
    if (!b) return;
    if (b->fd >= 0) CLOSE_FD(b->fd);
    if (b->owns_workspace) {
        free(b->gate); free(b->up); free(b->down); free(b->raw);
    }
    memset(b, 0, sizeof *b);
    b->fd = -1;
}
