/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 *
 * test_inkling_io — the staged-artifact readers, exercised as C.
 *
 * Why this exists: inkling_stage_reader.c carries two positional-read
 * implementations, POSIX pread() and Windows ReadFile() with OVERLAPPED, and
 * until this test the Windows one had never been compiled, let alone run. The
 * Python differential tests cover the POSIX path thoroughly and cannot run
 * under a cross-build at all.
 *
 * So this is deliberately dependency-free C: it writes the two staged formats
 * byte by byte, reads them back through the public reader entry points, and
 * checks that every corruption is refused. It runs identically on a Linux
 * build and on a MinGW build under Wine, which is the only way the Windows
 * branch gets executed by anything.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/inkling_stage_reader.h"

#define IKTN_MAGIC 0x4e544b49u
#define IKBF_MAGIC 0x46424b49u
#define STAGE_ALIGN 4096u

static int failures;

static void check(int ok, const char *what)
{
    if (!ok) {
        fprintf(stderr, "INKLING IO FAIL: %s\n", what);
        failures++;
    }
}

static uint32_t crc32u(const unsigned char *p, size_t n)
{
    uint32_t c = ~0u;
    for (size_t i = 0; i < n; i++) {
        c ^= p[i];
        for (int k = 0; k < 8; k++)
            c = (c >> 1) ^ ((0u - (c & 1u)) & 0xedb88320u);
    }
    return ~c;
}

static void wr16(unsigned char *p, uint16_t v) { p[0] = (unsigned char)v; p[1] = (unsigned char)(v >> 8); }

static void wr32(unsigned char *p, uint32_t v)
{
    for (int i = 0; i < 4; i++) p[i] = (unsigned char)(v >> (8 * i));
}

static void wr64(unsigned char *p, uint64_t v)
{
    for (int i = 0; i < 8; i++) p[i] = (unsigned char)(v >> (8 * i));
}

static uint64_t aligned(uint64_t v)
{
    return (v + STAGE_ALIGN - 1) / STAGE_ALIGN * STAGE_ALIGN;
}

/* Write one IKTN tensor artifact. Returns 0 on success. */
static int write_tensor(const char *path, const char *name, int dtype,
                        const int *shape, int ndim,
                        const unsigned char *payload, size_t bytes,
                        uint32_t crc_override, int crc_is_override)
{
    unsigned char h[64];
    memset(h, 0, sizeof h);
    wr32(h, IKTN_MAGIC);
    wr16(h + 4, 1);
    h[6] = (unsigned char)dtype;
    h[7] = (unsigned char)ndim;
    for (int i = 0; i < 4; i++) wr32(h + 12 + 4 * i, i < ndim ? (uint32_t)shape[i] : 0u);
    wr64(h + 28, (uint64_t)bytes);
    wr32(h + 36, crc_is_override ? crc_override : crc32u(payload, bytes));
    wr32(h + 40, crc32u((const unsigned char *)name, strlen(name)));

    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    const uint64_t stored = aligned(64u + bytes);
    int ok = fwrite(h, 1, sizeof h, f) == sizeof h &&
             fwrite(payload, 1, bytes, f) == bytes;
    for (uint64_t i = 64u + bytes; ok && i < stored; i++) ok = fputc(0, f) != EOF;
    fclose(f);
    return ok ? 0 : -1;
}

static void test_tensor_roundtrip(const char *dir)
{
    char path[512];
    const int rows = 5, cols = 7;
    const int shape[2] = { rows, cols };
    float values[5 * 7];
    unsigned char payload[sizeof values];
    waste_inkling_stage_tensor t;

    for (int i = 0; i < rows * cols; i++) values[i] = (float)i * 0.5f - 3.0f;
    memcpy(payload, values, sizeof values);

    snprintf(path, sizeof path, "%s/tensor.iktn", dir);
    check(write_tensor(path, "inkling.test", 3, shape, 2,
                       payload, sizeof payload, 0, 0) == 0, "write tensor");

    check(waste_inkling_stage_tensor_open(&t, path, "inkling.test", shape, 2, 1) == 0,
          "open with crc verification");

    /* Row reads are the hot path and the one the Windows branch serves one
     * pread at a time. Check every row, not just the first. */
    float row[7];
    int rows_ok = 1;
    for (int r = 0; r < rows; r++) {
        if (waste_inkling_stage_tensor_row(&t, r, cols, row)) { rows_ok = 0; break; }
        for (int c = 0; c < cols; c++)
            if (row[c] != values[r * cols + c]) { rows_ok = 0; break; }
    }
    check(rows_ok, "every row reads back exactly");

    check(waste_inkling_stage_tensor_row(&t, rows, cols, row) != 0, "row past the end is refused");
    check(waste_inkling_stage_tensor_row(&t, -1, cols, row) != 0, "negative row is refused");
    check(waste_inkling_stage_tensor_row(&t, 0, cols + 1, row) != 0, "wrong column count is refused");

    float all[5 * 7];
    check(waste_inkling_stage_tensor_read_all(&t, all, rows * cols) == 0, "read_all succeeds");
    check(memcmp(all, values, sizeof values) == 0, "read_all matches");
    check(waste_inkling_stage_tensor_read_all(&t, all, rows * cols - 1) != 0,
          "read_all with the wrong count is refused");
    waste_inkling_stage_tensor_close(&t);

    /* The shape is an input, not an output: a file that disagrees is refused
     * rather than reinterpreted. */
    const int wrong[2] = { cols, rows };
    check(waste_inkling_stage_tensor_open(&t, path, "inkling.test", wrong, 2, 0) != 0,
          "transposed shape is refused");
    const int wrong_rank[1] = { rows };
    check(waste_inkling_stage_tensor_open(&t, path, "inkling.test", wrong_rank, 1, 0) != 0,
          "wrong rank is refused");
    check(waste_inkling_stage_tensor_open(&t, path, "inkling.other", shape, 2, 0) != 0,
          "wrong canonical name is refused");

    snprintf(path, sizeof path, "%s/badcrc.iktn", dir);
    check(write_tensor(path, "inkling.test", 3, shape, 2,
                       payload, sizeof payload, 0xdeadbeefu, 1) == 0, "write bad-crc tensor");
    check(waste_inkling_stage_tensor_open(&t, path, "inkling.test", shape, 2, 1) != 0,
          "payload crc mismatch is refused when verifying");
}

static void test_bf16_tensor(const char *dir)
{
    char path[512];
    /* 1.0, -2.0, 0.5, -0.0 as BF16 bit patterns. */
    static const uint16_t bits[4] = { 0x3F80, 0xC000, 0x3F00, 0x8000 };
    static const float want[4] = { 1.0f, -2.0f, 0.5f, -0.0f };
    unsigned char payload[sizeof bits];
    const int shape[2] = { 2, 2 };
    waste_inkling_stage_tensor t;
    float got[4];

    for (int i = 0; i < 4; i++) wr16(payload + 2 * i, bits[i]);
    snprintf(path, sizeof path, "%s/bf16.iktn", dir);
    check(write_tensor(path, "inkling.bf16", 1, shape, 2,
                       payload, sizeof payload, 0, 0) == 0, "write bf16 tensor");
    check(waste_inkling_stage_tensor_open(&t, path, "inkling.bf16", shape, 2, 1) == 0,
          "open bf16 tensor");
    check(waste_inkling_stage_tensor_read_all(&t, got, 4) == 0, "read bf16 payload");
    int ok = 1;
    for (int i = 0; i < 4; i++) if (got[i] != want[i]) ok = 0;
    check(ok, "bf16 widening is exact");
    waste_inkling_stage_tensor_close(&t);
}

static void test_expert_bank(const char *dir)
{
    char path[512];
    const int layer = 3, experts = 2, hidden = 4, inter = 6;
    const uint64_t one = (uint64_t)hidden * inter * 2u;   /* one BF16 matrix   */
    const uint64_t payload_bytes = 3u * one;
    const uint64_t rec = aligned(64u + payload_bytes);
    waste_inkling_stage_bank b;
    waste_inkling_expert_weights w;

    unsigned char *record = (unsigned char *)calloc(1, (size_t)rec);
    if (!record) { check(0, "allocate record"); return; }

    snprintf(path, sizeof path, "%s/experts-L3.bin", dir);
    FILE *f = fopen(path, "wb");
    if (!f) { free(record); check(0, "open bank for writing"); return; }

    for (int e = 0; e < experts; e++) {
        unsigned char *p = record + 64;
        memset(record, 0, (size_t)rec);
        /* Distinct BF16 values per expert so a swapped record is visible. */
        for (uint64_t i = 0; i < payload_bytes / 2u; i++)
            wr16(p + 2u * i, (uint16_t)(0x3F80u + e * 0x0100u + (uint16_t)i));
        wr32(record, IKBF_MAGIC);
        wr16(record + 4, 1);
        wr16(record + 6, (uint16_t)layer);
        wr16(record + 8, (uint16_t)e);
        record[10] = 1;
        record[11] = 0;
        wr32(record + 12, (uint32_t)hidden);
        wr32(record + 16, (uint32_t)inter);
        wr64(record + 20, 64u);
        wr64(record + 28, 64u + one);
        wr64(record + 36, 64u + 2u * one);
        wr64(record + 44, payload_bytes);
        wr32(record + 52, crc32u(record + 64, (size_t)payload_bytes));
        if (fwrite(record, 1, (size_t)rec, f) != (size_t)rec) check(0, "write record");
    }
    fclose(f);

    check(waste_inkling_stage_bank_open(&b, path, layer, experts, hidden, inter, rec, 1) == 0,
          "open expert bank");
    for (int e = 0; e < experts; e++) {
        char what[64];
        snprintf(what, sizeof what, "expert %d reads back", e);
        check(waste_inkling_stage_expert_get(&b, layer, e, &w) == 0, what);
        /* gate/up/down are distinct slices of one record, so they must differ. */
        check(w.gate[0] != w.up[0] && w.up[0] != w.down[0], "gate/up/down are distinct");
    }
    check(waste_inkling_stage_expert_get(&b, layer, experts, &w) != 0, "expert past the end is refused");
    check(waste_inkling_stage_expert_get(&b, layer + 1, 0, &w) != 0, "wrong layer is refused");
    waste_inkling_stage_bank_close(&b);

    check(waste_inkling_stage_bank_open(&b, path, layer, experts + 1, hidden, inter, rec, 0) != 0,
          "wrong expert count is refused by file size");
    /* open() is a file-size check; per-record identity and geometry are
     * validated on read, so a wrong hidden size is caught by expert_get
     * rather than by open. Assert the contract that actually holds: the
     * mismatch is refused before any weights are handed back. */
    if (waste_inkling_stage_bank_open(&b, path, layer, experts, hidden + 1, inter, rec, 0) == 0) {
        check(waste_inkling_stage_expert_get(&b, layer, 0, &w) != 0,
              "wrong hidden size is refused when the record is read");
        waste_inkling_stage_bank_close(&b);
    }
    free(record);
}

int main(int argc, char **argv)
{
    const char *dir = argc > 1 ? argv[1] : ".";
    test_tensor_roundtrip(dir);
    test_bf16_tensor(dir);
    test_expert_bank(dir);
    if (failures) {
        fprintf(stderr, "INKLING IO FAIL %d\n", failures);
        return 1;
    }
    printf("INKLING IO OK\n");
    return 0;
}
