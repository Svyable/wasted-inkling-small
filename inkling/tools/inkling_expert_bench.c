/* SPDX-License-Identifier: Apache-2.0
 *
 * inkling_expert_bench.c — what one routed expert costs, two ways.
 *
 * This port's expert path and upstream's compute the same function by very
 * different means, and until now nobody had measured the difference.
 *
 *   EXPAND   `decode_matrix` in inkling_wexp.c reconstructs the whole
 *            quantized matrix into F32 — for Inkling that is
 *            3 x 2048 x 4096 floats, 100.7 MB per expert — and
 *            inkling_layer.c then runs an ordinary dense matvec over it.
 *            This is what the private runtime does today.
 *
 *   LUT      upstream's `vq_matvec` (src/model.c) never expands an expert.
 *            It builds a table from the ACTIVATION once per matrix,
 *            lut[v][stage][entry] = sum_d x[v*dim+d] * book[stage][entry][d],
 *            then each output row is `nv * stages` table lookups and adds.
 *            At Inkling's geometry the table is nv*stages*entries floats =
 *            1.5 MB, so it is L2-resident rather than L1-resident. That is
 *            still two orders of magnitude below the 100.7 MB the expanded
 *            expert streams through DRAM, which is the point.
 *
 * docs/WASTE-CONSTRAINTS.md §2 already said the scalar path "should not
 * become the hot path". This measures what that sentence is worth.
 *
 * Both implementations here are standalone so the benchmark builds without
 * the engine, and the FIRST thing the program does is check that they agree
 * to floating-point tolerance on the same weights and activation. A speed
 * comparison between two things that compute different functions would be
 * worthless, so if the check fails the program refuses to print a timing.
 *
 * Build: cc -O2 -std=gnu11 -o inkling_expert_bench inkling_expert_bench.c -lm
 * Usage: inkling_expert_bench [hidden] [intermediate] [reps]
 *
 * Prints one JSON object on stdout.
 */
#define _GNU_SOURCE
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define STAGES 3
#define ENTRIES 256
#define VEC_DIM 8
#define INDEX_BLOCK 64

/* Inkling-Small: 40 sparse layers x top-6 routed, plus 2 shared per layer. */
#define ROUTED_PER_TOKEN 240
#define SHARED_PER_TOKEN 80

static double now(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec / 1e9;
}

static uint32_t rng_state = 0x9e3779b9u;
static float frand(void)
{
    rng_state = rng_state * 1664525u + 1013904223u;
    return (float)((rng_state >> 8) & 0xffff) / 32768.0f - 1.0f;
}

/* Index layout, matching inkling_vq.py's `block_indices` and the reader in
 * inkling_wexp.c: [row_block][vector][row_in_block][stage]. */
static size_t idx_at(int row, int vector, int stage, int nv)
{
    const size_t block = (size_t)(row / INDEX_BLOCK);
    const size_t in_block = (size_t)(row % INDEX_BLOCK);
    return ((block * (size_t)nv + (size_t)vector) * INDEX_BLOCK + in_block)
           * STAGES + (size_t)stage;
}

/* ---- path EXPAND: reconstruct to F32, then a dense matvec ---------------- */

static void expand(const uint8_t *idx, const uint16_t *scale_h,
                   const float *books, int rows, int cols, float *out)
{
    const int nv = cols / VEC_DIM;
    for (int r = 0; r < rows; r++) {
        /* The real reader converts an fp16 scale here; a multiply by a float
         * is the same work and keeps this file free of an f16 decoder. */
        const float scale = (float)scale_h[r] * (1.0f / 32768.0f);
        for (int v = 0; v < nv; v++) {
            float *dst = out + ((size_t)r * nv + v) * VEC_DIM;
            for (int d = 0; d < VEC_DIM; d++) dst[d] = 0.0f;
            for (int s = 0; s < STAGES; s++) {
                const unsigned code = idx[idx_at(r, v, s, nv)];
                const float *book = books + ((size_t)s * ENTRIES + code) * VEC_DIM;
                for (int d = 0; d < VEC_DIM; d++) dst[d] += book[d];
            }
            for (int d = 0; d < VEC_DIM; d++) dst[d] *= scale;
        }
    }
}

static void dense_matvec(const float *W, const float *x, int rows, int cols,
                         float *y)
{
    for (int r = 0; r < rows; r++) {
        const float *w = W + (size_t)r * cols;
        float acc = 0.0f;
        for (int c = 0; c < cols; c++) acc += w[c] * x[c];
        y[r] = acc;
    }
}

/* ---- path LUT: build a table from the activation, then gather ------------ */

static void build_lut(const float *books, const float *x, int cols, float *lut)
{
    const int nv = cols / VEC_DIM;
    for (int v = 0; v < nv; v++) {
        const float *xv = x + (size_t)v * VEC_DIM;
        for (int s = 0; s < STAGES; s++) {
            const float *book = books + (size_t)s * ENTRIES * VEC_DIM;
            float *dst = lut + ((size_t)v * STAGES + s) * ENTRIES;
            for (int e = 0; e < ENTRIES; e++) {
                const float *b = book + (size_t)e * VEC_DIM;
                float acc = 0.0f;
                for (int d = 0; d < VEC_DIM; d++) acc += xv[d] * b[d];
                dst[e] = acc;
            }
        }
    }
}

static void lut_matvec(const uint8_t *idx, const uint16_t *scale_h,
                       const float *lut, int rows, int cols, float *y)
{
    const int nv = cols / VEC_DIM;
    for (int r = 0; r < rows; r++) {
        float acc = 0.0f;
        for (int v = 0; v < nv; v++) {
            const float *blk = lut + (size_t)v * STAGES * ENTRIES;
            for (int s = 0; s < STAGES; s++)
                acc += blk[(size_t)s * ENTRIES + idx[idx_at(r, v, s, nv)]];
        }
        y[r] = acc * ((float)scale_h[r] * (1.0f / 32768.0f));
    }
}

int main(int argc, char **argv)
{
    const int hidden = argc > 1 ? atoi(argv[1]) : 4096;
    const int inter = argc > 2 ? atoi(argv[2]) : 2048;
    const int reps = argc > 3 ? atoi(argv[3]) : 3;

    if (hidden <= 0 || inter <= 0 || hidden % VEC_DIM || inter % VEC_DIM ||
        hidden % INDEX_BLOCK || inter % INDEX_BLOCK || reps <= 0) {
        fprintf(stderr, "bad geometry\n");
        return 1;
    }

    /* One matrix stands for the expert; gate, up and down are all
     * rows x cols with the same total work, so the per-expert figure is
     * three times what is measured here. gate is [inter, hidden]. */
    const int rows = inter, cols = hidden;
    const int nv = cols / VEC_DIM;

    const size_t n_idx = (size_t)((rows + INDEX_BLOCK - 1) / INDEX_BLOCK)
                       * INDEX_BLOCK * (size_t)nv * STAGES;
    uint8_t *idx = malloc(n_idx);
    uint16_t *scale = malloc((size_t)rows * sizeof *scale);
    float *books = malloc((size_t)STAGES * ENTRIES * VEC_DIM * sizeof *books);
    float *x = malloc((size_t)cols * sizeof *x);
    float *W = malloc((size_t)rows * cols * sizeof *W);
    float *lut = malloc((size_t)nv * STAGES * ENTRIES * sizeof *lut);
    float *ya = malloc((size_t)rows * sizeof *ya);
    float *yb = malloc((size_t)rows * sizeof *yb);
    if (!idx || !scale || !books || !x || !W || !lut || !ya || !yb) {
        fprintf(stderr, "out of memory\n");
        return 1;
    }

    for (size_t i = 0; i < n_idx; i++) idx[i] = (uint8_t)(rng_state = rng_state * 1103515245u + 12345u) ;
    for (int r = 0; r < rows; r++) scale[r] = (uint16_t)(1000 + (r % 5000));
    for (size_t i = 0; i < (size_t)STAGES * ENTRIES * VEC_DIM; i++) books[i] = frand();
    for (int c = 0; c < cols; c++) x[c] = frand();

    /* Agreement first. A timing comparison between two implementations that
     * do not compute the same thing is not a measurement of anything. */
    expand(idx, scale, books, rows, cols, W);
    dense_matvec(W, x, rows, cols, ya);
    build_lut(books, x, cols, lut);
    lut_matvec(idx, scale, lut, rows, cols, yb);

    double worst = 0.0, denom = 0.0;
    for (int r = 0; r < rows; r++) {
        const double d = fabs((double)ya[r] - (double)yb[r]);
        if (d > worst) worst = d;
        if (fabs((double)ya[r]) > denom) denom = fabs((double)ya[r]);
    }
    const double rel = denom > 0 ? worst / denom : worst;
    if (!(rel < 1e-5)) {
        fprintf(stderr, "inkling_expert_bench: the two paths disagree "
                        "(rel %.3e) — refusing to report a timing\n", rel);
        return 2;
    }

    double t_expand = 1e30, t_lut = 1e30;
    for (int i = 0; i < reps; i++) {
        double t0 = now();
        expand(idx, scale, books, rows, cols, W);
        dense_matvec(W, x, rows, cols, ya);
        double dt = now() - t0;
        if (dt < t_expand) t_expand = dt;

        t0 = now();
        build_lut(books, x, cols, lut);
        lut_matvec(idx, scale, lut, rows, cols, yb);
        dt = now() - t0;
        if (dt < t_lut) t_lut = dt;
    }

    /* Per expert: three matrices of this shape. */
    const double exp_expert = t_expand * 3.0, lut_expert = t_lut * 3.0;
    const int experts = ROUTED_PER_TOKEN + SHARED_PER_TOKEN;
    const double f32_bytes = (double)rows * cols * 4.0 * 3.0;

    /* Memory traffic, which is the part no amount of SIMD or threading
     * changes. EXPAND writes the reconstructed F32 matrix and reads it
     * straight back, so it moves twice the expanded size through DRAM per
     * expert. LUT touches the index planes — bytes the engine had to read
     * from disk regardless — plus a table small enough to stay in cache. */
    const double expand_traffic = f32_bytes * 2.0;
    const double lut_bytes = (double)nv * STAGES * ENTRIES * 4.0;
    const double lut_traffic = (double)n_idx * 3.0 + lut_bytes;

    printf("{\"hidden\":%d,\"intermediate\":%d,\"reps\":%d,"
           "\"rel_disagreement\":%.3e,"
           "\"expand_matrix_s\":%.6f,\"lut_matrix_s\":%.6f,"
           "\"expand_expert_s\":%.6f,\"lut_expert_s\":%.6f,"
           "\"speedup\":%.2f,"
           "\"expert_f32_bytes\":%.0f,"
           "\"expand_traffic_per_expert\":%.0f,"
           "\"lut_traffic_per_expert\":%.0f,"
           "\"lut_table_bytes\":%.0f,"
           "\"traffic_ratio\":%.1f,"
           "\"experts_per_token\":%d,"
           "\"expand_token_traffic_gib\":%.1f,"
           "\"lut_token_traffic_gib\":%.2f,"
           "\"expand_token_s\":%.3f,\"lut_token_s\":%.3f}\n",
           hidden, inter, reps, rel,
           t_expand, t_lut, exp_expert, lut_expert,
           t_expand / t_lut, f32_bytes,
           expand_traffic, lut_traffic, lut_bytes,
           expand_traffic / lut_traffic, experts,
           expand_traffic * experts / 1073741824.0,
           lut_traffic * experts / 1073741824.0,
           exp_expert * experts, lut_expert * experts);

    free(idx); free(scale); free(books); free(x); free(W);
    free(lut); free(ya); free(yb);
    return 0;
}
