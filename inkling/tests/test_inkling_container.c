/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
/*
 * test_inkling_container.c — the public loader, against two containers that
 * hold the same weights at two widths.
 *
 *   ./test_inkling_container <f32-container> <quantized> [row-tol] [matvec-tol]
 *
 * Both are written by tools/make_inkling_container.py from one seed, so the
 * f32 one is the oracle and the quantized one is the thing under test. That
 * shape is deliberate: what this promotion added is a path where a matrix is
 * never materialized — it binds with a NULL pointer and reaches the
 * arithmetic through waste_matmul_t — and the only way to know that path
 * computes anything is to compare it against the same weights read the
 * ordinary way.
 *
 * It also drives the two vocabulary tables through the row callback the
 * binding installed. Those are left on disk when quantized, so the f32
 * container answers from memory and the quantized one from a pread and an
 * unpack, and they have to agree.
 *
 * What it does not do is run a token. That is the next promotion; here the
 * point is that asking for one is refused.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/inkling_container.h"
#include "../src/model.h"
#include "../src/waste.h"

static int fails;

static void check(int ok, const char *what)
{
    if (!ok) {
        fprintf(stderr, "INKLING CONTAINER: %s\n", what);
        fails++;
    }
}

/* One row of a bound vocabulary table, however it is stored: through the
 * callback when the load installed one, straight out of the resident table
 * when it did not. */
static int vocab_row(const waste_inkling_model_weights *w, int table, int row,
                     int cols, float *dst)
{
    const float *resident = table ? w->unembedding : w->embedding;
    waste_inkling_row_get_fn get = table ? w->unembedding_get : w->embedding_get;
    void *ctx = table ? w->unembedding_ctx : w->embedding_ctx;
    if (get) return get(ctx, row, cols, dst);
    if (!resident) return -1;
    memcpy(dst, resident + (size_t)row * cols, (size_t)cols * sizeof(float));
    return 0;
}

static float maxdiff(const float *a, const float *b, int n)
{
    float worst = 0;
    for (int i = 0; i < n; i++) {
        const float d = fabsf(a[i] - b[i]);
        if (d > worst) worst = d;
    }
    return worst;
}

/* Agreement measured against the reference's own scale.
 *
 * An absolute tolerance would have to be rewritten for every stored width —
 * a four-bit group scale is eighteen times coarser than an eight-bit one —
 * and a tolerance loose enough for four bits stops telling a rounding
 * difference from a wrong row. Relative to the reference, rounding stays
 * around the width's own step while reading the wrong row or the wrong
 * shared-expert block lands near 1, which is the distinction that matters. */
static float reldiff(const float *a, const float *b, int n)
{
    float scale = 1e-6f;
    for (int i = 0; i < n; i++) if (fabsf(a[i]) > scale) scale = fabsf(a[i]);
    return maxdiff(a, b, n) / scale;
}

static float worst_row, worst_mv;

static void track(float *slot, float v)
{
    if (v > *slot) *slot = v;
}

int main(int argc, char **argv)
{
    waste_model ref, cand;
    waste_load_opts opt;

    if (argc < 3) {
        fprintf(stderr, "usage: %s <f32-container> <quantized-container> "
                        "[row-tol] [matvec-tol]\n", argv[0]);
        return 2;
    }
    /* Relative, and the caller's to set, because it is a property of the
     * stored width rather than of the code under test. */
    const float row_tol = argc > 3 ? (float)atof(argv[3]) : 0.05f;
    const float mv_tol = argc > 4 ? (float)atof(argv[4]) : 0.15f;
    memset(&opt, 0, sizeof opt);

    check(waste_model_load(&ref, argv[1], 4096, &opt) == 0, "f32 container did not load");
    check(waste_model_load(&cand, argv[2], 4096, &opt) == 0, "quantized container did not load");
    if (fails) return 1;

    check(waste_inkling_container_present(&ref), "f32 model is not an Inkling container");
    check(waste_inkling_container_present(&cand), "quantized model is not an Inkling container");

    const waste_inkling_config *rc = waste_inkling_container_config(&ref);
    const waste_inkling_config *cc = waste_inkling_container_config(&cand);
    check(rc && cc, "no bound config");
    if (!rc || !cc) return 1;
    /* Same source config, so the geometry has to survive both widths
     * identically — a container whose stored width changed its shape would
     * mean the loader read a dimension out of the payload. */
    check(memcmp(rc, cc, sizeof *rc) == 0, "the two containers disagree about geometry");
    check(rc->n_layers >= 3 && rc->dense_layers >= 1 &&
          rc->dense_layers < rc->n_layers,
          "the fixture must carry a dense and a sparse layer");

    const waste_inkling_bound_weights *rw = waste_inkling_container_weights(&ref);
    const waste_inkling_bound_weights *cw = waste_inkling_container_weights(&cand);
    const waste_inkling_matrix_backend *rb = waste_inkling_container_backends(&ref);
    const waste_inkling_matrix_backend *cb = waste_inkling_container_backends(&cand);
    check(rw && cw && rb && cb, "no bound weights or backends");
    if (!rw || !cw || !rb || !cb) return 1;

    const int hid = rc->hidden;
    float *a = (float *)malloc((size_t)hid * sizeof(float));
    float *b = (float *)malloc((size_t)hid * sizeof(float));
    check(a && b, "out of memory");
    if (!a || !b) return 1;

    /* The quantized container leaves both tables on disk; the f32 one keeps
     * them resident. Neither fact is asserted here — what is asserted is
     * that the two answers are the same row. */
    check(cw->model.embedding_get != NULL && cw->model.unembedding_get != NULL,
          "a quantized vocabulary table was made resident");
    check(rw->model.embedding != NULL && rw->model.unembedding != NULL,
          "an f32 vocabulary table was not made resident");
    for (int t = 0; t < 2; t++) {
        const int rows = t ? rc->unpadded_vocab : rc->vocab;
        for (int r = 0; r < rows; r += (rows / 5 > 0 ? rows / 5 : 1)) {
            check(vocab_row(&rw->model, t, r, hid, a) == 0, "f32 vocabulary row failed");
            check(vocab_row(&cw->model, t, r, hid, b) == 0, "quantized vocabulary row failed");
            track(&worst_row, reldiff(a, b, hid));
            check(reldiff(a, b, hid) < row_tol,
                  "vocabulary rows disagree beyond quantization");
        }
        /* A row past the end of the table is an error, not a clamp: the
         * caller is a bound weight table, and a silently wrapped row is a
         * wrong token nobody would ever see reported. */
        check(vocab_row(&cw->model, t, rows, hid, b) != 0,
              "a row past the end of a vocabulary table was served");
    }

    /* Every layer matrix, both widths, against each other — and, on the f32
     * side, against a plain dot product, so the comparison is anchored to
     * arithmetic rather than to two backends agreeing with each other. */
    static const waste_inkling_matrix_kind common[] = {
        WASTE_IK_MAT_Q, WASTE_IK_MAT_K, WASTE_IK_MAT_V, WASTE_IK_MAT_R,
    };
    float *x = (float *)malloc((size_t)hid * sizeof(float));
    check(x != NULL, "out of memory");
    if (!x) return 1;
    for (int i = 0; i < hid; i++) x[i] = (float)((i % 7) - 3) * 0.25f;

    int probed = 0;
    for (int L = 0; L < rc->n_layers; L++) {
        const waste_inkling_layer_cfg *lc = &rc->layer[L];
        for (size_t k = 0; k < sizeof common / sizeof common[0]; k++) {
            const waste_inkling_matrix_kind kind = common[k];
            int rows = lc->num_heads * lc->head_dim;
            if (kind == WASTE_IK_MAT_K || kind == WASTE_IK_MAT_V)
                rows = lc->num_kv_heads * lc->head_dim;
            if (kind == WASTE_IK_MAT_R) rows = lc->num_heads * rc->d_rel;
            float *yr = (float *)calloc((size_t)rows, sizeof(float));
            float *yc = (float *)calloc((size_t)rows, sizeof(float));
            check(yr && yc, "out of memory");
            if (!yr || !yc) return 1;
            check(rb[L].matvec(rb[L].ctx, L, kind, 0, x, yr, rows, hid) == 0,
                  "f32 backend refused a matrix it bound");
            check(cb[L].matvec(cb[L].ctx, L, kind, 0, x, yc, rows, hid) == 0,
                  "quantized backend refused a matrix it bound");
            track(&worst_mv, reldiff(yr, yc, rows));
            check(reldiff(yr, yc, rows) < mv_tol,
                  "quantized matvec disagrees with the f32 one beyond rounding");
            {   /* The f32 side against arithmetic written out longhand. */
                const float *W = kind == WASTE_IK_MAT_Q ? rw->layer[L].wq
                               : kind == WASTE_IK_MAT_K ? rw->layer[L].wk
                               : kind == WASTE_IK_MAT_V ? rw->layer[L].wv
                                                        : rw->layer[L].wr;
                check(W != NULL, "an f32 layer matrix did not bind resident");
                if (W) {
                    float worst = 0;
                    for (int r = 0; r < rows; r++) {
                        float acc = 0;
                        for (int c = 0; c < hid; c++)
                            acc += W[(size_t)r * hid + c] * x[c];
                        const float d = fabsf(acc - yr[r]);
                        if (d > worst) worst = d;
                    }
                    check(worst < 1e-4f, "the f32 backend is not a matvec");
                }
            }
            free(yr); free(yc);
            probed++;
        }
        /* Shared experts are one stacked tensor and `index` selects a block.
         * Getting that offset wrong is silent — it returns another expert's
         * answer, not an error — so both blocks are compared and required to
         * differ from each other as well as to match across widths. */
        if (L >= rc->dense_layers && rc->n_shared_experts > 1) {
            const int rows = rc->moe_intermediate;
            float *y0 = (float *)calloc((size_t)rows, sizeof(float));
            float *y1 = (float *)calloc((size_t)rows, sizeof(float));
            float *q0 = (float *)calloc((size_t)rows, sizeof(float));
            check(y0 && y1 && q0, "out of memory");
            if (!y0 || !y1 || !q0) return 1;
            check(rb[L].matvec(rb[L].ctx, L, WASTE_IK_MAT_SHARED_GATE, 0, x, y0, rows, hid) == 0 &&
                  rb[L].matvec(rb[L].ctx, L, WASTE_IK_MAT_SHARED_GATE, 1, x, y1, rows, hid) == 0 &&
                  cb[L].matvec(cb[L].ctx, L, WASTE_IK_MAT_SHARED_GATE, 1, x, q0, rows, hid) == 0,
                  "a shared-expert block was refused");
            /* Self-scaling, so it means the same thing at every stored
             * width: the two blocks have to be several times further apart
             * than the width's own rounding, or this comparison could not
             * have caught a block-offset bug in the first place. */
            const float agree = reldiff(y1, q0, rows);
            track(&worst_mv, agree);
            check(reldiff(y0, y1, rows) > 4.0f * (agree + 1e-3f),
                  "two shared-expert blocks are not far enough apart to tell "
                  "a block-offset bug from rounding");
            check(agree < mv_tol,
                  "the quantized shared-expert block is not the same block");
            check(rb[L].matvec(rb[L].ctx, L, WASTE_IK_MAT_SHARED_GATE,
                               rc->n_shared_experts, x, y0, rows, hid) != 0,
                  "a shared-expert index past the end was served");
            free(y0); free(y1); free(q0);
        }
        /* A dense layer has no router and a sparse one has no dense MLP.
         * Asking anyway must be refused rather than answered from whatever
         * tensor happened to be at that slot. */
        const waste_inkling_matrix_kind absent =
            L < rc->dense_layers ? WASTE_IK_MAT_ROUTER : WASTE_IK_MAT_DENSE_GATE;
        float sink[4];
        check(cb[L].matvec(cb[L].ctx, L, absent, 0, x, sink, 1, hid) != 0,
              "a matrix this layer class does not have was served");
    }
    check(probed == rc->n_layers * 4, "not every layer was probed");

    /* Out-of-range arguments the caller controls. */
    float sink[4];
    check(cb[0].matvec(cb[0].ctx, -1, WASTE_IK_MAT_Q, 0, x, sink, 1, hid) != 0,
          "a negative layer was served");
    check(cb[0].matvec(cb[0].ctx, rc->n_layers, WASTE_IK_MAT_Q, 0, x, sink, 1, hid) != 0,
          "a layer past the end was served");
    check(cb[0].matvec(cb[0].ctx, 0, WASTE_IK_MAT_Q, 0, x, sink, 1, hid + 1) != 0,
          "a width the tensor does not have was served");

    /* Plan and load, against each other.
     *
     * The planner leaves the two vocabulary tables out of the resident floor
     * and the loader leaves them on disk, under one shared rule; everything
     * else it reports is the stored size of a tensor the loader made
     * resident. That makes the agreement exact rather than approximate, and
     * exact is worth asserting: a plan that is merely close is how a machine
     * ends up swapping halfway through a generation.
     *
     * WASTE_Q8=0 is the debug switch that dequantizes the whole trunk at
     * load. It changes the resident set without changing the manifest, so
     * the equality is not claimed under it. */
    if (!getenv("WASTE_Q8") || getenv("WASTE_Q8")[0] != '0') {
        for (int which = 0; which < 2; which++) {
            const waste_model *m = which ? &cand : &ref;
            waste_memplan plan;
            uint64_t resident = 0;
            check(waste_plan_memory(argv[1 + which], 4096, &plan) == WASTE_OK,
                  "the planner refused a container the loader accepted");
            for (int i = 0; i < m->n_tensors; i++) {
                const waste_tensor *t = &m->t[i];
                if (t->on_disk) continue;
                if (t->q) {
                    const int cols = t->shape[t->ndim - 1];
                    const uint64_t rows = (uint64_t)(t->n / (size_t)cols);
                    const uint64_t ng = (uint64_t)((cols + t->group - 1) / t->group);
                    resident += rows * (uint64_t)t->rowbytes + rows * ng * 2u;
                } else if (t->data) {
                    resident += (uint64_t)t->n * 4u;
                }
            }
            check(plan.trunk_bytes == resident,
                  "the planned resident trunk is not the one the load produced");
            if (plan.trunk_bytes != resident)
                fprintf(stderr, "  planned %llu, resident %llu\n",
                        (unsigned long long)plan.trunk_bytes,
                        (unsigned long long)resident);
        }
    }

    /* The refusal that is the point of this step. */
    check(waste_model_step(&cand, 0, 0, NULL) == NULL, "an Inkling step produced logits");
    {
        const int toks[2] = { 0, 1 };
        check(waste_model_prefill(&cand, toks, 2, 0) == NULL,
              "an Inkling prefill produced logits");
    }

    free(a); free(b); free(x);
    waste_model_free(&ref);
    waste_model_free(&cand);

    if (fails) {
        fprintf(stderr, "INKLING CONTAINER FAIL %d\n", fails);
        return 1;
    }
    printf("INKLING CONTAINER OK worst-row %.4f worst-matvec %.4f "
           "(tolerances %.3f / %.3f)\n", worst_row, worst_mv, row_tol, mv_tol);
    return 0;
}
