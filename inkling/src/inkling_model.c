/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
#include "inkling_model.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

static int addz(size_t *a, size_t b)
{
    if (!a || b > SIZE_MAX - *a) return -1;
    *a += b;
    return 0;
}

static int mulz(size_t a, size_t b, size_t *out)
{
    if (!out || (a && b > SIZE_MAX / a)) return -1;
    *out = a * b;
    return 0;
}

static int layer_capacity(const waste_inkling_config *cfg, int layer,
                          int context_capacity)
{
    if (!cfg || layer < 0 || layer >= cfg->n_layers || context_capacity < 1)
        return 0;
    if (!cfg->layer[layer].is_local) return context_capacity;
    return context_capacity < cfg->sliding_window
         ? context_capacity : cfg->sliding_window;
}

static void rmsnorm(float *out, const float *x, const float *weight,
                    int n, float eps)
{
    waste_inkling_rmsnorm_profile(out, x, weight, n, eps,
                                  WASTE_INKLING_NUMERIC_F32);
}

static float dot(const float *a, const float *b, int n)
{
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += (double)a[i] * b[i];
    return (float)sum;
}

size_t waste_inkling_model_state_floats(const waste_inkling_config *cfg,
                                        int context_capacity)
{
    if (!cfg || context_capacity < 1 || context_capacity > cfg->max_context)
        return 0;
    size_t total = 0;
    for (int i = 0; i < cfg->n_layers; i++) {
        const waste_inkling_layer_cfg *l = &cfg->layer[i];
        const int cap = layer_capacity(cfg, i, context_capacity);
        size_t kv, kv2, conv;
        if (!cap || mulz((size_t)cap, (size_t)l->num_kv_heads * l->head_dim, &kv) ||
            mulz(kv, 2u, &kv2) ||
            mulz((size_t)cfg->conv_kernel,
                 2u * (size_t)l->num_kv_heads * l->head_dim +
                 2u * (size_t)cfg->hidden, &conv) ||
            addz(&total, kv2) || addz(&total, conv))
            return 0;
    }
    return total;
}

size_t waste_inkling_model_scratch_floats(const waste_inkling_config *cfg,
                                          int context_capacity)
{
    if (!cfg || context_capacity < 1 || context_capacity > cfg->max_context)
        return 0;
    size_t largest = 0;
    for (int i = 0; i < cfg->n_layers; i++) {
        const size_t n = waste_inkling_layer_scratch_floats(
            cfg, i, layer_capacity(cfg, i, context_capacity));
        if (!n) return 0;
        if (n > largest) largest = n;
    }
    size_t total = 0;
    if (mulz((size_t)cfg->hidden, 2, &total) || addz(&total, largest)) return 0;
    return total;
}

size_t waste_inkling_model_scratch_ints(const waste_inkling_config *cfg)
{
    return waste_inkling_layer_scratch_ints(cfg);
}

int waste_inkling_model_state_init(waste_inkling_model_state *s,
                                   const waste_inkling_config *cfg,
                                   int context_capacity,
                                   float *buffer, size_t nfloat)
{
    if (!s || !cfg || !buffer) return -1;
    const size_t need = waste_inkling_model_state_floats(cfg, context_capacity);
    if (!need || nfloat < need) return -1;
    memset(s, 0, sizeof *s);
    memset(buffer, 0, need * sizeof(float));
    float *p = buffer;
    for (int i = 0; i < cfg->n_layers; i++) {
        const waste_inkling_layer_cfg *l = &cfg->layer[i];
        const int cap = layer_capacity(cfg, i, context_capacity);
        const size_t kdim = (size_t)l->num_kv_heads * l->head_dim;
        const size_t kv = (size_t)cap * kdim;
        const size_t kc = kdim * cfg->conv_kernel;
        const size_t hc = (size_t)cfg->hidden * cfg->conv_kernel;
        float *k_cache = p; p += kv;
        float *v_cache = p; p += kv;
        float *k_conv = p; p += kc;
        float *v_conv = p; p += kc;
        float *attn_conv = p; p += hc;
        float *mlp_conv = p; p += hc;
        if (waste_inkling_layer_state_init(&s->layer[i], cfg, i, cap,
                k_cache, v_cache, k_conv, v_conv, attn_conv, mlp_conv))
            return -1;
    }
    if ((size_t)(p - buffer) != need) return -1;
    s->buffer = buffer;
    s->buffer_floats = need;
    s->context_capacity = context_capacity;
    return 0;
}

int waste_inkling_model_scratch_init(waste_inkling_model_scratch *s,
                                     const waste_inkling_config *cfg,
                                     int context_capacity,
                                     float *buffer, size_t nfloat,
                                     int *ibuffer, size_t nint)
{
    if (!s || !cfg || !buffer || !ibuffer) return -1;
    const size_t need = waste_inkling_model_scratch_floats(cfg, context_capacity);
    const size_t ineed = waste_inkling_model_scratch_ints(cfg);
    if (!need || nfloat < need || nint < ineed) return -1;
    memset(s, 0, sizeof *s);
    s->x = buffer;
    s->row = buffer + cfg->hidden;
    float *layer_buf = buffer + 2u * (size_t)cfg->hidden;
    const size_t layer_n = need - 2u * (size_t)cfg->hidden;
    /* Initialize with the largest-capacity layer. Layer scratch pointers are
     * rebound before each step because projection dimensions differ. */
    int best = 0;
    size_t best_n = 0;
    for (int i = 0; i < cfg->n_layers; i++) {
        size_t n = waste_inkling_layer_scratch_floats(
            cfg, i, layer_capacity(cfg, i, context_capacity));
        if (n > best_n) { best_n = n; best = i; }
    }
    if (waste_inkling_layer_scratch_init(&s->layer, cfg, best,
            layer_capacity(cfg, best, context_capacity), layer_buf, layer_n,
            ibuffer, nint)) return -1;
    s->buffer = buffer;
    s->ibuffer = ibuffer;
    s->buffer_floats = need;
    s->ibuffer_ints = ineed;
    return 0;
}

void waste_inkling_model_reset(waste_inkling_model_state *s,
                               const waste_inkling_config *cfg)
{
    if (!s || !cfg) return;
    if (s->buffer && s->buffer_floats)
        memset(s->buffer, 0, s->buffer_floats * sizeof(float));
    for (int i = 0; i < cfg->n_layers; i++)
        waste_inkling_attention_reset(&s->layer[i].attention);
    s->next_position = 0;
}

static int get_row(const float *table, int rows, int cols,
                   waste_inkling_row_get_fn get, void *ctx,
                   int row, float *out)
{
    if (!out || row < 0 || row >= rows || cols <= 0) return -1;
    if (table) {
        memcpy(out, table + (size_t)row * cols, (size_t)cols * sizeof(float));
        return 0;
    }
    return get ? get(ctx, row, cols, out) : -1;
}

int waste_inkling_final_head_profile(
    const waste_inkling_config *cfg,
    const waste_inkling_model_weights *w,
    const waste_inkling_matrix_backend *backend,
    const float *hidden_state,
    const int *rows, int n_rows,
    float *logits, size_t logits_count,
    float *normalized, float *row_scratch,
    waste_inkling_numeric_profile profile,
    const waste_inkling_trace *trace)
{
    if (!cfg || !w || !hidden_state || !logits || !normalized ||
        !row_scratch || !w->final_norm ||
        cfg->hidden <= 0 || cfg->unpadded_vocab <= 0 ||
        n_rows <= 0 || logits_count < (size_t)n_rows ||
        !(cfg->logits_width_multiplier > 0.0f) ||
        !waste_inkling_numeric_profile_valid(profile))
        return -1;

    const int bf16 = profile == WASTE_INKLING_NUMERIC_BF16_REFERENCE;
    /* The resident table may be shorter than the vocabulary; a callback covers
     * exactly the unpadded vocabulary. Either way a selection that reaches
     * past what is bound fails closed rather than reading a neighboring row. */
    const int bound = w->unembedding ? w->unembedding_rows : cfg->unpadded_vocab;
    if (bf16) {
        if (!backend || !backend->matvec) return -1;
    } else if (!w->unembedding && !w->unembedding_get) {
        return -1;
    } else if (n_rows > bound) {
        return -1;
    }
    if (rows) {
        const int limit = bf16 ? cfg->unpadded_vocab : bound;
        for (int j = 0; j < n_rows; j++)
            if (rows[j] < 0 || rows[j] >= limit) return -1;
    }

    const int hidden = cfg->hidden;
    waste_inkling_rmsnorm_profile(normalized, hidden_state, w->final_norm,
                                  hidden, cfg->rms_eps, profile);
    if (trace && trace->emit_float &&
        trace->emit_float(trace->ctx, -1, "final_norm", normalized,
                          (size_t)hidden)) return -1;
    if (bf16) {
        /* The official forward divides the hidden state by this factor
         * immediately before the vocabulary projection, so divide here rather
         * than multiply by a reciprocal. The release multiplier is 16 and the
         * two agree there; they diverge for any multiplier whose reciprocal is
         * inexact, and the operation should not depend on that. */
        for (int i = 0; i < hidden; i++)
            normalized[i] = waste_inkling_bf16_round(
                normalized[i] / cfg->logits_width_multiplier);
    } else {
        const float inv_width = 1.0f / cfg->logits_width_multiplier;
        for (int i = 0; i < hidden; i++) normalized[i] *= inv_width;
    }
    if (trace && trace->emit_float &&
        trace->emit_float(trace->ctx, -1, "final_norm_scaled", normalized,
                          (size_t)hidden)) return -1;

    if (bf16) {
        if (backend->matvec(backend->ctx, -1, WASTE_IK_MAT_UNEMBED, 0,
                            normalized, logits, n_rows, hidden)) return -1;
    } else {
        for (int j = 0; j < n_rows; j++) {
            const int v = rows ? rows[j] : j;
            const float *row;
            if (w->unembedding) {
                row = w->unembedding + (size_t)v * hidden;
            } else {
                if (w->unembedding_get(w->unembedding_ctx, v, hidden,
                                       row_scratch)) return -1;
                row = row_scratch;
            }
            logits[j] = dot(row, normalized, hidden);
        }
    }
    if (trace && trace->emit_float &&
        trace->emit_float(trace->ctx, -1, "logits", logits,
                          (size_t)n_rows)) return -1;
    return 0;
}

int waste_inkling_model_step_backend_trace(
    const waste_inkling_config *cfg,
    const waste_inkling_model_weights *w,
    const waste_inkling_matrix_backend *backends,
    waste_inkling_model_state *state,
    waste_inkling_model_scratch *scratch,
    int token, int position, float *logits, size_t logits_count,
    waste_inkling_expert_get_fn expert_get, void *expert_ctx,
    const waste_inkling_trace *trace)
{
    if (!cfg || !w || !state || !scratch || !logits ||
        !w->embed_norm || !w->final_norm || !w->layer ||
        token < 0 || token >= cfg->vocab || position < 0 ||
        position != state->next_position ||
        position >= state->context_capacity || position >= cfg->max_context ||
        cfg->unpadded_vocab <= 0 || logits_count < (size_t)cfg->unpadded_vocab ||
        !(cfg->logits_width_multiplier > 0.0f))
        return -1;
    if (!w->embedding && !w->embedding_get) return -1;
    if (!w->unembedding && !w->unembedding_get) return -1;
    if (w->unembedding && w->unembedding_rows < cfg->unpadded_vocab) return -1;

    if (get_row(w->embedding, cfg->vocab, cfg->hidden,
                w->embedding_get, w->embedding_ctx,
                token, scratch->row)) return -1;
    rmsnorm(scratch->x, scratch->row, w->embed_norm,
            cfg->hidden, cfg->rms_eps);
    if (trace && trace->emit_float &&
        trace->emit_float(trace->ctx, -1, "embedding_norm", scratch->x,
                          (size_t)cfg->hidden)) return -1;

    float *layer_buf = scratch->buffer + 2u * (size_t)cfg->hidden;
    const size_t layer_n = scratch->buffer_floats - 2u * (size_t)cfg->hidden;
    for (int i = 0; i < cfg->n_layers; i++) {
        const int cap = layer_capacity(cfg, i, state->context_capacity);
        if (waste_inkling_layer_scratch_init(&scratch->layer, cfg, i, cap,
                layer_buf, layer_n, scratch->ibuffer, scratch->ibuffer_ints) ||
            waste_inkling_layer_step_backend_trace(cfg, i, &w->layer[i],
                backends ? &backends[i] : NULL, &state->layer[i],
                scratch->x, position, &scratch->layer, expert_get, expert_ctx,
                trace))
            return -1;
    }

    /* Public stepping stays on the checked-in F32 profile. scratch->x holds the
     * decoder output on the way in and is reused as the row staging buffer,
     * which the primitive documents as allowed. */
    if (waste_inkling_final_head_profile(cfg, w, NULL, scratch->x,
            NULL, cfg->unpadded_vocab, logits, logits_count,
            scratch->row, scratch->x, WASTE_INKLING_NUMERIC_F32, trace))
        return -1;
    state->next_position++;
    return 0;
}

int waste_inkling_model_step_backend(
    const waste_inkling_config *cfg,
    const waste_inkling_model_weights *weights,
    const waste_inkling_matrix_backend *backends,
    waste_inkling_model_state *state,
    waste_inkling_model_scratch *scratch,
    int token, int position, float *logits, size_t logits_count,
    waste_inkling_expert_get_fn expert_get, void *expert_ctx)
{
    return waste_inkling_model_step_backend_trace(cfg, weights, backends,
        state, scratch, token, position, logits, logits_count,
        expert_get, expert_ctx, NULL);
}

int waste_inkling_model_step(
    const waste_inkling_config *cfg,
    const waste_inkling_model_weights *weights,
    waste_inkling_model_state *state,
    waste_inkling_model_scratch *scratch,
    int token, int position, float *logits, size_t logits_count,
    waste_inkling_expert_get_fn expert_get, void *expert_ctx)
{
    return waste_inkling_model_step_backend(cfg, weights, NULL, state, scratch,
        token, position, logits, logits_count, expert_get, expert_ctx);
}
