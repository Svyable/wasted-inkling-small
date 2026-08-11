/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
#include "inkling_layer.h"
#include "inkling.h"

#include <math.h>
#include <stddef.h>
#include <string.h>

static void matvec(float *out, const float *weight, const float *x,
                   int rows, int cols)
{
    for (int r = 0; r < rows; r++) {
        double sum = 0.0;
        const float *w = weight + (size_t)r * cols;
        for (int c = 0; c < cols; c++) sum += (double)w[c] * x[c];
        out[r] = (float)sum;
    }
}

static int apply_matvec_profile(const waste_inkling_matrix_backend *backend,
                                int layer, waste_inkling_matrix_kind kind,
                                int index, float *out, const float *resident,
                                const float *x, int rows, int cols,
                                waste_inkling_numeric_profile profile)
{
    if (profile == WASTE_INKLING_NUMERIC_BF16_REFERENCE) {
        /* Exact BF16 evidence is kernel-sensitive.  Never manufacture it by
         * widening a resident BF16 matrix and using the scalar F32 fallback. */
        return backend && backend->matvec
             ? backend->matvec(backend->ctx, layer, kind, index,
                               x, out, rows, cols)
             : -1;
    }
    if (resident) { matvec(out, resident, x, rows, cols); return 0; }
    return backend && backend->matvec
         ? backend->matvec(backend->ctx, layer, kind, index, x, out, rows, cols)
         : -1;
}

static int apply_matvec(const waste_inkling_matrix_backend *backend,
                        int layer, waste_inkling_matrix_kind kind, int index,
                        float *out, const float *resident, const float *x,
                        int rows, int cols)
{
    return apply_matvec_profile(backend, layer, kind, index, out, resident,
                                x, rows, cols, WASTE_INKLING_NUMERIC_F32);
}

/* The policy itself lives in inkling_numeric.h so that the layer and the final
 * head share one definition. This wrapper only keeps the call sites short. */
static void rmsnorm_profile(float *out, const float *x, const float *weight,
                            int n, float eps,
                            waste_inkling_numeric_profile profile)
{
    waste_inkling_rmsnorm_profile(out, x, weight, n, eps, profile);
}

static float silu(float x) { return x / (1.0f + expf(-x)); }

static int trace_f(const waste_inkling_trace *trace, int layer,
                   const char *point, const float *data, size_t count)
{
    return trace && trace->emit_float
         ? trace->emit_float(trace->ctx, layer, point, data, count) : 0;
}

static int trace_i(const waste_inkling_trace *trace, int layer,
                   const char *point, const int *data, size_t count)
{
    return trace && trace->emit_int
         ? trace->emit_int(trace->ctx, layer, point, data, count) : 0;
}

static int expert_eval(float *out, float *gate, float *up,
                       const waste_inkling_expert_weights *w,
                       const float *x, int hidden, int intermediate)
{
    if (!out || !gate || !up || !w || !w->gate || !w->up || !w->down || !x)
        return -1;
    matvec(gate, w->gate, x, intermediate, hidden);
    matvec(up, w->up, x, intermediate, hidden);
    for (int i = 0; i < intermediate; i++) gate[i] = silu(gate[i]) * up[i];
    matvec(out, w->down, gate, hidden, intermediate);
    return 0;
}

static void bf16_gated_activation(float *gate, const float *up, int n)
{
    for (int i = 0; i < n; i++) {
        const float activated = waste_inkling_bf16_round(silu(gate[i]));
        gate[i] = waste_inkling_bf16_round(
            activated * waste_inkling_bf16_round(up[i]));
    }
}

static int get_routed(const waste_inkling_config *cfg,
                      const waste_inkling_layer_weights *w,
                      int layer, int expert,
                      waste_inkling_expert_get_fn get, void *ctx,
                      waste_inkling_expert_weights *out)
{
    if (expert < 0 || expert >= cfg->n_routed_experts || !out) return -1;
    if (w->routed_gate && w->routed_up && w->routed_down) {
        const size_t gh = (size_t)cfg->moe_intermediate * cfg->hidden;
        const size_t dh = (size_t)cfg->hidden * cfg->moe_intermediate;
        out->gate = w->routed_gate + (size_t)expert * gh;
        out->up = w->routed_up + (size_t)expert * gh;
        out->down = w->routed_down + (size_t)expert * dh;
        return 0;
    }
    return get ? get(ctx, layer, expert, out) : -1;
}

static int weights_common_ok(const waste_inkling_layer_weights *w)
{
    return w && w->input_norm && w->post_attention_norm &&
           w->wq && w->wk && w->wv && w->wr && w->wo &&
           w->q_norm && w->k_norm && w->relative_proj &&
           w->k_sconv && w->v_sconv && w->attn_sconv && w->mlp_sconv;
}

size_t waste_inkling_layer_scratch_floats(const waste_inkling_config *cfg,
                                           int layer, int capacity)
{
    if (!cfg || layer < 0 || layer >= cfg->n_layers || capacity < 1) return 0;
    const waste_inkling_layer_cfg *l = &cfg->layer[layer];
    const size_t H = (size_t)cfg->hidden;
    const size_t Q = (size_t)l->num_heads * l->head_dim;
    const size_t K = (size_t)l->num_kv_heads * l->head_dim;
    const size_t R = (size_t)l->num_heads * cfg->d_rel;
    const size_t I = (size_t)(layer < cfg->dense_layers
                           ? cfg->dense_intermediate : cfg->moe_intermediate);
    const size_t router = (size_t)cfg->n_routed_experts + cfg->n_shared_experts;
    return H + Q + K + K + R + (size_t)l->num_heads * capacity + Q +
           H + I + I + H + router + cfg->top_k + cfg->n_shared_experts;
}

size_t waste_inkling_layer_scratch_ints(const waste_inkling_config *cfg)
{
    return cfg && cfg->top_k > 0 ? (size_t)cfg->top_k : 0;
}

int waste_inkling_layer_scratch_init(waste_inkling_layer_scratch *s,
                                     const waste_inkling_config *cfg,
                                     int layer, int capacity,
                                     float *buf, size_t nfloat,
                                     int *ibuf, size_t nint)
{
    if (!s || !cfg || !buf || !ibuf || layer < 0 || layer >= cfg->n_layers)
        return -1;
    const size_t need = waste_inkling_layer_scratch_floats(cfg, layer, capacity);
    const size_t ineed = waste_inkling_layer_scratch_ints(cfg);
    if (!need || nfloat < need || nint < ineed) return -1;
    const waste_inkling_layer_cfg *l = &cfg->layer[layer];
    const size_t H = (size_t)cfg->hidden;
    const size_t Q = (size_t)l->num_heads * l->head_dim;
    const size_t K = (size_t)l->num_kv_heads * l->head_dim;
    const size_t R = (size_t)l->num_heads * cfg->d_rel;
    const size_t I = (size_t)(layer < cfg->dense_layers
                           ? cfg->dense_intermediate : cfg->moe_intermediate);
    const size_t router = (size_t)cfg->n_routed_experts + cfg->n_shared_experts;
    float *p = buf;
    memset(s, 0, sizeof *s);
#define TAKE(name, count) do { s->name = p; p += (count); } while (0)
    TAKE(norm, H);
    TAKE(q, Q);
    TAKE(k, K);
    TAKE(v, K);
    TAKE(relative, R);
    TAKE(scores, (size_t)l->num_heads * capacity);
    TAKE(attn_out, Q);
    TAKE(branch, H);
    TAKE(gate, I);
    TAKE(up, I);
    TAKE(ff, H);
    TAKE(router_logits, router);
    TAKE(routed_weight, cfg->top_k);
    TAKE(shared_weight, cfg->n_shared_experts);
#undef TAKE
    s->routed_index = ibuf;
    s->float_count = need;
    s->int_count = ineed;
    return (size_t)(p - buf) == need ? 0 : -1;
}

int waste_inkling_layer_state_init(
    waste_inkling_layer_state *s,
    const waste_inkling_config *cfg, int layer, int capacity,
    float *k_cache, float *v_cache,
    float *k_conv, float *v_conv, float *attn_conv, float *mlp_conv)
{
    if (!s || !cfg || layer < 0 || layer >= cfg->n_layers ||
        !k_conv || !v_conv || !attn_conv || !mlp_conv)
        return -1;
    const waste_inkling_layer_cfg *l = &cfg->layer[layer];
    memset(s, 0, sizeof *s);
    if (waste_inkling_attention_init(&s->attention, l->is_local,
            l->num_heads, l->num_kv_heads, l->head_dim, cfg->d_rel,
            l->relative_extent, capacity, cfg->rms_eps, k_cache, v_cache))
        return -1;
    s->k_conv_state = k_conv;
    s->v_conv_state = v_conv;
    s->attn_conv_state = attn_conv;
    s->mlp_conv_state = mlp_conv;
    return 0;
}

int waste_inkling_layer_step_backend_trace_profile(
    const waste_inkling_config *cfg, int layer,
    const waste_inkling_layer_weights *w,
    const waste_inkling_matrix_backend *backend,
    waste_inkling_layer_state *state,
    float *x, int position,
    waste_inkling_layer_scratch *s,
    waste_inkling_expert_get_fn expert_get, void *expert_ctx,
    const waste_inkling_trace *trace,
    waste_inkling_numeric_profile profile)
{
    if (!cfg || layer < 0 || layer >= cfg->n_layers || !w ||
        !waste_inkling_numeric_profile_valid(profile) ||
        !w->input_norm || !w->post_attention_norm ||
        (!backend && !weights_common_ok(w)) ||
        !w->q_norm || !w->k_norm || !w->relative_proj ||
        !w->k_sconv || !w->v_sconv || !w->attn_sconv || !w->mlp_sconv ||
        !state || !x || !s || position < 0) return -1;
    /* The dense/sparse schedule is architecture, not tensor metadata.  A
     * mismatched flag otherwise executes a numerically valid but different
     * network, which is more dangerous than a loud load failure. */
    const int expected_sparse = layer >= cfg->dense_layers;
    if ((w->sparse != 0 && w->sparse != 1) || w->sparse != expected_sparse)
        return -1;
    const waste_inkling_layer_cfg *l = &cfg->layer[layer];
    const int hidden = cfg->hidden;
    const int qdim = l->num_heads * l->head_dim;
    const int kdim = l->num_kv_heads * l->head_dim;
    const int rdim = l->num_heads * cfg->d_rel;
    const int bf16 = profile == WASTE_INKLING_NUMERIC_BF16_REFERENCE;

    /* The measured exact profile requires native BF16 matmuls for both dense
     * and sparse layers.  Sparse execution additionally reuses dead Q scratch
     * as the shared FP32 accumulator after attention; dense layers do not need
     * that alias, so only sparse geometry must satisfy Q >= H. */
    if (bf16 && (!backend || !backend->matvec || (expected_sparse && qdim < hidden)))
        return -1;

    rmsnorm_profile(s->norm, x, w->input_norm, hidden, cfg->rms_eps, profile);
    if (trace_f(trace, layer, "input_norm", s->norm, (size_t)hidden)) return -1;
    if (apply_matvec_profile(backend, layer, WASTE_IK_MAT_Q, 0,
            s->q, w->wq, s->norm, qdim, hidden, profile)) return -1;
    if (apply_matvec_profile(backend, layer, WASTE_IK_MAT_K, 0,
            s->k, w->wk, s->norm, kdim, hidden, profile)) return -1;
    if (apply_matvec_profile(backend, layer, WASTE_IK_MAT_V, 0,
            s->v, w->wv, s->norm, kdim, hidden, profile)) return -1;
    if (apply_matvec_profile(backend, layer, WASTE_IK_MAT_R, 0,
            s->relative, w->wr, s->norm, rdim, hidden, profile)) return -1;
    if (trace_f(trace, layer, "q_proj", s->q, (size_t)qdim) ||
        trace_f(trace, layer, "k_proj", s->k, (size_t)kdim) ||
        trace_f(trace, layer, "v_proj", s->v, (size_t)kdim) ||
        trace_f(trace, layer, "relative_proj_input", s->relative, (size_t)rdim))
        return -1;
    if (waste_inkling_sconv_step(s->k, state->k_conv_state, w->k_sconv,
                                 kdim, cfg->conv_kernel, s->k) ||
        waste_inkling_sconv_step(s->v, state->v_conv_state, w->v_sconv,
                                 kdim, cfg->conv_kernel, s->v))
        return -1;
    if (trace_f(trace, layer, "k_sconv", s->k, (size_t)kdim) ||
        trace_f(trace, layer, "v_sconv", s->v, (size_t)kdim)) return -1;
    if (waste_inkling_attention_step_profile(&state->attention,
            s->q, s->k, s->v, w->q_norm, w->k_norm,
            s->relative, w->relative_proj, position,
            l->is_local ? 0 : cfg->log_scaling_n_floor,
            cfg->log_scaling_alpha, s->scores, s->attn_out, profile))
        return -1;
    if (trace_f(trace, layer, "attention_out", s->attn_out, (size_t)qdim))
        return -1;
    if (apply_matvec_profile(backend, layer, WASTE_IK_MAT_O, 0,
            s->branch, w->wo, s->attn_out, hidden, qdim, profile)) return -1;
    if (waste_inkling_sconv_step(s->branch, state->attn_conv_state,
                                 w->attn_sconv, hidden, cfg->conv_kernel,
                                 s->branch))
        return -1;
    if (trace_f(trace, layer, "attention_branch", s->branch, (size_t)hidden))
        return -1;
    if (bf16) {
        for (int i = 0; i < hidden; i++) {
            const float residual = waste_inkling_bf16_round(x[i]);
            const float branch = waste_inkling_bf16_round(s->branch[i]);
            x[i] = waste_inkling_bf16_round(residual + branch);
        }
    } else {
        for (int i = 0; i < hidden; i++) x[i] += s->branch[i];
    }
    if (trace_f(trace, layer, "post_attention_residual", x, (size_t)hidden))
        return -1;

    rmsnorm_profile(s->norm, x, w->post_attention_norm,
                    hidden, cfg->rms_eps, profile);
    if (trace_f(trace, layer, "post_attention_norm", s->norm, (size_t)hidden))
        return -1;
    if (!w->sparse) {
        if ((!backend && (!w->dense_gate || !w->dense_up || !w->dense_down)) ||
            !w->dense_global_scale) return -1;
        const int inter = cfg->dense_intermediate;
        if (apply_matvec_profile(backend, layer, WASTE_IK_MAT_DENSE_GATE, 0,
                s->gate, w->dense_gate, s->norm, inter, hidden, profile)) return -1;
        if (apply_matvec_profile(backend, layer, WASTE_IK_MAT_DENSE_UP, 0,
                s->up, w->dense_up, s->norm, inter, hidden, profile)) return -1;
        if (bf16) {
            bf16_gated_activation(s->gate, s->up, inter);
        } else {
            for (int i = 0; i < inter; i++) s->gate[i] = silu(s->gate[i]) * s->up[i];
        }
        if (apply_matvec_profile(backend, layer, WASTE_IK_MAT_DENSE_DOWN, 0,
                s->ff, w->dense_down, s->gate, hidden, inter, profile)) return -1;
        if (bf16) {
            const float scale = waste_inkling_bf16_round(*w->dense_global_scale);
            for (int i = 0; i < hidden; i++)
                s->ff[i] = waste_inkling_bf16_round(
                    waste_inkling_bf16_round(s->ff[i]) * scale);
        } else {
            for (int i = 0; i < hidden; i++) s->ff[i] *= *w->dense_global_scale;
        }
        if (trace_f(trace, layer, "dense_mlp_out", s->ff, (size_t)hidden))
            return -1;
    } else {
        if ((!backend && (!w->router_weight || !w->shared_gate ||
                          !w->shared_up || !w->shared_down)) ||
            !w->router_bias || !w->router_global_scale) return -1;
        const int total = cfg->n_routed_experts + cfg->n_shared_experts;
        const int inter = cfg->moe_intermediate;
        if (apply_matvec_profile(backend, layer, WASTE_IK_MAT_ROUTER, 0,
                s->router_logits, w->router_weight, s->norm,
                total, hidden, profile)) return -1;
        if (waste_inkling_route_profile(s->router_logits, w->router_bias,
                cfg->n_routed_experts, cfg->n_shared_experts, cfg->top_k,
                cfg->route_scale, *w->router_global_scale,
                s->routed_index, s->routed_weight, s->shared_weight, profile))
            return -1;
        if (trace_f(trace, layer, "router_logits", s->router_logits,
                    (size_t)cfg->n_routed_experts) ||
            trace_i(trace, layer, "routed_index", s->routed_index,
                    (size_t)cfg->top_k) ||
            trace_f(trace, layer, "routed_weight", s->routed_weight,
                    (size_t)cfg->top_k) ||
            trace_f(trace, layer, "shared_weight", s->shared_weight,
                    (size_t)cfg->n_shared_experts)) return -1;

        if (bf16) {
            /* Official expert reductions are ordered by expert id, not router
             * score slot. Keep each canonical (id, weight) pair together. */
            for (int a = 0; a < cfg->top_k; a++) {
                for (int b = a + 1; b < cfg->top_k; b++) {
                    if (s->routed_index[b] < s->routed_index[a]) {
                        const int ti = s->routed_index[a];
                        const float tw = s->routed_weight[a];
                        s->routed_index[a] = s->routed_index[b];
                        s->routed_weight[a] = s->routed_weight[b];
                        s->routed_index[b] = ti;
                        s->routed_weight[b] = tw;
                    }
                }
            }
            for (int i = 0; i < hidden; i++) s->ff[i] = 0.0f;
            for (int k = 0; k < cfg->top_k; k++) {
                const int expert = s->routed_index[k];
                /* In the BF16 profile every routed matrix is supplied by the
                 * numeric backend. Do not require a second expert_get storage
                 * path merely to validate an ID; that would reject the same
                 * nonresident/quantized layout the loader is designed to bind. */
                if (expert < 0 || expert >= cfg->n_routed_experts) return -1;
                if (apply_matvec_profile(backend, layer, WASTE_IK_MAT_ROUTED_GATE,
                        expert, s->gate, NULL, s->norm,
                        inter, hidden, profile) ||
                    apply_matvec_profile(backend, layer, WASTE_IK_MAT_ROUTED_UP,
                        expert, s->up, NULL, s->norm,
                        inter, hidden, profile)) return -1;
                if (k == 0 &&
                    (trace_f(trace, layer, "routed_gate0", s->gate,
                             (size_t)inter) ||
                     trace_f(trace, layer, "routed_up0", s->up,
                             (size_t)inter))) return -1;
                bf16_gated_activation(s->gate, s->up, inter);
                if (k == 0 && trace_f(trace, layer, "routed_gated0",
                                      s->gate, (size_t)inter)) return -1;
                if (apply_matvec_profile(backend, layer, WASTE_IK_MAT_ROUTED_DOWN,
                        expert, s->branch, NULL, s->gate,
                        hidden, inter, profile)) return -1;
                if (k == 0 && trace_f(trace, layer, "routed_down0",
                                      s->branch, (size_t)hidden)) return -1;
                const float gamma = waste_inkling_bf16_round(s->routed_weight[k]);
                for (int i = 0; i < hidden; i++) {
                    const float contribution = waste_inkling_bf16_round(
                        waste_inkling_bf16_round(s->branch[i]) * gamma);
                    s->ff[i] = waste_inkling_bf16_round(
                        waste_inkling_bf16_round(s->ff[i]) + contribution);
                }
            }

            /* Attention Q scratch is dead after the O projection and is at
             * least H for the measured Inkling profiles (checked above). Reuse
             * it for the independent FP32 shared-expert accumulator so the
             * established ctypes scratch layout and public planning contract do
             * not move during arithmetic promotion. */
            float *shared_accum = s->q;
            for (int i = 0; i < hidden; i++) shared_accum[i] = 0.0f;
            for (int e = 0; e < cfg->n_shared_experts; e++) {
                if (apply_matvec_profile(backend, layer, WASTE_IK_MAT_SHARED_GATE,
                        e, s->gate, NULL, s->norm, inter, hidden, profile) ||
                    apply_matvec_profile(backend, layer, WASTE_IK_MAT_SHARED_UP,
                        e, s->up, NULL, s->norm, inter, hidden, profile))
                    return -1;
                if (e == 0 &&
                    (trace_f(trace, layer, "shared_gate0", s->gate,
                             (size_t)inter) ||
                     trace_f(trace, layer, "shared_up0", s->up,
                             (size_t)inter))) return -1;
                bf16_gated_activation(s->gate, s->up, inter);
                if (e == 0 && trace_f(trace, layer, "shared_gated0",
                                      s->gate, (size_t)inter)) return -1;
                const float gamma = waste_inkling_bf16_round(s->shared_weight[e]);
                for (int j = 0; j < inter; j++)
                    s->gate[j] = waste_inkling_bf16_round(
                        waste_inkling_bf16_round(s->gate[j]) * gamma);
                if (apply_matvec_profile(backend, layer, WASTE_IK_MAT_SHARED_DOWN,
                        e, s->branch, NULL, s->gate,
                        hidden, inter, profile)) return -1;
                if (e == 0 && trace_f(trace, layer, "shared_down0",
                                      s->branch, (size_t)hidden)) return -1;
                for (int i = 0; i < hidden; i++) shared_accum[i] += s->branch[i];
            }
            for (int i = 0; i < hidden; i++) {
                const float routed = waste_inkling_bf16_round(s->ff[i]);
                const float shared = waste_inkling_bf16_round(shared_accum[i]);
                s->ff[i] = waste_inkling_bf16_round(routed + shared);
            }
        } else {
            for (int i = 0; i < hidden; i++) s->ff[i] = 0.0f;
            for (int k = 0; k < cfg->top_k; k++) {
                waste_inkling_expert_weights ew;
                if (get_routed(cfg, w, layer, s->routed_index[k],
                               expert_get, expert_ctx, &ew) ||
                    expert_eval(s->branch, s->gate, s->up, &ew,
                                s->norm, hidden, inter)) return -1;
                const float gamma = s->routed_weight[k];
                for (int i = 0; i < hidden; i++) s->ff[i] += gamma * s->branch[i];
            }
            const size_t gh = (size_t)inter * hidden;
            const size_t dh = (size_t)hidden * inter;
            for (int e = 0; e < cfg->n_shared_experts; e++) {
                if (w->shared_gate && w->shared_up && w->shared_down) {
                    waste_inkling_expert_weights ew = {
                        w->shared_gate + (size_t)e * gh,
                        w->shared_up + (size_t)e * gh,
                        w->shared_down + (size_t)e * dh,
                    };
                    if (expert_eval(s->branch, s->gate, s->up, &ew,
                                    s->norm, hidden, inter)) return -1;
                } else {
                    if (apply_matvec(backend, layer, WASTE_IK_MAT_SHARED_GATE, e,
                                     s->gate, NULL, s->norm, inter, hidden) ||
                        apply_matvec(backend, layer, WASTE_IK_MAT_SHARED_UP, e,
                                     s->up, NULL, s->norm, inter, hidden)) return -1;
                    for (int j = 0; j < inter; j++)
                        s->gate[j] = silu(s->gate[j]) * s->up[j];
                    if (apply_matvec(backend, layer, WASTE_IK_MAT_SHARED_DOWN, e,
                                     s->branch, NULL, s->gate,
                                     hidden, inter)) return -1;
                }
                const float gamma = s->shared_weight[e];
                for (int i = 0; i < hidden; i++) s->ff[i] += gamma * s->branch[i];
            }
        }
        if (trace_f(trace, layer, "moe_out", s->ff, (size_t)hidden)) return -1;
    }
    if (waste_inkling_sconv_step(s->ff, state->mlp_conv_state,
                                 w->mlp_sconv, hidden, cfg->conv_kernel,
                                 s->ff))
        return -1;
    if (trace_f(trace, layer, "mlp_branch", s->ff, (size_t)hidden)) return -1;
    if (bf16) {
        for (int i = 0; i < hidden; i++) {
            const float residual = waste_inkling_bf16_round(x[i]);
            const float branch = waste_inkling_bf16_round(s->ff[i]);
            x[i] = waste_inkling_bf16_round(residual + branch);
        }
    } else {
        for (int i = 0; i < hidden; i++) x[i] += s->ff[i];
    }
    if (trace_f(trace, layer, "layer_out", x, (size_t)hidden)) return -1;
    return 0;
}

int waste_inkling_layer_step_backend_trace(
    const waste_inkling_config *cfg, int layer,
    const waste_inkling_layer_weights *w,
    const waste_inkling_matrix_backend *backend,
    waste_inkling_layer_state *state,
    float *x, int position,
    waste_inkling_layer_scratch *s,
    waste_inkling_expert_get_fn expert_get, void *expert_ctx,
    const waste_inkling_trace *trace)
{
    return waste_inkling_layer_step_backend_trace_profile(
        cfg, layer, w, backend, state, x, position, s,
        expert_get, expert_ctx, trace, WASTE_INKLING_NUMERIC_F32);
}

int waste_inkling_layer_step_backend(
    const waste_inkling_config *cfg, int layer,
    const waste_inkling_layer_weights *weights,
    const waste_inkling_matrix_backend *backend,
    waste_inkling_layer_state *state,
    float *x, int position,
    waste_inkling_layer_scratch *scratch,
    waste_inkling_expert_get_fn expert_get, void *expert_ctx)
{
    return waste_inkling_layer_step_backend_trace(cfg, layer, weights, backend,
        state, x, position, scratch, expert_get, expert_ctx, NULL);
}

int waste_inkling_layer_step(
    const waste_inkling_config *cfg, int layer,
    const waste_inkling_layer_weights *weights,
    waste_inkling_layer_state *state,
    float *x, int position,
    waste_inkling_layer_scratch *scratch,
    waste_inkling_expert_get_fn expert_get, void *expert_ctx)
{
    return waste_inkling_layer_step_backend(cfg, layer, weights, NULL, state,
        x, position, scratch, expert_get, expert_ctx);
}