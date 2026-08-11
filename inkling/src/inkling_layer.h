/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
#ifndef WASTE_INKLING_LAYER_H
#define WASTE_INKLING_LAYER_H

#include <stddef.h>

#include "inkling_attention.h"
#include "inkling_config.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    const float *gate; /* [intermediate][hidden] */
    const float *up;   /* [intermediate][hidden] */
    const float *down; /* [hidden][intermediate] */
} waste_inkling_expert_weights;

typedef int (*waste_inkling_expert_get_fn)(void *ctx, int layer,
                                            int expert,
                                            waste_inkling_expert_weights *out);

typedef struct {
    const float *input_norm;          /* [hidden] */
    const float *post_attention_norm; /* [hidden] */
    const float *wq;                  /* [heads*head_dim][hidden] */
    const float *wk;                  /* [kv_heads*head_dim][hidden] */
    const float *wv;                  /* [kv_heads*head_dim][hidden] */
    const float *wr;                  /* [heads*d_rel][hidden] */
    const float *wo;                  /* [hidden][heads*head_dim] */
    const float *q_norm;              /* [head_dim] */
    const float *k_norm;              /* [head_dim] */
    const float *relative_proj;       /* [d_rel][relative_extent] */
    const float *k_sconv;             /* [kv_heads*head_dim][conv_kernel] */
    const float *v_sconv;             /* [kv_heads*head_dim][conv_kernel] */
    const float *attn_sconv;          /* [hidden][conv_kernel] */
    const float *mlp_sconv;           /* [hidden][conv_kernel] */

    int sparse;

    /* Dense layer. */
    const float *dense_gate;          /* [dense_intermediate][hidden] */
    const float *dense_up;            /* [dense_intermediate][hidden] */
    const float *dense_down;          /* [hidden][dense_intermediate] */
    const float *dense_global_scale;  /* scalar */

    /* Sparse layer. Router rows include routed experts followed by shared. */
    const float *router_weight;       /* [routed+shared][hidden] */
    const float *router_bias;         /* [routed] */
    const float *router_global_scale; /* scalar */
    const float *shared_gate;         /* [shared][moe_intermediate][hidden] */
    const float *shared_up;           /* [shared][moe_intermediate][hidden] */
    const float *shared_down;         /* [shared][hidden][moe_intermediate] */

    /* Optional resident routed arrays for tests/small fixtures and the legacy
     * F32 path. F32 streaming may leave these NULL and provide expert_get;
     * BF16_REFERENCE routes expert matrices through the matrix backend only. */
    const float *routed_gate;         /* [routed][moe_intermediate][hidden] */
    const float *routed_up;
    const float *routed_down;         /* [routed][hidden][moe_intermediate] */
} waste_inkling_layer_weights;

typedef enum {
    WASTE_IK_MAT_Q = 0, WASTE_IK_MAT_K, WASTE_IK_MAT_V, WASTE_IK_MAT_R,
    WASTE_IK_MAT_O, WASTE_IK_MAT_DENSE_GATE, WASTE_IK_MAT_DENSE_UP,
    WASTE_IK_MAT_DENSE_DOWN, WASTE_IK_MAT_ROUTER, WASTE_IK_MAT_SHARED_GATE,
    WASTE_IK_MAT_SHARED_UP, WASTE_IK_MAT_SHARED_DOWN,
    /* Appended so every pre-promotion kind keeps its numeric value.  The BF16
     * evidence profile uses these callbacks to avoid silently evaluating a
     * routed expert through the scalar F32 resident matvec. */
    WASTE_IK_MAT_ROUTED_GATE, WASTE_IK_MAT_ROUTED_UP, WASTE_IK_MAT_ROUTED_DOWN,
    /* Vocabulary projection of the final head.  Appended for the same reason:
     * the BF16 evidence profile must reach the real kernel instead of widening
     * a resident table into the scalar F32 dot product.  Callbacks for this
     * kind receive layer -1, index 0, and exactly the selected vocabulary rows
     * in selection order. */
    WASTE_IK_MAT_UNEMBED
} waste_inkling_matrix_kind;

/* RMS normalization under a numeric profile. Exposed so the final head shares
 * the layer's definition instead of carrying a second copy of the policy.
 * F32 is the original checked-in expression; BF16_REFERENCE is the ordering the
 * retained official-weight evidence established. */
void waste_inkling_rmsnorm_profile(float *out, const float *x,
                                   const float *weight, int n, float eps,
                                   waste_inkling_numeric_profile profile);

typedef int (*waste_inkling_matvec_fn)(void *ctx, int layer,
    waste_inkling_matrix_kind kind, int index, const float *x, float *out,
    int rows, int cols);

typedef struct {
    waste_inkling_matvec_fn matvec;
    void *ctx;
} waste_inkling_matrix_backend;

typedef int (*waste_inkling_trace_float_fn)(void *ctx, int layer,
                                             const char *point,
                                             const float *data, size_t count);
typedef int (*waste_inkling_trace_int_fn)(void *ctx, int layer,
                                           const char *point,
                                           const int *data, size_t count);
typedef struct {
    waste_inkling_trace_float_fn emit_float;
    waste_inkling_trace_int_fn emit_int;
    void *ctx;
} waste_inkling_trace;

typedef struct {
    waste_inkling_attention_state attention;
    float *k_conv_state;
    float *v_conv_state;
    float *attn_conv_state;
    float *mlp_conv_state;
} waste_inkling_layer_state;

typedef struct {
    float *norm;
    float *q;
    float *k;
    float *v;
    float *relative;
    float *scores;
    float *attn_out;
    float *branch;
    float *gate;
    float *up;
    float *ff;
    float *router_logits;
    float *routed_weight;
    float *shared_weight;
    int *routed_index;
    size_t float_count;
    size_t int_count;
} waste_inkling_layer_scratch;

size_t waste_inkling_layer_scratch_floats(const waste_inkling_config *cfg,
                                           int layer, int attention_capacity);
size_t waste_inkling_layer_scratch_ints(const waste_inkling_config *cfg);
int waste_inkling_layer_scratch_init(waste_inkling_layer_scratch *scratch,
                                     const waste_inkling_config *cfg,
                                     int layer, int attention_capacity,
                                     float *float_buffer, size_t float_count,
                                     int *int_buffer, size_t int_count);

int waste_inkling_layer_state_init(
    waste_inkling_layer_state *state,
    const waste_inkling_config *cfg, int layer, int attention_capacity,
    float *k_cache, float *v_cache,
    float *k_conv_state, float *v_conv_state,
    float *attn_conv_state, float *mlp_conv_state);

/* Execute one F32 decoder layer in place. K/V inputs are short-convolved before
 * the attention core; attention and MLP branch outputs are short-convolved
 * before the outer residual addition, matching the official Inkling order.
 * Routed experts may be resident in `weights` or supplied by `expert_get`.
 * Returns 0 or -1. */
int waste_inkling_layer_step(
    const waste_inkling_config *cfg, int layer,
    const waste_inkling_layer_weights *weights,
    waste_inkling_layer_state *state,
    float *x, int position,
    waste_inkling_layer_scratch *scratch,
    waste_inkling_expert_get_fn expert_get, void *expert_ctx);

/* Same decoder semantics with optional non-resident matrix storage. A backend
 * is consulted only when the corresponding resident pointer is NULL. */
int waste_inkling_layer_step_backend(
    const waste_inkling_config *cfg, int layer,
    const waste_inkling_layer_weights *weights,
    const waste_inkling_matrix_backend *backend,
    waste_inkling_layer_state *state,
    float *x, int position,
    waste_inkling_layer_scratch *scratch,
    waste_inkling_expert_get_fn expert_get, void *expert_ctx);

int waste_inkling_layer_step_backend_trace(
    const waste_inkling_config *cfg, int layer,
    const waste_inkling_layer_weights *weights,
    const waste_inkling_matrix_backend *backend,
    waste_inkling_layer_state *state,
    float *x, int position,
    waste_inkling_layer_scratch *scratch,
    waste_inkling_expert_get_fn expert_get, void *expert_ctx,
    const waste_inkling_trace *trace);

/* Internal promotion seam. F32 delegates the exact legacy path. The measured
 * BF16_REFERENCE profile covers dense and sparse Inkling layer classes and
 * requires the numeric matrix backend for kernel-sensitive BF16 matmuls.
 * Unsupported backend/geometry paths fail closed rather than falling back to
 * a scalar F32 matvec and manufacturing a parity claim. */
int waste_inkling_layer_step_backend_trace_profile(
    const waste_inkling_config *cfg, int layer,
    const waste_inkling_layer_weights *weights,
    const waste_inkling_matrix_backend *backend,
    waste_inkling_layer_state *state,
    float *x, int position,
    waste_inkling_layer_scratch *scratch,
    waste_inkling_expert_get_fn expert_get, void *expert_ctx,
    const waste_inkling_trace *trace,
    waste_inkling_numeric_profile profile);

#ifdef __cplusplus
}
#endif
#endif