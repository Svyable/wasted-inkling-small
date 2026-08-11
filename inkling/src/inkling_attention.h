/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
#ifndef WASTE_INKLING_ATTENTION_H
#define WASTE_INKLING_ATTENTION_H

#include "inkling_numeric.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int is_local;
    int num_heads;
    int num_kv_heads;
    int head_dim;
    int d_rel;
    int relative_extent;
    int capacity;
    float rms_eps;
    int next_position;
    float *k_cache;  /* [capacity][num_kv_heads][head_dim], normalized */
    float *v_cache;  /* [capacity][num_kv_heads][head_dim]             */
} waste_inkling_attention_state;

/* Initialize a caller-owned one-token attention state. Local layers use a
 * ring of `capacity` positions; global layers use positions 0..capacity-1.
 * Cache storage is caller-owned and must hold capacity*num_kv_heads*head_dim
 * floats in each buffer. */
int waste_inkling_attention_init(waste_inkling_attention_state *state,
                                 int is_local, int num_heads,
                                 int num_kv_heads, int head_dim,
                                 int d_rel, int relative_extent,
                                 int capacity, float rms_eps,
                                 float *k_cache, float *v_cache);

void waste_inkling_attention_reset(waste_inkling_attention_state *state);

/* Exact one-token eager Inkling attention core after projection and K/V short
 * convolution, before output projection.
 *
 * q:              [num_heads][head_dim]
 * k, v:           [num_kv_heads][head_dim]
 * q_norm, k_norm: learned per-head-dimension RMSNorm weights [head_dim]
 * relative_state: [num_heads][d_rel]
 * relative_proj:  [d_rel][relative_extent]
 * scores:         scratch [num_heads][capacity]
 * out:            [num_heads][head_dim]
 *
 * `position` must be sequential. Set log_scaling_n_floor to zero for local
 * layers; global layers use the configured floor and alpha. Returns 0 or -1
 * for invalid geometry, non-sequential input, or a full global cache. */
int waste_inkling_attention_step(
    waste_inkling_attention_state *state,
    const float *q, const float *k, const float *v,
    const float *q_norm, const float *k_norm,
    const float *relative_state, const float *relative_proj,
    int position, int log_scaling_n_floor, float log_scaling_alpha,
    float *scores, float *out);

/* Internal numeric-profile variant. F32 is the exact legacy behavior above.
 * BF16_REFERENCE implements the retained portable attention policy: BF16-bound
 * head RMSNorm, K/V cache operands, score reductions/scaling, probability
 * operands, and completed attention output. */
int waste_inkling_attention_step_profile(
    waste_inkling_attention_state *state,
    const float *q, const float *k, const float *v,
    const float *q_norm, const float *k_norm,
    const float *relative_state, const float *relative_proj,
    int position, int log_scaling_n_floor, float log_scaling_alpha,
    float *scores, float *out,
    waste_inkling_numeric_profile profile);

#ifdef __cplusplus
}
#endif
#endif
