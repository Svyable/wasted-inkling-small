/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
#ifndef WASTE_INKLING_CONFIG_H
#define WASTE_INKLING_CONFIG_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define WASTE_INKLING_MAX_LAYERS 128

typedef struct {
    int is_local;
    int num_heads;
    int num_kv_heads;
    int head_dim;
    int relative_extent;
} waste_inkling_layer_cfg;

typedef struct {
    int n_layers;
    int hidden;
    int vocab;
    int unpadded_vocab;
    int max_context;

    int global_heads;
    int global_kv_heads;
    int global_head_dim;
    int local_heads;
    int local_kv_heads;
    int local_head_dim;
    int sliding_window;
    int d_rel;
    int rel_extent;
    int conv_kernel;

    int dense_layers;
    int dense_intermediate;
    int moe_intermediate;
    int n_routed_experts;
    int top_k;
    int n_shared_experts;

    float rms_eps;
    float route_scale;
    float logits_width_multiplier;
    int log_scaling_n_floor;
    float log_scaling_alpha;

    waste_inkling_layer_cfg layer[WASTE_INKLING_MAX_LAYERS];
} waste_inkling_config;

typedef struct {
    int n_layers;
    int hidden;
    int vocab;
    int unpadded_vocab;
    int max_context;

    int global_heads;
    int global_kv_heads;
    int global_head_dim;
    int local_heads;
    int local_kv_heads;
    int local_head_dim;
    int sliding_window;
    int d_rel;
    int rel_extent;
    int conv_kernel;

    int dense_layers;
    int dense_intermediate;
    int moe_intermediate;
    int n_routed_experts;
    int top_k;
    int n_shared_experts;

    float rms_eps;
    float route_scale;
    float logits_width_multiplier;
    int log_scaling_n_floor;
    float log_scaling_alpha;

    const int *local_layer_ids;
    int n_local_layers;
} waste_inkling_config_args;

typedef struct {
    uint64_t kv_bytes;
    uint64_t conv_bytes;
    uint64_t state_bytes;

    uint64_t token_vector_bytes;
    uint64_t projection_bytes;
    uint64_t attention_score_bytes;
    uint64_t router_bytes;
    uint64_t expert_workspace_bytes;
    uint64_t dense_workspace_bytes;
    uint64_t shared_workspace_bytes;
    uint64_t logits_bytes;
    uint64_t decode_scratch_bytes;
} waste_inkling_memory;

/* Validate a normalized Inkling text config and derive every layer's exact
 * local/global attention geometry. Returns 0, or -1 for invalid values. */
int waste_inkling_config_build(waste_inkling_config *out,
                               const waste_inkling_config_args *args);

/* Memory contract for a future one-token Inkling decode implementation.
 * State includes per-layer K/V caches and all four fp32 short-convolution
 * rings. Scratch is broken out so the eventual allocator can be checked field
 * by field instead of relying on a flat estimate. Returns 0, or -1 on invalid
 * input/overflow. */
int waste_inkling_plan_decode_memory(const waste_inkling_config *cfg,
                                     uint32_t ctx_tokens,
                                     waste_inkling_memory *out);

/* How many logits a caller must provide room for, and how many the model
 * writes. Exposed as a function rather than left to the caller reading the
 * struct, because every out-of-process consumer that wants one integer would
 * otherwise redeclare waste_inkling_config in its own binding — and a partial
 * redeclaration of a C struct is exactly what silently overran a buffer in
 * tools/inkling_layer_parity.py while every test still passed. Returns 0 for
 * a NULL config. */
int waste_inkling_config_unpadded_vocab(const waste_inkling_config *cfg);

#ifdef __cplusplus
}
#endif
#endif
