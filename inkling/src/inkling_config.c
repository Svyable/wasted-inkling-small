/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
#include "inkling_config.h"

#include <limits.h>
#include <stddef.h>
#include <string.h>

static int mul_u64(uint64_t a, uint64_t b, uint64_t *out)
{
    if (a && b > UINT64_MAX / a) return -1;
    *out = a * b;
    return 0;
}

static int add_u64(uint64_t *dst, uint64_t value)
{
    if (value > UINT64_MAX - *dst) return -1;
    *dst += value;
    return 0;
}

static int bytes4(uint64_t count, uint64_t *out)
{
    return mul_u64(count, 4, out);
}

static int positive(int v) { return v > 0; }

int waste_inkling_config_build(waste_inkling_config *out,
                               const waste_inkling_config_args *a)
{
    int seen[WASTE_INKLING_MAX_LAYERS];
    if (!out || !a) return -1;
    if (a->n_layers < 1 || a->n_layers > WASTE_INKLING_MAX_LAYERS ||
        !positive(a->hidden) || !positive(a->vocab) ||
        !positive(a->unpadded_vocab) || a->unpadded_vocab > a->vocab ||
        !positive(a->max_context) ||
        !positive(a->global_heads) || !positive(a->global_kv_heads) ||
        !positive(a->global_head_dim) ||
        !positive(a->local_heads) || !positive(a->local_kv_heads) ||
        !positive(a->local_head_dim) || !positive(a->sliding_window) ||
        !positive(a->d_rel) || !positive(a->rel_extent) ||
        a->conv_kernel < 1 || a->conv_kernel > 64 ||
        a->dense_layers < 0 || a->dense_layers > a->n_layers ||
        !positive(a->dense_intermediate) || !positive(a->moe_intermediate) ||
        !positive(a->n_routed_experts) || !positive(a->top_k) ||
        a->top_k > a->n_routed_experts || !positive(a->n_shared_experts) ||
        !(a->rms_eps > 0.0f) || !(a->route_scale > 0.0f) ||
        !(a->logits_width_multiplier > 0.0f) ||
        a->log_scaling_n_floor < 0 || a->log_scaling_alpha < 0.0f ||
        a->global_heads % a->global_kv_heads != 0 ||
        a->local_heads % a->local_kv_heads != 0 ||
        a->n_local_layers < 0 || a->n_local_layers > a->n_layers ||
        (a->n_local_layers && !a->local_layer_ids))
        return -1;

    memset(out, 0, sizeof *out);
    memset(seen, 0, sizeof seen);
    for (int i = 0; i < a->n_local_layers; i++) {
        const int layer = a->local_layer_ids[i];
        if (layer < 0 || layer >= a->n_layers || seen[layer]) return -1;
        seen[layer] = 1;
    }

    out->n_layers = a->n_layers;
    out->hidden = a->hidden;
    out->vocab = a->vocab;
    out->unpadded_vocab = a->unpadded_vocab;
    out->max_context = a->max_context;
    out->global_heads = a->global_heads;
    out->global_kv_heads = a->global_kv_heads;
    out->global_head_dim = a->global_head_dim;
    out->local_heads = a->local_heads;
    out->local_kv_heads = a->local_kv_heads;
    out->local_head_dim = a->local_head_dim;
    out->sliding_window = a->sliding_window;
    out->d_rel = a->d_rel;
    out->rel_extent = a->rel_extent;
    out->conv_kernel = a->conv_kernel;
    out->dense_layers = a->dense_layers;
    out->dense_intermediate = a->dense_intermediate;
    out->moe_intermediate = a->moe_intermediate;
    out->n_routed_experts = a->n_routed_experts;
    out->top_k = a->top_k;
    out->n_shared_experts = a->n_shared_experts;
    out->rms_eps = a->rms_eps;
    out->route_scale = a->route_scale;
    out->logits_width_multiplier = a->logits_width_multiplier;
    out->log_scaling_n_floor = a->log_scaling_n_floor;
    out->log_scaling_alpha = a->log_scaling_alpha;

    for (int i = 0; i < a->n_layers; i++) {
        waste_inkling_layer_cfg *l = &out->layer[i];
        l->is_local = seen[i];
        l->num_heads = seen[i] ? a->local_heads : a->global_heads;
        l->num_kv_heads = seen[i] ? a->local_kv_heads : a->global_kv_heads;
        l->head_dim = seen[i] ? a->local_head_dim : a->global_head_dim;
        l->relative_extent = seen[i] ? a->sliding_window : a->rel_extent;
    }
    return 0;
}

int waste_inkling_plan_decode_memory(const waste_inkling_config *cfg,
                                     uint32_t ctx_tokens,
                                     waste_inkling_memory *out)
{
    uint64_t max_attention = 0;
    uint64_t n, b;
    if (!cfg || !out || ctx_tokens == 0 ||
        ctx_tokens > (uint32_t)cfg->max_context ||
        cfg->n_layers < 1 || cfg->n_layers > WASTE_INKLING_MAX_LAYERS)
        return -1;
    memset(out, 0, sizeof *out);

    for (int i = 0; i < cfg->n_layers; i++) {
        const waste_inkling_layer_cfg *l = &cfg->layer[i];
        const uint64_t tokens = l->is_local && ctx_tokens > (uint32_t)cfg->sliding_window
                              ? (uint64_t)cfg->sliding_window : (uint64_t)ctx_tokens;
        if (!positive(l->num_heads) || !positive(l->num_kv_heads) ||
            !positive(l->head_dim) || !positive(l->relative_extent)) return -1;

        /* K and V are retained in fp32 at [tokens][kv_heads][head_dim]. */
        if (mul_u64(tokens, (uint64_t)2 * (uint64_t)l->num_kv_heads, &n) ||
            mul_u64(n, (uint64_t)l->head_dim, &n) || bytes4(n, &b) ||
            add_u64(&out->kv_bytes, b)) return -1;

        /* Four conv rings: K, V, attention branch and MLP branch. */
        n = (uint64_t)2 * (uint64_t)l->num_kv_heads * (uint64_t)l->head_dim
          + (uint64_t)2 * (uint64_t)cfg->hidden;
        if (mul_u64(n, (uint64_t)cfg->conv_kernel, &n) || bytes4(n, &b) ||
            add_u64(&out->conv_bytes, b)) return -1;

        if (mul_u64(tokens, (uint64_t)l->num_heads, &n) || n > max_attention)
            max_attention = n;
    }
    out->state_bytes = out->kv_bytes;
    if (add_u64(&out->state_bytes, out->conv_bytes)) return -1;

    /* Exact fields of the planned one-token decoder workspace contract. */
    if (bytes4((uint64_t)8 * (uint64_t)cfg->hidden, &out->token_vector_bytes)) return -1;
    {
        uint64_t max_projection = 0;
        for (int i = 0; i < cfg->n_layers; i++) {
            const waste_inkling_layer_cfg *l = &cfg->layer[i];
            n = (uint64_t)l->num_heads * (uint64_t)l->head_dim
              + (uint64_t)2 * (uint64_t)l->num_kv_heads * (uint64_t)l->head_dim
              + (uint64_t)l->num_heads * (uint64_t)cfg->d_rel;
            if (n > max_projection) max_projection = n;
        }
        if (bytes4(max_projection, &out->projection_bytes)) return -1;
    }
    if (bytes4(max_attention, &out->attention_score_bytes)) return -1;

    n = (uint64_t)2 * (uint64_t)cfg->n_routed_experts
      + (uint64_t)cfg->n_shared_experts
      + (uint64_t)2 * (uint64_t)cfg->top_k;
    if (bytes4(n, &out->router_bytes) ||
        add_u64(&out->router_bytes, (uint64_t)cfg->top_k * sizeof(int))) return -1;

    n = (uint64_t)3 * (uint64_t)cfg->hidden * (uint64_t)cfg->moe_intermediate
      + (uint64_t)2 * (uint64_t)cfg->moe_intermediate
      + (uint64_t)cfg->hidden;
    if (bytes4(n, &out->expert_workspace_bytes)) return -1;

    n = (uint64_t)2 * (uint64_t)cfg->dense_intermediate + (uint64_t)cfg->hidden;
    if (bytes4(n, &out->dense_workspace_bytes)) return -1;
    n = (uint64_t)2 * (uint64_t)cfg->moe_intermediate + (uint64_t)cfg->hidden;
    if (bytes4(n, &out->shared_workspace_bytes)) return -1;
    if (bytes4((uint64_t)cfg->unpadded_vocab, &out->logits_bytes)) return -1;

    out->decode_scratch_bytes = 0;
#define ADD_FIELD(name) do { if (add_u64(&out->decode_scratch_bytes, out->name)) return -1; } while (0)
    ADD_FIELD(token_vector_bytes);
    ADD_FIELD(projection_bytes);
    ADD_FIELD(attention_score_bytes);
    ADD_FIELD(router_bytes);
    ADD_FIELD(expert_workspace_bytes);
    ADD_FIELD(dense_workspace_bytes);
    ADD_FIELD(shared_workspace_bytes);
    ADD_FIELD(logits_bytes);
#undef ADD_FIELD
    return 0;
}

int waste_inkling_config_unpadded_vocab(const waste_inkling_config *cfg)
{
    return cfg ? cfg->unpadded_vocab : 0;
}
