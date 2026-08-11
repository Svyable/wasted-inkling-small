/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
/*
 * inkling_manifest.c — the manifest's `config` object, read once.
 *
 * This used to live inside the planner, which was fine while planning was
 * the only public capability Inkling had. It is not any more: the loader
 * builds the same geometry from the same object, and two readers of one
 * schema is how a container gets planned as one model and opened as
 * another. So the reader moved to the seam between them and neither owns
 * it.
 *
 * No manifest schema is invented here. Every field read below already
 * exists in format v0.
 */

#include "inkling_manifest.h"

#include <string.h>

/* Read an integer that must be present and positive. */
static int req_int(const js_doc *d, int obj, const char *key, int *out)
{
    const int tok = js_get(d, obj, key);
    if (tok < 0) return -1;
    const int64_t v = js_int(d, tok, 0);
    if (v <= 0 || v > 0x7fffffff) return -1;
    *out = (int)v;
    return 0;
}

/* Read an integer that may be absent, in which case the default stands. */
static void opt_int(const js_doc *d, int obj, const char *key, int *out)
{
    const int tok = js_get(d, obj, key);
    if (tok >= 0) {
        const int64_t v = js_int(d, tok, *out);
        if (v >= 0 && v <= 0x7fffffff) *out = (int)v;
    }
}

static int req_num(const js_doc *d, int obj, const char *key, float *out)
{
    const int tok = js_get(d, obj, key);
    if (tok < 0) return -1;
    const double v = js_num(d, tok, 0.0);
    if (!(v > 0.0) || v > 1e30) return -1;
    *out = (float)v;
    return 0;
}

int waste_inkling_manifest_config_token(const js_doc *d)
{
    if (!d) return -1;
    const int cfg = js_get(d, 0, "config");
    const int nested = js_get(d, cfg, "text_config");
    return nested >= 0 ? nested : cfg;
}

int waste_inkling_manifest_config(const js_doc *d, int cfg,
                                  waste_inkling_config *out)
{
    waste_inkling_config_args a;
    int local_ids[WASTE_INKLING_MAX_LAYERS];
    int n_local = 0;

    if (!d || cfg < 0 || !out) return -1;
    memset(&a, 0, sizeof a);
    if (req_int(d, cfg, "num_hidden_layers", &a.n_layers) ||
        req_int(d, cfg, "hidden_size", &a.hidden) ||
        req_int(d, cfg, "vocab_size", &a.vocab) ||
        req_int(d, cfg, "model_max_length", &a.max_context) ||
        req_int(d, cfg, "num_attention_heads", &a.global_heads) ||
        req_int(d, cfg, "num_key_value_heads", &a.global_kv_heads) ||
        req_int(d, cfg, "head_dim", &a.global_head_dim) ||
        req_int(d, cfg, "sliding_window_size", &a.sliding_window) ||
        req_int(d, cfg, "d_rel", &a.d_rel) ||
        req_int(d, cfg, "rel_extent", &a.rel_extent) ||
        req_int(d, cfg, "sconv_kernel_size", &a.conv_kernel) ||
        req_int(d, cfg, "dense_intermediate_size", &a.dense_intermediate) ||
        req_int(d, cfg, "intermediate_size", &a.moe_intermediate) ||
        req_int(d, cfg, "n_routed_experts", &a.n_routed_experts) ||
        req_int(d, cfg, "num_experts_per_tok", &a.top_k) ||
        req_int(d, cfg, "n_shared_experts", &a.n_shared_experts) ||
        req_num(d, cfg, "rms_norm_eps", &a.rms_eps) ||
        req_num(d, cfg, "route_scale", &a.route_scale) ||
        req_num(d, cfg, "logits_mup_width_multiplier", &a.logits_width_multiplier))
        return -1;

    /* The sliding-window tower may restate the head geometry; when it does
     * not, it shares the global values. Absent is not the same as zero, so
     * these are seeded before the optional read. */
    a.local_heads = a.global_heads;
    a.local_kv_heads = a.global_kv_heads;
    a.local_head_dim = a.global_head_dim;
    opt_int(d, cfg, "swa_num_attention_heads", &a.local_heads);
    opt_int(d, cfg, "swa_num_key_value_heads", &a.local_kv_heads);
    opt_int(d, cfg, "swa_head_dim", &a.local_head_dim);

    a.unpadded_vocab = a.vocab;
    opt_int(d, cfg, "unpadded_vocab_size", &a.unpadded_vocab);
    opt_int(d, cfg, "dense_mlp_idx", &a.dense_layers);
    opt_int(d, cfg, "log_scaling_n_floor", &a.log_scaling_n_floor);
    {
        const int tok = js_get(d, cfg, "log_scaling_alpha");
        if (tok >= 0) {
            const double v = js_num(d, tok, 0.0);
            if (v < 0.0 || v > 1e30) return -1;
            a.log_scaling_alpha = (float)v;
        }
    }

    {
        const int arr = js_get(d, cfg, "local_layer_ids");
        const int n = js_size(d, arr);
        if (n > WASTE_INKLING_MAX_LAYERS) return -1;
        for (int i = 0; i < n; i++) {
            const int64_t v = js_int(d, js_at(d, arr, i), -1);
            if (v < 0 || v >= a.n_layers) return -1;
            local_ids[n_local++] = (int)v;
        }
    }
    a.local_layer_ids = local_ids;
    a.n_local_layers = n_local;

    return waste_inkling_config_build(out, &a) ? -1 : 0;
}
