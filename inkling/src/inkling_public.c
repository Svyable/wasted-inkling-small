/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
/*
 * inkling_public.c — manifest to memory plan, and nothing else.
 *
 * The public loader refuses Inkling because a forward path that has never
 * seen the official weights should refuse. Planning is a different kind of
 * claim: it reads dimensions out of a manifest and does arithmetic that is
 * tested field by field in waste_inkling_plan_decode_memory(). The worst a
 * bug here produces is a wrong byte count, which the caller finds out about
 * immediately, rather than a wrong token, which it does not.
 *
 * No new manifest schema is invented. Every field read below already exists
 * in format v0: `config` for architecture, `trunk[]` for resident bytes,
 * `layers[]` for the expert record size. That is the point — the private
 * runtime index was always a rehearsal of the public manifest.
 *
 * Everything fails closed. A missing dimension, an out-of-range value, or a
 * config the builder rejects returns WASTE_E_FORMAT.
 */

#include "inkling_public.h"

#include <string.h>

#include "inkling_config.h"
#include "json.h"

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

static int add64(uint64_t *dst, uint64_t v)
{
    if (v > UINT64_MAX - *dst) return -1;
    *dst += v;
    return 0;
}

static int mul64(uint64_t a, uint64_t b, uint64_t *out)
{
    if (a && b > UINT64_MAX / a) return -1;
    *out = a * b;
    return 0;
}

/* The vocabulary tables stay on disk and are read one row per token, so they
 * are not part of the resident floor. Inkling keeps *both* of them row-backed
 * — the embedding and the independent unembedding — which is one table more
 * than the Kimi path leaves out. */
static int row_backed(const char *name)
{
    const size_t n = strlen(name);
    static const char *tails[] = { ".embed", ".unembed",
                                   "embed_tokens.weight", "unembed.weight" };
    for (size_t i = 0; i < sizeof tails / sizeof tails[0]; i++) {
        const size_t t = strlen(tails[i]);
        if (n >= t && memcmp(name + n - t, tails[i], t) == 0) return 1;
    }
    return 0;
}

static waste_status plan_config(const js_doc *d, int cfg,
                                waste_inkling_config *out)
{
    waste_inkling_config_args a;
    int local_ids[WASTE_INKLING_MAX_LAYERS];
    int n_local = 0;

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
        return WASTE_E_FORMAT;

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
            if (v < 0.0 || v > 1e30) return WASTE_E_FORMAT;
            a.log_scaling_alpha = (float)v;
        }
    }

    {
        const int arr = js_get(d, cfg, "local_layer_ids");
        const int n = js_size(d, arr);
        if (n > WASTE_INKLING_MAX_LAYERS) return WASTE_E_FORMAT;
        for (int i = 0; i < n; i++) {
            const int64_t v = js_int(d, js_at(d, arr, i), -1);
            if (v < 0 || v >= a.n_layers) return WASTE_E_FORMAT;
            local_ids[n_local++] = (int)v;
        }
    }
    a.local_layer_ids = local_ids;
    a.n_local_layers = n_local;

    /* The builder is the validator: divisibility, bounds, duplicate local
     * ids, top_k against the routed count. Nothing here second-guesses it. */
    if (waste_inkling_config_build(out, &a)) return WASTE_E_FORMAT;
    return WASTE_OK;
}

waste_status waste_inkling_plan_memory_json(const char *manifest_json,
                                            uint32_t ctx_tokens,
                                            waste_memplan *out)
{
    js_doc d;
    waste_inkling_config cfg;
    waste_inkling_memory mem;
    waste_status st;

    if (!manifest_json || !out || ctx_tokens == 0) return WASTE_E_ARG;
    if (js_parse(&d, manifest_json) < 0) return WASTE_E_FORMAT;

    {
        /* The converter flattens the release's text config into `config`,
         * with the multimodal wrapper kept under `_outer`, exactly as K3
         * does. A nested text_config is accepted for a raw release config. */
        int cfg_tok = js_get(&d, 0, "config");
        const int nested = js_get(&d, cfg_tok, "text_config");
        if (nested >= 0) cfg_tok = nested;
        if (cfg_tok < 0) { js_free(&d); return WASTE_E_FORMAT; }
        st = plan_config(&d, cfg_tok, &cfg);
        if (st != WASTE_OK) { js_free(&d); return st; }
    }

    if (ctx_tokens > (uint32_t)cfg.max_context) { js_free(&d); return WASTE_E_ARG; }
    if (waste_inkling_plan_decode_memory(&cfg, ctx_tokens, &mem)) {
        js_free(&d);
        return WASTE_E_FORMAT;
    }

    memset(out, 0, sizeof *out);
    out->state_bytes = mem.state_bytes;
    out->scratch_bytes = mem.decode_scratch_bytes;

    {   /* Resident trunk: every stored tensor except the two the engine
         * reads one row at a time. */
        const int trunk = js_get(&d, 0, "trunk");
        const int n = js_size(&d, trunk);
        for (int i = 0; i < n; i++) {
            const int e = js_at(&d, trunk, i);
            char name[160];
            js_str(&d, js_get(&d, e, "name"), name, sizeof name);
            if (row_backed(name)) continue;
            const int64_t bytes = js_int(&d, js_get(&d, e, "bytes"), 0);
            if (bytes < 0 || add64(&out->trunk_bytes, (uint64_t)bytes)) {
                js_free(&d);
                return WASTE_E_FORMAT;
            }
        }
    }

    {   /* One layer's top-k routed experts, double buffered — the same
         * definition of "smallest cache that can run a layer" the Kimi path
         * uses, read from the first bank's record size. */
        const int layers = js_get(&d, 0, "layers");
        uint64_t rec = 0;
        if (js_size(&d, layers) > 0) {
            const int first = js_at(&d, layers, 0);
            const int64_t bytes = js_int(&d, js_get(&d, first, "bytes"), 0);
            const int64_t count = js_int(&d, js_get(&d, first, "experts"), 0);
            if (bytes < 0 || count < 0) { js_free(&d); return WASTE_E_FORMAT; }
            rec = count ? (uint64_t)bytes / (uint64_t)count : 0;
        }
        if (mul64(rec, (uint64_t)cfg.top_k * 2u, &out->min_expert_cache)) {
            js_free(&d);
            return WASTE_E_FORMAT;
        }
        out->floor_bytes = 0;
        if (add64(&out->floor_bytes, out->trunk_bytes) ||
            add64(&out->floor_bytes, out->state_bytes) ||
            add64(&out->floor_bytes, out->scratch_bytes) ||
            add64(&out->floor_bytes, out->min_expert_cache)) {
            js_free(&d);
            return WASTE_E_FORMAT;
        }
        /* A cache below one token's working set keeps nothing alive to the
         * next token, so "recommended" starts at three layers' worth — the
         * same shape as the Kimi recommendation, with Inkling's top-k. */
        uint64_t extra;
        out->recommended_bytes = out->floor_bytes;
        if (mul64(rec, (uint64_t)cfg.top_k * (uint64_t)cfg.n_layers * 3u, &extra) ||
            add64(&out->recommended_bytes, extra)) {
            js_free(&d);
            return WASTE_E_FORMAT;
        }
    }

    js_free(&d);
    return WASTE_OK;
}
