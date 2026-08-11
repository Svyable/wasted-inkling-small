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

#include "arch.h"
#include "inkling_config.h"
#include "inkling_manifest.h"
#include "json.h"

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

waste_status waste_inkling_plan_memory_json(const char *manifest_json,
                                            uint32_t ctx_tokens,
                                            waste_memplan *out)
{
    js_doc d;
    waste_inkling_config cfg;
    waste_inkling_memory mem;

    if (!manifest_json || !out || ctx_tokens == 0) return WASTE_E_ARG;
    if (js_parse(&d, manifest_json) < 0) return WASTE_E_FORMAT;

    {
        /* The same reader the loader uses, so a container cannot be planned
         * as one model and opened as another. */
        const int cfg_tok = waste_inkling_manifest_config_token(&d);
        if (cfg_tok < 0 || waste_inkling_manifest_config(&d, cfg_tok, &cfg)) {
            js_free(&d);
            return WASTE_E_FORMAT;
        }
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
         * reads one row at a time.
         *
         * "Reads one row at a time" is a property of the loader, not of the
         * name alone: it leaves a vocabulary table on disk when the table is
         * quantized, which is what a real container stores, and materializes
         * an F32 one because there is no row unpacker for F32 on disk. The
         * exclusion carries the same condition, so the floor this reports is
         * the resident set the load actually produces rather than the one a
         * naming convention implies. */
        const int trunk = js_get(&d, 0, "trunk");
        const int n = js_size(&d, trunk);
        for (int i = 0; i < n; i++) {
            const int e = js_at(&d, trunk, i);
            char name[160];
            js_str(&d, js_get(&d, e, "name"), name, sizeof name);
            const int64_t fmt = js_int(&d, js_get(&d, e, "fmt"), 0);
            if (fmt != 0 && waste_arch_row_backed(name)) continue;
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
