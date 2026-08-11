/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
/*
 * inkling_container.c — a public Inkling container, placed in memory.
 *
 * The planner was promoted first because a plan is geometry: the worst a
 * bug in it produces is a wrong byte count, which the caller finds out
 * about at once. Loading sits one step further along the same argument. It
 * decides where bytes live and which name refers to which matrix; it does
 * not decide what a token is. So it can be promoted while the forward path
 * is still refused, and the refusal moves from "Inkling is not supported"
 * to the one claim this port has not yet earned — that these weights
 * produce these tokens.
 *
 * Three properties are worth stating because they are what makes this a
 * branch rather than a second engine:
 *
 *   - The trunk is read by upstream's own loader. `trunk.bin` is one
 *     format with one set of offsets, scale tables and packed widths, and
 *     a second reader for it is how two readers come to disagree.
 *   - Quantized matrices are never materialized. They bind as non-resident
 *     and reach the arithmetic through waste_matmul_t, which is WASTE's
 *     optimized kernel with its thread pool, not the scalar path.
 *   - Nothing defaults. A missing dimension, a tensor that was skipped, a
 *     bank that does not divide into records — each is an error, because a
 *     model that loads with a hole in it is worse than one that does not
 *     load.
 *
 * Experts are deliberately absent. The bank metadata is validated and
 * recorded, which is what the cache promotion needs, but no descriptor is
 * opened and no codebook is read: that is the next step, and it is the one
 * that makes a routed layer computable.
 */

#include "inkling_container.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "arch.h"
#include "inkling_bind.h"
#include "inkling_manifest.h"

#define IK_MAT_COUNT (WASTE_IK_MAT_SHARED_DOWN + 1)

typedef struct {
    waste_model *m;
    const waste_tensor *t;
} inkling_row_ctx;

typedef struct {
    waste_inkling_config cfg;
    waste_inkling_bound_weights bound;
    waste_inkling_tensor_view *views;
    size_t nviews;
    /* Resolved once at load: the backend runs per matrix per token and has
     * no business doing string work there. NULL entries are matrices this
     * layer does not have, or ones that stayed resident F32 and are read
     * through their pointer without consulting a backend at all. */
    const waste_tensor *matrix[WASTE_INKLING_MAX_LAYERS][IK_MAT_COUNT];
    waste_inkling_matrix_backend backend[WASTE_INKLING_MAX_LAYERS];
    inkling_row_ctx embed, unembed;
    waste_model *m;
} waste_inkling_container;

static int fail(const char *what)
{
    fprintf(stderr, "waste: Inkling container: %s\n", what);
    return -2;
}

/* ---- canonical names ---------------------------------------------------- */

/* The public container names its tensors exactly as the private runtime
 * index does. That is not a coincidence to be tidied away later: the index
 * was always a rehearsal of the manifest, and one name set means
 * inkling_bind.c binds both without a translation table in between. */
static const char *matrix_role(waste_inkling_matrix_kind kind)
{
    switch (kind) {
    case WASTE_IK_MAT_Q: return "q";
    case WASTE_IK_MAT_K: return "k";
    case WASTE_IK_MAT_V: return "v";
    case WASTE_IK_MAT_R: return "r";
    case WASTE_IK_MAT_O: return "o";
    case WASTE_IK_MAT_DENSE_GATE: return "mlp.gate";
    case WASTE_IK_MAT_DENSE_UP: return "mlp.up";
    case WASTE_IK_MAT_DENSE_DOWN: return "mlp.down";
    case WASTE_IK_MAT_ROUTER: return "router.weight";
    case WASTE_IK_MAT_SHARED_GATE: return "shared.gate";
    case WASTE_IK_MAT_SHARED_UP: return "shared.up";
    case WASTE_IK_MAT_SHARED_DOWN: return "shared.down";
    /* Routed experts are not trunk matrices in G6 step 2. Their bank
     * metadata is validated above, but record opening/cache dispatch is a
     * later promotion. Keep these enum values explicit so a future matrix
     * kind still trips -Wswitch instead of disappearing behind a default. */
    case WASTE_IK_MAT_ROUTED_GATE:
    case WASTE_IK_MAT_ROUTED_UP:
    case WASTE_IK_MAT_ROUTED_DOWN:
    /* The vocabulary projection is not a per-layer trunk matrix either: the
     * loader binds `inkling.unembed` through a row callback, and the final
     * head reaches a matrix backend only in the private BF16 evidence
     * profile. There is no container name to hand back here. */
    case WASTE_IK_MAT_UNEMBED:
        return NULL;
    }
    return NULL;
}

static int matrix_name(char *out, size_t cap, int layer,
                       waste_inkling_matrix_kind kind)
{
    const char *role = matrix_role(kind);
    if (!role) return -1;
    const int n = snprintf(out, cap, "inkling.layer.%d.%s", layer, role);
    return n < 0 || (size_t)n >= cap ? -1 : 0;
}

static int shared_kind(waste_inkling_matrix_kind kind)
{
    return kind == WASTE_IK_MAT_SHARED_GATE ||
           kind == WASTE_IK_MAT_SHARED_UP ||
           kind == WASTE_IK_MAT_SHARED_DOWN;
}

/* ---- runtime callbacks -------------------------------------------------- */

static int container_row_get(void *ctx, int row, int cols, float *out)
{
    inkling_row_ctx *r = (inkling_row_ctx *)ctx;
    if (!r || !r->m || !r->t) return -1;
    return waste_trunk_row_f32(r->m, r->t, row, cols, out);
}

/* One matrix-vector product against a trunk tensor that was left in its
 * stored width. The shared-expert matrices are one tensor of `n_shared`
 * stacked blocks, so `index` selects a block by advancing the payload and
 * its scale rows together — which is the same row arithmetic every other
 * reader of this tensor does, done once instead of materializing a copy. */
static int container_matvec(void *ctx, int layer,
                            waste_inkling_matrix_kind kind, int index,
                            const float *x, float *out, int rows, int cols)
{
    waste_inkling_container *c = (waste_inkling_container *)ctx;
    if (!c || !x || !out || rows < 1 || cols < 1) return -1;
    if (layer < 0 || layer >= c->cfg.n_layers) return -1;
    if ((int)kind < 0 || (int)kind >= IK_MAT_COUNT) return -1;

    const waste_tensor *t = c->matrix[layer][kind];
    if (!t || t->on_disk || (!t->data && !t->q)) return -1;
    if (t->shape[t->ndim - 1] != cols) return -1;

    int blocks = 1;
    if (shared_kind(kind)) {
        if (t->ndim != 3 || t->shape[1] != rows) return -1;
        blocks = t->shape[0];
    } else {
        if (t->ndim != 2 || t->shape[0] != rows) return -1;
    }
    if (index < 0 || index >= blocks) return -1;

    waste_tensor sub = *t;
    const size_t r0 = (size_t)index * (size_t)rows;
    sub.ndim = 2;
    sub.shape[0] = rows;
    sub.shape[1] = cols;
    sub.shape[2] = sub.shape[3] = 0;
    sub.n = (size_t)rows * (size_t)cols;
    if (t->data) sub.data = t->data + r0 * (size_t)cols;
    if (t->q) {
        const int ng = (cols + t->group - 1) / t->group;
        sub.q = t->q + r0 * t->rowbytes;
        sub.qs = t->qs + r0 * (size_t)ng;
    }
    waste_matmul_t(c->m, out, &sub, x, rows, cols, 1);
    return 0;
}

/* ---- load --------------------------------------------------------------- */

static const waste_tensor *find_unique(const waste_model *m, const char *name)
{
    const waste_tensor *found = NULL;
    for (int i = 0; i < m->n_tensors; i++) {
        if (strcmp(m->t[i].name, name)) continue;
        if (found) return NULL;             /* duplicate is always invalid */
        found = &m->t[i];
    }
    return found;
}

/* Mirror the geometry into the Kimi-shaped config the generic parts of the
 * engine read — `waste info`, the parameter report, the quantization
 * summary. These are descriptions of the container, not inputs to a Kimi
 * forward path, which is never reached: what would select it is refused
 * before this runs. */
static void mirror_config(waste_model *m, const js_doc *d, int cfg_tok,
                          const waste_inkling_config *c)
{
    waste_config *k = &m->cfg;
    k->n_layers = c->n_layers;
    k->hidden = c->hidden;
    k->vocab = c->vocab;
    k->n_experts = c->n_routed_experts;
    k->top_k = c->top_k;
    k->moe_inter = c->moe_intermediate;
    k->dense_inter = c->dense_intermediate;
    k->n_shared = c->n_shared_experts;
    k->first_dense = c->dense_layers;
    k->n_heads = c->global_heads;
    k->conv_k = c->conv_kernel;
    k->eps = c->rms_eps;
    k->routed_scale = c->route_scale;
    k->eos_token_id = (int)js_int(d, js_get(d, cfg_tok, "eos_token_id"), 0);
    js_str(d, js_get(d, 0, "arch"), k->arch, sizeof k->arch);
}

/* expert_quant, read under the same bounds the Kimi loader applies. Nothing
 * here consumes a codebook yet; recording the format is what lets the next
 * promotion open the banks without re-reading the manifest, and what lets
 * `waste info` name the quantization it is holding. */
static int load_expert_quant(waste_model *m, const js_doc *d)
{
    const int eq = js_get(d, 0, "expert_quant");
    m->stages = (int)js_int(d, js_get(d, eq, "stages"), 3);
    m->vec_dim = (int)js_int(d, js_get(d, eq, "vec_dim"), 8);
    m->cb_entries = (int)js_int(d, js_get(d, eq, "entries"), 256);
    m->index_bits = (int)js_int(d, js_get(d, eq, "index_bits"), 8);
    if (m->stages < 1 || m->stages > 8 || m->vec_dim < 1 || m->vec_dim > 64 ||
        m->cb_entries < 1 || m->cb_entries > 256)
        return fail("expert_quant is out of range");
    if (m->index_bits != 8 &&
        (m->index_bits != 6 || m->stages != 4 || m->cb_entries != 64))
        return fail("expert_quant index_bits is unsupported");
    return 0;
}

/* Bank metadata for every sparse layer, validated and recorded without
 * opening a descriptor. The expert path is the next promotion; what this
 * has to guarantee now is that a container missing its experts, or
 * declaring a record size that does not divide, fails at open rather than
 * at the first routed token. */
static int safe_bank_file(const char *name)
{
    return name[0] && !strchr(name, '/') && !strchr(name, '\\') &&
           strcmp(name, ".") && strcmp(name, "..");
}

static int load_bank_meta(waste_model *m, const js_doc *d,
                          const waste_inkling_config *c)
{
    /* An array, one entry per sparse layer, ascending — the shape the
     * planner already reads the first record size out of. The Kimi path
     * keys an object by layer number; Inkling does not need that freedom
     * and it costs a way for a container to be missing a layer without
     * saying so, because a short array is short and a sparse object is
     * merely quiet. */
    const int layers = js_get(d, 0, "layers");
    const int n = js_size(d, layers);
    const int want = c->n_layers - c->dense_layers;
    if (js_at(d, layers, 0) < 0 && n > 0)
        return fail("`layers` is not an array");
    if (n != want)
        return fail("`layers` does not carry one bank per sparse layer");
    for (int i = 0; i < n; i++) {
        const int L = c->dense_layers + i;
        const int e = js_at(d, layers, i);
        char file[128];
        const int64_t experts = js_int(d, js_get(d, e, "experts"), 0);
        const int64_t bytes = js_int(d, js_get(d, e, "bytes"), 0);
        const int64_t base = js_int(d, js_get(d, e, "codebook_base"), 0);
        const int lt = js_get(d, e, "layer");
        js_str(d, js_get(d, e, "file"), file, sizeof file);
        if (lt >= 0 && js_int(d, lt, -1) != L)
            return fail("expert banks are not in ascending sparse-layer order");
        if (!safe_bank_file(file))
            return fail("an expert bank names no file, or names one outside the container");
        if (experts != c->n_routed_experts || bytes <= 0 ||
            bytes % experts != 0 || base < 0 || base > (1 << 24))
            return fail("expert bank geometry does not match the config");
        m->bank[L].n_experts = (int)experts;
        m->bank[L].cb_base = (int)base;
        m->bank[L].rec_bytes = bytes / experts;
    }
    return 0;
}

/* One row of scratch for whichever tensors stayed on disk, sized from the
 * widest of them. The Kimi tail does the same thing further down its own
 * load; this path does not run that tail, and without the buffers the two
 * vocabulary tables have nowhere to land. */
static int alloc_row_scratch(waste_model *m)
{
    size_t rb = 0, sb = 0;
    for (int i = 0; i < m->n_tensors; i++) {
        const waste_tensor *t = &m->t[i];
        if (!t->on_disk || !t->group) continue;
        const int ng = (t->shape[t->ndim - 1] + t->group - 1) / t->group;
        if (t->rowbytes > rb) rb = t->rowbytes;
        if ((size_t)ng > sb) sb = (size_t)ng;
    }
    if (!rb) return 0;
    m->embrow = (int8_t *)malloc(rb);
    m->embsc = (uint16_t *)malloc(sb * sizeof(uint16_t));
    return m->embrow && m->embsc ? 0 : -1;
}

int waste_inkling_container_load(waste_model *m, const char *dir,
                                 const js_doc *d, int ctx_tokens)
{
    waste_inkling_container *c;
    int rc;

    if (!m || !dir || !d) return -2;
    waste_inkling_container_free(m);

    {   /* Canonical names carry no prefix. A container that declares one
         * is describing tensors this binding would not find, and a lookup
         * silently one prefix short is exactly the failure the Kimi loader
         * documents having had. */
        char prefix[64];
        js_str(d, js_get(d, 0, "tensor_prefix"), prefix, sizeof prefix);
        if (prefix[0])
            return fail("tensor_prefix is set; Inkling tensor names are canonical");
    }

    c = (waste_inkling_container *)calloc(1, sizeof *c);
    if (!c) return -1;
    c->m = m;
    m->inkling = c;

    {
        const int cfg_tok = waste_inkling_manifest_config_token(d);
        if (cfg_tok < 0 || waste_inkling_manifest_config(d, cfg_tok, &c->cfg))
            return fail("config is missing a required dimension or out of range");
        if (ctx_tokens < 1 || ctx_tokens > c->cfg.max_context)
            return fail("requested context exceeds the model maximum");
        mirror_config(m, d, cfg_tok, &c->cfg);
    }

    if ((rc = load_expert_quant(m, d))) return rc;
    if ((rc = load_bank_meta(m, d, &c->cfg))) return rc;

    if (waste_trunk_load(m, dir, d, js_get(d, 0, "trunk")) < 0) return -1;
    if (m->n_tensors < 1) return fail("the trunk declares no tensors");
    if (alloc_row_scratch(m)) return -1;

    /* Views over what the trunk loader produced. A quantized matrix has no
     * float pointer and binds non-resident; everything else must be
     * resident F32 and the binding says so by name if it is not. */
    c->views = (waste_inkling_tensor_view *)calloc((size_t)m->n_tensors,
                                                   sizeof *c->views);
    if (!c->views) return -1;
    c->nviews = (size_t)m->n_tensors;
    for (int i = 0; i < m->n_tensors; i++) {
        const waste_tensor *t = &m->t[i];
        c->views[i].name = t->name;
        c->views[i].ndim = t->ndim;
        for (int k = 0; k < 4; k++) c->views[i].shape[k] = t->shape[k];
        c->views[i].data = t->on_disk ? NULL : t->data;
    }

    {   /* The two vocabulary tables, read one row per token. Either width
         * is accepted — quantized and left on disk, which is what a real
         * container stores, or F32 and resident, which is what a small one
         * does — and the planner excludes them from the resident floor
         * under exactly the same condition. */
        const waste_tensor *embed = find_unique(m, "inkling.embed");
        const waste_tensor *unembed = find_unique(m, "inkling.unembed");
        if (!embed || !unembed)
            return fail("the embedding or unembedding table is missing or duplicated");
        c->embed.m = c->unembed.m = m;
        c->embed.t = embed;
        c->unembed.t = unembed;
        if (waste_inkling_bind_weights_ex_backend(
                &c->bound, &c->cfg, c->views, c->nviews,
                embed->data && !embed->on_disk ? NULL : container_row_get, &c->embed,
                unembed->data && !unembed->on_disk ? NULL : container_row_get, &c->unembed,
                1))
            return fail("canonical tensors do not bind to the Inkling model");
    }

    /* Resolve every matrix the backend may be asked for. Binding accepted a
     * NULL data pointer for these; this is where that permission is paid
     * for, by proving the bytes are actually here and reachable. A tensor
     * the trunk loader skipped would otherwise reach waste_matmul_t, which
     * answers a missing tensor with a vector of zeros — a wrong token
     * rather than an error, which is the whole failure mode this port
     * exists to refuse. */
    for (int L = 0; L < c->cfg.n_layers; L++) {
        const int sparse = L >= c->cfg.dense_layers;
        for (int k = 0; k < IK_MAT_COUNT; k++) {
            const waste_inkling_matrix_kind kind = (waste_inkling_matrix_kind)k;
            const int dense_only = kind == WASTE_IK_MAT_DENSE_GATE ||
                                   kind == WASTE_IK_MAT_DENSE_UP ||
                                   kind == WASTE_IK_MAT_DENSE_DOWN;
            const int sparse_only = kind == WASTE_IK_MAT_ROUTER || shared_kind(kind);
            if ((dense_only && sparse) || (sparse_only && !sparse)) continue;
            char name[128];
            if (matrix_name(name, sizeof name, L, kind)) return -1;
            const waste_tensor *t = find_unique(m, name);
            if (!t || t->on_disk || (!t->data && !t->q))
                return fail("a layer matrix is missing, duplicated, or was not loaded");
            c->matrix[L][k] = t;
        }
        c->backend[L].matvec = container_matvec;
        c->backend[L].ctx = c;
    }

    return 0;
}

void waste_inkling_container_free(waste_model *m)
{
    waste_inkling_container *c;
    if (!m || !m->inkling) return;
    c = (waste_inkling_container *)m->inkling;
    free(c->views);
    free(c);
    m->inkling = NULL;
}

int waste_inkling_container_present(const waste_model *m)
{
    return m && m->inkling ? 1 : 0;
}

const waste_inkling_config *waste_inkling_container_config(const waste_model *m)
{
    return m && m->inkling
         ? &((const waste_inkling_container *)m->inkling)->cfg : NULL;
}

const waste_inkling_bound_weights *waste_inkling_container_weights(
    const waste_model *m)
{
    return m && m->inkling
         ? &((const waste_inkling_container *)m->inkling)->bound : NULL;
}

const waste_inkling_matrix_backend *waste_inkling_container_backends(
    const waste_model *m)
{
    return m && m->inkling
         ? ((const waste_inkling_container *)m->inkling)->backend : NULL;
}
