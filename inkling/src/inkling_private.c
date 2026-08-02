/* SPDX-License-Identifier: Apache-2.0 */
#include "inkling_private.h"

#include "inkling_bind.h"
#include "inkling_config.h"
#include "inkling_stage_reader.h"
#include "inkling_wexp.h"
#include "inkling_qtensor.h"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define IKRT_MAGIC 0x54524b49u
#define IKRT_VERSION_BF16 1u
#define IKRT_VERSION_VQ 2u
#define IKRT_VERSION_QTRUNK_VQ 3u
#define IKRT_BANK_BF16 0u
#define IKRT_BANK_VQ 1u
#define IKRT_HEADER_BYTES 256u
#define IKRT_LAYER_BYTES 32u
#define IKRT_TENSOR_BYTES 256u
#define IKRT_BANK_BYTES 128u
#define IKRT_FLAG_OFFICIAL_SMALL 1u
#define IKRT_TENSOR_ROW_BACKED 1u
#define IKRT_TENSOR_QUANTIZED 2u
#define IKRT_MAX_TENSORS 2048u
#define IKRT_PATH_MAX 1024u

typedef struct {
    char name[128];
    char path[80];
    int dtype, ndim, flags;
    int shape[4];
    uint64_t payload_bytes, stored_bytes;
    uint32_t crc32;
    float *data;
    int quantized;
    waste_inkling_stage_tensor stage;
    waste_inkling_qtensor qtensor;
} private_tensor;

typedef struct {
    char path[80];
    int layer, experts, hidden, intermediate, codebook_base;
    uint64_t record_bytes, bytes;
} private_bank_meta;

typedef struct {
    int format, stages, entries, vec_dim, index_block;
} private_expert_format;

struct waste_inkling_private {
    waste_inkling_config cfg;
    waste_inkling_bound_weights bound;
    waste_inkling_tensor_view *views;
    private_tensor *tensor;
    size_t ntensors;
    float *resident;
    size_t resident_floats;

    waste_inkling_stage_bank bank[WASTE_INKLING_MAX_LAYERS];
    waste_inkling_wexp_bank vq_bank[WASTE_INKLING_MAX_LAYERS];
    int bank_present[WASTE_INKLING_MAX_LAYERS];
    private_expert_format expert_format;
    float *expert_gate, *expert_up, *expert_down;
    unsigned char *expert_raw;
    size_t expert_matrix_floats, expert_raw_bytes;

    waste_inkling_model_state state;
    waste_inkling_model_scratch scratch;
    float *state_buffer, *scratch_buffer;
    int *scratch_ints;
    size_t state_floats, scratch_floats, scratch_nints;
    waste_inkling_matrix_backend backend[WASTE_INKLING_MAX_LAYERS];
    size_t quantized_resident_bytes;
    int verify_crc;
    char detail[256];
};

static uint16_t rd16(const unsigned char *p)
{
    return (uint16_t)(p[0] | ((uint16_t)p[1] << 8));
}

static uint32_t rd32(const unsigned char *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static uint64_t rd64(const unsigned char *p)
{
    return (uint64_t)rd32(p) | ((uint64_t)rd32(p + 4) << 32);
}

static float rdf32(const unsigned char *p)
{
    union { uint32_t u; float f; } v;
    v.u = rd32(p);
    return v.f;
}

static int addz(size_t a, size_t b, size_t *out)
{
    if (!out || b > SIZE_MAX - a) return -1;
    *out = a + b;
    return 0;
}

static int mulz(size_t a, size_t b, size_t *out)
{
    if (!out || (a && b > SIZE_MAX / a)) return -1;
    *out = a * b;
    return 0;
}

static void copy_detail(char *dst, size_t cap, const char *text)
{
    if (!dst || !cap) return;
    if (!text) text = "";
    snprintf(dst, cap, "%s", text);
}

static waste_inkling_private_status fail_open(
    waste_inkling_private *p, waste_inkling_private_status status,
    char *detail, size_t cap, const char *message)
{
    copy_detail(detail, cap, message);
    if (p) {
        snprintf(p->detail, sizeof p->detail, "%s", message ? message : "");
        waste_inkling_private_close(p);
    }
    return status;
}

static int fixed_string(const unsigned char *src, size_t n,
                        char *dst, size_t cap)
{
    size_t len = 0;
    if (!src || !dst || cap == 0) return -1;
    while (len < n && src[len]) len++;
    if (len == 0 || len == n || len >= cap) return -1;
    memcpy(dst, src, len);
    dst[len] = 0;
    for (size_t i = len + 1; i < n; i++) if (src[i]) return -1;
    return 0;
}

static int safe_relative(const char *path)
{
    if (!path || !path[0] || path[0] == '/' || path[0] == '\\' ||
        (path[0] && path[1] == ':') || strchr(path, '\\')) return 0;
    const char *p = path;
    while (*p) {
        const char *q = strchr(p, '/');
        const size_t n = q ? (size_t)(q - p) : strlen(p);
        if (n == 0 || (n == 1 && p[0] == '.') ||
            (n == 2 && p[0] == '.' && p[1] == '.')) return 0;
        if (!q) break;
        p = q + 1;
    }
    return 1;
}

static int join_path(char *out, size_t cap, const char *dir, const char *rel)
{
    if (!out || !cap || !dir || !safe_relative(rel)) return -1;
    const size_t n = strlen(dir);
    const char sep = n && (dir[n - 1] == '/' || dir[n - 1] == '\\') ? 0 : '/';
    const int wrote = snprintf(out, cap, sep ? "%s/%s" : "%s%s", dir, rel);
    return wrote < 0 || (size_t)wrote >= cap ? -1 : 0;
}

static unsigned char *read_file(const char *path, size_t *size)
{
    FILE *f;
    long n;
    unsigned char *data;
    if (!path || !size || !(f = fopen(path, "rb"))) return NULL;
    if (fseek(f, 0, SEEK_END) || (n = ftell(f)) < 0 || fseek(f, 0, SEEK_SET)) {
        fclose(f); return NULL;
    }
    data = (unsigned char *)malloc((size_t)n ? (size_t)n : 1u);
    if (!data || fread(data, 1, (size_t)n, f) != (size_t)n) {
        free(data); fclose(f); return NULL;
    }
    fclose(f);
    *size = (size_t)n;
    return data;
}

static size_t expected_tensor_count(const waste_inkling_config *cfg)
{
    size_t total = 4;
    if (!cfg || cfg->n_layers < 1 || cfg->dense_layers < 0 ||
        cfg->dense_layers > cfg->n_layers) return 0;
    total += (size_t)cfg->n_layers * 14u;
    total += (size_t)cfg->dense_layers * 4u;
    total += (size_t)(cfg->n_layers - cfg->dense_layers) * 6u;
    return total;
}

static int is_official_small_geometry(const waste_inkling_config *c)
{
    static const int global_layers[] = {5, 11, 17, 23, 29, 35, 41};
    if (!c || c->n_layers != 42 || c->hidden != 4096 ||
        c->vocab != 201024 || c->unpadded_vocab != 200058 ||
        c->max_context != 1048576 || c->global_heads != 32 ||
        c->global_kv_heads != 8 || c->global_head_dim != 128 ||
        c->local_heads != 32 || c->local_kv_heads != 8 ||
        c->local_head_dim != 128 || c->sliding_window != 512 ||
        c->d_rel != 16 || c->rel_extent != 1024 || c->conv_kernel != 4 ||
        c->dense_layers != 2 || c->dense_intermediate != 16384 ||
        c->moe_intermediate != 2048 || c->n_routed_experts != 256 ||
        c->top_k != 6 || c->n_shared_experts != 2 ||
        c->rms_eps != 1e-6f || c->logits_width_multiplier != 16.0f ||
        c->route_scale != 8.0f ||
        c->log_scaling_n_floor != 128000 || c->log_scaling_alpha != 0.1f)
        return 0;
    for (int i = 0; i < c->n_layers; i++) {
        int global = 0;
        for (size_t j = 0; j < sizeof global_layers / sizeof global_layers[0]; j++)
            if (i == global_layers[j]) global = 1;
        if (!!c->layer[i].is_local == !!global) return 0;
    }
    return 1;
}

static int parse_config(const unsigned char *data, size_t size,
                        waste_inkling_config *cfg, uint32_t *flags,
                        uint32_t *ntensors, uint32_t *nbanks,
                        private_expert_format *expert_format,
                        size_t *layers_off, size_t *tensors_off, size_t *banks_off)
{
    uint32_t n_layers, layer_bytes, tensor_bytes, bank_bytes;
    uint16_t version;
    size_t need, x;
    int local_ids[WASTE_INKLING_MAX_LAYERS];
    int nlocal = 0;
    if (!data || !cfg || !flags || !ntensors || !nbanks || !expert_format ||
        size < IKRT_HEADER_BYTES || rd32(data) != IKRT_MAGIC ||
        rd16(data + 6) != IKRT_HEADER_BYTES) return -1;
    version = rd16(data + 4);
    if (version != IKRT_VERSION_BF16 && version != IKRT_VERSION_VQ &&
        version != IKRT_VERSION_QTRUNK_VQ) return -1;
    memset(expert_format, 0, sizeof *expert_format);
    *flags = rd32(data + 8);
    if (*flags & ~IKRT_FLAG_OFFICIAL_SMALL) return -1;
    layer_bytes = rd32(data + 12);
    tensor_bytes = rd32(data + 16);
    bank_bytes = rd32(data + 20);
    n_layers = rd32(data + 24);
    *ntensors = rd32(data + 28);
    *nbanks = rd32(data + 32);
    if (layer_bytes != IKRT_LAYER_BYTES || tensor_bytes != IKRT_TENSOR_BYTES ||
        bank_bytes != IKRT_BANK_BYTES || n_layers < 1 ||
        n_layers > WASTE_INKLING_MAX_LAYERS || *ntensors < 1 ||
        *ntensors > IKRT_MAX_TENSORS || *nbanks > WASTE_INKLING_MAX_LAYERS)
        return -1;
    if (version == IKRT_VERSION_BF16) {
        for (int i = 248; i < 256; i++) if (data[i]) return -1;
        expert_format->format = IKRT_BANK_BF16;
    } else {
        expert_format->format = data[248];
        expert_format->stages = data[249];
        expert_format->vec_dim = data[250];
        expert_format->index_block = data[251];
        expert_format->entries = (int)rd16(data + 252);
        if (expert_format->format != IKRT_BANK_VQ ||
            (expert_format->stages != 2 && expert_format->stages != 3) ||
            expert_format->vec_dim < 1 || expert_format->vec_dim > 64 ||
            expert_format->index_block < 1 ||
            expert_format->entries < 2 || expert_format->entries > 256 ||
            rd16(data + 254) != 0) return -1;
    }
    *layers_off = IKRT_HEADER_BYTES;
    if (mulz((size_t)n_layers, IKRT_LAYER_BYTES, &x) ||
        addz(*layers_off, x, tensors_off) ||
        mulz((size_t)*ntensors, IKRT_TENSOR_BYTES, &x) ||
        addz(*tensors_off, x, banks_off) ||
        mulz((size_t)*nbanks, IKRT_BANK_BYTES, &x) ||
        addz(*banks_off, x, &need) || need != size) return -1;

    for (uint32_t i = 0; i < n_layers; i++) {
        const unsigned char *e = data + *layers_off + (size_t)i * IKRT_LAYER_BYTES;
        if (rd32(e) != i || rd32(e + 28) != 0) return -1;
        if (rd32(e + 4)) local_ids[nlocal++] = (int)i;
    }
    waste_inkling_config_args a;
    memset(&a, 0, sizeof a);
    a.n_layers = (int)n_layers;
    a.hidden = (int)rd32(data + 36);
    a.vocab = (int)rd32(data + 40);
    a.unpadded_vocab = (int)rd32(data + 44);
    a.max_context = (int)rd32(data + 48);
    a.global_heads = (int)rd32(data + 52);
    a.global_kv_heads = (int)rd32(data + 56);
    a.global_head_dim = (int)rd32(data + 60);
    a.local_heads = (int)rd32(data + 64);
    a.local_kv_heads = (int)rd32(data + 68);
    a.local_head_dim = (int)rd32(data + 72);
    a.sliding_window = (int)rd32(data + 76);
    a.d_rel = (int)rd32(data + 80);
    a.rel_extent = (int)rd32(data + 84);
    a.conv_kernel = (int)rd32(data + 88);
    a.dense_layers = (int)rd32(data + 92);
    a.dense_intermediate = (int)rd32(data + 96);
    a.moe_intermediate = (int)rd32(data + 100);
    a.n_routed_experts = (int)rd32(data + 104);
    a.top_k = (int)rd32(data + 108);
    a.n_shared_experts = (int)rd32(data + 112);
    a.rms_eps = rdf32(data + 116);
    a.route_scale = rdf32(data + 120);
    a.logits_width_multiplier = rdf32(data + 124);
    a.log_scaling_n_floor = (int)rd32(data + 128);
    a.log_scaling_alpha = rdf32(data + 132);
    a.local_layer_ids = local_ids;
    a.n_local_layers = nlocal;
    if (waste_inkling_config_build(cfg, &a)) return -1;
    for (int i = 0; i < cfg->n_layers; i++) {
        const unsigned char *e = data + *layers_off + (size_t)i * IKRT_LAYER_BYTES;
        const waste_inkling_layer_cfg *l = &cfg->layer[i];
        if (rd32(e + 4) != (uint32_t)l->is_local ||
            rd32(e + 8) != (uint32_t)l->num_heads ||
            rd32(e + 12) != (uint32_t)l->num_kv_heads ||
            rd32(e + 16) != (uint32_t)l->head_dim ||
            rd32(e + 20) != (uint32_t)l->relative_extent ||
            rd32(e + 24) != (uint32_t)(i >= cfg->dense_layers)) return -1;
    }
    return 0;
}

static int tensor_product(const private_tensor *t, size_t *out)
{
    size_t n = 1;
    if (!t || !out || t->ndim < 1 || t->ndim > 4) return -1;
    for (int i = 0; i < t->ndim; i++)
        if (t->shape[i] <= 0 || mulz(n, (size_t)t->shape[i], &n)) return -1;
    *out = n;
    return 0;
}

static int parse_tensors(waste_inkling_private *p, const unsigned char *data,
                         size_t off, uint32_t count)
{
    p->tensor = (private_tensor *)calloc(count, sizeof *p->tensor);
    p->views = (waste_inkling_tensor_view *)calloc(count, sizeof *p->views);
    if (!p->tensor || !p->views) return -1;
    p->ntensors = count;
    for (uint32_t i = 0; i < count; i++) {
        p->tensor[i].stage.fd = -1;
        p->tensor[i].qtensor.fd = -1;
    }
    size_t resident = 0;
    for (uint32_t i = 0; i < count; i++) {
        private_tensor *t = &p->tensor[i];
        const unsigned char *e = data + off + (size_t)i * IKRT_TENSOR_BYTES;
        if (fixed_string(e, 128, t->name, sizeof t->name) ||
            fixed_string(e + 128, 80, t->path, sizeof t->path) ||
            !safe_relative(t->path)) return -1;
        t->dtype = e[208];
        t->ndim = e[209];
        t->flags = (int)rd16(e + 210);
        if ((t->dtype < 1 || t->dtype > 5) || t->ndim < 1 || t->ndim > 4 ||
            (t->flags & ~(IKRT_TENSOR_ROW_BACKED | IKRT_TENSOR_QUANTIZED))) return -1;
        t->quantized = !!(t->flags & IKRT_TENSOR_QUANTIZED);
        if (t->quantized != (t->dtype == 4 || t->dtype == 5)) return -1;
        if (t->quantized && t->ndim < 2) return -1;
        for (int d = 0; d < 4; d++) {
            t->shape[d] = (int)rd32(e + 212 + 4 * d);
            if ((d < t->ndim && t->shape[d] <= 0) ||
                (d >= t->ndim && t->shape[d] != 0)) return -1;
        }
        t->payload_bytes = rd64(e + 228);
        t->stored_bytes = rd64(e + 236);
        t->crc32 = rd32(e + 244);
        for (int j = 248; j < 256; j++) if (e[j]) return -1;
        for (uint32_t j = 0; j < i; j++)
            if (!strcmp(t->name, p->tensor[j].name)) return -1;
        const int row_name = !strcmp(t->name, "inkling.embed") ||
                             !strcmp(t->name, "inkling.unembed");
        if (!!(t->flags & IKRT_TENSOR_ROW_BACKED) != !!row_name) return -1;
        if (!row_name && !t->quantized) {
            size_t n;
            if (tensor_product(t, &n) || addz(resident, n, &resident)) return -1;
        }
    }
    if (resident > SIZE_MAX / sizeof(float)) return -1;
    p->resident_floats = resident;
    p->resident = (float *)malloc(resident ? resident * sizeof(float) : sizeof(float));
    return p->resident ? 0 : -1;
}

static int load_tensors(waste_inkling_private *p, const char *dir)
{
    size_t cursor = 0;
    char path[IKRT_PATH_MAX];
    for (size_t i = 0; i < p->ntensors; i++) {
        private_tensor *t = &p->tensor[i];
        if (join_path(path, sizeof path, dir, t->path)) return -1;
        p->views[i].name = t->name;
        p->views[i].ndim = t->ndim;
        for (int d = 0; d < 4; d++) p->views[i].shape[d] = t->shape[d];
        if (t->quantized) {
            if (waste_inkling_qtensor_open(&t->qtensor, path, t->name,
                    t->shape, t->ndim, p->verify_crc) ||
                t->qtensor.qbytes != t->payload_bytes ||
                t->qtensor.q_crc32 != t->crc32 ||
                ((t->dtype == 4) != (t->qtensor.fmt == 2)) ||
                ((t->dtype == 5) != (t->qtensor.fmt == 3))) return -1;
            size_t stored = waste_inkling_qtensor_resident_bytes(&t->qtensor);
            if (!stored || stored > SIZE_MAX - 96u) return -1;
            stored += 96u;
            stored = (stored + 4095u) & ~(size_t)4095u;
            if ((uint64_t)stored != t->stored_bytes) return -1;
            if (!(t->flags & IKRT_TENSOR_ROW_BACKED)) {
                const size_t bytes = waste_inkling_qtensor_resident_bytes(&t->qtensor);
                if (!bytes || waste_inkling_qtensor_load_resident(&t->qtensor) ||
                    addz(p->quantized_resident_bytes, bytes,
                         &p->quantized_resident_bytes)) return -1;
            }
            p->views[i].data = NULL;
            continue;
        }
        if (waste_inkling_stage_tensor_open(&t->stage, path, t->name,
                t->shape, t->ndim, p->verify_crc) ||
            t->stage.dtype != t->dtype || t->stage.payload_bytes != t->payload_bytes ||
            t->stage.stored_bytes != t->stored_bytes ||
            t->stage.payload_crc32 != t->crc32) return -1;
        if (t->flags & IKRT_TENSOR_ROW_BACKED) {
            p->views[i].data = NULL;
        } else {
            size_t n;
            if (tensor_product(t, &n) || n > p->resident_floats - cursor ||
                waste_inkling_stage_tensor_read_all(&t->stage, p->resident + cursor, n))
                return -1;
            t->data = p->resident + cursor;
            p->views[i].data = t->data;
            cursor += n;
            waste_inkling_stage_tensor_close(&t->stage);
        }
    }
    return cursor == p->resident_floats ? 0 : -1;
}

static private_tensor *find_tensor(waste_inkling_private *p, const char *name)
{
    private_tensor *found = NULL;
    for (size_t i = 0; i < p->ntensors; i++) {
        if (strcmp(p->tensor[i].name, name)) continue;
        if (found) return NULL;
        found = &p->tensor[i];
    }
    return found;
}

static int private_row_get(void *ctx, int row, int cols, float *out)
{
    private_tensor *t = (private_tensor *)ctx;
    if (!t) return -1;
    if (t->quantized)
        return waste_inkling_qtensor_row(&t->qtensor, (uint64_t)row, out,
                                         (size_t)cols);
    return waste_inkling_stage_tensor_row(&t->stage, row, cols, out);
}

static int matrix_name(char *out, size_t cap, int layer,
                       waste_inkling_matrix_kind kind)
{
    const char *role = NULL;
    switch (kind) {
    case WASTE_IK_MAT_Q: role = "q"; break;
    case WASTE_IK_MAT_K: role = "k"; break;
    case WASTE_IK_MAT_V: role = "v"; break;
    case WASTE_IK_MAT_R: role = "r"; break;
    case WASTE_IK_MAT_O: role = "o"; break;
    case WASTE_IK_MAT_DENSE_GATE: role = "mlp.gate"; break;
    case WASTE_IK_MAT_DENSE_UP: role = "mlp.up"; break;
    case WASTE_IK_MAT_DENSE_DOWN: role = "mlp.down"; break;
    case WASTE_IK_MAT_ROUTER: role = "router.weight"; break;
    case WASTE_IK_MAT_SHARED_GATE: role = "shared.gate"; break;
    case WASTE_IK_MAT_SHARED_UP: role = "shared.up"; break;
    case WASTE_IK_MAT_SHARED_DOWN: role = "shared.down"; break;
    default: return -1;
    }
    const int n = snprintf(out, cap, "inkling.layer.%d.%s", layer, role);
    return n < 0 || (size_t)n >= cap ? -1 : 0;
}

static int private_matvec(void *ctx, int layer, waste_inkling_matrix_kind kind,
                          int index, const float *x, float *out,
                          int rows, int cols)
{
    waste_inkling_private *p = (waste_inkling_private *)ctx;
    char name[128];
    if (!p || !x || !out || rows < 1 || cols < 1 ||
        matrix_name(name, sizeof name, layer, kind)) return -1;
    private_tensor *t = find_tensor(p, name);
    if (!t || !t->quantized || t->qtensor.fd < 0 ||
        (uint64_t)cols != t->qtensor.cols) return -1;
    size_t row0 = 0;
    if (kind == WASTE_IK_MAT_SHARED_GATE || kind == WASTE_IK_MAT_SHARED_UP ||
        kind == WASTE_IK_MAT_SHARED_DOWN) {
        if (index < 0 || index >= p->cfg.n_shared_experts) return -1;
        row0 = (size_t)index * (size_t)rows;
    } else if (index != 0) return -1;
    return waste_inkling_qtensor_matvec_rows(&t->qtensor, x, out, row0,
                                              (size_t)rows, (size_t)cols);
}

static int parse_banks(waste_inkling_private *p, const unsigned char *data,
                       size_t off, uint32_t count, const char *dir)
{
    private_bank_meta meta[WASTE_INKLING_MAX_LAYERS];
    char full[IKRT_PATH_MAX], codebooks[IKRT_PATH_MAX], rel[80];
    uint64_t max_record = 0;
    if (count != (uint32_t)(p->cfg.n_layers - p->cfg.dense_layers)) return -1;
    for (uint32_t i = 0; i < count; i++) {
        const unsigned char *e = data + off + (size_t)i * IKRT_BANK_BYTES;
        if (fixed_string(e, 80, meta[i].path, sizeof meta[i].path) ||
            !safe_relative(meta[i].path)) return -1;
        meta[i].layer = (int)rd32(e + 80);
        meta[i].experts = (int)rd32(e + 84);
        meta[i].hidden = (int)rd32(e + 88);
        meta[i].intermediate = (int)rd32(e + 92);
        meta[i].record_bytes = rd64(e + 96);
        meta[i].bytes = rd64(e + 104);
        meta[i].codebook_base = (int)rd32(e + 112);
        for (int j = 116; j < 128; j++) if (e[j]) return -1;
        if (p->expert_format.format == IKRT_BANK_BF16 && meta[i].codebook_base != 0)
            return -1;
        if (meta[i].layer < p->cfg.dense_layers || meta[i].layer >= p->cfg.n_layers ||
            p->bank_present[meta[i].layer] || meta[i].experts != p->cfg.n_routed_experts ||
            meta[i].hidden != p->cfg.hidden ||
            meta[i].intermediate != p->cfg.moe_intermediate ||
            meta[i].record_bytes == 0 || meta[i].record_bytes > SIZE_MAX ||
            (uint64_t)meta[i].experts > UINT64_MAX / meta[i].record_bytes ||
            meta[i].bytes != meta[i].record_bytes * (uint64_t)meta[i].experts)
            return -1;
        if (meta[i].record_bytes > max_record) max_record = meta[i].record_bytes;
        p->bank_present[meta[i].layer] = 1;
    }
    for (int i = p->cfg.dense_layers; i < p->cfg.n_layers; i++)
        if (!p->bank_present[i]) return -1;

    size_t matrix;
    if (mulz((size_t)p->cfg.hidden, (size_t)p->cfg.moe_intermediate, &matrix) ||
        matrix > SIZE_MAX / sizeof(float)) return -1;
    p->expert_matrix_floats = matrix;
    if (p->expert_format.format == IKRT_BANK_BF16) {
        if (matrix > SIZE_MAX / 6u) return -1;
        p->expert_raw_bytes = matrix * 6u;
    } else {
        p->expert_raw_bytes = (size_t)max_record;
    }
    p->expert_gate = (float *)malloc(matrix * sizeof(float));
    p->expert_up = (float *)malloc(matrix * sizeof(float));
    p->expert_down = (float *)malloc(matrix * sizeof(float));
    p->expert_raw = (unsigned char *)malloc(p->expert_raw_bytes);
    if (!p->expert_gate || !p->expert_up || !p->expert_down || !p->expert_raw)
        return -1;
    for (uint32_t i = 0; i < count; i++) {
        if (join_path(full, sizeof full, dir, meta[i].path)) return -1;
        if (p->expert_format.format == IKRT_BANK_BF16) {
            if (waste_inkling_stage_bank_open_with_workspace(
                    &p->bank[meta[i].layer], full, meta[i].layer, meta[i].experts,
                    meta[i].hidden, meta[i].intermediate, meta[i].record_bytes,
                    p->verify_crc, p->expert_gate, p->expert_up, p->expert_down,
                    matrix, p->expert_raw, p->expert_raw_bytes)) return -1;
        } else {
            const int wrote = snprintf(rel, sizeof rel, "codebooks-L%d.bin", meta[i].layer);
            if (wrote < 0 || (size_t)wrote >= sizeof rel ||
                join_path(codebooks, sizeof codebooks, dir, rel) ||
                waste_inkling_wexp_bank_open_with_workspace(
                    &p->vq_bank[meta[i].layer], full, codebooks,
                    meta[i].layer, meta[i].experts, meta[i].hidden,
                    meta[i].intermediate, p->expert_format.stages,
                    p->expert_format.entries, p->expert_format.vec_dim,
                    p->expert_format.index_block, meta[i].codebook_base,
                    p->verify_crc, p->expert_gate, p->expert_up, p->expert_down,
                    matrix, p->expert_raw, p->expert_raw_bytes) ||
                p->vq_bank[meta[i].layer].record_bytes != meta[i].record_bytes)
                return -1;
        }
    }
    return 0;
}

static int expert_get(void *ctx, int layer, int expert,
                      waste_inkling_expert_weights *out)
{
    waste_inkling_private *p = (waste_inkling_private *)ctx;
    if (!p || layer < 0 || layer >= p->cfg.n_layers || !p->bank_present[layer])
        return -1;
    if (p->expert_format.format == IKRT_BANK_VQ)
        return waste_inkling_wexp_expert_get(&p->vq_bank[layer], layer, expert, out);
    return waste_inkling_stage_expert_get(&p->bank[layer], layer, expert, out);
}

void waste_inkling_private_options_init(waste_inkling_private_options *o)
{
    if (!o) return;
    o->context_capacity = 4096;
    o->verify_crc = 1;
    o->require_official_small = 1;
}

waste_inkling_private_status waste_inkling_private_open(
    const char *dir, const waste_inkling_private_options *options,
    waste_inkling_private **out, char *detail, size_t detail_capacity)
{
    waste_inkling_private_options opt;
    waste_inkling_private *p = NULL;
    unsigned char *index = NULL;
    size_t index_size = 0, lo, to, bo;
    uint32_t flags, ntensors, nbanks;
    char path[IKRT_PATH_MAX];
    if (!dir || !out) return WASTE_INKLING_PRIVATE_E_ARG;
    *out = NULL;
    if (options) opt = *options; else waste_inkling_private_options_init(&opt);
    if (opt.context_capacity < 1) return WASTE_INKLING_PRIVATE_E_ARG;
    p = (waste_inkling_private *)calloc(1, sizeof *p);
    if (!p) return fail_open(NULL, WASTE_INKLING_PRIVATE_E_OOM,
                            detail, detail_capacity, "out of memory");
    p->verify_crc = !!opt.verify_crc;
    for (int i = 0; i < WASTE_INKLING_MAX_LAYERS; i++) {
        p->bank[i].fd = -1;
        p->vq_bank[i].bank_fd = -1;
    }
    if (join_path(path, sizeof path, dir, "runtime-stage.bin") ||
        !(index = read_file(path, &index_size)))
        return fail_open(p, WASTE_INKLING_PRIVATE_E_IO, detail, detail_capacity,
                         "cannot read runtime-stage.bin");
    if (parse_config(index, index_size, &p->cfg, &flags, &ntensors, &nbanks,
                     &p->expert_format, &lo, &to, &bo)) {
        free(index);
        return fail_open(p, WASTE_INKLING_PRIVATE_E_FORMAT, detail, detail_capacity,
                         "invalid Inkling runtime-stage index or config");
    }
    if (opt.require_official_small &&
        (!(flags & IKRT_FLAG_OFFICIAL_SMALL) || !is_official_small_geometry(&p->cfg))) {
        free(index);
        return fail_open(p, WASTE_INKLING_PRIVATE_E_UNSUPPORTED,
                         detail, detail_capacity,
                         "runtime stage is not the verified official Inkling-Small profile");
    }
    if ((size_t)ntensors != expected_tensor_count(&p->cfg) ||
        parse_tensors(p, index, to, ntensors)) {
        free(index);
        return fail_open(p, WASTE_INKLING_PRIVATE_E_FORMAT, detail, detail_capacity,
                         "invalid canonical tensor registry");
    }
    if (load_tensors(p, dir)) {
        free(index);
        return fail_open(p, WASTE_INKLING_PRIVATE_E_FORMAT, detail, detail_capacity,
                         "staged trunk tensor failed identity, shape, size, or CRC validation");
    }
    private_tensor *embed = find_tensor(p, "inkling.embed");
    private_tensor *unembed = find_tensor(p, "inkling.unembed");
    if (!embed || !unembed ||
        (!embed->quantized && embed->stage.fd < 0) ||
        (!unembed->quantized && unembed->stage.fd < 0) ||
        (embed->quantized && embed->qtensor.fd < 0) ||
        (unembed->quantized && unembed->qtensor.fd < 0) ||
        waste_inkling_bind_weights_ex_backend(&p->bound, &p->cfg,
            p->views, p->ntensors, private_row_get, embed,
            private_row_get, unembed, 1)) {
        free(index);
        return fail_open(p, WASTE_INKLING_PRIVATE_E_FORMAT, detail, detail_capacity,
                         "canonical tensors do not bind to the Inkling model");
    }
    if (parse_banks(p, index, bo, nbanks, dir)) {
        free(index);
        return fail_open(p, WASTE_INKLING_PRIVATE_E_FORMAT, detail, detail_capacity,
                         "staged expert bank validation failed");
    }
    for (int i = 0; i < p->cfg.n_layers; i++) {
        p->backend[i].matvec = private_matvec;
        p->backend[i].ctx = p;
    }
    free(index);
    if (opt.context_capacity > p->cfg.max_context)
        return fail_open(p, WASTE_INKLING_PRIVATE_E_ARG, detail, detail_capacity,
                         "context capacity exceeds model maximum");
    p->state_floats = waste_inkling_model_state_floats(&p->cfg, opt.context_capacity);
    p->scratch_floats = waste_inkling_model_scratch_floats(&p->cfg, opt.context_capacity);
    p->scratch_nints = waste_inkling_model_scratch_ints(&p->cfg);
    if (!p->state_floats || !p->scratch_floats || !p->scratch_nints)
        return fail_open(p, WASTE_INKLING_PRIVATE_E_FORMAT, detail, detail_capacity,
                         "invalid Inkling runtime memory contract");
    if (p->state_floats > SIZE_MAX / sizeof(float) ||
        p->scratch_floats > SIZE_MAX / sizeof(float) ||
        p->scratch_nints > SIZE_MAX / sizeof(int))
        return fail_open(p, WASTE_INKLING_PRIVATE_E_OOM, detail, detail_capacity,
                         "Inkling state or scratch size overflows this platform");
    p->state_buffer = (float *)malloc(p->state_floats * sizeof(float));
    p->scratch_buffer = (float *)malloc(p->scratch_floats * sizeof(float));
    p->scratch_ints = (int *)malloc(p->scratch_nints * sizeof(int));
    if (!p->state_buffer || !p->scratch_buffer || !p->scratch_ints)
        return fail_open(p, WASTE_INKLING_PRIVATE_E_OOM, detail, detail_capacity,
                         "out of memory allocating Inkling state or scratch");
    if (waste_inkling_model_state_init(&p->state, &p->cfg, opt.context_capacity,
            p->state_buffer, p->state_floats) ||
        waste_inkling_model_scratch_init(&p->scratch, &p->cfg, opt.context_capacity,
            p->scratch_buffer, p->scratch_floats,
            p->scratch_ints, p->scratch_nints))
        return fail_open(p, WASTE_INKLING_PRIVATE_E_FORMAT, detail, detail_capacity,
                         "Inkling state or scratch initialization failed");
    p->detail[0] = 0;
    copy_detail(detail, detail_capacity, "");
    *out = p;
    return WASTE_INKLING_PRIVATE_OK;
}

waste_inkling_private_status waste_inkling_private_step_trace(
    waste_inkling_private *p, int token, int position,
    float *logits, size_t logits_count, const waste_inkling_trace *trace)
{
    if (!p || !logits) return WASTE_INKLING_PRIVATE_E_ARG;
    if (waste_inkling_model_step_backend_trace(&p->cfg, &p->bound.model,
            p->backend, &p->state, &p->scratch, token, position,
            logits, logits_count, expert_get, p, trace)) {
        snprintf(p->detail, sizeof p->detail,
                 "Inkling step failed at token %d position %d", token, position);
        return WASTE_INKLING_PRIVATE_E_RUNTIME;
    }
    p->detail[0] = 0;
    return WASTE_INKLING_PRIVATE_OK;
}

waste_inkling_private_status waste_inkling_private_step(
    waste_inkling_private *p, int token, int position,
    float *logits, size_t logits_count)
{
    return waste_inkling_private_step_trace(p, token, position, logits,
                                             logits_count, NULL);
}

void waste_inkling_private_reset(waste_inkling_private *p)
{
    if (!p) return;
    waste_inkling_model_reset(&p->state, &p->cfg);
    p->detail[0] = 0;
}

void waste_inkling_private_close(waste_inkling_private *p)
{
    if (!p) return;
    if (p->tensor) {
        for (size_t i = 0; i < p->ntensors; i++) {
            waste_inkling_stage_tensor_close(&p->tensor[i].stage);
            waste_inkling_qtensor_close(&p->tensor[i].qtensor);
        }
    }
    for (int i = 0; i < WASTE_INKLING_MAX_LAYERS; i++) {
        waste_inkling_stage_bank_close(&p->bank[i]);
        waste_inkling_wexp_bank_close(&p->vq_bank[i]);
    }
    free(p->expert_gate); free(p->expert_up); free(p->expert_down); free(p->expert_raw);
    free(p->state_buffer); free(p->scratch_buffer); free(p->scratch_ints);
    free(p->resident); free(p->views); free(p->tensor);
    memset(p, 0, sizeof *p);
    free(p);
}

const waste_inkling_config *waste_inkling_private_config(
    const waste_inkling_private *p)
{
    return p ? &p->cfg : NULL;
}

const char *waste_inkling_private_error(const waste_inkling_private *p)
{
    return p && p->detail[0] ? p->detail : NULL;
}

size_t waste_inkling_private_quantized_trunk_bytes(
    const waste_inkling_private *p)
{
    return p ? p->quantized_resident_bytes : 0;
}

size_t waste_inkling_private_resident_f32_bytes(
    const waste_inkling_private *p)
{
    return p && p->resident_floats <= SIZE_MAX / sizeof(float)
         ? p->resident_floats * sizeof(float) : 0;
}
