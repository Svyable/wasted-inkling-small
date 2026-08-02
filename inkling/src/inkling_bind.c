/* SPDX-License-Identifier: Apache-2.0 */
#include "inkling_bind.h"
#include <stdio.h>
#include <string.h>

static const waste_inkling_tensor_view *findv(const waste_inkling_tensor_view *v,
                                               size_t n, const char *name)
{
    const waste_inkling_tensor_view *found = NULL;
    for (size_t i = 0; i < n; i++) {
        if (!v[i].name || strcmp(v[i].name, name)) continue;
        if (found) return NULL; /* duplicate is always invalid */
        found = &v[i];
    }
    return found;
}

static const waste_inkling_tensor_view *need_view(
    const waste_inkling_tensor_view *v, size_t n, const char *name,
    int ndim, int a, int b, int c)
{
    const waste_inkling_tensor_view *t = findv(v, n, name);
    if (!t || t->ndim != ndim) return NULL;
    const int want[3] = {a, b, c};
    for (int i = 0; i < ndim; i++) if (t->shape[i] != want[i]) return NULL;
    return t;
}

static const float *need(const waste_inkling_tensor_view *v, size_t n,
                         const char *name, int ndim,
                         int a, int b, int c)
{
    const waste_inkling_tensor_view *t = need_view(v, n, name, ndim, a, b, c);
    return t && t->data ? t->data : NULL;
}

int waste_inkling_bind_weights_ex_backend(
    waste_inkling_bound_weights *out,
    const waste_inkling_config *cfg,
    const waste_inkling_tensor_view *v,
    size_t n,
    waste_inkling_row_get_fn embedding_get, void *embedding_ctx,
    waste_inkling_row_get_fn unembedding_get, void *unembedding_ctx,
    int allow_nonresident_matrices)
{
    if (!out || !cfg || !v || !n) return -1;
    memset(out, 0, sizeof *out);

    const waste_inkling_tensor_view *embed = need_view(
        v, n, "inkling.embed", 2, cfg->vocab, cfg->hidden, 0);
    const waste_inkling_tensor_view *unembed = need_view(
        v, n, "inkling.unembed", 2, cfg->unpadded_vocab, cfg->hidden, 0);
    out->model.embedding = embed ? embed->data : NULL;
    out->model.embed_norm = need(v,n,"inkling.embed_norm",1,cfg->hidden,0,0);
    out->model.final_norm = need(v,n,"inkling.final_norm",1,cfg->hidden,0,0);
    out->model.unembedding = unembed ? unembed->data : NULL;
    out->model.unembedding_rows = cfg->unpadded_vocab;
    out->model.embedding_get = embedding_get;
    out->model.embedding_ctx = embedding_ctx;
    out->model.unembedding_get = unembedding_get;
    out->model.unembedding_ctx = unembedding_ctx;
    if (!embed || !unembed || (!embed->data && !embedding_get) ||
        (!unembed->data && !unembedding_get) || !out->model.embed_norm ||
        !out->model.final_norm) return -1;

    char name[128];
    for (int i = 0; i < cfg->n_layers; i++) {
        waste_inkling_layer_weights *w = &out->layer[i];
        const waste_inkling_layer_cfg *lc = &cfg->layer[i];
#define N1(field,role,d0) do { snprintf(name,sizeof name,"inkling.layer.%d.%s",i,role); \
    w->field=need(v,n,name,1,d0,0,0); if(!w->field)return -1; } while(0)
#define N2(field,role,d0,d1) do { snprintf(name,sizeof name,"inkling.layer.%d.%s",i,role); \
    w->field=need(v,n,name,2,d0,d1,0); if(!w->field)return -1; } while(0)
#define N2Q(field,role,d0,d1) do { snprintf(name,sizeof name,"inkling.layer.%d.%s",i,role); \
    const waste_inkling_tensor_view *tv_=need_view(v,n,name,2,d0,d1,0); \
    if(!tv_||(!tv_->data&&!allow_nonresident_matrices)){return -1;} w->field=tv_->data; } while(0)
#define N3Q(field,role,d0,d1,d2) do { snprintf(name,sizeof name,"inkling.layer.%d.%s",i,role); \
    const waste_inkling_tensor_view *tv_=need_view(v,n,name,3,d0,d1,d2); \
    if(!tv_||(!tv_->data&&!allow_nonresident_matrices)){return -1;} w->field=tv_->data; } while(0)
        N1(input_norm,"input_norm",cfg->hidden);
        N1(post_attention_norm,"post_attention_norm",cfg->hidden);
        N2Q(wq,"q",lc->num_heads*lc->head_dim,cfg->hidden);
        N2Q(wk,"k",lc->num_kv_heads*lc->head_dim,cfg->hidden);
        N2Q(wv,"v",lc->num_kv_heads*lc->head_dim,cfg->hidden);
        N2Q(wr,"r",lc->num_heads*cfg->d_rel,cfg->hidden);
        N2Q(wo,"o",cfg->hidden,lc->num_heads*lc->head_dim);
        N1(q_norm,"q_norm",lc->head_dim);
        N1(k_norm,"k_norm",lc->head_dim);
        N2(relative_proj,"rel_proj",cfg->d_rel,lc->relative_extent);
        N2(k_sconv,"k_sconv",lc->num_kv_heads*lc->head_dim,cfg->conv_kernel);
        N2(v_sconv,"v_sconv",lc->num_kv_heads*lc->head_dim,cfg->conv_kernel);
        N2(attn_sconv,"attn_sconv",cfg->hidden,cfg->conv_kernel);
        N2(mlp_sconv,"mlp_sconv",cfg->hidden,cfg->conv_kernel);
        w->sparse = i >= cfg->dense_layers;
        if (!w->sparse) {
            N2Q(dense_gate,"mlp.gate",cfg->dense_intermediate,cfg->hidden);
            N2Q(dense_up,"mlp.up",cfg->dense_intermediate,cfg->hidden);
            N2Q(dense_down,"mlp.down",cfg->hidden,cfg->dense_intermediate);
            N1(dense_global_scale,"mlp.global_scale",1);
        } else {
            N2Q(router_weight,"router.weight",cfg->n_routed_experts+cfg->n_shared_experts,cfg->hidden);
            N1(router_bias,"router.correction_bias",cfg->n_routed_experts);
            N1(router_global_scale,"router.global_scale",1);
            N3Q(shared_gate,"shared.gate",cfg->n_shared_experts,cfg->moe_intermediate,cfg->hidden);
            N3Q(shared_up,"shared.up",cfg->n_shared_experts,cfg->moe_intermediate,cfg->hidden);
            N3Q(shared_down,"shared.down",cfg->n_shared_experts,cfg->hidden,cfg->moe_intermediate);
        }
#undef N1
#undef N2
#undef N2Q
#undef N3Q
    }
    out->model.layer = out->layer;
    return 0;
}

int waste_inkling_bind_weights_ex(
    waste_inkling_bound_weights *out,
    const waste_inkling_config *cfg,
    const waste_inkling_tensor_view *v, size_t n,
    waste_inkling_row_get_fn embedding_get, void *embedding_ctx,
    waste_inkling_row_get_fn unembedding_get, void *unembedding_ctx)
{
    return waste_inkling_bind_weights_ex_backend(out, cfg, v, n,
        embedding_get, embedding_ctx, unembedding_get, unembedding_ctx, 0);
}

int waste_inkling_bind_weights(waste_inkling_bound_weights *out,
                               const waste_inkling_config *cfg,
                               const waste_inkling_tensor_view *v,
                               size_t n)
{
    return waste_inkling_bind_weights_ex_backend(out, cfg, v, n,
        NULL, NULL, NULL, NULL, 0);
}
