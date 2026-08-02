/* SPDX-License-Identifier: Apache-2.0 */
#ifndef WASTE_INKLING_BIND_H
#define WASTE_INKLING_BIND_H
#include <stddef.h>
#include "inkling_model.h"
#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    const char *name;
    const float *data;
    int shape[4];
    int ndim;
} waste_inkling_tensor_view;

typedef struct {
    waste_inkling_model_weights model;
    waste_inkling_layer_weights layer[WASTE_INKLING_MAX_LAYERS];
} waste_inkling_bound_weights;

/* Bind canonical `inkling.*` tensor names to the private runtime structures.
 * Every required tensor must appear exactly once with its exact shape. Routed
 * expert matrices are intentionally absent: production supplies them through
 * the expert callback. */
int waste_inkling_bind_weights(waste_inkling_bound_weights *out,
                               const waste_inkling_config *cfg,
                               const waste_inkling_tensor_view *views,
                               size_t nviews);

/* Row-backed variant used by the staged-directory opener. Embedding and
 * unembedding views still have to be present with the exact official shapes,
 * but their data pointer may be NULL when the corresponding row callback is
 * supplied. All other tensors must be resident F32 views. */
int waste_inkling_bind_weights_ex(
    waste_inkling_bound_weights *out,
    const waste_inkling_config *cfg,
    const waste_inkling_tensor_view *views,
    size_t nviews,
    waste_inkling_row_get_fn embedding_get, void *embedding_ctx,
    waste_inkling_row_get_fn unembedding_get, void *unembedding_ctx);

/* Quantized-matrix variant. Matrix views may have data == NULL when
 * allow_nonresident_matrices is nonzero; vectors/scalars still must be
 * resident. The caller must supply a matrix backend at execution time. */
int waste_inkling_bind_weights_ex_backend(
    waste_inkling_bound_weights *out,
    const waste_inkling_config *cfg,
    const waste_inkling_tensor_view *views, size_t nviews,
    waste_inkling_row_get_fn embedding_get, void *embedding_ctx,
    waste_inkling_row_get_fn unembedding_get, void *unembedding_ctx,
    int allow_nonresident_matrices);
#ifdef __cplusplus
}
#endif
#endif
