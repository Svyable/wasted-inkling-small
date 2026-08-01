/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
#ifndef WASTE_INKLING_MODEL_H
#define WASTE_INKLING_MODEL_H

#include <stddef.h>

#include "inkling_layer.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef int (*waste_inkling_row_get_fn)(void *ctx, int row, int cols,
                                         float *out);

typedef struct {
    /* Either a resident row-major table or a callback must be supplied for
     * each of embedding and unembedding. The two tables are independent in
     * official Inkling checkpoints. */
    const float *embedding;       /* [vocab][hidden], optional */
    const float *embed_norm;      /* [hidden] */
    const float *final_norm;      /* [hidden] */
    const float *unembedding;     /* [unembedding_rows][hidden], optional */
    int unembedding_rows;
    const waste_inkling_layer_weights *layer; /* [n_layers] */

    waste_inkling_row_get_fn embedding_get;
    void *embedding_ctx;
    waste_inkling_row_get_fn unembedding_get;
    void *unembedding_ctx;
} waste_inkling_model_weights;

typedef struct {
    waste_inkling_layer_state layer[WASTE_INKLING_MAX_LAYERS];
    float *buffer;
    size_t buffer_floats;
    int context_capacity;
    int next_position;
} waste_inkling_model_state;

typedef struct {
    float *x;
    float *row;
    waste_inkling_layer_scratch layer;
    float *buffer;
    int *ibuffer;
    size_t buffer_floats;
    size_t ibuffer_ints;
} waste_inkling_model_scratch;

/* Exact caller-owned storage requirements. Local attention uses at most the
 * smaller of context_capacity and sliding_window; global attention uses the
 * requested context_capacity. Returns zero for invalid input or overflow. */
size_t waste_inkling_model_state_floats(const waste_inkling_config *cfg,
                                        int context_capacity);
size_t waste_inkling_model_scratch_floats(const waste_inkling_config *cfg,
                                          int context_capacity);
size_t waste_inkling_model_scratch_ints(const waste_inkling_config *cfg);

int waste_inkling_model_state_init(waste_inkling_model_state *state,
                                   const waste_inkling_config *cfg,
                                   int context_capacity,
                                   float *buffer, size_t buffer_floats);
int waste_inkling_model_scratch_init(waste_inkling_model_scratch *scratch,
                                     const waste_inkling_config *cfg,
                                     int context_capacity,
                                     float *buffer, size_t buffer_floats,
                                     int *ibuffer, size_t ibuffer_ints);
void waste_inkling_model_reset(waste_inkling_model_state *state,
                               const waste_inkling_config *cfg);

/* Execute one token and write exactly cfg->unpadded_vocab logits. Position
 * must be sequential and below both context_capacity and cfg->max_context.
 * Routed experts can still be supplied through the layer callback, keeping
 * this wrapper compatible with WASTE's streamed expert banks. */
int waste_inkling_model_step(
    const waste_inkling_config *cfg,
    const waste_inkling_model_weights *weights,
    waste_inkling_model_state *state,
    waste_inkling_model_scratch *scratch,
    int token, int position, float *logits, size_t logits_count,
    waste_inkling_expert_get_fn expert_get, void *expert_ctx);

/* Full-model entry point with one optional matrix backend per layer. The
 * legacy resident-F32 API above remains ABI-compatible and delegates here. */
int waste_inkling_model_step_backend(
    const waste_inkling_config *cfg,
    const waste_inkling_model_weights *weights,
    const waste_inkling_matrix_backend *backends,
    waste_inkling_model_state *state,
    waste_inkling_model_scratch *scratch,
    int token, int position, float *logits, size_t logits_count,
    waste_inkling_expert_get_fn expert_get, void *expert_ctx);

int waste_inkling_model_step_backend_trace(
    const waste_inkling_config *cfg,
    const waste_inkling_model_weights *weights,
    const waste_inkling_matrix_backend *backends,
    waste_inkling_model_state *state,
    waste_inkling_model_scratch *scratch,
    int token, int position, float *logits, size_t logits_count,
    waste_inkling_expert_get_fn expert_get, void *expert_ctx,
    const waste_inkling_trace *trace);

#ifdef __cplusplus
}
#endif
#endif
