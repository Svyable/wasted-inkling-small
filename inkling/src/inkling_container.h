/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
#ifndef WASTE_INKLING_CONTAINER_H
#define WASTE_INKLING_CONTAINER_H

#include "inkling_bind.h"
#include "inkling_config.h"
#include "json.h"
#include "model.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Open a public Inkling container into `m`.
 *
 * This is the loader half of the promotion the planner started: the guard in
 * waste_model_load() becomes a branch, and an Inkling manifest reaches an
 * Inkling config build and an Inkling tensor binding instead of a refusal.
 * What it does NOT do is execute — the expert cache and the forward path are
 * separate promotions and stay refused, so a model this returns is one that
 * has been placed in memory, not one that has produced a token.
 *
 * The container is the ordinary WASTE one: manifest.json for geometry,
 * trunk.bin for the named tensors at their declared offsets and widths, and
 * one expert bank per sparse layer. Nothing is read that format v0 does not
 * already define. `d` is the already-parsed manifest, `dir` the directory it
 * came from, `ctx_tokens` the context the caller asked for.
 *
 * Returns 0, -1 for an I/O failure, or -2 for a manifest this build cannot
 * read. On failure the model is left for waste_model_free() to clean up,
 * which is upstream's convention here and not an oversight: the load
 * allocates tensors before it can discover the manifest is wrong, and
 * freeing only the caller's struct is how a bad manifest used to cost tens
 * of gigabytes. Nothing bound is usable — waste_inkling_container_weights()
 * returns a zeroed table until the whole load has succeeded.
 */
int waste_inkling_container_load(waste_model *m, const char *dir,
                                 const js_doc *d, int ctx_tokens);

/* Release whatever the load attached. Safe on a Kimi model, on a partially
 * built one, and twice. */
void waste_inkling_container_free(waste_model *m);

/* Nonzero when `m` holds an Inkling container. The forward path tests this
 * to refuse; nothing else should need it. */
int waste_inkling_container_present(const waste_model *m);

/* The bound geometry, or NULL. */
const waste_inkling_config *waste_inkling_container_config(const waste_model *m);

/* What the load produced: the weight table the model step takes, and one
 * matrix backend per layer for the matrices that stayed in their stored
 * width. These are the two arguments waste_inkling_model_step_backend()
 * wants, which is why they are exposed in this shape rather than a shape
 * convenient for anything else — the promotion that runs a token consumes
 * them unchanged, and until then the suite drives them directly so the
 * row reader and the quantized matvec are exercised rather than merely
 * compiled. NULL when `m` holds no Inkling container. */
const waste_inkling_bound_weights *waste_inkling_container_weights(
    const waste_model *m);
const waste_inkling_matrix_backend *waste_inkling_container_backends(
    const waste_model *m);

/* --- defined by src/model.c, declared here because this is who they are for.
 *
 * Both are the loader's own machinery, exposed rather than copied. trunk.bin
 * is one format with one set of offsets, scale tables and packed widths; a
 * second reader for it in Inkling code would be a second thing to keep in
 * step with the converter, and the first divergence would show up as wrong
 * numbers rather than as a build failure. */

/* Read the manifest's `trunk` array into m->t, exactly as the Kimi path
 * does. Returns 0 or -1. */
int waste_trunk_load(waste_model *m, const char *dir, const js_doc *d,
                     int trunk);

/* One row of any trunk tensor as f32 — resident or left on disk, at
 * whatever width it is stored. waste_embed_row is this operation bound to
 * Kimi's embedding name; Inkling's two tables are named differently and
 * there is no reason for a second unpacker. Returns 0 or -1. */
int waste_trunk_row_f32(waste_model *m, const waste_tensor *t, long row,
                        int cols, float *dst);

#ifdef __cplusplus
}
#endif
#endif
