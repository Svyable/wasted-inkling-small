/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 *
 * Inkling routed-expert storage over WASTE's existing bounded expert cache.
 *
 * This interface is deliberately below model execution: it opens the bank
 * files already validated by inkling_container.c and returns raw native WEXP
 * records.  No expert is dequantized here and public Inkling stepping remains
 * refused.  The next compute tranche can consume these same cached records
 * through WASTE's native VQ kernels without introducing a second cache or a
 * second on-disk format.
 */
#ifndef WASTE_INKLING_EXPERT_STORE_H
#define WASTE_INKLING_EXPERT_STORE_H

#include <stdint.h>

#include "json.h"
#include "model.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Open every sparse-layer bank named by the already-validated Inkling
 * manifest and initialize m->cache/m->miss_buf from the caller's normal WASTE
 * load options.  Bank geometry must already be present in m->bank[]. */
int waste_inkling_expert_store_open(waste_model *m, const char *dir,
                                    const js_doc *d,
                                    const waste_load_opts *opt);

/* Return one raw WEXP record through the normal WASTE ecache.  With a zero
 * cache budget this uses the model's aligned miss buffer, matching the native
 * WASTE path.  The returned pointer remains owned by the model/cache. */
const uint8_t *waste_inkling_expert_record_get(waste_model *m,
                                                int layer, int expert);

/* Existing ecache hint/hold semantics, exposed at the architecture boundary
 * so routed top-k can issue reads before matrix work begins in a later gate. */
void waste_inkling_expert_hint(waste_model *m, int layer,
                               const int *experts, int n);
const uint8_t *waste_inkling_expert_record_hold(waste_model *m,
                                                 int layer, int expert);
void waste_inkling_expert_release(waste_model *m);

#ifdef __cplusplus
}
#endif
#endif
