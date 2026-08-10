/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
#ifndef WASTE_INKLING_MANIFEST_H
#define WASTE_INKLING_MANIFEST_H

#include "inkling_config.h"
#include "json.h"

#ifdef __cplusplus
extern "C" {
#endif

/* The `config` object of a WASTE container manifest, or -1.
 *
 * The converter flattens the release's text config into `config` and keeps
 * the multimodal wrapper under `_outer`, exactly as the K3 path does; a raw
 * release config with a nested `text_config` is also accepted. Both the
 * planner and the loader have to resolve it the same way, or one of them
 * plans a different model from the one the other opens.
 */
int waste_inkling_manifest_config_token(const js_doc *d);

/* Build the geometry contract from that object.
 *
 * Every dimension is required and no default stands in for a missing one:
 * a config this build cannot read has to be an error rather than a plausible
 * model. Validation — divisibility, bounds, duplicate local-layer ids, top_k
 * against the routed count — belongs to waste_inkling_config_build(), and
 * nothing here second-guesses it. Returns 0, or -1.
 */
int waste_inkling_manifest_config(const js_doc *d, int cfg_tok,
                                  waste_inkling_config *out);

#ifdef __cplusplus
}
#endif
#endif
