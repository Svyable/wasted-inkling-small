/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
#ifndef WASTE_INKLING_PUBLIC_H
#define WASTE_INKLING_PUBLIC_H

#include <stdint.h>

#include "waste.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Memory planning for an Inkling container.
 *
 * This is the only public capability Inkling has earned, and the boundary is
 * deliberate: planning answers a geometry question, so the worst it can get
 * wrong is a byte count. Inference is a claim about a model that has not been
 * run against its official weights, so `waste_open` still refuses.
 *
 * `manifest_json` is the raw text of manifest.json; the caller has already
 * read it and checked format_version and arch. Returns WASTE_OK, or
 * WASTE_E_FORMAT for a manifest this build cannot plan from — never a
 * plausible number derived from a missing field.
 */
waste_status waste_inkling_plan_memory_json(const char *manifest_json,
                                            uint32_t ctx_tokens,
                                            waste_memplan *out);

#ifdef __cplusplus
}
#endif
#endif
