/* SPDX-License-Identifier: Apache-2.0
 * Converter-private staged Inkling runtime. Not part of the public WASTE API.
 */
#ifndef WASTE_INKLING_PRIVATE_H
#define WASTE_INKLING_PRIVATE_H

#include <stddef.h>
#include "inkling_model.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct waste_inkling_private waste_inkling_private;

typedef enum {
    WASTE_INKLING_PRIVATE_OK = 0,
    WASTE_INKLING_PRIVATE_E_ARG = -1,
    WASTE_INKLING_PRIVATE_E_IO = -2,
    WASTE_INKLING_PRIVATE_E_FORMAT = -3,
    WASTE_INKLING_PRIVATE_E_OOM = -4,
    WASTE_INKLING_PRIVATE_E_UNSUPPORTED = -5,
    WASTE_INKLING_PRIVATE_E_RUNTIME = -6
} waste_inkling_private_status;

typedef struct {
    int context_capacity;
    int verify_crc;
    int require_official_small;
} waste_inkling_private_options;

void waste_inkling_private_options_init(waste_inkling_private_options *options);

/* Open a directory containing runtime-stage.bin plus the trunk and expert
 * staging artifacts it indexes. The fixed binary index is converter-private;
 * this function deliberately does not accept manifest.json. */
waste_inkling_private_status waste_inkling_private_open(
    const char *stage_dir,
    const waste_inkling_private_options *options,
    waste_inkling_private **out,
    char *detail, size_t detail_capacity);

waste_inkling_private_status waste_inkling_private_step(
    waste_inkling_private *runtime,
    int token, int position,
    float *logits, size_t logits_count);

waste_inkling_private_status waste_inkling_private_step_trace(
    waste_inkling_private *runtime,
    int token, int position,
    float *logits, size_t logits_count,
    const waste_inkling_trace *trace);

void waste_inkling_private_reset(waste_inkling_private *runtime);
void waste_inkling_private_close(waste_inkling_private *runtime);
const waste_inkling_config *waste_inkling_private_config(
    const waste_inkling_private *runtime);
const char *waste_inkling_private_error(const waste_inkling_private *runtime);
/* Storage accounting for the private parity runtime. Quantized bytes are the
 * resident Q8/Q4 payload plus scales; resident_f32 excludes state/scratch and
 * includes only canonical non-matrix tensors. */
size_t waste_inkling_private_quantized_trunk_bytes(
    const waste_inkling_private *runtime);
size_t waste_inkling_private_resident_f32_bytes(
    const waste_inkling_private *runtime);

#ifdef __cplusplus
}
#endif
#endif
