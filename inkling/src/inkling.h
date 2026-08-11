/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
#ifndef WASTE_INKLING_H
#define WASTE_INKLING_H

#include "inkling_numeric.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Exact scalar pieces of the Inkling forward pass.  These are separated from
 * model.c so they can be differential-tested before architecture dispatch,
 * cache allocation, and tensor loading are wired. */

/* Select top_k routed experts from sigmoid(logit)+correction_bias, then jointly
 * normalize the selected routed logits and every shared logit using
 * log-sigmoid probabilities.  The normalized values are multiplied by
 * route_scale * global_scale.
 *
 * logits: [n_routed + n_shared]
 * correction_bias: [n_routed]
 * routed_index/routed_weight: [top_k]
 * shared_weight: [n_shared]
 *
 * Selected entries are returned in descending choice-score order; ties use the
 * lower expert id so routing itself is deterministic.  Reduction order is a
 * layer/numeric-profile contract, not a property of this router: the measured
 * BF16 layer profile reorders each selected (expert id, weight) pair by expert
 * id before accumulation, while the legacy F32 layer path retains its existing
 * ordering.
 */
int waste_inkling_route(const float *logits, const float *correction_bias,
                        int n_routed, int n_shared, int top_k,
                        float route_scale, float global_scale,
                        int *routed_index, float *routed_weight,
                        float *shared_weight);

/* Internal numeric-profile variant.  F32 is exactly the legacy function above.
 * BF16_REFERENCE applies the measured BF16 completion policy to router
 * choice-score and normalized-weight arithmetic while deliberately keeping
 * WASTE's deterministic low-ID cutoff tie rule.  It does not pretend that a
 * platform-specific official top-k choice among numerically tied experts is
 * portable; that ambiguity remains a separate evidence contract. */
int waste_inkling_route_profile(
    const float *logits, const float *correction_bias,
    int n_routed, int n_shared, int top_k,
    float route_scale, float global_scale,
    int *routed_index, float *routed_weight, float *shared_weight,
    waste_inkling_numeric_profile profile);

/* One causal depthwise short-convolution update, matching Inkling's fp32
 * cached-decode path.  state is [channels][kernel] and is updated in place.
 * out[c] = x[c] + dot(state_after_shift[c], weight[c]). */
int waste_inkling_sconv_step(const float *x, float *state,
                             const float *weight, int channels, int kernel,
                             float *out);

/* Materialize one query token's learned relative-position bias.
 * relative_state is [heads][d_rel], projection is [d_rel][extent], and out is
 * [heads][kv_len].  Bias is zero when distance is outside [0, extent). */
int waste_inkling_relative_bias(const float *relative_state,
                                const float *projection,
                                int heads, int d_rel, int extent,
                                int query_pos, int key_pos0, int kv_len,
                                float *out);

/* Global attention scaling tau = 1 + alpha*log(max((pos+1)/floor, 1)). */
float waste_inkling_log_tau(int position, int n_floor, float alpha);

#ifdef __cplusplus
}
#endif
#endif