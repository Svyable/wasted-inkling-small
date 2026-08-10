/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 *
 * Inkling equations independently implemented from the official Apache-2.0
 * Transformers model.  No third-party kernel code is copied here.
 */
#include "inkling.h"

#include <float.h>
#include <math.h>
#include <stddef.h>

static float inkling_sigmoid(float x)
{
    if (x >= 0.0f) {
        const float z = expf(-x);
        return 1.0f / (1.0f + z);
    }
    const float z = expf(x);
    return z / (1.0f + z);
}

static float inkling_logsigmoid(float x)
{
    /* -softplus(-x), with branches that avoid overflow. */
    if (x >= 0.0f) return -log1pf(expf(-x));
    return x - log1pf(expf(x));
}

static int route_args_ok(const float *logits, const float *correction_bias,
                         int n_routed, int n_shared, int top_k,
                         int *routed_index, float *routed_weight,
                         float *shared_weight,
                         waste_inkling_numeric_profile profile)
{
    return logits && correction_bias && routed_index && routed_weight &&
           (!n_shared || shared_weight) && n_routed > 0 && n_shared >= 0 &&
           top_k > 0 && top_k <= n_routed &&
           waste_inkling_numeric_profile_valid(profile);
}

static int choose_routed(const float *logits, const float *correction_bias,
                         int n_routed, int top_k,
                         int *routed_index, float *choice_score,
                         waste_inkling_numeric_profile profile)
{
    /* Descending insertion into a fixed top-k set. Inkling-Small has k=6,
     * so this is cheaper and simpler than allocating/sorting 256 entries.
     *
     * BF16_REFERENCE completes the sigmoid and corrected choice score to BF16
     * before comparison. Exact cutoff ties still use the lower expert id:
     * that is WASTE's portable contract, intentionally distinct from a named
     * platform-specific official top-k route. */
    for (int j = 0; j < top_k; j++) {
        routed_index[j] = -1;
        choice_score[j] = -FLT_MAX;
    }
    for (int expert = 0; expert < n_routed; expert++) {
        float sigmoid = inkling_sigmoid(logits[expert]);
        float bias = correction_bias[expert];
        if (profile == WASTE_INKLING_NUMERIC_BF16_REFERENCE) {
            sigmoid = waste_inkling_bf16_round(sigmoid);
            bias = waste_inkling_bf16_round(bias);
        }
        float choice = sigmoid + bias;
        if (profile == WASTE_INKLING_NUMERIC_BF16_REFERENCE)
            choice = waste_inkling_bf16_round(choice);
        int pos = top_k;
        for (int j = 0; j < top_k; j++) {
            if (choice > choice_score[j] ||
                (choice == choice_score[j] &&
                 (routed_index[j] < 0 || expert < routed_index[j]))) {
                pos = j;
                break;
            }
        }
        if (pos == top_k) continue;
        for (int j = top_k - 1; j > pos; j--) {
            choice_score[j] = choice_score[j - 1];
            routed_index[j] = routed_index[j - 1];
        }
        choice_score[pos] = choice;
        routed_index[pos] = expert;
    }
    for (int j = 0; j < top_k; j++) if (routed_index[j] < 0) return -1;
    return 0;
}

static int normalize_route_f32(const float *logits,
                               int n_routed, int n_shared, int top_k,
                               float route_scale, float global_scale,
                               const int *routed_index,
                               float *routed_weight, float *shared_weight)
{
    float max_logp = -FLT_MAX;
    for (int j = 0; j < top_k; j++) {
        const float v = inkling_logsigmoid(logits[routed_index[j]]);
        routed_weight[j] = v;
        if (v > max_logp) max_logp = v;
    }
    for (int j = 0; j < n_shared; j++) {
        const float v = inkling_logsigmoid(logits[n_routed + j]);
        shared_weight[j] = v;
        if (v > max_logp) max_logp = v;
    }

    float denom = 0.0f;
    for (int j = 0; j < top_k; j++)
        denom += expf(routed_weight[j] - max_logp);
    for (int j = 0; j < n_shared; j++)
        denom += expf(shared_weight[j] - max_logp);
    if (!(denom > 0.0f) || !isfinite(denom)) return -1;

    const float scale = route_scale * global_scale / denom;
    for (int j = 0; j < top_k; j++)
        routed_weight[j] = expf(routed_weight[j] - max_logp) * scale;
    for (int j = 0; j < n_shared; j++)
        shared_weight[j] = expf(shared_weight[j] - max_logp) * scale;
    return 0;
}

static int normalize_route_bf16(const float *logits,
                                int n_routed, int n_shared, int top_k,
                                float route_scale, float global_scale,
                                const int *routed_index,
                                float *routed_weight, float *shared_weight)
{
    /* Retained #50/#51 post-reduction policy (0x7f): BF16 log-sigmoid,
     * stabilized deltas and exp terms; F32 denominator reduction; one BF16
     * denominator completion; then BF16 logdenom/LSE/output delta/exp and
     * BF16 completion after route/global scaling. */
    float max_logp = -FLT_MAX;
    for (int j = 0; j < top_k; j++) {
        const float selected = waste_inkling_bf16_round(logits[routed_index[j]]);
        const float value = waste_inkling_bf16_round(inkling_logsigmoid(selected));
        routed_weight[j] = value;
        if (value > max_logp) max_logp = value;
    }
    for (int j = 0; j < n_shared; j++) {
        const float selected = waste_inkling_bf16_round(logits[n_routed + j]);
        const float value = waste_inkling_bf16_round(inkling_logsigmoid(selected));
        shared_weight[j] = value;
        if (value > max_logp) max_logp = value;
    }

    float denom_f32 = 0.0f;
    for (int j = 0; j < top_k; j++) {
        const float delta = waste_inkling_bf16_round(routed_weight[j] - max_logp);
        denom_f32 += waste_inkling_bf16_round(expf(delta));
    }
    for (int j = 0; j < n_shared; j++) {
        const float delta = waste_inkling_bf16_round(shared_weight[j] - max_logp);
        denom_f32 += waste_inkling_bf16_round(expf(delta));
    }
    const float denom = waste_inkling_bf16_round(denom_f32);
    if (!(denom > 0.0f) || !isfinite(denom)) return -1;

    const float logdenom = waste_inkling_bf16_round(logf(denom));
    const float lse = waste_inkling_bf16_round(max_logp + logdenom);
    const float route = waste_inkling_bf16_round(route_scale);
    const float global = waste_inkling_bf16_round(global_scale);

    for (int j = 0; j < top_k; j++) {
        const float delta = waste_inkling_bf16_round(routed_weight[j] - lse);
        float value = waste_inkling_bf16_round(expf(delta));
        value = waste_inkling_bf16_round(value * route);
        value = waste_inkling_bf16_round(value * global);
        routed_weight[j] = waste_inkling_bf16_round(value);
    }
    for (int j = 0; j < n_shared; j++) {
        const float delta = waste_inkling_bf16_round(shared_weight[j] - lse);
        float value = waste_inkling_bf16_round(expf(delta));
        value = waste_inkling_bf16_round(value * route);
        value = waste_inkling_bf16_round(value * global);
        shared_weight[j] = waste_inkling_bf16_round(value);
    }
    return 0;
}

int waste_inkling_route_profile(
    const float *logits, const float *correction_bias,
    int n_routed, int n_shared, int top_k,
    float route_scale, float global_scale,
    int *routed_index, float *routed_weight, float *shared_weight,
    waste_inkling_numeric_profile profile)
{
    if (!route_args_ok(logits, correction_bias, n_routed, n_shared, top_k,
                       routed_index, routed_weight, shared_weight, profile))
        return -1;
    if (choose_routed(logits, correction_bias, n_routed, top_k,
                      routed_index, routed_weight, profile))
        return -1;
    return profile == WASTE_INKLING_NUMERIC_BF16_REFERENCE
         ? normalize_route_bf16(logits, n_routed, n_shared, top_k,
                                 route_scale, global_scale, routed_index,
                                 routed_weight, shared_weight)
         : normalize_route_f32(logits, n_routed, n_shared, top_k,
                                route_scale, global_scale, routed_index,
                                routed_weight, shared_weight);
}

int waste_inkling_route(const float *logits, const float *correction_bias,
                        int n_routed, int n_shared, int top_k,
                        float route_scale, float global_scale,
                        int *routed_index, float *routed_weight,
                        float *shared_weight)
{
    return waste_inkling_route_profile(
        logits, correction_bias, n_routed, n_shared, top_k,
        route_scale, global_scale, routed_index, routed_weight, shared_weight,
        WASTE_INKLING_NUMERIC_F32);
}

int waste_inkling_sconv_step(const float *x, float *state,
                             const float *weight, int channels, int kernel,
                             float *out)
{
    if (!x || !state || !weight || !out || channels <= 0 || kernel <= 0)
        return -1;
    for (int c = 0; c < channels; c++) {
        float *s = state + (size_t)c * (size_t)kernel;
        const float *w = weight + (size_t)c * (size_t)kernel;
        for (int k = 0; k + 1 < kernel; k++) s[k] = s[k + 1];
        s[kernel - 1] = x[c];
        float y = x[c]; /* residual inside InklingShortConvolution */
        for (int k = 0; k < kernel; k++) y += s[k] * w[k];
        out[c] = y;
    }
    return 0;
}

int waste_inkling_relative_bias(const float *relative_state,
                                const float *projection,
                                int heads, int d_rel, int extent,
                                int query_pos, int key_pos0, int kv_len,
                                float *out)
{
    if (!relative_state || !projection || !out || heads <= 0 || d_rel <= 0 ||
        extent <= 0 || kv_len < 0)
        return -1;
    for (int h = 0; h < heads; h++) {
        const float *r = relative_state + (size_t)h * (size_t)d_rel;
        float *dst = out + (size_t)h * (size_t)kv_len;
        for (int k = 0; k < kv_len; k++) {
            const int distance = query_pos - (key_pos0 + k);
            if (distance < 0 || distance >= extent) {
                dst[k] = 0.0f;
                continue;
            }
            float v = 0.0f;
            for (int d = 0; d < d_rel; d++)
                v += r[d] * projection[(size_t)d * (size_t)extent + (size_t)distance];
            dst[k] = v;
        }
    }
    return 0;
}

float waste_inkling_log_tau(int position, int n_floor, float alpha)
{
    if (position < 0 || n_floor <= 0) return NAN;
    const float ratio = (float)(position + 1) / (float)n_floor;
    return 1.0f + alpha * logf(ratio > 1.0f ? ratio : 1.0f);
}
