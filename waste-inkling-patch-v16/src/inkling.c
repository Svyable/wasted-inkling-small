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

int waste_inkling_route(const float *logits, const float *correction_bias,
                        int n_routed, int n_shared, int top_k,
                        float route_scale, float global_scale,
                        int *routed_index, float *routed_weight,
                        float *shared_weight)
{
    if (!logits || !correction_bias || !routed_index || !routed_weight ||
        (n_shared && !shared_weight) || n_routed <= 0 || n_shared < 0 ||
        top_k <= 0 || top_k > n_routed)
        return -1;

    /* Descending insertion into a fixed top-k set.  Inkling-Small has k=6,
     * so this is cheaper and simpler than allocating/sorting 256 entries. */
    for (int j = 0; j < top_k; j++) {
        routed_index[j] = -1;
        routed_weight[j] = -FLT_MAX; /* choice score until normalization */
    }
    for (int expert = 0; expert < n_routed; expert++) {
        const float choice = inkling_sigmoid(logits[expert]) + correction_bias[expert];
        int pos = top_k;
        for (int j = 0; j < top_k; j++) {
            if (choice > routed_weight[j] ||
                (choice == routed_weight[j] &&
                 (routed_index[j] < 0 || expert < routed_index[j]))) {
                pos = j;
                break;
            }
        }
        if (pos == top_k) continue;
        for (int j = top_k - 1; j > pos; j--) {
            routed_weight[j] = routed_weight[j - 1];
            routed_index[j] = routed_index[j - 1];
        }
        routed_weight[pos] = choice;
        routed_index[pos] = expert;
    }

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
    for (int j = 0; j < top_k; j++) denom += expf(routed_weight[j] - max_logp);
    for (int j = 0; j < n_shared; j++) denom += expf(shared_weight[j] - max_logp);
    if (!(denom > 0.0f) || !isfinite(denom)) return -1;

    const float scale = route_scale * global_scale / denom;
    for (int j = 0; j < top_k; j++)
        routed_weight[j] = expf(routed_weight[j] - max_logp) * scale;
    for (int j = 0; j < n_shared; j++)
        shared_weight[j] = expf(shared_weight[j] - max_logp) * scale;
    return 0;
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
