/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
#include "inkling_attention.h"
#include "inkling.h"

#include <math.h>
#include <stddef.h>
#include <string.h>

static int positive(int value) { return value > 0; }

static void rmsnorm_head(float *out, const float *x, const float *weight,
                         int dim, float eps)
{
    double ss = 0.0;
    for (int i = 0; i < dim; i++) ss += (double)x[i] * (double)x[i];
    const float scale = 1.0f / sqrtf((float)(ss / dim) + eps);
    for (int i = 0; i < dim; i++) out[i] = x[i] * scale * weight[i];
}

int waste_inkling_attention_init(waste_inkling_attention_state *s,
                                 int is_local, int num_heads,
                                 int num_kv_heads, int head_dim,
                                 int d_rel, int relative_extent,
                                 int capacity, float rms_eps,
                                 float *k_cache, float *v_cache)
{
    if (!s || !k_cache || !v_cache ||
        (is_local != 0 && is_local != 1) ||
        !positive(num_heads) || !positive(num_kv_heads) ||
        num_heads % num_kv_heads != 0 || !positive(head_dim) ||
        !positive(d_rel) || !positive(relative_extent) ||
        !positive(capacity) || !(rms_eps > 0.0f))
        return -1;
    memset(s, 0, sizeof *s);
    s->is_local = is_local;
    s->num_heads = num_heads;
    s->num_kv_heads = num_kv_heads;
    s->head_dim = head_dim;
    s->d_rel = d_rel;
    s->relative_extent = relative_extent;
    s->capacity = capacity;
    s->rms_eps = rms_eps;
    s->k_cache = k_cache;
    s->v_cache = v_cache;
    return 0;
}

void waste_inkling_attention_reset(waste_inkling_attention_state *s)
{
    if (s) s->next_position = 0;
}

int waste_inkling_attention_step(
    waste_inkling_attention_state *s,
    const float *q, const float *k, const float *v,
    const float *q_norm, const float *k_norm,
    const float *relative_state, const float *relative_proj,
    int position, int log_scaling_n_floor, float log_scaling_alpha,
    float *scores, float *out)
{
    if (!s || !q || !k || !v || !q_norm || !k_norm ||
        !relative_state || !relative_proj || !scores || !out ||
        position < 0 || position != s->next_position ||
        log_scaling_n_floor < 0 || log_scaling_alpha < 0.0f)
        return -1;
    if (!s->is_local && position >= s->capacity) return -1;

    const int H = s->num_heads;
    const int KH = s->num_kv_heads;
    const int D = s->head_dim;
    const int group = H / KH;
    const int slot = s->is_local ? position % s->capacity : position;
    const size_t kv_stride = (size_t)KH * (size_t)D;
    float *kdst = s->k_cache + (size_t)slot * kv_stride;
    float *vdst = s->v_cache + (size_t)slot * kv_stride;

    /* `out` temporarily stores normalized queries. Every score row is built
     * before that head's output overwrites its query row. */
    for (int h = 0; h < H; h++)
        rmsnorm_head(out + (size_t)h * D, q + (size_t)h * D,
                     q_norm, D, s->rms_eps);
    for (int h = 0; h < KH; h++)
        rmsnorm_head(kdst + (size_t)h * D, k + (size_t)h * D,
                     k_norm, D, s->rms_eps);
    memcpy(vdst, v, kv_stride * sizeof(float));

    const int begin = s->is_local && position + 1 > s->capacity
                    ? position + 1 - s->capacity : 0;
    const int kv_len = position - begin + 1;
    const float tau = log_scaling_n_floor > 0
                    ? waste_inkling_log_tau(position, log_scaling_n_floor,
                                            log_scaling_alpha)
                    : 1.0f;
    const float dot_scale = tau / (float)D;

    for (int h = 0; h < H; h++) {
        const int kh = h / group;
        const float *qh = out + (size_t)h * D;
        float *row = scores + (size_t)h * s->capacity;
        float max_score = -INFINITY;
        for (int j = 0; j < kv_len; j++) {
            const int key_pos = begin + j;
            const int key_slot = s->is_local ? key_pos % s->capacity : key_pos;
            const float *kc = s->k_cache + (size_t)key_slot * kv_stride
                            + (size_t)kh * D;
            float score = 0.0f;
            for (int d = 0; d < D; d++) score += qh[d] * kc[d];
            score *= dot_scale;
            const int distance = position - key_pos;
            if (distance >= 0 && distance < s->relative_extent) {
                const float *rs = relative_state + (size_t)h * s->d_rel;
                float bias = 0.0f;
                for (int r = 0; r < s->d_rel; r++)
                    bias += rs[r] * relative_proj[(size_t)r * s->relative_extent + distance];
                score += tau * bias;
            }
            row[j] = score;
            if (score > max_score) max_score = score;
        }
        double sum = 0.0;
        for (int j = 0; j < kv_len; j++) {
            row[j] = expf(row[j] - max_score);
            sum += row[j];
        }
        if (!(sum > 0.0) || !isfinite(sum)) return -1;
        const float inv_sum = (float)(1.0 / sum);
        float *oh = out + (size_t)h * D;
        for (int d = 0; d < D; d++) oh[d] = 0.0f;
        for (int j = 0; j < kv_len; j++) {
            const int key_pos = begin + j;
            const int key_slot = s->is_local ? key_pos % s->capacity : key_pos;
            const float *vc = s->v_cache + (size_t)key_slot * kv_stride
                            + (size_t)kh * D;
            const float weight = row[j] * inv_sum;
            for (int d = 0; d < D; d++) oh[d] += weight * vc[d];
        }
    }
    s->next_position++;
    return 0;
}
