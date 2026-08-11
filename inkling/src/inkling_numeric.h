/* SPDX-License-Identifier: Apache-2.0
 * Internal numeric execution profiles for the Inkling runtime.
 *
 * This header is not part of the public WASTE ABI.  F32 preserves the
 * checked-in synthetic/runtime behavior.  BF16_REFERENCE names only the
 * arithmetic completion policy established by the retained official-weight
 * evidence; selecting it does not change routing tie semantics or promote
 * public Inkling execution.
 */
#ifndef WASTE_INKLING_NUMERIC_H
#define WASTE_INKLING_NUMERIC_H

#include <math.h>
#include <stdint.h>
#include <string.h>

typedef enum {
    WASTE_INKLING_NUMERIC_F32 = 0,
    WASTE_INKLING_NUMERIC_BF16_REFERENCE = 1
} waste_inkling_numeric_profile;

static inline int waste_inkling_numeric_profile_valid(
    waste_inkling_numeric_profile profile)
{
    return profile == WASTE_INKLING_NUMERIC_F32 ||
           profile == WASTE_INKLING_NUMERIC_BF16_REFERENCE;
}

/* Round an IEEE-754 float to the nearest-even bfloat16 value, returned as an
 * exactly representable float.  NaN/Inf payloads are left untouched, matching
 * the evidence helpers used to derive the BF16 completion boundaries. */
static inline float waste_inkling_bf16_round(float value)
{
    uint32_t bits = 0;
    memcpy(&bits, &value, sizeof bits);
    if ((bits & 0x7f800000u) != 0x7f800000u) {
        bits += 0x7fffu + ((bits >> 16) & 1u);
        bits &= 0xffff0000u;
        memcpy(&value, &bits, sizeof value);
    }
    return value;
}

/* The single checked-in RMS normalization policy, shared by every Inkling
 * normalization site so the layer and the final head cannot drift apart.
 *
 * F32 keeps the original checked-in expression exactly.  BF16_REFERENCE is the
 * ordering the retained official-weight evidence established: normalize,
 * complete to BF16, multiply the BF16 weight, then complete the output again.
 * The sum of squares is accumulated in double in both profiles; that is the
 * pre-existing behavior and the evidence was gathered against it. */
static inline void waste_inkling_rmsnorm_profile(
    float *out, const float *x, const float *weight, int n, float eps,
    waste_inkling_numeric_profile profile)
{
    double ss = 0.0;
    for (int i = 0; i < n; i++) ss += (double)x[i] * x[i];
    const float scale = 1.0f / sqrtf((float)(ss / n) + eps);
    if (profile == WASTE_INKLING_NUMERIC_BF16_REFERENCE) {
        for (int i = 0; i < n; i++) {
            const float normalized = waste_inkling_bf16_round(x[i] * scale);
            out[i] = waste_inkling_bf16_round(normalized * weight[i]);
        }
    } else {
        for (int i = 0; i < n; i++) out[i] = x[i] * scale * weight[i];
    }
}

#endif
