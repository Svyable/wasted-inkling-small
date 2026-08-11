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

#endif
