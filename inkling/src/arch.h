/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
#ifndef WASTE_ARCH_H
#define WASTE_ARCH_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    WASTE_ARCH_LEGACY = 0,
    WASTE_ARCH_INKLING = 1,
} waste_arch_kind;

/* Format-v0 Kimi containers historically treated `arch` as descriptive, so
 * unknown/empty names retain legacy behavior. Inkling is singled out because
 * feeding its manifest into Kimi memory formulas or forward dispatch can
 * produce plausible but invalid plans before tensor validation runs. */
waste_arch_kind waste_arch_classify(const char *arch);

#ifdef __cplusplus
}
#endif
#endif
