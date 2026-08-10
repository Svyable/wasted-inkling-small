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

/* Is this canonical tensor name one of the two vocabulary tables the engine
 * reads a single row of per token?
 *
 * Three places have to agree about that set or they disagree about memory:
 * the planner leaves these out of the resident floor, the loader leaves them
 * on disk, and the parameter report counts them as stored but not active. It
 * lives on the seam because a second copy of the rule is how the plan and the
 * load start describing different containers.
 *
 * Inkling keeps the embedding and the unembedding independent, so unlike the
 * Kimi path there are two of them. */
int waste_arch_row_backed(const char *name);

#ifdef __cplusplus
}
#endif
#endif
