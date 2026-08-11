/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
#include "arch.h"

#include <stddef.h>
#include <string.h>

static unsigned char ascii_lower(unsigned char c)
{
    return c >= 'A' && c <= 'Z' ? (unsigned char)(c + ('a' - 'A')) : c;
}

waste_arch_kind waste_arch_classify(const char *arch)
{
    static const char inkling[] = "inkling";
    if (!arch) return WASTE_ARCH_LEGACY;
    for (size_t i = 0; i < sizeof inkling - 1; i++) {
        if (!arch[i] || ascii_lower((unsigned char)arch[i]) !=
                        (unsigned char)inkling[i])
            return WASTE_ARCH_LEGACY;
    }
    return WASTE_ARCH_INKLING;
}

int waste_arch_row_backed(const char *name)
{
    static const char *tails[] = { ".embed", ".unembed",
                                   "embed_tokens.weight", "unembed.weight" };
    if (!name) return 0;
    const size_t n = strlen(name);
    for (size_t i = 0; i < sizeof tails / sizeof tails[0]; i++) {
        const size_t t = strlen(tails[i]);
        if (n >= t && memcmp(name + n - t, tails[i], t) == 0) return 1;
    }
    return 0;
}
