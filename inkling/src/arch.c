/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
#include "arch.h"

#include <stddef.h>

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
