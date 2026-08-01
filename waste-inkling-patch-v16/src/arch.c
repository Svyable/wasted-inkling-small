/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
#include "arch.h"

#include <string.h>

waste_arch_kind waste_arch_classify(const char *arch)
{
    if (arch && strncmp(arch, "inkling", 7) == 0)
        return WASTE_ARCH_INKLING;
    return WASTE_ARCH_LEGACY;
}
