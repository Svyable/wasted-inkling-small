#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
#
# Generate the integration patch and prove it is the reviewed tree.
#
#   verify.sh [workdir]
#
# Checks, in order:
#   1. the generated tree hash equals EXPECTED_APPLIED_TREE;
#   2. every Inkling translation unit compiles with -Werror;
#   3. the upstream suite passes (make check).
#
# Set WASTE_VERIFY_SKIP_CHECK=1 to stop after the compile step.
set -eu

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
work=${1:-$(mktemp -d)/waste}

# shellcheck source=/dev/null
. "$here/baseline.env"

tree=$("$here/generate.sh" "$work")

if [ "$tree" != "$EXPECTED_APPLIED_TREE" ]; then
    printf 'verify.sh: FAIL generated tree %s, expected %s\n' \
        "$tree" "$EXPECTED_APPLIED_TREE" >&2
    printf 'verify.sh: if the change is intended, update EXPECTED_APPLIED_TREE\n' >&2
    exit 1
fi
printf 'verify.sh: PASS tree %s\n' "$tree"

units=0
objects=$(mktemp -d)
for source in "$work"/src/arch.c "$work"/src/inkling*.c; do
    cc -std=gnu11 -Wall -Wextra -Werror -I"$work/src" \
        -c "$source" -o "$objects/$(basename "${source%.c}").o"
    units=$((units + 1))
done
rm -rf "$objects"
printf 'verify.sh: PASS %d Inkling translation units compiled with -Werror\n' "$units"

if [ "${WASTE_VERIFY_SKIP_CHECK:-0}" = "1" ]; then
    printf 'verify.sh: skipping make check (WASTE_VERIFY_SKIP_CHECK=1)\n'
    exit 0
fi

PATH=/usr/bin:/bin make -C "$work" check
printf 'verify.sh: PASS make check\n'
