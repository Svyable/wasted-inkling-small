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
#   3. the pinned WASTE ABI, conversion and storage-measurement contracts hold;
#   4. the upstream suite passes (make check).
#
# Set WASTE_VERIFY_SKIP_CHECK=1 to stop after the contract step.
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
    if [ "${GITHUB_ACTIONS:-}" = "true" ]; then
        printf '::error file=integration/waste/baseline.env,title=Generated tree mismatch::generated tree %s; expected %s\n' \
            "$tree" "$EXPECTED_APPLIED_TREE"
    fi
    exit 1
fi
printf 'verify.sh: PASS tree %s\n' "$tree"

units=0
objects=$(mktemp -d)
for source in "$work"/src/arch.c "$work"/src/inkling*.c; do
    compile_log="$objects/compile.err"
    if ! cc -std=gnu11 -Wall -Wextra -Werror -I"$work/src" \
        -c "$source" -o "$objects/$(basename "${source%.c}").o" \
        2>"$compile_log"; then
        cat "$compile_log" >&2
        if [ "${GITHUB_ACTIONS:-}" = "true" ]; then
            detail=$(tr '\n' ' ' < "$compile_log" | sed 's/%/%25/g; s/\r/%0D/g; s/\n/%0A/g')
            printf '::error file=%s,title=Inkling -Werror compile failed::%s\n' \
                "${source#"$work/"}" "$detail"
        fi
        rm -rf "$objects"
        exit 1
    fi
    units=$((units + 1))
done
rm -rf "$objects"
printf 'verify.sh: PASS %d Inkling translation units compiled with -Werror\n' "$units"

if ! python3 "$here/check_upstream_contracts.py" "$work"; then
    if [ "${GITHUB_ACTIONS:-}" = "true" ]; then
        printf '::error file=integration/waste/check_upstream_contracts.py,title=WASTE contract verification failed::generated tree compiled, but an upstream ABI/conversion/storage contract failed\n'
    fi
    exit 1
fi
printf 'verify.sh: PASS WASTE 0.6.6 upstream contracts\n'

if [ "${WASTE_VERIFY_SKIP_CHECK:-0}" = "1" ]; then
    printf 'verify.sh: skipping make check (WASTE_VERIFY_SKIP_CHECK=1)\n'
    exit 0
fi

if ! PATH=/usr/bin:/bin make -C "$work" check; then
    if [ "${GITHUB_ACTIONS:-}" = "true" ]; then
        printf '::error file=integration/waste/verify.sh,title=Upstream make check failed::generated tree compiled and upstream contracts passed; make check failed\n'
    fi
    exit 1
fi
printf 'verify.sh: PASS make check\n'