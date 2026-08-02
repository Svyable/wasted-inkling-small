#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
#
# Generate the WASTE Inkling integration patch from the source-of-truth tree.
#
#   generate.sh <workdir> [output-dir]
#
# The patch is a build product. Edit inkling/ and integration/waste/overlay/;
# never edit a generated patch. Prints the applied tree hash on success.
set -eu

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo=$(CDPATH= cd -- "$here/../.." && pwd)
work=${1:?usage: generate.sh <workdir> [output-dir]}
outdir=${2:-}

# shellcheck source=/dev/null
. "$here/baseline.env"

if [ -e "$work" ]; then
    printf 'generate.sh: %s already exists; refusing to overwrite\n' "$work" >&2
    exit 1
fi

# A cached clone keeps repeated runs off the network. WASTE_UPSTREAM_CACHE may
# point at an existing clone; it is only ever read from.
if [ -n "${WASTE_UPSTREAM_CACHE:-}" ] && [ -d "$WASTE_UPSTREAM_CACHE/.git" ]; then
    git clone --quiet --shared "$WASTE_UPSTREAM_CACHE" "$work"
else
    git clone --quiet "$UPSTREAM_REPOSITORY" "$work"
fi
git -C "$work" -c advice.detachedHead=false checkout --quiet --detach "$UPSTREAM_COMMIT"

actual_upstream=$(git -C "$work" rev-parse 'HEAD^{tree}')
if [ "$actual_upstream" != "$UPSTREAM_TREE" ]; then
    printf 'generate.sh: upstream tree %s, expected %s\n' \
        "$actual_upstream" "$UPSTREAM_TREE" >&2
    exit 1
fi

# 1. New files: a directory copy, because that is all they have ever been.
mkdir -p "$work/src" "$work/tools" "$work/tests/data" "$work/docs"
cp "$repo"/inkling/src/*.c "$repo"/inkling/src/*.h "$work/src/"
cp "$repo"/inkling/tools/*.py "$work/tools/"
cp "$repo"/inkling/tests/*.py "$repo"/inkling/tests/*.c "$work/tests/"
cp "$repo"/inkling/tests/data/*.json "$work/tests/data/"
cp "$repo"/inkling/docs/*.md "$work/docs/"

# 2. Files the bundle places at the upstream root. These are provenance notes
#    that v18 shipped inside the applied tree; they are kept here so the
#    generated tree stays byte-identical to the reviewed one.
if [ -d "$here/tree-extras" ]; then
    for extra in "$here"/tree-extras/*; do
        [ -e "$extra" ] || continue
        cp "$extra" "$work/"
    done
fi

# 3. Upstream edits: five small fragments, applied in a fixed order. A failure
#    here is the whole point — it names the one upstream file that moved.
for fragment in "$here"/overlay/*.diff; do
    [ -e "$fragment" ] || continue
    if ! git -C "$work" apply --whitespace=nowarn "$fragment"; then
        printf 'generate.sh: overlay fragment failed: %s\n' "$fragment" >&2
        printf 'generate.sh: re-resolve it against %s and commit the new fragment\n' \
            "$UPSTREAM_COMMIT" >&2
        exit 1
    fi
done

# --force because upstream's .gitignore lists data/, which would otherwise
# silently drop tests/data/inkling-small-*.json from the tree. `git am` never
# consulted .gitignore, so this is what keeps the generated tree equal to the
# replayed one.
git -C "$work" add -A --force
GIT_AUTHOR_DATE="$PATCH_DATE" GIT_COMMITTER_DATE="$PATCH_DATE" \
git -C "$work" \
    -c user.name="$PATCH_AUTHOR_NAME" \
    -c user.email="$PATCH_AUTHOR_EMAIL" \
    commit --quiet -m "$PATCH_SUBJECT"

tree=$(git -C "$work" rev-parse 'HEAD^{tree}')

if [ -n "$outdir" ]; then
    mkdir -p "$outdir"
    git -C "$work" format-patch --quiet -1 -o "$(CDPATH= cd -- "$outdir" && pwd)"
fi

printf '%s\n' "$tree"
