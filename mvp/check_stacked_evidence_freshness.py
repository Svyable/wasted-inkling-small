#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Fail closed when stacked evidence no longer matches its semantic parent.

The evidence archive records the exact Git blob IDs of parent files that
produced or define its semantics. Unrelated parent commits do not invalidate
evidence; changes to those load-bearing files do.

The recorded parent head must remain an ancestor of the current parent ref, and
every declared dependency path must resolve to the recorded blob SHA.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


class FreshnessError(RuntimeError):
    pass


def _git(repo: str, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    command = ["git", "-C", repo, *args]
    result = subprocess.run(command, capture_output=True, text=True)
    if check and result.returncode:
        detail = result.stderr.strip() or result.stdout.strip() or "git command failed"
        raise FreshnessError(f"{' '.join(command)}: {detail}")
    return result


def _sha(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or len(value) != 40:
        raise FreshnessError(f"{label}: expected a 40-hex Git object id")
    try:
        int(value, 16)
    except ValueError as exc:
        raise FreshnessError(f"{label}: invalid Git object id") from exc
    return value.lower()


def check_freshness(
    manifest_path: str | os.PathLike[str],
    parent_ref: str,
    *,
    repo: str = ".",
) -> dict[str, Any]:
    try:
        archive = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FreshnessError(f"cannot read {manifest_path}: {exc}") from exc
    parent = archive.get("stacked_parent")
    if not isinstance(parent, dict):
        raise FreshnessError("archive has no stacked_parent contract")
    if parent.get("freshness_mode") != "dependency_blobs":
        raise FreshnessError("unsupported stacked_parent freshness_mode")
    recorded_head = _sha(parent.get("recorded_head"), label="recorded_head")
    dependencies = parent.get("dependencies")
    if not isinstance(dependencies, list) or not dependencies:
        raise FreshnessError("stacked_parent.dependencies must be non-empty")

    current_head = _git(repo, "rev-parse", f"{parent_ref}^{{commit}}").stdout.strip()
    ancestor = _git(
        repo,
        "merge-base",
        "--is-ancestor",
        recorded_head,
        current_head,
        check=False,
    )
    if ancestor.returncode != 0:
        raise FreshnessError(
            f"recorded parent head {recorded_head} is not an ancestor of "
            f"current parent {parent_ref} ({current_head})"
        )

    checked: list[dict[str, str]] = []
    for index, item in enumerate(dependencies):
        if not isinstance(item, dict):
            raise FreshnessError(f"dependency {index}: expected object")
        path = item.get("path")
        if (
            not isinstance(path, str)
            or not path
            or path.startswith("/")
            or path == ".."
            or path.startswith("../")
        ):
            raise FreshnessError(f"dependency {index}: invalid path")
        expected = _sha(item.get("blob_sha"), label=f"dependency {path}")
        actual_result = _git(
            repo, "rev-parse", f"{parent_ref}:{path}", check=False
        )
        if actual_result.returncode:
            raise FreshnessError(
                f"dependency {path}: missing from current parent {parent_ref}"
            )
        actual = actual_result.stdout.strip().lower()
        if actual != expected:
            raise FreshnessError(
                f"dependency {path}: current blob {actual} != recorded {expected}"
            )
        checked.append({"path": path, "blob_sha": actual})

    return {
        "status": "fresh",
        "parent_ref": parent_ref,
        "recorded_head": recorded_head,
        "current_head": current_head,
        "dependencies_checked": checked,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", help="evidence archive JSON")
    parser.add_argument("--parent-ref", required=True, help="Git ref of stacked parent")
    parser.add_argument("--repo", default=".", help="repository worktree")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = check_freshness(
            args.manifest, args.parent_ref, repo=os.path.abspath(args.repo)
        )
    except FreshnessError as exc:
        result = {"status": "stale", "error": str(exc)}
        if args.json:
            print(json.dumps(result, sort_keys=True))
        else:
            print(f"STALE EVIDENCE: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        print(
            f"FRESH {len(result['dependencies_checked'])} dependencies "
            f"against {result['parent_ref']} @ {result['current_head']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
