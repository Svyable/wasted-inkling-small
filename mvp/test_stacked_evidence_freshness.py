#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Model-free contracts for semantic stacked-evidence freshness."""
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

from check_stacked_evidence_freshness import FreshnessError, check_freshness


def run(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def commit(repo: Path, message: str) -> str:
    run(repo, "add", "-A")
    run(
        repo,
        "-c",
        "user.name=Port Evidence Test",
        "-c",
        "user.email=port-evidence@example.invalid",
        "commit",
        "-m",
        message,
    )
    return run(repo, "rev-parse", "HEAD")


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="stacked-evidence-") as directory:
        repo = Path(directory)
        run(repo, "init", "-q")
        (repo / "semantic.txt").write_text("v1\n")
        (repo / "packaging.txt").write_text("p1\n")
        parent = commit(repo, "parent")
        semantic_blob = run(repo, "rev-parse", f"{parent}:semantic.txt")
        archive = {
            "stacked_parent": {
                "recorded_head": parent,
                "freshness_mode": "dependency_blobs",
                "dependencies": [
                    {"path": "semantic.txt", "blob_sha": semantic_blob},
                ],
            }
        }
        manifest = repo / "archive.json"
        manifest.write_text(json.dumps(archive) + "\n")

        result = check_freshness(manifest, "HEAD", repo=str(repo))
        assert result["status"] == "fresh"
        assert result["recorded_head"] == parent
        assert len(result["dependencies_checked"]) == 1

        # Unrelated parent evolution is allowed.
        (repo / "packaging.txt").write_text("p2\n")
        commit(repo, "unrelated packaging")
        result = check_freshness(manifest, "HEAD", repo=str(repo))
        assert result["status"] == "fresh"

        # A load-bearing blob change invalidates the evidence.
        (repo / "semantic.txt").write_text("v2\n")
        commit(repo, "semantic change")
        try:
            check_freshness(manifest, "HEAD", repo=str(repo))
        except FreshnessError as exc:
            assert "semantic.txt" in str(exc)
            assert "current blob" in str(exc)
        else:
            raise AssertionError("semantic parent drift was accepted")

        # A rewritten parent line also fails, even if the dependency bytes happen
        # to return to the old value.
        run(repo, "checkout", "-q", "--orphan", "rewritten")
        for path in repo.iterdir():
            if path.name != ".git" and path.is_file():
                path.unlink()
        (repo / "semantic.txt").write_text("v1\n")
        rewritten = commit(repo, "rewritten parent")
        assert run(repo, "rev-parse", f"{rewritten}:semantic.txt") == semantic_blob
        manifest.write_text(json.dumps(archive) + "\n")
        try:
            check_freshness(manifest, "HEAD", repo=str(repo))
        except FreshnessError as exc:
            assert "not an ancestor" in str(exc)
        else:
            raise AssertionError("rewritten parent ancestry was accepted")

    print(
        "PASS stacked evidence freshness: unrelated commits allowed; "
        "semantic drift and rewritten ancestry refused"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
