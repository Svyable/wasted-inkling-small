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
        root = Path(directory)
        repo = root / "repo"
        repo.mkdir()
        run(repo, "init", "-q")
        (repo / "semantic.txt").write_text("v1\n")
        (repo / "packaging.txt").write_text("p1\n")
        parent = commit(repo, "parent")
        run(repo, "branch", "parent-base", parent)
        semantic_blob = run(repo, "rev-parse", f"{parent}:semantic.txt")

        run(repo, "checkout", "-q", "-b", "evidence", parent)
        (repo / "producer.txt").write_text("producer-v1\n")
        evidence = commit(repo, "evidence producer")
        run(repo, "branch", "evidence-original", evidence)
        producer_blob = run(repo, "rev-parse", f"{evidence}:producer.txt")

        archive = {
            "stacked_parent": {
                "recorded_head": parent,
                "freshness_mode": "dependency_blobs",
                "dependencies": [
                    {
                        "scope": "parent",
                        "path": "semantic.txt",
                        "blob_sha": semantic_blob,
                    },
                    {
                        "scope": "current",
                        "path": "producer.txt",
                        "blob_sha": producer_blob,
                    },
                ],
            }
        }
        manifest = root / "archive.json"
        manifest.write_text(json.dumps(archive) + "\n")

        result = check_freshness(
            manifest,
            "parent-base",
            repo=str(repo),
            current_ref="evidence-original",
        )
        assert result["status"] == "fresh"
        assert result["recorded_head"] == parent
        assert [item["scope"] for item in result["dependencies_checked"]] == [
            "parent",
            "current",
        ]

        # Unrelated parent evolution is allowed.
        run(repo, "checkout", "-q", "parent-base")
        (repo / "packaging.txt").write_text("p2\n")
        unrelated = commit(repo, "unrelated packaging")
        run(repo, "branch", "parent-unrelated", unrelated)
        result = check_freshness(
            manifest,
            "parent-unrelated",
            repo=str(repo),
            current_ref="evidence-original",
        )
        assert result["status"] == "fresh"

        # A load-bearing parent blob change invalidates the evidence.
        (repo / "semantic.txt").write_text("v2\n")
        semantic_drift = commit(repo, "semantic change")
        run(repo, "branch", "parent-semantic-drift", semantic_drift)
        try:
            check_freshness(
                manifest,
                "parent-semantic-drift",
                repo=str(repo),
                current_ref="evidence-original",
            )
        except FreshnessError as exc:
            assert "semantic.txt" in str(exc)
            assert "parent" in str(exc)
            assert "current blob" in str(exc)
        else:
            raise AssertionError("semantic parent drift was accepted")

        # A load-bearing producer change on the evidence branch is also stale.
        run(repo, "checkout", "-q", "evidence-original")
        (repo / "producer.txt").write_text("producer-v2\n")
        producer_drift = commit(repo, "producer change")
        run(repo, "branch", "evidence-drift", producer_drift)
        try:
            check_freshness(
                manifest,
                "parent-unrelated",
                repo=str(repo),
                current_ref="evidence-drift",
            )
        except FreshnessError as exc:
            assert "producer.txt" in str(exc)
            assert "current" in str(exc)
            assert "current blob" in str(exc)
        else:
            raise AssertionError("current producer drift was accepted")

        # Rewritten parent ancestry fails even when the semantic bytes return to
        # the recorded value.
        run(repo, "checkout", "-q", "--orphan", "parent-rewritten")
        for path in repo.iterdir():
            if path.name != ".git" and path.is_file():
                path.unlink()
        (repo / "semantic.txt").write_text("v1\n")
        (repo / "packaging.txt").write_text("rewritten\n")
        rewritten = commit(repo, "rewritten parent")
        assert run(repo, "rev-parse", f"{rewritten}:semantic.txt") == semantic_blob
        try:
            check_freshness(
                manifest,
                "parent-rewritten",
                repo=str(repo),
                current_ref="evidence-original",
            )
        except FreshnessError as exc:
            assert "not an ancestor" in str(exc)
        else:
            raise AssertionError("rewritten parent ancestry was accepted")

    print(
        "PASS stacked evidence freshness: parent/current producers scoped; "
        "unrelated commits allowed; semantic drift and rewritten ancestry refused"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
