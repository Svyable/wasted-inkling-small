#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare layer archives after canonicalizing router expert/weight pairs."""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TOOLS = REPO / "inkling" / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from inkling_parity import (  # noqa: E402
    ParityError,
    compare_activation_archives,
    read_activation_archive,
    write_activation_archive,
)
from inkling_release_config import (  # noqa: E402
    ReleaseConfigError,
    canonicalize_router_pairs,
)


def compare_pairwise(reference: Path | str, candidate: Path | str, *,
                     atol: float = 1e-5, rtol: float = 1e-5) -> dict:
    ref, ref_meta = read_activation_archive(reference)
    got, got_meta = read_activation_archive(candidate)
    ref_pairs = canonicalize_router_pairs(ref)
    got_pairs = canonicalize_router_pairs(got)
    with tempfile.TemporaryDirectory() as rd, tempfile.TemporaryDirectory() as cd:
        write_activation_archive(rd, ref, metadata=ref_meta)
        write_activation_archive(cd, got, metadata=got_meta)
        report = compare_activation_archives(rd, cd, atol=atol, rtol=rtol)
    report["router_pair_canonicalization"] = {
        "reference": ref_pairs,
        "candidate": got_pairs,
        "order": "expert_id_ascending",
    }
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare-reference", required=True)
    ap.add_argument("--compare-candidate", required=True)
    ap.add_argument("--report")
    ap.add_argument("--atol", type=float, default=1e-5)
    ap.add_argument("--rtol", type=float, default=1e-5)
    args = ap.parse_args(argv)
    try:
        report = compare_pairwise(
            args.compare_reference,
            args.compare_candidate,
            atol=args.atol,
            rtol=args.rtol,
        )
    except (ParityError, ReleaseConfigError, OSError, ValueError) as exc:
        ap.error(str(exc))
        return 2
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report:
        Path(args.report).write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
