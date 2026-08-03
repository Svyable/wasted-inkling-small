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


def compare_pairwise(
    reference: Path | str,
    candidate: Path | str,
    *,
    atol: float = 1e-5,
    rtol: float = 1e-5,
    allow_candidate_extras: bool = False,
) -> dict:
    """Compare archives while keeping router IDs attached to their weights.

    The official prefill hook records full per-position layer outputs but only
    last-position intermediates. The C trace records intermediates at every
    position. ``allow_candidate_extras`` projects only those disclosed C-only
    names out of the numerical comparison; every official name remains
    mandatory and candidate extras are listed in the report.
    """
    ref, ref_meta = read_activation_archive(reference)
    got, got_meta = read_activation_archive(candidate)
    ref_pairs = canonicalize_router_pairs(ref)
    got_pairs = canonicalize_router_pairs(got)
    ref_names = set(ref)
    got_names = set(got)
    candidate_only = sorted(got_names - ref_names)
    reference_only = sorted(ref_names - got_names)
    compared_candidate = (
        {name: got[name] for name in ref if name in got}
        if allow_candidate_extras
        else got
    )
    with tempfile.TemporaryDirectory() as rd, tempfile.TemporaryDirectory() as cd:
        write_activation_archive(rd, ref, metadata=ref_meta)
        write_activation_archive(cd, compared_candidate, metadata=got_meta)
        report = compare_activation_archives(rd, cd, atol=atol, rtol=rtol)
    report["router_pair_canonicalization"] = {
        "reference": ref_pairs,
        "candidate": got_pairs,
        "order": "expert_id_ascending",
    }
    report["comparison_scope"] = {
        "mode": "reference" if allow_candidate_extras else "strict",
        "reference_names": len(ref_names),
        "candidate_names": len(got_names),
        "candidate_only_ignored": candidate_only if allow_candidate_extras else [],
        "candidate_only_count": len(candidate_only),
        "reference_only_missing": reference_only,
    }
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare-reference", required=True)
    ap.add_argument("--compare-candidate", required=True)
    ap.add_argument("--report")
    ap.add_argument("--atol", type=float, default=1e-5)
    ap.add_argument("--rtol", type=float, default=1e-5)
    ap.add_argument(
        "--allow-candidate-extras",
        action="store_true",
        help=(
            "compare every official reference name while listing and ignoring "
            "candidate-only trace names"
        ),
    )
    args = ap.parse_args(argv)
    try:
        report = compare_pairwise(
            args.compare_reference,
            args.compare_candidate,
            atol=args.atol,
            rtol=args.rtol,
            allow_candidate_extras=args.allow_candidate_extras,
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
