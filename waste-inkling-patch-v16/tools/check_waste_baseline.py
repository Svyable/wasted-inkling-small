#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check the upstream WASTE seams the Inkling integration plans to reuse.

This is intentionally a structural drift check, not a compiler or semantic
compatibility test. It makes a changed public API or container contract visible
before another Inkling patch is stacked on stale assumptions.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path


BASELINE_COMMIT = "c7cb64022963871506908317a661338f1794f70e"
BASELINE_VERSION = (0, 6, 2)
EXPECTED_API_VERSION = 1
EXPECTED_FORMAT_VERSION = 0
EXPECTED_ALIGNMENT = 4096

REQUIRED_PATHS = (
    "src/waste.h",
    "src/waste.c",
    "src/model.c",
    "src/ecache.c",
    "src/waste_format.h",
    "tools/convert.py",
    "tools/make_test_container.py",
    "tools/pipeline.sh",
    "tools/fuzz_container.py",
    "tests/run.sh",
    ".github/workflows/ci.yml",
)

REQUIRED_PUBLIC_SYMBOLS = (
    "waste_plan_memory",
    "waste_cfg_init",
    "waste_open",
    "waste_close",
    "waste_tokenize",
    "waste_tokenize_markup",
    "waste_detokenize",
    "waste_generate",
    "waste_eval",
    "waste_state_save",
    "waste_state_load",
    "waste_state_reset",
    "waste_model_get_info",
    "waste_get_stats",
)

REQUIRED_FORMAT_MARKERS = (
    "WASTE_MAGIC_MANIFEST",
    "WASTE_MAGIC_EXPERT",
    "WASTE_MAGIC_TRUNK",
    "WASTE_MAGIC_CODEBOOK",
    "WQ_Q8G",
    "WQ_Q4G",
    "WQ_VQ3R",
    "WQ_VQ2R",
)


@dataclass(frozen=True)
class BaselineReport:
    root: str
    baseline_commit: str
    version: tuple[int, int, int] | None
    api_version: int | None
    format_version: int | None
    alignment: int | None
    missing_paths: tuple[str, ...]
    missing_public_symbols: tuple[str, ...]
    missing_format_markers: tuple[str, ...]
    warnings: tuple[str, ...]
    compatible: bool


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return ""


def _define_int(text: str, name: str) -> int | None:
    match = re.search(
        rf"^\s*#\s*define\s+{re.escape(name)}\s+"
        rf"(0[xX][0-9A-Fa-f]+|[0-9]+)(?:[uUlL]+)?\b",
        text,
        re.MULTILINE,
    )
    return int(match.group(1), 0) if match else None


def inspect_baseline(root: Path | str, *, strict_version: bool = False) -> BaselineReport:
    root = Path(root).resolve()
    missing_paths = tuple(path for path in REQUIRED_PATHS if not (root / path).is_file())
    public = _read(root / "src/waste.h")
    fmt = _read(root / "src/waste_format.h")

    major = _define_int(public, "WASTE_VERSION_MAJOR")
    minor = _define_int(public, "WASTE_VERSION_MINOR")
    patch = _define_int(public, "WASTE_VERSION_PATCH")
    version = None if None in (major, minor, patch) else (major, minor, patch)
    api_version = _define_int(public, "WASTE_API_VERSION")
    format_version = _define_int(fmt, "WASTE_FORMAT_VERSION")
    alignment = _define_int(fmt, "WASTE_ALIGN")

    missing_public_symbols = tuple(
        symbol for symbol in REQUIRED_PUBLIC_SYMBOLS
        if not re.search(rf"\b{re.escape(symbol)}\s*\(", public)
    )
    missing_format_markers = tuple(
        marker for marker in REQUIRED_FORMAT_MARKERS
        if not re.search(rf"\b{re.escape(marker)}\b", fmt)
    )

    warnings: list[str] = []
    if version is not None and version > BASELINE_VERSION:
        warnings.append(
            "upstream is newer than the reviewed 0.6.2 baseline; repeat the "
            "semantic comparison before integrating"
        )
    if version is not None and version < BASELINE_VERSION:
        warnings.append("upstream predates the reviewed 0.6.2 baseline")

    version_ok = version == BASELINE_VERSION if strict_version else (
        version is not None and version >= BASELINE_VERSION
    )
    compatible = not any((
        missing_paths,
        missing_public_symbols,
        missing_format_markers,
        not version_ok,
        api_version != EXPECTED_API_VERSION,
        format_version != EXPECTED_FORMAT_VERSION,
        alignment != EXPECTED_ALIGNMENT,
    ))

    return BaselineReport(
        root=str(root),
        baseline_commit=BASELINE_COMMIT,
        version=version,
        api_version=api_version,
        format_version=format_version,
        alignment=alignment,
        missing_paths=missing_paths,
        missing_public_symbols=missing_public_symbols,
        missing_format_markers=missing_format_markers,
        warnings=tuple(warnings),
        compatible=compatible,
    )


def _human(report: BaselineReport) -> str:
    status = "PASS" if report.compatible else "FAIL"
    version = ".".join(map(str, report.version)) if report.version else "unknown"
    lines = [
        f"WASTE upstream baseline: {status}",
        f"  checkout: {report.root}",
        f"  reviewed commit: {report.baseline_commit}",
        f"  version: {version}",
        f"  public API: {report.api_version}",
        f"  container format: {report.format_version}",
        f"  record alignment: {report.alignment}",
    ]
    for title, values in (
        ("missing paths", report.missing_paths),
        ("missing public symbols", report.missing_public_symbols),
        ("missing format markers", report.missing_format_markers),
    ):
        if values:
            lines.append(f"  {title}: {', '.join(values)}")
    lines.extend(f"  warning: {warning}" for warning in report.warnings)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", type=Path, help="local sqliteai/waste checkout")
    parser.add_argument("--json", action="store_true", help="emit a JSON report")
    parser.add_argument(
        "--strict-version",
        action="store_true",
        help="require exactly WASTE 0.6.2 instead of accepting newer versions",
    )
    args = parser.parse_args(argv)
    report = inspect_baseline(args.checkout, strict_version=args.strict_version)
    print(json.dumps(asdict(report), indent=2, sort_keys=True) if args.json else _human(report))
    return 0 if report.compatible else 1


if __name__ == "__main__":
    raise SystemExit(main())
