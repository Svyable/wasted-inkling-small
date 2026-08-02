# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from check_waste_baseline import (  # noqa: E402
    REQUIRED_FORMAT_MARKERS,
    REQUIRED_PATHS,
    REQUIRED_PUBLIC_SYMBOLS,
    inspect_baseline,
)


class WasteBaselineTests(unittest.TestCase):
    def make_checkout(self, *, version: tuple[int, int, int] = (0, 6, 2)) -> Path:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        for rel in REQUIRED_PATHS:
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()

        public = root / "src/waste.h"
        public.write_text(
            "\n".join((
                "#define WASTE_API_VERSION 1",
                f"#define WASTE_VERSION_MAJOR {version[0]}",
                f"#define WASTE_VERSION_MINOR {version[1]}",
                f"#define WASTE_VERSION_PATCH {version[2]}",
                *(f"int {symbol}(void);" for symbol in REQUIRED_PUBLIC_SYMBOLS),
            )),
            encoding="utf-8",
        )
        fmt = root / "src/waste_format.h"
        fmt.write_text(
            "\n".join((
                "#define WASTE_FORMAT_VERSION 0",
                "#define WASTE_ALIGN 4096u",
                *REQUIRED_FORMAT_MARKERS,
            )),
            encoding="utf-8",
        )
        return root

    def test_reviewed_baseline_passes(self) -> None:
        report = inspect_baseline(self.make_checkout())
        self.assertTrue(report.compatible)
        self.assertEqual(report.version, (0, 6, 2))
        self.assertFalse(report.warnings)

    def test_missing_public_symbol_fails(self) -> None:
        root = self.make_checkout()
        header = root / "src/waste.h"
        header.write_text(
            header.read_text(encoding="utf-8").replace("int waste_open(void);", ""),
            encoding="utf-8",
        )
        report = inspect_baseline(root)
        self.assertFalse(report.compatible)
        self.assertIn("waste_open", report.missing_public_symbols)

    def test_changed_format_fails(self) -> None:
        root = self.make_checkout()
        header = root / "src/waste_format.h"
        header.write_text(
            header.read_text(encoding="utf-8").replace(
                "#define WASTE_FORMAT_VERSION 0",
                "#define WASTE_FORMAT_VERSION 1",
            ),
            encoding="utf-8",
        )
        self.assertFalse(inspect_baseline(root).compatible)

    def test_newer_version_warns_unless_strict(self) -> None:
        root = self.make_checkout(version=(0, 7, 0))
        report = inspect_baseline(root)
        self.assertTrue(report.compatible)
        self.assertTrue(report.warnings)
        self.assertFalse(inspect_baseline(root, strict_version=True).compatible)


if __name__ == "__main__":
    unittest.main()
