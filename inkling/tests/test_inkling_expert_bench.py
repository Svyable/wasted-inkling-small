#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Tests for the expert-matvec benchmark.

The benchmark's whole value rests on one property: the two paths it times
compute the same function. A speed comparison between two implementations
that disagree measures nothing, so that check is the benchmark's first act
and these tests confirm it is load-bearing rather than decorative.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "tools" / "inkling_expert_bench.c"


def _cc() -> str | None:
    return os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc")


@unittest.skipUnless(_cc() and SOURCE.is_file(), "no C compiler or source")
class ExpertBenchTest(unittest.TestCase):
    tmp: tempfile.TemporaryDirectory
    binary: Path

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.binary = Path(cls.tmp.name) / "bench"
        proc = subprocess.run(
            [_cc(), "-O2", "-std=gnu11", "-Wall", "-Wextra", "-Werror",
             "-o", str(cls.binary), str(SOURCE), "-lm"],
            capture_output=True, text=True)
        if proc.returncode != 0:
            raise AssertionError(f"benchmark does not compile clean:\n{proc.stderr}")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_bench(self, hidden=512, inter=256, reps=1):
        proc = subprocess.run(
            [str(self.binary), str(hidden), str(inter), str(reps)],
            capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        return json.loads(proc.stdout)

    def test_the_two_paths_agree(self):
        """If this ever fails the LUT formulation is not a drop-in and the
        speedup is meaningless."""
        r = self.run_bench()
        self.assertLess(r["rel_disagreement"], 1e-5)

    def test_agreement_holds_across_geometries(self):
        for hidden, inter in ((512, 256), (1024, 512), (2048, 512)):
            r = self.run_bench(hidden, inter)
            self.assertLess(r["rel_disagreement"], 1e-5, f"{hidden}x{inter}")

    def test_expanded_size_is_the_three_matrices(self):
        r = self.run_bench(512, 256)
        self.assertEqual(r["expert_f32_bytes"], 3 * 256 * 512 * 4)

    def test_expand_moves_the_expansion_twice(self):
        """Write it, then read it straight back. This is the part no amount
        of SIMD or threading changes."""
        r = self.run_bench(512, 256)
        self.assertEqual(r["expand_traffic_per_expert"], 2 * r["expert_f32_bytes"])

    def test_lut_moves_far_less_than_expand(self):
        r = self.run_bench(4096, 2048)
        self.assertGreater(r["traffic_ratio"], 10.0)
        self.assertLess(r["lut_traffic_per_expert"], r["expand_traffic_per_expert"])

    def test_lut_traffic_is_dominated_by_bytes_already_read_from_disk(self):
        """The index planes are what the expert record IS, so the LUT path
        adds almost nothing to what streaming already paid for."""
        r = self.run_bench(4096, 2048)
        self.assertLess(r["lut_table_bytes"], r["lut_traffic_per_expert"])

    def test_reports_the_released_expert_count(self):
        r = self.run_bench(512, 256)
        self.assertEqual(r["experts_per_token"], 240 + 80)

    def test_rejects_geometry_it_cannot_lay_out(self):
        for hidden, inter in ((100, 256), (512, 100), (0, 256), (512, 0)):
            proc = subprocess.run(
                [str(self.binary), str(hidden), str(inter), "1"],
                capture_output=True, text=True)
            self.assertNotEqual(proc.returncode, 0, f"{hidden}x{inter} accepted")

    def test_rejects_zero_reps(self):
        proc = subprocess.run([str(self.binary), "512", "256", "0"],
                              capture_output=True, text=True)
        self.assertNotEqual(proc.returncode, 0)

    def test_timings_are_positive_and_expert_is_three_matrices(self):
        r = self.run_bench(1024, 512, reps=2)
        self.assertGreater(r["expand_matrix_s"], 0)
        self.assertGreater(r["lut_matrix_s"], 0)
        # Both fields are printed with %.6f, so the per-expert figure is
        # three times an unrounded value while matrix_s*3 is three times a
        # rounded one. Half a printed ulp times three is the honest tolerance.
        tol = 3 * 0.5e-6
        self.assertAlmostEqual(r["expand_expert_s"], r["expand_matrix_s"] * 3, delta=tol)
        self.assertAlmostEqual(r["lut_expert_s"], r["lut_matrix_s"] * 3, delta=tol)


if __name__ == "__main__":
    unittest.main()
