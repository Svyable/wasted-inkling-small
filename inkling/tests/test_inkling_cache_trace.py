#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Dependency-light tests for the routing-trace generator.

The generator's job is to be honest about being a model: two parameters, each
fitted to one upstream measurement, and a structure that cannot accidentally
claim more than that. These tests pin the fit, the trace format, and the
dedup semantics that the batching lever depends on.
"""

import struct
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

from inkling_cache_trace import (
    DEFAULT_B,
    DEFAULT_G,
    DEFAULT_STICKINESS,
    GATE2_COVERAGE,
    GATE2_REUSE,
    TRACE_MAGIC,
    Router,
    coverage,
    generate,
    main,
    measured_reuse,
    weights,
    write_trace,
)


class ConcentrationFitTest(unittest.TestCase):
    def test_reproduces_gate2_coverage_within_two_points(self):
        w = weights(256)
        for frac, want in GATE2_COVERAGE:
            self.assertAlmostEqual(coverage(w, frac), want, delta=0.02,
                                   msg=f"top {frac:.1%}")

    def test_weights_are_a_normalised_decreasing_distribution(self):
        w = weights(256)
        self.assertAlmostEqual(sum(w), 1.0, places=9)
        self.assertEqual(w, sorted(w, reverse=True))
        self.assertTrue(all(x > 0 for x in w))

    def test_coverage_is_monotone(self):
        w = weights(256)
        last = 0.0
        for frac in (0.05, 0.1, 0.25, 0.5, 0.9, 1.0):
            got = coverage(w, frac)
            self.assertGreaterEqual(got, last)
            last = got
        self.assertAlmostEqual(coverage(w, 1.0), 1.0, places=9)

    def test_rejects_degenerate_parameters(self):
        for kwargs in ({"g": 0}, {"g": -1}, {"b": 0}, {"b": -2}):
            with self.assertRaises(ValueError):
                weights(256, **kwargs)
        with self.assertRaises(ValueError):
            weights(0)


class ReuseFitTest(unittest.TestCase):
    def test_reproduces_gate2_reuse(self):
        trace = generate(26, 256, 8, 400, stickiness=DEFAULT_STICKINESS)
        self.assertAlmostEqual(measured_reuse(trace), GATE2_REUSE, delta=0.03)

    def test_concentration_alone_does_not_explain_the_reuse(self):
        """The finding that justifies the second parameter existing: an IID
        draw from the fitted distribution reuses far less than was measured."""
        iid = measured_reuse(generate(26, 256, 8, 400, stickiness=0.0))
        self.assertLess(iid, GATE2_REUSE - 0.15)

    def test_reuse_rises_with_stickiness(self):
        low = measured_reuse(generate(8, 256, 6, 200, stickiness=0.1, seed=3))
        high = measured_reuse(generate(8, 256, 6, 200, stickiness=0.6, seed=3))
        self.assertGreater(high, low)

    def test_rejects_out_of_range_stickiness(self):
        for s in (-0.1, 1.0, 1.5):
            with self.assertRaises(ValueError):
                Router.make(256, 6, s, DEFAULT_G, DEFAULT_B)


class TraceShapeTest(unittest.TestCase):
    def test_each_step_selects_exactly_top_k(self):
        trace = generate(5, 64, 6, 50, seed=2)
        for step in trace:
            self.assertEqual(len(step), 5)
            for ids in step:
                self.assertEqual(len(ids), 6)
                self.assertEqual(len(set(ids)), 6, "duplicate expert in one layer")
                self.assertTrue(all(0 <= e < 64 for e in ids))

    def test_is_deterministic_for_a_seed(self):
        a = generate(4, 64, 6, 30, seed=11)
        b = generate(4, 64, 6, 30, seed=11)
        self.assertEqual(a, b)
        c = generate(4, 64, 6, 30, seed=12)
        self.assertNotEqual(a, c)

    def test_layers_route_independently(self):
        """A shared RNG must not make every layer pick the same experts."""
        trace = generate(8, 256, 6, 40, seed=5)
        same = sum(1 for step in trace if len(set(map(tuple, step))) == 1)
        self.assertEqual(same, 0)

    def test_rejects_impossible_geometry(self):
        for args in ((5, 64, 65, 10), (5, 64, 0, 10), (0, 64, 6, 10), (5, 64, 6, 0)):
            with self.assertRaises(ValueError):
                generate(*args)


class ChunkDedupTest(unittest.TestCase):
    """The batching lever depends entirely on this being a real union."""

    def test_chunk_one_is_top_k(self):
        trace = generate(4, 256, 6, 20, chunk=1, seed=4)
        self.assertTrue(all(len(ids) == 6 for step in trace for ids in step))

    def test_a_chunk_never_exceeds_its_positions_and_never_undercuts_one(self):
        for chunk in (2, 4, 16):
            trace = generate(4, 256, 6, 20, chunk=chunk, seed=4)
            for step in trace:
                for ids in step:
                    self.assertLessEqual(len(ids), 6 * chunk)
                    self.assertGreaterEqual(len(ids), 6)
                    self.assertEqual(len(set(ids)), len(ids))

    def test_dedup_strengthens_with_chunk_size(self):
        def per_position(chunk):
            trace = generate(8, 256, 6, 16, chunk=chunk, seed=9)
            recs = sum(len(i) for s in trace for i in s) / len(trace)
            return recs / chunk

        self.assertLess(per_position(32), per_position(8))
        self.assertLess(per_position(8), per_position(1) + 1e-9)

    def test_a_chunk_cannot_exceed_the_expert_count(self):
        trace = generate(2, 16, 6, 10, chunk=64, seed=1)
        for step in trace:
            for ids in step:
                self.assertLessEqual(len(ids), 16)


class TraceFileTest(unittest.TestCase):
    def test_roundtrip_header_and_payload(self):
        trace = generate(3, 32, 6, 7, seed=6)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.bin"
            write_trace(path, trace, 32)
            raw = path.read_bytes()
        magic, steps, layers, slots = struct.unpack("<IIII", raw[:16])
        self.assertEqual(magic, TRACE_MAGIC)
        self.assertEqual((steps, layers, slots), (7, 3, 32))

        off = 16
        for step in trace:
            for ids in step:
                (n,) = struct.unpack("<I", raw[off:off + 4])
                off += 4
                self.assertEqual(n, len(ids))
                got = struct.unpack(f"<{n}I", raw[off:off + 4 * n])
                off += 4 * n
                self.assertEqual(list(got), ids)
        self.assertEqual(off, len(raw), "trailing bytes in trace")

    def test_refuses_an_out_of_range_expert(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                write_trace(Path(tmp) / "t.bin", [[[0, 99]]], 32)


class CliTest(unittest.TestCase):
    def test_check_reports_both_fits(self):
        import io
        from contextlib import redirect_stdout

        with redirect_stdout(io.StringIO()) as out:
            self.assertEqual(main(["check"]), 0)
        text = out.getvalue()
        self.assertIn("Concentration", text)
        self.assertIn("Reuse", text)
        self.assertIn("FITTED", text)

    def test_generate_writes_a_file(self):
        import io
        from contextlib import redirect_stdout

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "g.bin"
            with redirect_stdout(io.StringIO()):
                self.assertEqual(
                    main(["generate", "--out", str(path), "--layers", "4",
                          "--top-k", "6", "--steps", "20"]), 0)
            self.assertTrue(path.is_file())

    def test_bad_parameters_exit_nonzero(self):
        import io
        from contextlib import redirect_stdout, redirect_stderr

        with tempfile.TemporaryDirectory() as tmp:
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                rc = main(["--stickiness", "1.5", "generate",
                           "--out", str(Path(tmp) / "x.bin"), "--steps", "5"])
        self.assertEqual(rc, 2)


if __name__ == "__main__":
    unittest.main()
