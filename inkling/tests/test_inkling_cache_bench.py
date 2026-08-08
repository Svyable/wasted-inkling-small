#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Tests for the expert-cache benchmark.

These compile `inkling_cache_sim.c` against upstream's real `ecache.c` and
run it, so they skip when that file is not beside them — which is the case in
this repository's own tree and not the case in an applied WASTE tree, where
CI runs them.

The properties checked are the ones every number in docs/THROUGHPUT.md rests
on: that the cache below one token's working set hits zero (upstream's most
predictive measurement), that the stand-in record used to keep a 32 GiB cache
off a 15 GiB machine does not change the answer, and that chunk dedup reduces
reads.
"""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import inkling_cache_trace as trace_mod
from inkling_cache_bench import (
    GATE5,
    INKLING,
    STANDIN_REC,
    BuildError,
    build,
    main,
    run_slots,
    slots_for,
)

# The applied tree puts ecache.c beside our sources; this repository does not.
WASTE_SRC = REPO / "src"


def _have_ecache() -> bool:
    return (WASTE_SRC / "ecache.c").is_file()


@unittest.skipUnless(_have_ecache(), "ecache.c not present (run in an applied WASTE tree)")
class SimulatorTest(unittest.TestCase):
    binary: Path
    tmp: tempfile.TemporaryDirectory

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.binary = build(WASTE_SRC, Path(cls.tmp.name) / "sim")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _trace(self, name, **kw):
        kw.setdefault("layers", 8)
        kw.setdefault("experts", 256)
        kw.setdefault("top_k", 6)
        kw.setdefault("steps", 60)
        experts = kw.pop("experts")
        t = trace_mod.generate(kw.pop("layers"), experts, kw.pop("top_k"),
                               kw.pop("steps"), **kw)
        path = Path(self.tmp.name) / name
        trace_mod.write_trace(path, t, experts)
        return path

    def test_zero_slots_is_all_misses(self):
        path = self._trace("z.bin")
        r = run_slots(self.binary, path, 0)
        self.assertEqual(r["hits"], 0)
        self.assertEqual(r["misses"], r["accesses"])

    def test_a_cache_below_one_working_set_hits_zero(self):
        """docs/GATES.md Gate 5's most predictive finding, reproduced against
        the real cache: one token touches layers*top_k records, and a cache
        that cannot hold them keeps nothing alive across a token."""
        layers, top_k = 8, 6
        path = self._trace("w.bin", layers=layers, top_k=top_k, steps=80)
        working_set = layers * top_k
        r = run_slots(self.binary, path, working_set // 3)
        self.assertEqual(r["hits"], 0, "a sub-working-set cache produced hits")

    def test_hit_rate_rises_with_slots(self):
        path = self._trace("m.bin", steps=120)
        last = -1.0
        for slots in (48, 128, 512, 1024, 2048):
            r = run_slots(self.binary, path, slots)
            self.assertGreaterEqual(r["hit_rate"], last)
            last = r["hit_rate"]

    def test_a_cache_holding_the_whole_bank_never_evicts(self):
        path = self._trace("f.bin", layers=4, experts=64, steps=60)
        r = run_slots(self.binary, path, 4 * 64)
        self.assertEqual(r["evictions"], 0)

    def test_accounting_is_closed(self):
        path = self._trace("a.bin")
        r = run_slots(self.binary, path, 256)
        self.assertEqual(r["hits"] + r["misses"], r["accesses"])
        self.assertEqual(r["fetches"], r["misses"])
        self.assertEqual(r["bytes_read"], r["misses"] * r["rec_bytes"])

    def test_the_standin_record_does_not_change_the_answer(self):
        """Hit rate is a function of the slot count and the trace alone, which
        is what lets a 32 GiB cache be measured on a 15 GiB machine. If this
        ever fails, every GiB/token figure in THROUGHPUT.md is wrong."""
        path = self._trace("s.bin", steps=80)
        slots = 300
        small = run_slots(self.binary, path, slots)
        proc = subprocess.run(
            [str(self.binary), str(path), str(slots * 4 * STANDIN_REC),
             str(4 * STANDIN_REC), "0"],
            capture_output=True, text=True, check=True)
        big = json.loads(proc.stdout)
        self.assertEqual(big["n_slots"], slots)
        self.assertEqual(big["hits"], small["hits"])
        self.assertEqual(big["misses"], small["misses"])

    def test_chunk_dedup_reduces_reads_per_position(self):
        """Measured with NO cache, so the effect is dedup alone. With a cache
        large enough to hold the short-range reuse the two levers overlap and
        chunking adds little — which is itself a finding, recorded in
        docs/THROUGHPUT.md, and the reason this test does not use one."""
        per_position = {}
        for chunk in (1, 16):
            path = self._trace(f"c{chunk}.bin", steps=max(1, 128 // chunk),
                               chunk=chunk)
            r = run_slots(self.binary, path, 0)
            per_position[chunk] = r["misses"] / r["steps"] / chunk
        self.assertLess(per_position[16], per_position[1])

    def test_dedup_and_caching_overlap(self):
        """The substitution the line above depends on: given a cache that
        already holds several working sets, chunking buys much less."""
        def per_position(chunk, slots):
            path = self._trace(f"o{chunk}_{slots}.bin",
                               steps=max(1, 128 // chunk), chunk=chunk)
            r = run_slots(self.binary, path, slots)
            return r["misses"] / r["steps"] / chunk

        gain_cold = per_position(1, 0) / per_position(16, 0)
        gain_warm = per_position(1, 600) / per_position(16, 600)
        self.assertGreater(gain_cold, gain_warm)

    def test_lru_and_lfru_are_both_accepted_and_differ(self):
        path = self._trace("p.bin", steps=120)
        lfru = run_slots(self.binary, path, 200, 0)
        lru = run_slots(self.binary, path, 200, 1)
        self.assertEqual(lfru["policy"], 0)
        self.assertEqual(lru["policy"], 1)
        self.assertNotEqual(lfru["hits"], lru["hits"])

    def test_rejects_an_expert_outside_the_declared_range(self):
        path = Path(self.tmp.name) / "bad.bin"
        trace_mod.write_trace(path, [[[0, 1]]], 256)
        raw = bytearray(path.read_bytes())
        raw[-4:] = (999).to_bytes(4, "little")     # expert 999 of 256
        path.write_bytes(bytes(raw))
        proc = subprocess.run([str(self.binary), str(path),
                               str(64 * STANDIN_REC), str(STANDIN_REC)],
                              capture_output=True, text=True)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("outside", proc.stderr)

    def test_rejects_a_truncated_trace(self):
        path = self._trace("t.bin", steps=20)
        raw = path.read_bytes()
        path.write_bytes(raw[:20])          # header plus one partial count
        proc = subprocess.run([str(self.binary), str(path),
                               str(64 * STANDIN_REC), str(STANDIN_REC)],
                              capture_output=True, text=True)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("truncated", proc.stderr)

    def test_rejects_a_bad_magic(self):
        path = Path(self.tmp.name) / "magic.bin"
        trace_mod.write_trace(path, [[[0]]], 256)
        raw = bytearray(path.read_bytes())
        raw[0] ^= 0xFF
        path.write_bytes(bytes(raw))
        proc = subprocess.run([str(self.binary), str(path),
                               str(64 * STANDIN_REC), str(STANDIN_REC)],
                              capture_output=True, text=True)
        self.assertNotEqual(proc.returncode, 0)

    def test_rejects_a_record_that_is_not_page_aligned(self):
        path = self._trace("r.bin", steps=5)
        proc = subprocess.run([str(self.binary), str(path), "100000", "5000"],
                              capture_output=True, text=True)
        self.assertNotEqual(proc.returncode, 0)


class DriverTest(unittest.TestCase):
    """These need no compiler."""

    def test_slots_for_matches_the_cache_formula(self):
        self.assertEqual(slots_for(6.34 * (1 << 30), INKLING["rec_bytes"]), 719)
        self.assertEqual(slots_for(0, INKLING["rec_bytes"]), 0)

    def test_gate5_table_slot_counts_match_its_published_ones(self):
        """If our record size for Kimi-Linear were wrong the validation would
        compare against the wrong row."""
        from inkling_cache_bench import GATE5_REC_BYTES

        for budget, want_slots, _frac, _hit in GATE5:
            got = slots_for(budget, GATE5_REC_BYTES)
            self.assertLessEqual(abs(got - want_slots) / want_slots, 0.005)

    def test_inkling_geometry_matches_the_throughput_tool(self):
        from inkling_throughput import geometry_from_config, load_config

        geo = geometry_from_config(load_config(REPO / "tests" / "data" /
                                               "inkling-small-config.json"))
        self.assertEqual(INKLING["layers"], geo.sparse_layers)
        self.assertEqual(INKLING["top_k"], geo.top_k)
        self.assertEqual(INKLING["experts"], geo.n_routed_experts)
        self.assertEqual(INKLING["rec_bytes"], geo.record_bytes)

    def test_build_names_a_missing_ecache(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(BuildError) as ctx:
                build(Path(tmp), Path(tmp) / "sim")
            self.assertIn("ecache.c", str(ctx.exception))

    def test_missing_ecache_exits_two_without_a_traceback(self):
        import io
        from contextlib import redirect_stdout, redirect_stderr

        with tempfile.TemporaryDirectory() as tmp:
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as err:
                rc = main(["--waste-src", tmp, "validate"])
        self.assertEqual(rc, 2)
        self.assertIn("ecache.c", err.getvalue())


if __name__ == "__main__":
    unittest.main()
