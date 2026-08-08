#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Dependency-light tests for the decode-cost model.

No torch is required. Where torch *is* available, the record-size arithmetic
is checked against a record `inkling_vq.write_expert_record` actually writes —
because a size model that drifts from the writer would silently move every
throughput number in the repository.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

from inkling_throughput import (
    BUDGET_CAP_FRACTION,
    CAL,
    GIB,
    Calibration,
    VQSpec,
    expert_record_bytes,
    geometry_from_config,
    k3_self_check,
    load_config,
    main,
    matrix_index_bytes,
    project,
)

CONFIG = REPO / "tests" / "data" / "inkling-small-config.json"

# The released Inkling-Small text geometry, restated here so a change to the
# recorded config cannot quietly move these tests with it.
RELEASE = {
    "num_hidden_layers": 42,
    "hidden_size": 4096,
    "intermediate_size": 2048,
    "dense_intermediate_size": 16384,
    "n_routed_experts": 256,
    "num_experts_per_tok": 6,
    "n_shared_experts": 2,
    "dense_mlp_idx": 2,
}


class RecordSizeTest(unittest.TestCase):
    def test_matches_the_published_vq3r_record(self):
        """9,457,664 B is quoted in ROADMAP-V19.md, WASTE-CONSTRAINTS.md and
        the memory plan promoted to waste_plan_memory()."""
        self.assertEqual(expert_record_bytes(4096, 2048, VQSpec()), 9_457_664)

    def test_record_is_page_aligned(self):
        for hidden, inter in ((4096, 2048), (2048, 512), (1024, 256)):
            self.assertEqual(expert_record_bytes(hidden, inter, VQSpec()) % 4096, 0)

    def test_index_bytes_pad_rows_to_the_index_block(self):
        spec = VQSpec(stages=3, vec_dim=8, index_block=64)
        # 100 rows pads to 128; 128 rows does not move.
        self.assertEqual(matrix_index_bytes(100, 64, spec), 128 * 8 * 3)
        self.assertEqual(matrix_index_bytes(128, 64, spec), 128 * 8 * 3)

    def test_two_stage_records_are_smaller(self):
        three = expert_record_bytes(4096, 2048, VQSpec(stages=3))
        two = expert_record_bytes(4096, 2048, VQSpec(stages=2))
        self.assertLess(two, three)

    def test_rejects_columns_not_divisible_by_vec_dim(self):
        with self.assertRaises(ValueError):
            matrix_index_bytes(64, 60, VQSpec(vec_dim=8))

    def test_rejects_invalid_spec(self):
        for spec in (VQSpec(stages=1), VQSpec(vec_dim=0), VQSpec(index_block=0)):
            with self.assertRaises(ValueError):
                expert_record_bytes(64, 64, spec)


class RecordSizeAgainstWriterTest(unittest.TestCase):
    """The model above duplicates `inkling_vq`'s layout so this tool can run
    without torch. This is the check that the duplicate stays honest."""

    def setUp(self):
        try:
            import torch  # noqa: F401
            import inkling_vq  # noqa: F401
        except ImportError as exc:
            self.skipTest(f"torch/inkling_vq unavailable: {exc}")

    def test_writer_produces_the_modelled_size(self):
        import torch
        import inkling_vq

        spec = inkling_vq.VQSpec()
        # Small, so the test is fast; the layout rule is size-independent and
        # test_matches_the_published_vq3r_record pins the real geometry.
        hidden, inter = 128, 64
        shapes = ((inter, hidden), (inter, hidden), (hidden, inter))
        matrices = [
            inkling_vq.QuantizedMatrix(
                indices=torch.zeros(
                    (spec.stages, rows * cols // spec.vec_dim), dtype=torch.uint8
                ),
                scales=torch.zeros(rows, dtype=torch.float16),
                shape=(rows, cols),
            )
            for rows, cols in shapes
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rec.bin"
            with path.open("wb") as f:
                written = inkling_vq.write_expert_record(f, 0, 0, 0, matrices, spec)
        self.assertEqual(
            written,
            expert_record_bytes(hidden, inter, VQSpec(
                stages=spec.stages, vec_dim=spec.vec_dim, index_block=spec.index_block
            )),
        )


class GeometryTest(unittest.TestCase):
    def setUp(self):
        self.geo = geometry_from_config(dict(RELEASE))

    def test_reads_the_recorded_config(self):
        self.assertEqual(geometry_from_config(load_config(CONFIG)).record_bytes,
                         self.geo.record_bytes)

    def test_layer_split(self):
        self.assertEqual(self.geo.dense_layers, 2)
        self.assertEqual(self.geo.sparse_layers, 40)

    def test_records_per_token_is_sparse_layers_times_top_k(self):
        self.assertEqual(self.geo.records_per_token, 240)

    def test_bytes_per_token(self):
        self.assertEqual(self.geo.bytes_per_token, 240 * 9_457_664)
        self.assertAlmostEqual(self.geo.working_set_gib, 2.114, places=3)

    def test_expert_bank_matches_the_published_total(self):
        """docs/ROADMAP-V19.md §G3: 96,846,479,360 B across 40 sparse layers."""
        self.assertEqual(self.geo.expert_bank_bytes, 96_846_479_360)
        self.assertAlmostEqual(self.geo.expert_bank_bytes / GIB, 90.2, places=1)

    def test_shared_experts_are_not_streamed(self):
        """They bind as trunk tensors (inkling_bind.c), so they must not
        appear in the per-token record count."""
        wider = dict(RELEASE, n_shared_experts=8)
        self.assertEqual(
            geometry_from_config(wider).records_per_token,
            self.geo.records_per_token,
        )

    def test_prefers_moe_intermediate_size_when_present(self):
        cfg = dict(RELEASE)
        del cfg["dense_intermediate_size"]
        cfg["moe_intermediate_size"] = 3072
        self.assertEqual(geometry_from_config(cfg).moe_intermediate, 3072)

    def test_rejects_a_config_carrying_both_width_keys(self):
        """The two schemas disagree about which key holds the routed width.
        A config with both is ambiguous and must not be guessed at."""
        cfg = dict(RELEASE, moe_intermediate_size=3072)
        with self.assertRaises(ValueError):
            geometry_from_config(cfg)

    def test_accepts_num_experts_as_an_alias(self):
        cfg = dict(RELEASE)
        del cfg["n_routed_experts"]
        cfg["num_experts"] = 256
        self.assertEqual(geometry_from_config(cfg).n_routed_experts, 256)


class FailClosedTest(unittest.TestCase):
    # Fields the decode cost genuinely does not depend on. `dense_mlp_idx`
    # decides how many layers stream experts, but the dense *width* only sizes
    # the trunk, which this tool takes from the published plan rather than
    # recomputing. `n_routed_experts` has an alias, tested separately.
    NOT_REQUIRED = {"dense_intermediate_size", "n_routed_experts"}

    def test_every_required_field_is_required(self):
        for key in RELEASE:
            if key in self.NOT_REQUIRED:
                continue
            cfg = dict(RELEASE)
            del cfg[key]
            with self.assertRaises(KeyError, msg=f"{key} was defaulted"):
                geometry_from_config(cfg)

    def test_dense_width_does_not_change_the_streaming_cost(self):
        """Stated as a test rather than left implicit: the dense MLP is
        trunk-resident, so its width cannot move bytes per token."""
        cfg = dict(RELEASE)
        del cfg["dense_intermediate_size"]
        self.assertEqual(
            geometry_from_config(cfg).bytes_per_token,
            geometry_from_config(dict(RELEASE)).bytes_per_token,
        )

    def test_rejects_non_positive_and_non_integer_fields(self):
        for bad in (0, -1, 2.5, "4096", True, None):
            with self.assertRaises((ValueError, KeyError)):
                geometry_from_config(dict(RELEASE, hidden_size=bad))

    def test_rejects_top_k_above_the_expert_count(self):
        with self.assertRaises(ValueError):
            geometry_from_config(dict(RELEASE, num_experts_per_tok=999))

    def test_rejects_dense_prefix_longer_than_the_model(self):
        with self.assertRaises(ValueError):
            geometry_from_config(dict(RELEASE, dense_mlp_idx=42))


class BudgetLadderTest(unittest.TestCase):
    """docs/GATES.md Gate 7: budget = floor + working_set * k, largest k in
    3..1 fitting under seven eighths of RAM."""

    def setUp(self):
        self.geo = geometry_from_config(dict(RELEASE))

    def test_eight_gib_machine_clears_one_working_set(self):
        budget, k = self.geo.budget(8)
        self.assertEqual(k, 1)
        self.assertLessEqual(budget, 8 * BUDGET_CAP_FRACTION)

    def test_sixteen_gib_machine_reaches_the_resolver_maximum(self):
        _budget, k = self.geo.budget(16)
        self.assertEqual(k, 3)

    def test_ladder_is_monotonic_in_ram(self):
        last = -1.0
        for ram in (4, 6, 8, 12, 16, 32, 64, 128):
            budget, _k = self.geo.budget(ram)
            self.assertGreaterEqual(budget, last)
            last = budget

    def test_below_the_floor_reports_zero_working_sets(self):
        """Gate 5 measured this regime at exactly zero demand hit rate, so it
        must be reported as its own case rather than as a small cache."""
        budget, k = self.geo.budget(4)
        self.assertEqual(k, 0)
        self.assertEqual(budget, self.geo.model_state_floor_gib)

    def test_budget_never_exceeds_the_cap(self):
        for ram in (4, 8, 16, 64, 128):
            budget, _k = self.geo.budget(ram)
            if budget > self.geo.model_state_floor_gib:
                self.assertLessEqual(budget, ram * BUDGET_CAP_FRACTION)


class TrunkConstantsTest(unittest.TestCase):
    """The trunk and floor are constants for the released Small, not
    derivations. Reusing them silently for another geometry would be exactly
    the confident-wrong-number this port refuses everywhere else."""

    def test_released_config_is_recognised(self):
        self.assertTrue(geometry_from_config(dict(RELEASE)).trunk_is_trustworthy)

    def test_a_different_geometry_is_not_vouched_for(self):
        for field, value in (
            ("num_hidden_layers", 24),
            ("hidden_size", 2048),
            ("intermediate_size", 1024),
            ("num_experts_per_tok", 8),
            ("n_shared_experts", 1),
        ):
            geo = geometry_from_config(dict(RELEASE, **{field: value}))
            self.assertFalse(geo.trunk_is_trustworthy, field)

    def test_overrides_make_another_geometry_trustworthy(self):
        geo = geometry_from_config(
            dict(RELEASE, num_hidden_layers=24), trunk_gib=2.0, floor_gib=2.5
        )
        self.assertTrue(geo.trunk_is_trustworthy)
        self.assertEqual(geo.trunk_gib, 2.0)
        self.assertEqual(geo.model_state_floor_gib, 2.5)

    def test_overrides_must_come_as_a_pair(self):
        for kwargs in ({"trunk_gib": 2.0}, {"floor_gib": 2.0}):
            with self.assertRaises(ValueError):
                geometry_from_config(dict(RELEASE), **kwargs)

    def test_overrides_must_be_positive(self):
        with self.assertRaises(ValueError):
            geometry_from_config(dict(RELEASE), trunk_gib=0.0, floor_gib=1.0)

    def test_cli_warns_on_an_unvouched_config(self):
        import io
        from contextlib import redirect_stdout, redirect_stderr

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "other.json"
            path.write_text(json.dumps(dict(RELEASE, num_hidden_layers=24)))
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as err:
                self.assertEqual(main(["--config", str(path), "geometry"]), 0)
            self.assertIn("WARNING", err.getvalue())

    def test_cli_is_silent_on_the_released_config(self):
        import io
        from contextlib import redirect_stdout, redirect_stderr

        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as err:
            main(["geometry"])
        self.assertNotIn("WARNING", err.getvalue())


class CalibrationTest(unittest.TestCase):
    def test_calibration_is_internally_consistent(self):
        CAL.check()

    def test_check_catches_a_transcription_error(self):
        with self.assertRaises(ValueError):
            Calibration(k3_io_seconds=5.0).check()

    def test_model_reproduces_k3s_measured_throughput(self):
        """The whole projection is worthless if the model cannot reproduce
        the one decode it was fitted against."""
        tok_s, _step = k3_self_check()
        low, high = CAL.k3_measured_tok_s
        self.assertGreaterEqual(tok_s, low - 0.05)
        self.assertLessEqual(tok_s, high + 0.05)


class ProjectionTest(unittest.TestCase):
    def setUp(self):
        self.geo = geometry_from_config(dict(RELEASE))

    def test_io_falls_with_hit_rate(self):
        cold = project(self.geo, 3.0, 0.0)
        warm = project(self.geo, 3.0, 0.5)
        self.assertLess(warm.io_seconds, cold.io_seconds)
        self.assertEqual(warm.expert_matmul_seconds, cold.expert_matmul_seconds)

    def test_io_falls_with_bandwidth(self):
        slow = project(self.geo, 1.0, 0.0)
        fast = project(self.geo, 10.0, 0.0)
        self.assertLess(fast.io_seconds, slow.io_seconds)

    def test_pipelining_never_exceeds_the_serial_step(self):
        p = project(self.geo, 3.0, 0.0)
        self.assertLessEqual(p.pipelined(p.other_low), p.serial_low)

    def test_stays_io_bound_on_a_laptop_class_disk(self):
        """The interesting structural claim: at 7 GiB/s and no cache hits the
        step is still dominated by reads, so prefetch and cache work continue
        to pay on Inkling. It is only on a top-tier SSD that the balance
        tips."""
        self.assertTrue(project(self.geo, 7.0, 0.0).io_bound)
        self.assertFalse(project(self.geo, 12.89, 0.42).io_bound)

    def test_tok_s_band_is_ordered(self):
        p = project(self.geo, 3.0, 0.29)
        self.assertLess(p.tok_s_low, p.tok_s_high)

    def test_rejects_impossible_inputs(self):
        for bw, hit in ((0.0, 0.0), (-1.0, 0.0), (3.0, 1.0), (3.0, -0.1)):
            with self.assertRaises(ValueError):
                project(self.geo, bw, hit)


class CliTest(unittest.TestCase):
    def test_subcommands_exit_zero(self):
        import io
        from contextlib import redirect_stdout

        for argv in (["geometry"], ["project"], ["compare"]):
            with redirect_stdout(io.StringIO()) as out:
                self.assertEqual(main(argv), 0, argv)
            self.assertIn("GiB", out.getvalue())

    def test_projection_output_is_labelled_as_an_estimate(self):
        """A projection that does not say so is the failure mode this whole
        repository is organised against."""
        import io
        from contextlib import redirect_stdout

        with redirect_stdout(io.StringIO()) as out:
            main(["project"])
        text = out.getvalue()
        self.assertIn("PROJECTION", text)
        self.assertIn("estimate, not a measurement", text)

    def test_bad_config_exits_nonzero_without_a_traceback(self):
        import io
        from contextlib import redirect_stdout, redirect_stderr

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text(json.dumps({"hidden_size": 4096}))
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as err:
                self.assertEqual(main(["--config", str(path), "geometry"]), 2)
            self.assertIn("missing required field", err.getvalue())


if __name__ == "__main__":
    unittest.main()
