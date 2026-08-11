#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib
import unittest

import diagnose_inkling_portable_bf16_composed_moe as composed
import diagnose_inkling_portable_bf16_stateful_sparse_layer as stateful


class CheckedStatefulRunnerIsolationTest(unittest.TestCase):
    def test_checked_stateful_import_does_not_rebind_shared_execution(self):
        before = (
            stateful.run_c_stateful,
            composed.transform_aggregation_source,
            composed.ExactWeightCollector,
            composed.FixedWeightHelper,
            composed.build_helper,
        )
        runner = importlib.import_module("run_inkling_checked_bf16_stateful_layer")
        importlib.reload(runner)
        after = (
            stateful.run_c_stateful,
            composed.transform_aggregation_source,
            composed.ExactWeightCollector,
            composed.FixedWeightHelper,
            composed.build_helper,
        )
        self.assertEqual(after, before)

    def test_checked_stateful_runner_uses_explicit_analysis_adapter(self):
        runner = importlib.import_module("run_inkling_checked_bf16_stateful_layer")
        analysis = importlib.import_module("checked_bf16_stateful_analysis")
        self.assertIs(runner.analyze_layer, analysis.analyze_layer)
        self.assertTrue(callable(runner.run_c_stateful_profile))


if __name__ == "__main__":
    unittest.main()
