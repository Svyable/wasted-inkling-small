#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib
import unittest

import diagnose_inkling_portable_bf16_composed_moe as composed


class CheckedRunnerIsolationTest(unittest.TestCase):
    def test_checked_runner_import_does_not_mutate_rewrite_era_globals(self):
        before = (
            composed.transform_aggregation_source,
            composed.ExactWeightCollector,
            composed.FixedWeightHelper,
            composed.build_helper,
        )
        runner = importlib.import_module("run_inkling_checked_bf16_profile")
        importlib.reload(runner)
        after = (
            composed.transform_aggregation_source,
            composed.ExactWeightCollector,
            composed.FixedWeightHelper,
            composed.build_helper,
        )
        self.assertEqual(after, before)


if __name__ == "__main__":
    unittest.main()
