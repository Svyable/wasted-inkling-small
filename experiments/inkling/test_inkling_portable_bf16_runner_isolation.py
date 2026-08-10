#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib
import unittest

import diagnose_inkling_portable_bf16_composed_moe as implementation
import run_inkling_portable_bf16_composed_moe as composed_runner
import run_inkling_portable_bf16_complete_sparse_layer as complete_runner


class RunnerIsolationTest(unittest.TestCase):
    def test_complete_runner_import_does_not_poison_composed_transform(self):
        self.assertIs(
            implementation.transform_aggregation_source,
            composed_runner.transform_aggregation_source,
        )
        importlib.reload(complete_runner)
        self.assertIs(
            implementation.transform_aggregation_source,
            composed_runner.transform_aggregation_source,
        )

    def test_complete_main_override_is_scoped_and_exception_safe(self):
        before_transform = implementation.transform_aggregation_source
        before_collector = implementation.ExactWeightCollector
        original_main = implementation.main
        seen = {}

        def fake_main(argv=None):
            seen["argv"] = argv
            seen["transform"] = implementation.transform_aggregation_source
            seen["collector"] = implementation.ExactWeightCollector
            raise RuntimeError("sentinel")

        implementation.main = fake_main
        try:
            with self.assertRaisesRegex(RuntimeError, "sentinel"):
                complete_runner.main(["probe"])
        finally:
            implementation.main = original_main

        self.assertEqual(seen["argv"], ["probe"])
        self.assertIs(
            seen["transform"],
            complete_runner.transform_complete_sparse_layer_source,
        )
        self.assertIs(seen["collector"], composed_runner.ExactWeightCollector)
        self.assertIs(implementation.transform_aggregation_source, before_transform)
        self.assertIs(implementation.ExactWeightCollector, before_collector)

    def test_complete_build_override_is_scoped(self):
        before_transform = implementation.transform_aggregation_source
        before_collector = implementation.ExactWeightCollector
        original_build = implementation.build_composed_library
        seen = {}

        def fake_build(out):
            seen["out"] = out
            seen["transform"] = implementation.transform_aggregation_source
            seen["collector"] = implementation.ExactWeightCollector
            return out, {"production_source_unchanged": True}

        implementation.build_composed_library = fake_build
        try:
            out, source = complete_runner.build_complete_sparse_layer_library(
                "sentinel.so"
            )
        finally:
            implementation.build_composed_library = original_build

        self.assertEqual(out, "sentinel.so")
        self.assertTrue(source["production_source_unchanged"])
        self.assertEqual(seen["out"], "sentinel.so")
        self.assertIs(
            seen["transform"],
            complete_runner.transform_complete_sparse_layer_source,
        )
        self.assertIs(seen["collector"], composed_runner.ExactWeightCollector)
        self.assertIs(implementation.transform_aggregation_source, before_transform)
        self.assertIs(implementation.ExactWeightCollector, before_collector)


if __name__ == "__main__":
    unittest.main()
