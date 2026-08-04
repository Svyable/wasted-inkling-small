#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import copy
import unittest

from validate_inkling_official_weight_evidence import (
    EvidenceValidationError,
    validate_report,
)


class OfficialWeightEvidenceValidationTest(unittest.TestCase):
    def evidence(self):
        return {
            "format": "inkling-official-weight-parity-evidence",
            "version": 1,
            "model_id": "model",
            "revision": "abc1234",
            "config_sha256": "c",
            "index_sha256": "i",
            "inputs_sha256": "x",
            "tokens": 2,
            "input_dtype": "bfloat16",
            "passed": False,
            "layers": {
                "0": {
                    "routing": None,
                    "max_output_abs_across_positions": 2.0,
                    "final_position": {"input_norm": 0.5, "output": 1.5},
                },
                "2": {
                    "routing": {"passed": True},
                    "max_output_abs_across_positions": 3.0,
                    "final_position": {"input_norm": 0.75, "output": 2.5},
                },
            },
        }

    @staticmethod
    def comparison(layer: int, *, passed: bool = False):
        status = "pass" if passed else "mismatch"
        return {
            "format": "inkling-parity-report",
            "version": 1,
            "passed": passed,
            "comparison_scope": {"reference_only_missing": []},
            "results": [
                {
                    "name": f"token.0.layer.{layer}.output",
                    "status": status,
                    "max_abs": 1.0 if layer == 0 else 2.0,
                    "max_rel": 1.0,
                },
                {
                    "name": f"token.1.layer.{layer}.input_norm",
                    "status": status,
                    "max_abs": 0.4 if layer == 0 else 0.7,
                    "max_rel": 1.0,
                },
                {
                    "name": f"token.1.layer.{layer}.output",
                    "status": status,
                    "max_abs": 1.4 if layer == 0 else 2.4,
                    "max_rel": 1.0,
                },
            ],
        }

    def report(self, *, passed: bool = False):
        report = {
            "format": "inkling-official-weight-layer-parity",
            "version": 1,
            "model_id": "model",
            "revision": "abc1234",
            "config_sha256": "c",
            "index_sha256": "i",
            "inputs_sha256": "x",
            "tokens": 2,
            "input_dtype": "bfloat16",
            "passed": passed,
            "layers": {
                "0": {
                    "passed": passed,
                    "activation_comparison": self.comparison(0, passed=passed),
                    "routing": None,
                },
                "2": {
                    "passed": passed,
                    "activation_comparison": self.comparison(2, passed=passed),
                    "routing": {
                        "passed": True,
                        "exact_indices": True,
                        "weights_within_tolerance": True,
                        "positions": 2,
                    },
                },
            },
        }
        return report

    def test_accepts_complete_bounded_mismatch(self):
        result = validate_report(self.report(), self.evidence())
        self.assertEqual(result["status"], "expected-bounded-mismatch")
        self.assertTrue(result["passed"])

    def test_accepts_full_parity(self):
        result = validate_report(self.report(passed=True), self.evidence())
        self.assertEqual(result["status"], "parity")

    def test_require_parity_rejects_mismatch(self):
        with self.assertRaisesRegex(EvidenceValidationError, "parity is required"):
            validate_report(self.report(), self.evidence(), require_parity=True)

    def test_rejects_structural_comparison_failure(self):
        report = self.report()
        report["layers"]["0"]["activation_comparison"]["results"][0]["status"] = "missing"
        with self.assertRaisesRegex(EvidenceValidationError, "structural result"):
            validate_report(report, self.evidence())

    def test_rejects_source_identity_drift(self):
        report = self.report()
        report["revision"] = "different"
        with self.assertRaisesRegex(EvidenceValidationError, "source identity drift"):
            validate_report(report, self.evidence())

    def test_rejects_numerical_regression_above_envelope(self):
        report = self.report()
        report["layers"]["0"]["activation_comparison"]["results"][0]["max_abs"] = 3.0
        with self.assertRaisesRegex(EvidenceValidationError, "output regression"):
            validate_report(report, self.evidence(), growth_limit=1.25)

    def test_rejects_sparse_route_regression(self):
        report = self.report()
        report["layers"]["2"]["routing"]["exact_indices"] = False
        with self.assertRaisesRegex(EvidenceValidationError, "canonical route drift"):
            validate_report(report, self.evidence())

    def test_improvements_remain_valid(self):
        report = self.report()
        improved = copy.deepcopy(report)
        for layer in improved["layers"].values():
            for item in layer["activation_comparison"]["results"]:
                item["max_abs"] *= 0.1
        result = validate_report(improved, self.evidence())
        self.assertEqual(result["status"], "expected-bounded-mismatch")


if __name__ == "__main__":
    unittest.main()
