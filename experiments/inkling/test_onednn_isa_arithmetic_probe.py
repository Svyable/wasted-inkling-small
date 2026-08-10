#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Contracts for the fixture-free oneDNN ISA probe.

These exercise the classifier against constructed arm outputs. The probe's
measurement itself needs Torch and a real host, so it is exercised by the
workflow rather than here; what must hold everywhere is that the classifier
refuses to over-read whatever it is handed.
"""
from __future__ import annotations

import unittest

import probe_onednn_isa_arithmetic as probe


class IsaProbeClassifierTest(unittest.TestCase):
    @staticmethod
    def arms(**digests: str):
        """Build a full arm set, defaulting every unnamed arm to 'native'."""
        return {
            name: {
                "output_sha256": digests.get(name, "native"),
                "cpu_capability": "AVX512",
            }
            for name, _ in probe.ARMS
        }

    def test_observed_signature_names_the_isa_cap(self):
        # The signature measured on an AVX-512 host with neither amx_bf16 nor
        # avx512_bf16: ATen forced to AVX2 is bitwise identical to native, the
        # AVX2 cap and mkldnn-off agree with each other and differ from native,
        # and every AVX-512 rung tracks native.
        result = probe.classify(
            self.arms(onednn_avx2="capped", mkldnn_off="capped"),
            ["avx512f", "avx512vl"],
        )
        self.assertEqual(result["classification"], "onednn_isa_cap_changes_the_result")
        self.assertEqual(
            result["first_rung_differing_from_avx2_baseline"],
            "onednn_avx512_core",
        )
        self.assertTrue(result["monotone"])
        self.assertTrue(result["onednn_isa_sensitive"])
        self.assertFalse(result["aten_dispatch_sensitive"])
        self.assertFalse(result["amx_bf16_available"])
        self.assertFalse(result["avx512_bf16_available"])

    def test_a_host_that_answers_every_arm_alike_claims_nothing(self):
        result = probe.classify(self.arms(), ["avx2"])
        self.assertEqual(
            result["classification"], "host_answers_every_arm_identically"
        )
        self.assertIsNone(result["first_rung_differing_from_avx2_baseline"])
        self.assertFalse(result["onednn_isa_sensitive"])

    def test_aten_sensitivity_is_reported_as_its_own_phenomenon(self):
        # The reference matrix excluded ATen dispatch. A host where forcing
        # ATen *does* change the bits is a different finding and must not be
        # folded into the oneDNN verdict.
        result = probe.classify(
            self.arms(
                aten_avx2="forced",
                onednn_avx2="capped",
                mkldnn_off="capped",
            ),
            ["avx512f"],
        )
        self.assertEqual(
            result["classification"],
            "aten_dispatch_changes_the_result_on_this_host",
        )
        self.assertTrue(result["aten_dispatch_sensitive"])

    def test_only_the_uncapped_path_differing_is_distinguished(self):
        # Every cap agrees, including the AMX rung, and only the uncapped run
        # differs. That is not "the cap changes the result" — it means
        # something outside the ladder's control is moving.
        result = probe.classify(
            self.arms(
                onednn_avx512_core="capped",
                onednn_avx512_core_bf16="capped",
                onednn_avx512_core_amx="capped",
                onednn_avx2="capped",
                mkldnn_off="capped",
            ),
            ["avx512f", "amx_bf16"],
        )
        self.assertEqual(result["classification"], "only_the_uncapped_path_differs")
        self.assertEqual(
            result["first_rung_differing_from_avx2_baseline"], "native"
        )
        self.assertTrue(result["monotone"])
        self.assertTrue(result["amx_bf16_available"])

    def test_nonmonotone_ladder_is_refused_rather_than_named(self):
        # Agrees with the AVX2 baseline, then differs at AVX512_CORE, then
        # agrees again at the BF16 rung. The cap is not the variable being
        # measured, so no rung may be named.
        arms = self.arms(onednn_avx2="capped", mkldnn_off="capped")
        arms["onednn_avx512_core_bf16"]["output_sha256"] = "capped"
        result = probe.classify(arms, ["avx512f"])
        self.assertEqual(result["classification"], "isa_ladder_nonmonotone")
        self.assertFalse(result["monotone"])

    def test_a_missing_arm_fails_closed(self):
        arms = self.arms(onednn_avx2="capped")
        del arms["onednn_avx512_core"]
        with self.assertRaisesRegex(probe.IsaProbeError, "missing arms"):
            probe.classify(arms, ["avx512f"])

    def test_every_ladder_rung_is_an_arm(self):
        names = {name for name, _ in probe.ARMS}
        for rung in probe.LADDER:
            self.assertIn(rung, names)


if __name__ == "__main__":
    unittest.main()
