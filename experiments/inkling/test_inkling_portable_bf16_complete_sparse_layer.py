#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from copy import deepcopy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import diagnose_inkling_portable_bf16_composed_moe as implementation
from run_inkling_portable_bf16_complete_sparse_layer import (
    _FINAL_RESIDUAL_NEW,
    _FINAL_RESIDUAL_OLD,
    apply_final_residual_source,
)


class CompleteSparseLayerTest(unittest.TestCase):
    def test_execution_profile_binds_runner_cpu_and_thread_controls(self):
        cpuinfo = """processor : 0
vendor_id : GenuineIntel
cpu family : 6
model : 143
model name : Intel(R) Xeon(R) Platinum 8481C
stepping : 8
microcode : 0x2b000643
flags : fpu sse4_2 avx avx2 avx512f avx512bw

processor : 1
model name : ignored second processor
"""
        fake_torch = SimpleNamespace(
            __version__="2.13.0+cu130",
            backends=SimpleNamespace(
                cpu=SimpleNamespace(get_cpu_capability=lambda: "AVX512")
            ),
            get_num_threads=lambda: 1,
            get_num_interop_threads=lambda: 1,
        )
        environment = {
            "GITHUB_EVENT_NAME": "workflow_dispatch",
            "GITHUB_JOB": "complete-sparse-layer",
            "GITHUB_REF": "refs/heads/agent/evidence-profile",
            "GITHUB_RUN_ATTEMPT": "2",
            "GITHUB_RUN_ID": "123456",
            "GITHUB_SHA": "a" * 40,
            "RUNNER_NAME": "GitHub Actions 1000000000",
            "RUNNER_OS": "Linux",
            "RUNNER_ARCH": "X64",
            "ImageOS": "ubuntu24",
            "ImageVersion": "20260720.247.2",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cpuinfo"
            path.write_text(cpuinfo, encoding="utf-8")
            profile = implementation.collect_execution_profile(
                environ=environment,
                cpuinfo_path=path,
                torch_module=fake_torch,
                logical_cpu_count=4,
            )
            second_profile = implementation.collect_execution_profile(
                environ={
                    **environment,
                    "GITHUB_RUN_ID": "654321",
                    "RUNNER_NAME": "GitHub Actions 2000000000",
                },
                cpuinfo_path=path,
                torch_module=fake_torch,
                logical_cpu_count=4,
            )
        self.assertEqual(profile["workflow"]["run_id"], "123456")
        self.assertEqual(profile["workflow"]["run_attempt"], "2")
        self.assertEqual(
            profile["runner"]["runner_id"], "GitHub Actions 1000000000"
        )
        self.assertEqual(profile["runner"]["image_version"], "20260720.247.2")
        self.assertEqual(
            profile["cpu"]["model_name"],
            "Intel(R) Xeon(R) Platinum 8481C",
        )
        self.assertEqual(profile["cpu"]["model"], "143")
        self.assertIn("avx512f", profile["cpu"]["flags"])
        self.assertEqual(len(profile["cpu"]["flags_sha256"]), 64)
        self.assertEqual(profile["cpu"]["logical_cpu_count"], 4)
        self.assertEqual(profile["schema_version"], 3)
        self.assertEqual(profile["torch"]["cpu_capability"], "AVX512")
        self.assertEqual(profile["thread_environment"]["OMP_NUM_THREADS"], "1")
        self.assertEqual(len(profile["host_class_sha256"]), 64)
        self.assertEqual(len(profile["reference_profile_sha256"]), 64)
        self.assertEqual(
            profile["host_class_sha256"], second_profile["host_class_sha256"]
        )
        self.assertEqual(
            profile["reference_profile_sha256"],
            second_profile["reference_profile_sha256"],
        )

        avx2_torch = SimpleNamespace(
            __version__="2.13.0+cu130",
            backends=SimpleNamespace(
                cpu=SimpleNamespace(get_cpu_capability=lambda: "AVX2")
            ),
            get_num_threads=lambda: 1,
            get_num_interop_threads=lambda: 1,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cpuinfo"
            path.write_text(cpuinfo, encoding="utf-8")
            avx2_profile = implementation.collect_execution_profile(
                environ={**environment, "ATEN_CPU_CAPABILITY": "avx2"},
                cpuinfo_path=path,
                torch_module=avx2_torch,
                logical_cpu_count=4,
            )
            limited_path = Path(directory) / "cpuinfo-without-avx512"
            limited_path.write_text(
                cpuinfo.replace(" avx512f avx512bw", ""),
                encoding="utf-8",
            )
            limited_hardware_profile = implementation.collect_execution_profile(
                environ={**environment, "ATEN_CPU_CAPABILITY": "avx2"},
                cpuinfo_path=limited_path,
                torch_module=avx2_torch,
                logical_cpu_count=4,
            )
        self.assertEqual(
            profile["host_class_sha256"], avx2_profile["host_class_sha256"]
        )
        self.assertNotEqual(
            profile["reference_profile_sha256"],
            avx2_profile["reference_profile_sha256"],
        )
        self.assertNotEqual(
            profile["host_class_sha256"],
            limited_hardware_profile["host_class_sha256"],
        )

    def test_tensor_payload_preserves_float32_and_bfloat16_bits(self):
        tensor = implementation.torch.tensor(
            [1.0, -0.0, 1.00390625],
            dtype=implementation.torch.float32,
        )
        payload = implementation.tensor_payload(tensor)
        self.assertEqual(payload["shape"], [3])
        self.assertEqual(len(payload["float32_bits"]), 3)
        self.assertEqual(len(payload["bfloat16_bits"]), 3)
        self.assertEqual(len(payload["float32_sha256"]), 64)
        self.assertEqual(len(payload["bfloat16_sha256"]), 64)

    @staticmethod
    def _pair_arm(
        *,
        exact: bool,
        official: list[int],
        candidate: list[int],
        first_nonexact_stage: str = "post_attention_norm",
        routing_weights_match: bool = True,
    ):
        classification = (
            "complete_sparse_layers_are_bfloat16_exact"
            if exact
            else "complete_sparse_layer_mismatch_observed"
        )
        payloads = {
            "official": {
                "float32_bits": official,
                "bfloat16_bits": official,
            },
            "candidate": {
                "float32_bits": candidate,
                "bfloat16_bits": candidate,
            },
        }
        return {
            "execution_profile": {
                "schema_version": 3,
                "host_class_sha256": "h" * 64,
                "reference_profile_sha256": "r" * 64,
            },
            "evidence_outcome": {"classification": classification},
            "layers": {
                "2": {
                    "decision": {
                        "first_nonexact_stage": (
                            None if exact else first_nonexact_stage
                        )
                    },
                    "official_route": {
                        "routed_weights": [[1.0]],
                        "shared_gammas": [[2.0]],
                    },
                    "exact_routed_weight": (
                        [1.0] if routing_weights_match else [9.0]
                    ),
                    "exact_shared_weight": (
                        [2.0] if routing_weights_match else [8.0]
                    ),
                    "layer_out_payloads": payloads,
                }
            },
        }

    def test_dispatch_pair_truth_table_requires_causal_stage(self):
        native = self._pair_arm(exact=False, official=[1], candidate=[2])
        forced = self._pair_arm(exact=True, official=[3], candidate=[3])
        forced["execution_profile"]["reference_profile_sha256"] = "a" * 64
        result = implementation.classify_dispatch_pair(native, forced)
        self.assertEqual(result["classification"], "forced_avx2_closes_native_mismatch")
        self.assertTrue(result["reference_profile_bound"])
        self.assertFalse(result["denominator_defect_branch_live"])

        forced_nonexact = self._pair_arm(
            exact=False,
            official=[1],
            candidate=[2],
        )
        forced_nonexact["execution_profile"]["reference_profile_sha256"] = (
            "a" * 64
        )
        result = implementation.classify_dispatch_pair(native, forced_nonexact)
        self.assertEqual(
            result["classification"],
            "dispatch_invariant_pre_router_mismatch_excludes_denominator_cause",
        )
        self.assertTrue(result["routing_weights_match_official"])
        self.assertFalse(result["denominator_defect_branch_live"])

        routing_mismatch = self._pair_arm(
            exact=False,
            official=[1],
            candidate=[2],
            first_nonexact_stage="routed_gate0",
            routing_weights_match=False,
        )
        routing_mismatch["execution_profile"]["reference_profile_sha256"] = (
            "a" * 64
        )
        result = implementation.classify_dispatch_pair(native, routing_mismatch)
        self.assertEqual(
            result["classification"],
            "dispatch_invariant_routing_weight_mismatch_keeps_denominator_defect_live",
        )
        self.assertFalse(result["routing_weights_match_official"])
        self.assertTrue(result["denominator_defect_branch_live"])

        variant = deepcopy(forced_nonexact)
        variant["layers"]["2"]["layer_out_payloads"]["official"][
            "float32_bits"
        ] = [9]
        result = implementation.classify_dispatch_pair(native, variant)
        self.assertEqual(
            result["classification"],
            "dispatch_variant_mismatch_remains_profile_bound",
        )
        self.assertFalse(result["denominator_defect_branch_live"])

        invalid_pair = deepcopy(forced_nonexact)
        invalid_pair["execution_profile"]["reference_profile_sha256"] = "r" * 64
        with self.assertRaisesRegex(
            implementation.ComposedMoeError,
            "reference profiles are identical",
        ):
            implementation.classify_dispatch_pair(native, invalid_pair)

    def test_complete_outcome_names_nonexact_layers(self):
        analyses = {
            "2": {
                "decision": {
                    "classification": "single_token_sparse_layer_is_bfloat16_exact"
                }
            },
            "5": {
                "decision": {"classification": "composed_moe_mismatch_isolated"}
            },
        }
        self.assertEqual(
            implementation.complete_evidence_outcome(analyses),
            {
                "classification": "complete_sparse_layer_mismatch_observed",
                "exact_layers": [2],
                "nonexact_layers": [5],
            },
        )

    def test_final_residual_transform_is_explicit_and_fail_closed(self):
        transformed = apply_final_residual_source(_FINAL_RESIDUAL_OLD)
        self.assertIn("bf16_round_probe(x[i])", transformed)
        self.assertIn("bf16_round_probe(s->ff[i])", transformed)
        self.assertIn("bf16_round_probe(residual + branch)", transformed)
        with self.assertRaisesRegex(
            implementation.ComposedMoeError,
            "final layer residual",
        ):
            apply_final_residual_source(transformed)

    def test_final_residual_transform_does_not_touch_attention_residual(self):
        source = (
            "    for (int i = 0; i < hidden; i++) x[i] += s->branch[i];\n"
            + _FINAL_RESIDUAL_OLD
        )
        transformed = apply_final_residual_source(source)
        self.assertIn("x[i] += s->branch[i]", transformed)
        self.assertIn(_FINAL_RESIDUAL_NEW, transformed)

    def test_complete_candidate_compiles_and_preserves_production(self):
        with tempfile.TemporaryDirectory() as directory:
            library, source = implementation.build_composed_library(
                Path(directory) / "libinkling-complete-sparse-layer.so"
            )
            self.assertTrue(library.is_file())
            self.assertTrue(source["production_source_unchanged"])
            self.assertNotEqual(
                source["production_layer_sha256"],
                source["probe_layer_sha256"],
            )


if __name__ == "__main__":
    unittest.main()
