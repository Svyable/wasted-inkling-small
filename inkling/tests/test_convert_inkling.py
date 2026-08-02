#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.

import json
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "tests"))

import convert_inkling
from test_inkling_plan import make_checkpoint
from test_inspect_inkling import inkling_config


class ConvertInklingEntryPointTest(unittest.TestCase):
    def test_default_refuses_to_publish_partial_container(self):
        td, src = make_checkpoint(inkling_config())
        self.addCleanup(td.cleanup)
        out_td = tempfile.TemporaryDirectory()
        self.addCleanup(out_td.cleanup)
        out = Path(out_td.name) / "model.waste"
        rc = convert_inkling.main(["--src", str(src), "--out", str(out)])
        self.assertEqual(rc, 2)
        self.assertFalse(out.exists())

    def test_plan_only_writes_only_plan(self):
        td, src = make_checkpoint(inkling_config())
        self.addCleanup(td.cleanup)
        out_td = tempfile.TemporaryDirectory()
        self.addCleanup(out_td.cleanup)
        out = Path(out_td.name) / "model.waste"
        rc = convert_inkling.main(["--src", str(src), "--out", str(out), "--plan-only"])
        self.assertEqual(rc, 0)
        self.assertEqual(sorted(p.name for p in out.iterdir()), ["conversion-plan.json"])
        plan = json.loads((out / "conversion-plan.json").read_text())
        self.assertFalse(plan["status"]["container_written"])
        self.assertNotIn("manifest.json", {p.name for p in out.iterdir()})

    def test_index_only_requires_explicit_opt_in(self):
        td, src = make_checkpoint(inkling_config(), with_payload=False)
        self.addCleanup(td.cleanup)
        out_td = tempfile.TemporaryDirectory()
        self.addCleanup(out_td.cleanup)
        out = Path(out_td.name) / "model.waste"
        self.assertEqual(
            convert_inkling.main(["--src", str(src), "--out", str(out), "--plan-only"]),
            2,
        )
        self.assertFalse(out.exists())
        self.assertEqual(
            convert_inkling.main(
                ["--src", str(src), "--out", str(out), "--plan-only", "--allow-index-only"]
            ),
            0,
        )
        plan = json.loads((out / "conversion-plan.json").read_text())
        self.assertFalse(plan["status"]["payload_headers_complete"])


class InklingStageCliTest(unittest.TestCase):
    def test_stage_experts_writes_stage_not_manifest(self):
        from test_inkling_weights import make_weight_checkpoint

        td, root, _ = make_weight_checkpoint()
        self.addCleanup(td.cleanup)
        out_td = tempfile.TemporaryDirectory()
        self.addCleanup(out_td.cleanup)
        out = Path(out_td.name)
        plan = {
            "source_dialect": "provider_raw",
            "model": {"sparse_mlp_layers": [1], "top_k": 2, "shared_experts": 2},
        }
        with patch("inkling_stage.build_plan", return_value=plan):
            rc = convert_inkling.main(
                ["--src", str(root), "--out", str(out), "--stage-experts", "--layers", "1", "--experts", "2"]
            )
        self.assertEqual(rc, 0)
        self.assertTrue((out / "stage.json").exists())
        self.assertFalse((out / "manifest.json").exists())


    def test_quantize_experts_writes_final_layer_artifacts_not_manifest(self):
        from test_inkling_weights import make_weight_checkpoint

        td, root, _ = make_weight_checkpoint()
        self.addCleanup(td.cleanup)
        out_td = tempfile.TemporaryDirectory()
        self.addCleanup(out_td.cleanup)
        out = Path(out_td.name)
        plan = {"source_dialect": "provider_raw", "official_small": False}
        with patch("inkling_vq.build_plan", return_value=plan):
            rc = convert_inkling.main([
                "--src", str(root), "--out", str(out), "--quantize-experts",
                "--layers", "1", "--experts", "2", "--vq-stages", "2",
                "--vq-entries", "16", "--codebook-sample", "2",
                "--train-vectors", "32", "--kmeans-iters", "2",
                "--assign-chunk", "16",
            ])
        self.assertEqual(rc, 0)
        self.assertTrue((out / "experts-L1.bin").exists())
        self.assertTrue((out / "codebooks-L1.bin").exists())
        self.assertTrue((out / "vq-stage.json").exists())
        self.assertFalse((out / "manifest.json").exists())

    def test_publish_private_runtime_stage_never_writes_manifest(self):
        from inkling_stage import stage_expert_banks
        from inkling_trunk import stage_trunk
        from test_inkling_trunk import make_full_checkpoint

        td, root, _ = make_full_checkpoint()
        self.addCleanup(td.cleanup)
        out_td = tempfile.TemporaryDirectory()
        self.addCleanup(out_td.cleanup)
        out = Path(out_td.name)
        stage_trunk(root, out, chunk_bytes=256)
        stage_expert_banks(root, out)
        rc = convert_inkling.main([
            "--src", str(root), "--out", str(out),
            "--publish-runtime-stage", "--allow-generic-inkling",
        ])
        self.assertEqual(rc, 0)
        self.assertTrue((out / "runtime-stage.bin").exists())
        self.assertTrue((out / "runtime-stage.json").exists())
        self.assertFalse((out / "manifest.json").exists())

    def test_publish_final_vq_runtime_stage_never_requires_bf16_expert_stage(self):
        from inkling_trunk import stage_trunk
        from inkling_vq import VQSpec, quantize_expert_banks
        from test_inkling_trunk import make_full_checkpoint

        td, root, _ = make_full_checkpoint()
        self.addCleanup(td.cleanup)
        out_td = tempfile.TemporaryDirectory()
        self.addCleanup(out_td.cleanup)
        out = Path(out_td.name)
        stage_trunk(root, out, chunk_bytes=256)
        quantize_expert_banks(
            root, out, spec=VQSpec(stages=2, vec_dim=4, entries=4, index_block=4),
            codebook_sample=2, train_vectors=32, kmeans_iterations=1,
            assign_chunk=64, verify=True,
        )
        rc = convert_inkling.main([
            "--src", str(root), "--out", str(out),
            "--publish-runtime-vq-stage", "--allow-generic-inkling",
        ])
        self.assertEqual(rc, 0)
        meta = json.loads((out / "runtime-stage.json").read_text())
        self.assertEqual(meta["expert_format"], "WEXP/VQ")
        self.assertFalse(meta["status"]["bf16_expert_stage_required"])
        self.assertFalse((out / "stage.json").exists())
        self.assertFalse((out / "manifest.json").exists())


    def test_publish_quantized_trunk_v3_stage(self):
        from inkling_trunk import stage_trunk
        from inkling_qtrunk import quantize_trunk
        from inkling_vq import VQSpec, quantize_expert_banks
        from test_inkling_trunk import make_full_checkpoint

        td, root, _ = make_full_checkpoint()
        self.addCleanup(td.cleanup)
        out_td = tempfile.TemporaryDirectory()
        self.addCleanup(out_td.cleanup)
        out = Path(out_td.name)
        stage_trunk(root, out, chunk_bytes=256)
        quantize_trunk(out, bits=4, group=4, chunk_rows=2)
        quantize_expert_banks(
            root, out, spec=VQSpec(stages=2, vec_dim=4, entries=4, index_block=4),
            codebook_sample=2, train_vectors=32, kmeans_iterations=1,
            assign_chunk=64, verify=True,
        )
        rc = convert_inkling.main([
            "--src", str(root), "--out", str(out),
            "--publish-runtime-qtrunk-stage", "--allow-generic-inkling",
        ])
        self.assertEqual(rc, 0)
        meta = json.loads((out / "runtime-stage.json").read_text())
        self.assertEqual(meta["schema"], "waste.inkling-private-runtime-stage.v3")
        self.assertTrue(meta["status"]["quantized_trunk"])
        self.assertFalse((out / "manifest.json").exists())

    def test_quantize_trunk_writes_private_artifacts_not_manifest(self):
        from inkling_trunk import stage_trunk
        from test_inkling_trunk import make_full_checkpoint

        td, root, _ = make_full_checkpoint()
        self.addCleanup(td.cleanup)
        out_td = tempfile.TemporaryDirectory()
        self.addCleanup(out_td.cleanup)
        out = Path(out_td.name)
        stage_trunk(root, out, chunk_bytes=256)
        rc = convert_inkling.main([
            "--src", str(root), "--out", str(out), "--quantize-trunk",
            "--trunk-bits", "4", "--trunk-group", "8",
            "--trunk-chunk-rows", "2",
        ])
        self.assertEqual(rc, 0)
        self.assertTrue((out / "qtrunk-stage.json").exists())
        self.assertTrue((out / "qtrunk-stage").is_dir())
        self.assertFalse((out / "manifest.json").exists())

    def test_stage_trunk_writes_internal_artifacts_only(self):
        from test_inkling_trunk import make_full_checkpoint

        td, root, _ = make_full_checkpoint()
        self.addCleanup(td.cleanup)
        out_td = tempfile.TemporaryDirectory()
        self.addCleanup(out_td.cleanup)
        out = Path(out_td.name)
        rc = convert_inkling.main(
            [
                "--src", str(root), "--out", str(out), "--stage-trunk",
                "--chunk-mib", "1",
            ]
        )
        self.assertEqual(rc, 0)
        self.assertTrue((out / "trunk-stage.json").exists())
        self.assertTrue((out / "trunk-stage").is_dir())
        self.assertFalse((out / "trunk.bin").exists())
        self.assertFalse((out / "manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
