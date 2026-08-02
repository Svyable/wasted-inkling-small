#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.

import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "tests"))

from inkling_parity import (
    ParityError,
    compare_activation_archives,
    extract_parity_fixture,
    parse_expert_selection,
    read_activation_archive,
    write_activation_archive,
)
from test_inkling_weights import write_safetensors


class InklingParityTest(unittest.TestCase):
    def test_activation_archive_roundtrip_and_corruption(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            values = {
                "layer.0.hidden": torch.arange(12, dtype=torch.float32).reshape(3, 4),
                "layer.0.route": torch.tensor([[2, 1], [0, 3]], dtype=torch.int64),
            }
            write_activation_archive(root, values, metadata={"prompt": "fixture"})
            got, meta = read_activation_archive(root)
            self.assertEqual(meta["prompt"], "fixture")
            for name in values:
                self.assertTrue(torch.equal(got[name], values[name].to(got[name].dtype)))
            manifest = json.loads((root / "activations.json").read_text())
            path = root / manifest["entries"][0]["path"]
            data = bytearray(path.read_bytes()); data[-1] ^= 0x55; path.write_bytes(data)
            with self.assertRaisesRegex(ParityError, "corruption"):
                read_activation_archive(root)

    def test_activation_comparison_reports_float_and_integer_mismatch(self):
        with tempfile.TemporaryDirectory() as rd, tempfile.TemporaryDirectory() as cd:
            ref = {"x": torch.tensor([1.0, 2.0]), "route": torch.tensor([1, 2])}
            got = {"x": torch.tensor([1.0, 2.01]), "route": torch.tensor([1, 3])}
            write_activation_archive(rd, ref)
            write_activation_archive(cd, got)
            report = compare_activation_archives(rd, cd, atol=1e-4, rtol=1e-4)
            self.assertFalse(report["passed"])
            by_name = {item["name"]: item for item in report["results"]}
            self.assertGreater(by_name["x"]["max_abs"], 0.009)
            self.assertEqual(by_name["route"]["mismatches"], 1)

    def _checkpoint(self, root: Path):
        tensors = {
            "model.llm.embed_norm.weight": torch.arange(4, dtype=torch.float32).to(torch.bfloat16),
            "model.llm.norm.weight": torch.arange(4, dtype=torch.float32).to(torch.bfloat16),
            "model.llm.layers.0.attn_norm.weight": torch.arange(4, dtype=torch.float32).to(torch.bfloat16),
            "model.llm.layers.0.attn.wq_du.weight": torch.arange(16, dtype=torch.float32).reshape(4,4).to(torch.bfloat16),
            "model.llm.layers.1.attn_norm.weight": torch.arange(4, dtype=torch.float32).to(torch.bfloat16),
            "model.llm.layers.1.mlp.experts.w13_weight": torch.arange(4*8*4, dtype=torch.float32).reshape(4,8,4).to(torch.bfloat16),
            "model.llm.layers.1.mlp.experts.w2_weight": torch.arange(4*4*4, dtype=torch.float32).reshape(4,4,4).to(torch.bfloat16),
            "model.llm.embed.weight": torch.zeros(32,4,dtype=torch.bfloat16),
            "model.llm.unembed.weight": torch.zeros(32,4,dtype=torch.bfloat16),
        }
        shard = "model-00001-of-00001.safetensors"
        write_safetensors(root / shard, tensors)
        (root / "model.safetensors.index.json").write_text(json.dumps({
            "metadata": {"total_size": sum(t.numel()*t.element_size() for t in tensors.values())},
            "weight_map": {name: shard for name in tensors},
        }))
        (root / "config.json").write_text(json.dumps({"text_config": {"num_hidden_layers": 2}}))
        return tensors

    def test_bounded_fixture_copies_selected_layers_and_experts(self):
        with tempfile.TemporaryDirectory() as td, tempfile.TemporaryDirectory() as od:
            root, out = Path(td), Path(od)
            tensors = self._checkpoint(root)
            manifest = extract_parity_fixture(root, out, layers=[0,1], experts={1:[2]},
                                              require_official=False, max_total_bytes=1<<20)
            names = {(e["name"], e.get("axis0")) for e in manifest["entries"]}
            self.assertIn(("model.llm.layers.0.attn.wq_du.weight", None), names)
            self.assertIn(("model.llm.layers.1.mlp.experts.w13_weight", 2), names)
            self.assertNotIn(("model.llm.embed.weight", None), names)
            full = sum(t.numel()*t.element_size() for t in tensors.values())
            self.assertLess(manifest["reader_payload_bytes"], full)
            self.assertTrue((out / "fixture.json").exists())

    def test_fixture_limits_and_selection_fail_closed(self):
        self.assertEqual(parse_expert_selection("2:1,3;4:0"), {2:[1,3], 4:[0]})
        with self.assertRaises(ParityError):
            parse_expert_selection("bad")
        with tempfile.TemporaryDirectory() as td, tempfile.TemporaryDirectory() as od:
            root, out = Path(td), Path(od)
            self._checkpoint(root)
            with self.assertRaisesRegex(ParityError, "above per-tensor limit"):
                extract_parity_fixture(root, out, layers=[0], require_official=False,
                                       max_tensor_bytes=8, max_total_bytes=1<<20)
            with self.assertRaisesRegex(ParityError, "unselected layer"):
                extract_parity_fixture(root, out, layers=[0], experts={1:[0]},
                                       require_official=False, max_total_bytes=1<<20)


if __name__ == "__main__":
    unittest.main()
