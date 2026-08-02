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

from inkling_plan import build_plan
from inkling_trunk import HEADER, TrunkStageError, stage_trunk, verify_tensor
from inkling_weights import SafeTensorReader
from test_inkling_weights import source_from_gate_up, write_safetensors
from test_inspect_inkling import (
    inkling_config,
    official_tokenizer_config,
    processor_config,
    raw_tensors,
)


def _values(shape, dtype, seed):
    n = 1
    for dim in shape:
        n *= dim
    value = (torch.arange(n, dtype=torch.float32) + seed).reshape(shape)
    return value.to(dtype)


def make_full_checkpoint():
    td = tempfile.TemporaryDirectory()
    root = Path(td.name)
    cfg = inkling_config(layers=3)
    (root / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    (root / "tiktoken").mkdir()
    (root / "tiktoken" / "tokenizer.model").write_bytes(b"tokenizer")
    (root / "chat_template.jinja").write_text("{{ messages }}", encoding="utf-8")
    (root / "tokenizer_config.json").write_text(
        json.dumps(official_tokenizer_config()), encoding="utf-8"
    )
    (root / "processor_config.json").write_text(
        json.dumps(processor_config()), encoding="utf-8"
    )

    specs = raw_tensors(cfg)
    tensors = {}
    dtype_map = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}
    seed = 0
    for name, (dtype_name, shape) in specs.items():
        tensors[name] = _values(shape, dtype_map[dtype_name], seed)
        seed += 17

    text = cfg["text_config"]
    h = text["hidden_size"]
    dense = text["dense_intermediate_size"]
    inter = text["intermediate_size"]
    shared = text["n_shared_experts"]
    expected = {}

    gate = _values([dense, h], torch.bfloat16, 1000)
    up = _values([dense, h], torch.bfloat16, 2000)
    tensors["model.llm.layers.0.mlp.w13_dn.weight"] = source_from_gate_up(gate, up, 0)
    expected["inkling.layer.0.mlp.gate"] = gate
    expected["inkling.layer.0.mlp.up"] = up

    for layer in (1, 2):
        gate = _values([shared, inter, h], torch.bfloat16, 3000 + layer * 100)
        up = _values([shared, inter, h], torch.bfloat16, 4000 + layer * 100)
        tensors[f"model.llm.layers.{layer}.mlp.shared_experts.shared_w13_weight"] = source_from_gate_up(
            gate, up, 1
        )
        expected[f"inkling.layer.{layer}.shared.gate"] = gate
        expected[f"inkling.layer.{layer}.shared.up"] = up

    shard = "model-00001-of-00001.safetensors"
    write_safetensors(root / shard, tensors)
    (root / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {
                    "total_size": sum(t.numel() * t.element_size() for t in tensors.values())
                },
                "weight_map": {name: shard for name in tensors},
            }
        ),
        encoding="utf-8",
    )
    return td, root, expected


def read_stage_tensor(out, meta):
    path = out / "trunk-stage" / meta["file"]
    raw = path.read_bytes()
    payload = raw[HEADER.size : HEADER.size + meta["payload_bytes"]]
    dtype = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}[meta["dtype"]]
    return torch.frombuffer(bytearray(payload), dtype=dtype).reshape(meta["shape"])


class InklingTrunkStageTest(unittest.TestCase):
    def test_complete_stage_is_bounded_and_splits_fused_tensors(self):
        td, root, expected = make_full_checkpoint()
        self.addCleanup(td.cleanup)
        out_td = tempfile.TemporaryDirectory()
        self.addCleanup(out_td.cleanup)
        out = Path(out_td.name)

        plan = build_plan(root, require_payloads=True)
        reader = SafeTensorReader(root)
        expected_source_bytes = sum(
            reader.location(name).nbytes
            for name in {item["source"] for item in plan["trunk"]}
        )
        stage = stage_trunk(root, out, chunk_bytes=64)
        self.assertFalse((out / "manifest.json").exists())
        self.assertFalse((out / "trunk.bin").exists())
        self.assertTrue((out / "trunk-stage.json").exists())
        self.assertEqual(stage["totals"]["tensors"], 62)
        self.assertEqual(stage["totals"]["source_bytes_read"], expected_source_bytes)
        self.assertTrue(all(not item["reused"] for item in stage["tensors"]))

        by_target = {item["target"]: item for item in stage["tensors"]}
        for target, want in expected.items():
            got = read_stage_tensor(out, by_target[target])
            self.assertTrue(torch.equal(got, want), target)

    def test_matching_sidecars_resume_without_source_payload_reads(self):
        td, root, _ = make_full_checkpoint()
        self.addCleanup(td.cleanup)
        out_td = tempfile.TemporaryDirectory()
        self.addCleanup(out_td.cleanup)
        out = Path(out_td.name)
        stage_trunk(root, out, chunk_bytes=128)
        second = stage_trunk(root, out, chunk_bytes=128)
        self.assertTrue(all(item["reused"] for item in second["tensors"]))
        self.assertEqual(second["totals"]["source_bytes_read"], 0)

    def test_crc_verification_detects_corruption(self):
        td, root, _ = make_full_checkpoint()
        self.addCleanup(td.cleanup)
        out_td = tempfile.TemporaryDirectory()
        self.addCleanup(out_td.cleanup)
        out = Path(out_td.name)
        stage = stage_trunk(root, out)
        meta = stage["tensors"][0]
        path = out / "trunk-stage" / meta["file"]
        with path.open("r+b") as f:
            f.seek(HEADER.size + 3)
            old = f.read(1)
            f.seek(HEADER.size + 3)
            f.write(bytes([old[0] ^ 1]))
        with self.assertRaisesRegex(TrunkStageError, "CRC mismatch"):
            verify_tensor(path, meta)

    def test_raw_copy_iterator_honors_chunk_bound(self):
        td, root, _ = make_full_checkpoint()
        self.addCleanup(td.cleanup)
        reader = SafeTensorReader(root)
        chunks = list(reader.iter_bytes("model.llm.embed.weight", chunk_bytes=31))
        self.assertTrue(chunks)
        self.assertLessEqual(max(map(len, chunks)), 31)
        self.assertEqual(sum(map(len, chunks)), reader.location("model.llm.embed.weight").nbytes)


if __name__ == "__main__":
    unittest.main()
