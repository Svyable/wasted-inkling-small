#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.

import json
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import zlib
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "tests"))

from inkling_stage import HEADER, StageError, parse_header, stage_expert_banks, verify_bank, write_bank
from inkling_weights import ProviderRawInklingWeights
from test_inkling_weights import make_weight_checkpoint


def minimal_plan():
    return {
        "source_dialect": "provider_raw",
        "model": {
            "sparse_mlp_layers": [1],
            "top_k": 2,
            "shared_experts": 2,
        },
    }


class InklingStageTest(unittest.TestCase):
    def test_aligned_bank_round_trip_and_commit_order(self):
        td, root, expected = make_weight_checkpoint()
        self.addCleanup(td.cleanup)
        out_td = tempfile.TemporaryDirectory()
        self.addCleanup(out_td.cleanup)
        out = Path(out_td.name)

        with patch("inkling_stage.build_plan", return_value=minimal_plan()):
            stage = stage_expert_banks(root, out, layers=[1])
        self.assertFalse((out / "manifest.json").exists())
        self.assertFalse((out / "trunk.bin").exists())
        self.assertTrue((out / "stage.json").exists())
        meta = stage["banks"][0]
        bank = out / meta["file"]
        self.assertEqual(bank.stat().st_size, meta["record_bytes"] * 4)
        self.assertEqual(meta["record_bytes"] % 4096, 0)
        verify_bank(bank, meta)

        inter = 8
        hidden = 16
        with bank.open("rb") as f:
            for expert in range(4):
                rec = f.read(meta["record_bytes"])
                header = parse_header(rec)
                self.assertEqual((header.layer, header.expert), (1, expert))
                payload = rec[HEADER.size : HEADER.size + header.payload_bytes]
                self.assertEqual(zlib.crc32(payload) & 0xFFFFFFFF, header.crc32)
                gate_n = inter * hidden * 2
                down_n = hidden * inter * 2
                gate = torch.frombuffer(bytearray(payload[:gate_n]), dtype=torch.bfloat16).reshape(inter, hidden)
                up = torch.frombuffer(bytearray(payload[gate_n : 2 * gate_n]), dtype=torch.bfloat16).reshape(inter, hidden)
                down = torch.frombuffer(bytearray(payload[2 * gate_n : 2 * gate_n + down_n]), dtype=torch.bfloat16).reshape(hidden, inter)
                self.assertTrue(torch.equal(gate.float(), expected["routed_gate"][expert]))
                self.assertTrue(torch.equal(up.float(), expected["routed_up"][expert]))
                self.assertTrue(torch.equal(down.float(), expected["routed_down"][expert]))

        published = json.loads((out / "stage.json").read_text())
        self.assertEqual(published["schema"], "waste.inkling-bf16-expert-stage.v1")
        self.assertFalse(published["status"]["waste_manifest_written"])

    def test_writer_removes_temporary_bank_on_failure(self):
        td, root, _ = make_weight_checkpoint()
        self.addCleanup(td.cleanup)
        source = ProviderRawInklingWeights(root)
        original = source.routed_expert

        def fail(layer, expert, *, float32=True):
            if expert == 2:
                raise RuntimeError("injected failure")
            return original(layer, expert, float32=float32)

        source.routed_expert = fail  # type: ignore[method-assign]
        out_td = tempfile.TemporaryDirectory()
        self.addCleanup(out_td.cleanup)
        bank = Path(out_td.name) / "experts-L1.bf16.stage"
        with self.assertRaisesRegex(RuntimeError, "injected failure"):
            write_bank(source, bank, 1)
        self.assertFalse(bank.exists())
        self.assertFalse(bank.with_name(bank.name + ".tmp").exists())

    def test_rejects_dense_layer_and_non_bf16_source(self):
        td, root, _ = make_weight_checkpoint()
        self.addCleanup(td.cleanup)
        source = ProviderRawInklingWeights(root)
        out_td = tempfile.TemporaryDirectory()
        self.addCleanup(out_td.cleanup)
        with self.assertRaisesRegex(StageError, "not a sparse"):
            write_bank(source, Path(out_td.name) / "bad", 0)

        original_location = source.reader.location
        source.reader.location = lambda name: SimpleNamespace(dtype="F16")  # type: ignore[method-assign]
        try:
            with self.assertRaisesRegex(StageError, "requires BF16"):
                write_bank(source, Path(out_td.name) / "f16", 1, expert_limit=1)
        finally:
            source.reader.location = original_location  # type: ignore[method-assign]

    def test_crc_verification_detects_corruption(self):
        td, root, _ = make_weight_checkpoint()
        self.addCleanup(td.cleanup)
        source = ProviderRawInklingWeights(root)
        out_td = tempfile.TemporaryDirectory()
        self.addCleanup(out_td.cleanup)
        bank = Path(out_td.name) / "experts-L1.bf16.stage"
        meta = write_bank(source, bank, 1, expert_limit=1)
        with bank.open("r+b") as f:
            f.seek(HEADER.size + 7)
            old = f.read(1)
            f.seek(HEADER.size + 7)
            f.write(bytes([old[0] ^ 1]))
        with self.assertRaisesRegex(StageError, "CRC mismatch"):
            verify_bank(bank, meta)

    def test_matching_bank_sidecar_resumes_without_rewrite(self):
        td, root, _ = make_weight_checkpoint()
        self.addCleanup(td.cleanup)
        out_td = tempfile.TemporaryDirectory()
        self.addCleanup(out_td.cleanup)
        out = Path(out_td.name)
        with patch("inkling_stage.build_plan", return_value=minimal_plan()):
            first = stage_expert_banks(root, out, layers=[1], expert_limit=2)
        self.assertFalse(first["banks"][0]["reused"])
        self.assertTrue((out / "experts-L1.bf16.stage.json").exists())
        with patch("inkling_stage.build_plan", return_value=minimal_plan()), \
             patch("inkling_stage.write_bank", side_effect=AssertionError("bank rewritten")):
            second = stage_expert_banks(root, out, layers=[1], expert_limit=2)
        self.assertTrue(second["banks"][0]["reused"])

    def test_expert_limit_marks_stage_incomplete(self):
        td, root, _ = make_weight_checkpoint()
        self.addCleanup(td.cleanup)
        out_td = tempfile.TemporaryDirectory()
        self.addCleanup(out_td.cleanup)
        with patch("inkling_stage.build_plan", return_value=minimal_plan()):
            stage = stage_expert_banks(root, Path(out_td.name), layers=[1], expert_limit=2)
        self.assertEqual(stage["banks"][0]["experts"], 2)
        self.assertFalse(stage["banks"][0]["complete"])
        self.assertFalse(stage["status"]["stage_complete_for_requested_layers"])


if __name__ == "__main__":
    unittest.main()
