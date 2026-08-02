#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.

import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "tests"))

from inkling_weights import (
    ProviderRawInklingWeights,
    SafeTensorReader,
    WeightError,
    interleave,
    split_fused_gate_up,
)
from test_inspect_inkling import inkling_config


DTYPE_NAME = {
    torch.bfloat16: "BF16",
    torch.float16: "F16",
    torch.float32: "F32",
    torch.int64: "I64",
}


def tensor_bytes(tensor: torch.Tensor) -> bytes:
    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy()
    return raw.tobytes()


def write_safetensors(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    payloads = []
    header = {}
    offset = 0
    for name, tensor in tensors.items():
        data = tensor_bytes(tensor)
        header[name] = {
            "dtype": DTYPE_NAME[tensor.dtype],
            "shape": list(tensor.shape),
            "data_offsets": [offset, offset + len(data)],
        }
        payloads.append(data)
        offset += len(data)
    raw_header = json.dumps(header, separators=(",", ":")).encode("utf-8")
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(raw_header)))
        f.write(raw_header)
        for data in payloads:
            f.write(data)


def source_from_gate_up(gate: torch.Tensor, up: torch.Tensor, dim: int) -> torch.Tensor:
    target = torch.cat([gate, up], dim=dim)
    return interleave(target, dim, inverse=True)


def make_weight_checkpoint():
    td = tempfile.TemporaryDirectory()
    root = Path(td.name)
    cfg = inkling_config(layers=3)
    (root / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    text = cfg["text_config"]
    h = text["hidden_size"]
    dense = text["dense_intermediate_size"]
    inter = text["intermediate_size"]
    experts = text["n_routed_experts"]
    shared = text["n_shared_experts"]

    dense_gate = torch.arange(dense * h, dtype=torch.float32).reshape(dense, h).to(torch.bfloat16)
    dense_up = (dense_gate.float() + 1000).to(torch.bfloat16)
    dense_down = torch.arange(h * dense, dtype=torch.float32).reshape(h, dense).to(torch.bfloat16)

    routed_gate = torch.arange(experts * inter * h, dtype=torch.float32).reshape(experts, inter, h).to(torch.bfloat16)
    routed_up = (routed_gate.float() + 2000).to(torch.bfloat16)
    routed_down = torch.arange(experts * h * inter, dtype=torch.float32).reshape(experts, h, inter).to(torch.bfloat16)

    shared_gate = torch.arange(shared * inter * h, dtype=torch.float32).reshape(shared, inter, h).to(torch.bfloat16)
    shared_up = (shared_gate.float() + 3000).to(torch.bfloat16)
    shared_down = torch.arange(shared * h * inter, dtype=torch.float32).reshape(shared, h, inter).to(torch.bfloat16)

    tensors = {
        "model.llm.layers.0.mlp.w13_dn.weight": source_from_gate_up(dense_gate, dense_up, 0),
        "model.llm.layers.0.mlp.w2_md.weight": dense_down,
        "model.llm.layers.1.mlp.experts.w13_weight": source_from_gate_up(routed_gate, routed_up, 1),
        "model.llm.layers.1.mlp.experts.w2_weight": routed_down,
        "model.llm.layers.1.mlp.shared_experts.shared_w13_weight": source_from_gate_up(shared_gate, shared_up, 1),
        "model.llm.layers.1.mlp.shared_experts.shared_w2_weight": shared_down,
    }
    shard = "model-00001-of-00001.safetensors"
    write_safetensors(root / shard, tensors)
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": sum(t.numel() * t.element_size() for t in tensors.values())},
                    "weight_map": {name: shard for name in tensors}}),
        encoding="utf-8",
    )
    expected = {
        "dense_gate": dense_gate.float(),
        "dense_up": dense_up.float(),
        "dense_down": dense_down.float(),
        "routed_gate": routed_gate.float(),
        "routed_up": routed_up.float(),
        "routed_down": routed_down.float(),
        "shared_gate": shared_gate.float(),
        "shared_up": shared_up.float(),
        "shared_down": shared_down.float(),
    }
    return td, root, expected


class InklingWeightsTest(unittest.TestCase):
    def test_interleave_matches_inverse_round_trip(self):
        x = torch.arange(2 * 6 * 4).reshape(2, 6, 4)
        y = interleave(x, 1)
        self.assertTrue(torch.equal(interleave(y, 1, inverse=True), x))
        gate, up = split_fused_gate_up(interleave(torch.cat([x[:, :3], x[:, 3:]], 1), 1, inverse=True), 1)
        self.assertTrue(torch.equal(gate, x[:, :3]))
        self.assertTrue(torch.equal(up, x[:, 3:]))

    def test_axis0_slice_reads_only_one_expert(self):
        td, root, expected = make_weight_checkpoint()
        self.addCleanup(td.cleanup)
        reader = SafeTensorReader(root)
        name = "model.llm.layers.1.mlp.experts.w13_weight"
        loc = reader.location(name)
        before = reader.bytes_read
        got = reader.slice0(name, 2)
        read = reader.bytes_read - before
        self.assertEqual(read, loc.nbytes // 4)
        self.assertEqual(list(got.shape), [16, 16])

    def test_provider_adapter_splits_dense_routed_and_shared(self):
        td, root, expected = make_weight_checkpoint()
        self.addCleanup(td.cleanup)
        source = ProviderRawInklingWeights(root)

        dense = source.dense(0)
        self.assertTrue(torch.equal(dense.gate, expected["dense_gate"]))
        self.assertTrue(torch.equal(dense.up, expected["dense_up"]))
        self.assertTrue(torch.equal(dense.down, expected["dense_down"]))

        expert = source.routed_expert(1, 2)
        self.assertTrue(torch.equal(expert.gate, expected["routed_gate"][2]))
        self.assertTrue(torch.equal(expert.up, expected["routed_up"][2]))
        self.assertTrue(torch.equal(expert.down, expected["routed_down"][2]))

        shared = source.shared(1)
        self.assertTrue(torch.equal(shared.gate, expected["shared_gate"]))
        self.assertTrue(torch.equal(shared.up, expected["shared_up"]))
        self.assertTrue(torch.equal(shared.down, expected["shared_down"]))

    def test_shape_and_dtype_fail_closed(self):
        td, root, _ = make_weight_checkpoint()
        self.addCleanup(td.cleanup)
        source = ProviderRawInklingWeights(root)
        with self.assertRaisesRegex(WeightError, "not a dense"):
            source.dense(1)
        with self.assertRaisesRegex(WeightError, "outside"):
            source.routed_expert(1, 99)

        # Add an unsupported tensor to a separate valid safetensors file.
        bad = root / "bad.safetensors"
        write_safetensors(bad, {"bad": torch.arange(4, dtype=torch.int64)})
        index = json.loads((root / "model.safetensors.index.json").read_text())
        index["weight_map"]["bad"] = bad.name
        (root / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
        reader = SafeTensorReader(root)
        with self.assertRaisesRegex(WeightError, "unsupported source dtype"):
            reader.tensor("bad")


if __name__ == "__main__":
    unittest.main()
