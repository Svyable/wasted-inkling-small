#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Tests for the fixture-backed C layer harness.

The C decoder layer is already validated against PyTorch by
test_inkling_layer_c.py. What is new here is everything *around* it: reading a
bounded fixture, resolving source tensor names through inkling_plan, splitting
the provider's row-interleaved gate/up, placing routed expert slices at their
own indices, and keeping every ctypes buffer alive.

So the central test is an equivalence: the same weights, run through the
already-validated direct path and through the fixture path, must produce
identical output. Any defect in the new code moves those numbers.
"""

import ctypes
import json
import sys
import tempfile
import unittest
import zlib
from array import array
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "tests"))

from inkling_fixture import load_fixture
from inkling_layer_parity import (
    AttentionState, Config as ParityConfig, LayerBinding, LayerParityError,
    LayerScratch, LayerState, LayerWeights, bind_layer, build_library,
    configure_library, run_layer_trace, split_fused_gate_up,
)
from inkling_plan import layer_attention_names, layer_sparse_mlp_names

FP = ctypes.POINTER(ctypes.c_float)

CONFIG = {
    "n_layers": 2, "hidden": 8, "vocab": 32, "unpadded_vocab": 30,
    "max_context": 16,
    "global_heads": 2, "global_kv_heads": 1, "global_head_dim": 4,
    "local_heads": 2, "local_kv_heads": 1, "local_head_dim": 4,
    "sliding_window": 3, "d_rel": 2, "rel_extent": 5, "conv_kernel": 3,
    "dense_layers": 1, "dense_intermediate": 12, "moe_intermediate": 6,
    "n_routed_experts": 3, "top_k": 2, "n_shared_experts": 1,
    "rms_eps": 1e-6, "route_scale": 2.0, "logits_width_multiplier": 2.0,
    "log_scaling_n_floor": 2, "log_scaling_alpha": 0.2,
    "local_layer_ids": [0],
}


def fuse_gate_up(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Inverse of split_fused_gate_up: interleave rows as the provider ships."""
    inter, hidden = gate.shape
    out = torch.empty(2 * inter, hidden, dtype=gate.dtype)
    out[0::2] = gate
    out[1::2] = up
    return out


def make_tensors(layer: int, seed: int = 7) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed + layer)
    c = CONFIG
    h = c["hidden"]
    is_local = layer in c["local_layer_ids"]
    heads = c["local_heads"] if is_local else c["global_heads"]
    kv = c["local_kv_heads"] if is_local else c["global_kv_heads"]
    hd = c["local_head_dim"] if is_local else c["global_head_dim"]
    extent = c["sliding_window"] if is_local else c["rel_extent"]
    k = c["conv_kernel"]
    t = {
        "input_norm": torch.rand(h) + 0.5,
        "post_attention_norm": torch.rand(h) + 0.5,
        "wq": torch.randn(heads * hd, h) * 0.15,
        "wk": torch.randn(kv * hd, h) * 0.15,
        "wv": torch.randn(kv * hd, h) * 0.15,
        "wr": torch.randn(heads * c["d_rel"], h) * 0.15,
        "wo": torch.randn(h, heads * hd) * 0.15,
        "q_norm": torch.rand(hd) + 0.5,
        "k_norm": torch.rand(hd) + 0.5,
        "relative_proj": torch.randn(c["d_rel"], extent) * 0.2,
        "k_sconv": torch.randn(kv * hd, k) * 0.2,
        "v_sconv": torch.randn(kv * hd, k) * 0.2,
        "attn_sconv": torch.randn(h, k) * 0.2,
        "mlp_sconv": torch.randn(h, k) * 0.2,
    }
    if layer < c["dense_layers"]:
        di = c["dense_intermediate"]
        t["dense_gate"] = torch.randn(di, h) * 0.15
        t["dense_up"] = torch.randn(di, h) * 0.15
        t["dense_down"] = torch.randn(h, di) * 0.15
        t["dense_global_scale"] = torch.tensor([1.25])
    else:
        mi, ne, ns = c["moe_intermediate"], c["n_routed_experts"], c["n_shared_experts"]
        t["router_weight"] = torch.randn(ne + ns, h) * 0.2
        t["router_bias"] = torch.randn(ne) * 0.1
        t["router_global_scale"] = torch.tensor([0.75])
        t["routed_gate"] = torch.randn(ne, mi, h) * 0.15
        t["routed_up"] = torch.randn(ne, mi, h) * 0.15
        t["routed_down"] = torch.randn(ne, h, mi) * 0.15
        t["shared_gate"] = torch.randn(ns, mi, h) * 0.15
        t["shared_up"] = torch.randn(ns, mi, h) * 0.15
        t["shared_down"] = torch.randn(ns, h, mi) * 0.15
    return t


def write_fixture(root: Path, layer: int, t: dict[str, torch.Tensor]) -> Path:
    """Write a fixture in provider_raw source naming, F32 so the comparison is
    about the harness rather than about BF16 rounding."""
    entries = []

    def add(name: str, tensor: torch.Tensor, axis0: int | None = None) -> None:
        payload = tensor.detach().float().contiguous().view(torch.uint8).numpy().tobytes()
        rel = f"e{len(entries)}.bin"
        (root / rel).write_bytes(payload)
        entry = {
            "name": name, "kind": "tensor" if axis0 is None else "axis0-slice",
            "dtype": "F32", "shape": list(tensor.shape), "bytes": len(payload),
            "crc32": zlib.crc32(payload) & 0xFFFFFFFF, "path": rel,
        }
        if axis0 is not None:
            entry["axis0"] = axis0
        entries.append(entry)

    names = layer_attention_names(layer, "provider_raw")
    for role, key in (("input_norm", "input_norm"),
                      ("post_attention_norm", "post_attention_norm"),
                      ("q", "wq"), ("k", "wk"), ("v", "wv"), ("r", "wr"),
                      ("o", "wo"), ("q_norm", "q_norm"), ("k_norm", "k_norm"),
                      ("rel_proj", "relative_proj"), ("k_sconv", "k_sconv"),
                      ("v_sconv", "v_sconv"), ("attn_sconv", "attn_sconv"),
                      ("mlp_sconv", "mlp_sconv")):
        add(names[role], t[key])

    experts: dict[str, list[int]] = {}
    if layer < CONFIG["dense_layers"]:
        from inkling_plan import layer_dense_mlp_names
        mlp = layer_dense_mlp_names(layer, "provider_raw")
        add(mlp["fused_gate_up"], fuse_gate_up(t["dense_gate"], t["dense_up"]))
        add(mlp["down"], t["dense_down"])
        add(mlp["global_scale"], t["dense_global_scale"])
    else:
        sparse = layer_sparse_mlp_names(layer, "provider_raw")
        add(sparse["router"]["weight"], t["router_weight"])
        add(sparse["router"]["correction_bias"], t["router_bias"])
        add(sparse["router"]["global_scale"], t["router_global_scale"])
        shared = torch.stack([
            fuse_gate_up(t["shared_gate"][s], t["shared_up"][s])
            for s in range(CONFIG["n_shared_experts"])
        ])
        add(sparse["shared"]["fused_gate_up"], shared)
        add(sparse["shared"]["down"], t["shared_down"])
        ids = list(range(CONFIG["n_routed_experts"]))
        for e in ids:
            add(sparse["routed"]["fused_gate_up"],
                fuse_gate_up(t["routed_gate"][e], t["routed_up"][e]), axis0=e)
            add(sparse["routed"]["down"], t["routed_down"][e], axis0=e)
        experts[str(layer)] = ids

    manifest = {
        "format": "inkling-parity-fixture", "version": 1,
        "model_id": "synthetic", "layers": [layer], "experts": experts,
        "source": {"config_sha256": "0" * 64, "index_sha256": "1" * 64},
        "total_payload_bytes": sum(e["bytes"] for e in entries),
        "entries": entries,
    }
    (root / "fixture.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return root


def direct_binding(t: dict[str, torch.Tensor], layer: int) -> LayerBinding:
    """Weights bound straight from the tensors, bypassing the fixture.

    Same struct and same executor as the fixture path; the only difference is
    where the numbers came from. That is precisely the variable under test.
    """
    binding = LayerBinding()
    flat = lambda name: t[name].detach().float().contiguous().view(-1).tolist()
    for field in ("input_norm", "post_attention_norm", "wq", "wk", "wv", "wr",
                  "wo", "q_norm", "k_norm", "relative_proj", "k_sconv",
                  "v_sconv", "attn_sconv", "mlp_sconv"):
        binding.set(field, flat(field))
    sparse = layer >= CONFIG["dense_layers"]
    binding.weights.sparse = int(sparse)
    fields = (("router_weight", "router_bias", "router_global_scale",
               "shared_gate", "shared_up", "shared_down",
               "routed_gate", "routed_up", "routed_down") if sparse else
              ("dense_gate", "dense_up", "dense_down", "dense_global_scale"))
    for field in fields:
        binding.set(field, flat(field))
    return binding


class LayerParityHarnessTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.td = tempfile.TemporaryDirectory()
        try:
            lib_path = build_library(REPO / "src", Path(cls.td.name) / "lib.so")
        except LayerParityError as exc:
            cls.td.cleanup()
            raise unittest.SkipTest(str(exc))
        cls.lib = ctypes.CDLL(str(lib_path))
        configure_library(cls.lib)

    @classmethod
    def tearDownClass(cls):
        cls.td.cleanup()

    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.root = Path(self.td.name)
        self.addCleanup(self.td.cleanup)

    def direct_run(self, layer, tensors, inputs):
        binding = direct_binding(tensors, layer)
        return run_layer_trace(self.lib, CONFIG, layer, binding, inputs)["outputs"]

    def fixture_run(self, layer, tensors, inputs):
        write_fixture(self.root, layer, tensors)
        fixture = load_fixture(self.root)
        binding = bind_layer(fixture, CONFIG, layer)
        return run_layer_trace(self.lib, CONFIG, layer, binding, inputs)

    def test_dense_layer_matches_the_direct_path(self):
        tensors = make_tensors(0)
        inputs = [torch.randn(CONFIG["hidden"]).tolist() for _ in range(3)]
        want = self.direct_run(0, tensors, inputs)
        got = self.fixture_run(0, tensors, inputs)["outputs"]
        self.assertEqual(got, want)

    def test_sparse_layer_matches_the_direct_path(self):
        tensors = make_tensors(1)
        inputs = [torch.randn(CONFIG["hidden"]).tolist() for _ in range(4)]
        want = self.direct_run(1, tensors, inputs)
        got = self.fixture_run(1, tensors, inputs)["outputs"]
        self.assertEqual(got, want)

    def test_trace_names_match_the_private_runtime_convention(self):
        tensors = make_tensors(1)
        inputs = [torch.randn(CONFIG["hidden"]).tolist() for _ in range(2)]
        traced = self.fixture_run(1, tensors, inputs)
        names = set(traced["values"])
        self.assertIn("token.0.layer.1.output", names)
        self.assertIn("token.1.layer.1.output", names)
        for name in names:
            self.assertTrue(name.startswith("token."), name)
            self.assertIn(".layer.1.", name)
        # routing indices are traced as integers, not floats
        routing = [n for n in names if traced["dtypes"][n] == "I32"]
        self.assertTrue(routing, "expected at least one integer trace record")

    def test_state_advances_across_positions(self):
        tensors = make_tensors(1)
        row = torch.randn(CONFIG["hidden"]).tolist()
        traced = self.fixture_run(1, tensors, [row, list(row)])
        self.assertNotEqual(traced["outputs"][0], traced["outputs"][1],
                            "identical inputs at different positions must differ: "
                            "the conv rings and KV cache carry state")


class StructLayoutTest(unittest.TestCase):
    """The ctypes declarations in inkling_layer_parity are a hand transcription
    of three C headers. A wrong one is not a test failure — it is a buffer
    overrun that produces correct numbers and crashes later, somewhere else.
    So ask the C compiler."""

    PROBE = """
    #include <stdio.h>
    #include "inkling_layer.h"
    int main(void) {
        printf("%zu %zu %zu %zu %zu\\n",
               sizeof(waste_inkling_attention_state),
               sizeof(waste_inkling_layer_state),
               sizeof(waste_inkling_layer_scratch),
               sizeof(waste_inkling_layer_weights),
               sizeof(waste_inkling_config));
        return 0;
    }
    """

    def test_struct_layout_matches_c(self):
        import shutil
        import subprocess

        cc = shutil.which("cc") or shutil.which("gcc")
        if not cc:
            self.skipTest("no C compiler")
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "probe.c"
            exe = Path(td) / "probe"
            src.write_text(self.PROBE)
            build = subprocess.run(
                [cc, "-std=c11", f"-I{REPO / 'src'}", str(src), "-o", str(exe)],
                capture_output=True, text=True)
            self.assertEqual(build.returncode, 0, build.stderr)
            out = subprocess.run([str(exe)], capture_output=True, text=True)
            self.assertEqual(out.returncode, 0, out.stderr)
        sizes = [int(x) for x in out.stdout.split()]
        for got, want, name in zip(
            (ctypes.sizeof(AttentionState), ctypes.sizeof(LayerState),
             ctypes.sizeof(LayerScratch), ctypes.sizeof(LayerWeights),
             ctypes.sizeof(ParityConfig)),
            sizes,
            ("attention_state", "layer_state", "layer_scratch",
             "layer_weights", "config"),
        ):
            self.assertEqual(got, want, f"{name}: ctypes {got} bytes, C {want}")


class FusedSplitTest(unittest.TestCase):
    def test_split_is_the_inverse_of_the_provider_interleave(self):
        gate = torch.randn(5, 3)
        up = torch.randn(5, 3)
        fused = fuse_gate_up(gate, up)
        values = array("f", fused.contiguous().view(-1).tolist())
        got_gate, got_up = split_fused_gate_up(values, 5, 3)
        self.assertEqual(list(got_gate), gate.contiguous().view(-1).tolist())
        self.assertEqual(list(got_up), up.contiguous().view(-1).tolist())

    def test_wrong_length_is_refused(self):
        with self.assertRaises(LayerParityError):
            split_fused_gate_up(array("f", [0.0] * 7), 2, 3)


class BindingRejectionTest(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.root = Path(self.td.name)
        self.addCleanup(self.td.cleanup)

    def test_missing_tensor_names_itself(self):
        tensors = make_tensors(0)
        write_fixture(self.root, 0, tensors)
        manifest = json.loads((self.root / "fixture.json").read_text())
        names = layer_attention_names(0, "provider_raw")
        manifest["entries"] = [e for e in manifest["entries"] if e["name"] != names["q"]]
        manifest.pop("total_payload_bytes")
        (self.root / "fixture.json").write_text(json.dumps(manifest))
        fixture = load_fixture(self.root)
        with self.assertRaises(Exception) as ctx:
            bind_layer(fixture, CONFIG, 0)
        self.assertIn(names["q"], str(ctx.exception))

    def test_shape_disagreement_is_refused(self):
        tensors = make_tensors(0)
        tensors["wq"] = torch.randn(CONFIG["local_heads"] * CONFIG["local_head_dim"],
                                    CONFIG["hidden"] + 1)
        write_fixture(self.root, 0, tensors)
        fixture = load_fixture(self.root)
        with self.assertRaises(LayerParityError) as ctx:
            bind_layer(fixture, CONFIG, 0)
        self.assertIn("config implies", str(ctx.exception))

    def test_uncovered_layer_is_refused(self):
        write_fixture(self.root, 0, make_tensors(0))
        fixture = load_fixture(self.root)
        with self.assertRaises(Exception):
            bind_layer(fixture, CONFIG, 1)

    def test_sparse_layer_without_experts_is_refused(self):
        tensors = make_tensors(1)
        write_fixture(self.root, 1, tensors)
        manifest = json.loads((self.root / "fixture.json").read_text())
        manifest["experts"] = {}
        (self.root / "fixture.json").write_text(json.dumps(manifest))
        fixture = load_fixture(self.root)
        with self.assertRaises(LayerParityError) as ctx:
            bind_layer(fixture, CONFIG, 1)
        self.assertIn("no routed experts", str(ctx.exception))

    def test_requesting_an_absent_expert_is_refused(self):
        write_fixture(self.root, 1, make_tensors(1))
        fixture = load_fixture(self.root)
        with self.assertRaises(Exception) as ctx:
            bind_layer(fixture, CONFIG, 1, experts=[0, 99])
        self.assertIn("99", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
