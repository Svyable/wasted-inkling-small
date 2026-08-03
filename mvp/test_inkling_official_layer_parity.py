#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Our C decoder layer against the official transformers implementation.

This is the independent oracle G1 needs. Every other differential in this
repository compares the C against a PyTorch reference written *in this
repository*; if that reference shares a misreading with the C, both agree and
both are wrong. `transformers` cannot share our misreadings.

It needs no checkpoint: the fixture is synthesized, but it is the single shared
source of weights for both sides, which is what makes the comparison mean
something.

Two harness bugs this test was written to pin down, both of which produce a
large, plausible, position-dependent divergence that looks like a bug in the C:

  * `attention_mask=None` is UNMASKED in the eager path, not causal;
  * a local layer's attention ring capacity IS its sliding window, so defaulting
    that capacity to the token count silently makes a windowed layer
    full-context.
"""

import ctypes
import json
import sys
import tempfile
import unittest
import zlib
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "mvp"))
sys.path.insert(0, str(REPO / "inkling" / "tools"))

from inkling_release_config import build_transformers_text_config
from inkling_fixture_reference import run_layer_reference
from inkling_fixture import load_fixture
from inkling_layer_parity import bind_layer, build_library, configure_library, run_layer_trace
from inkling_plan import layer_attention_names, layer_dense_mlp_names, layer_sparse_mlp_names


H, HEADS, KV, HD, DREL = 8, 2, 1, 4, 2
SW, RELEXT, K = 3, 5, 3
DENSE_INTER, MOE_INTER = 12, 6
NROUTED, TOPK, NSHARED = 3, 2, 1

RELEASE = {
    "hidden_size": H, "num_hidden_layers": 2,
    "num_attention_heads": HEADS, "num_key_value_heads": KV, "head_dim": HD,
    "swa_num_attention_heads": HEADS, "swa_num_key_value_heads": KV, "swa_head_dim": HD,
    "d_rel": DREL, "rel_extent": RELEXT, "sliding_window_size": SW,
    "sconv_kernel_size": K,
    "dense_intermediate_size": DENSE_INTER,   # release schema: dense
    "intermediate_size": MOE_INTER,           # release schema: routed
    "n_routed_experts": NROUTED, "num_experts_per_tok": TOPK,
    "n_shared_experts": NSHARED, "dense_mlp_idx": 1,
    "local_layer_ids": [0],
    "vocab_size": 32, "unpadded_vocab_size": 30, "model_max_length": 16,
    "rms_norm_eps": 1e-6, "route_scale": 2.0,
    "logits_mup_width_multiplier": 2.0,
    "log_scaling_n_floor": 2, "log_scaling_alpha": 0.2,
    "shared_expert_sink": True, "norm_after_topk": True,
    "use_gate_bias": True, "gate_activation": "sigmoid",
}

C_CONFIG = {
    "n_layers": 2, "hidden": H, "vocab": 32, "unpadded_vocab": 30,
    "max_context": 16,
    "global_heads": HEADS, "global_kv_heads": KV, "global_head_dim": HD,
    "local_heads": HEADS, "local_kv_heads": KV, "local_head_dim": HD,
    "sliding_window": SW, "d_rel": DREL, "rel_extent": RELEXT, "conv_kernel": K,
    "dense_layers": 1, "dense_intermediate": DENSE_INTER, "moe_intermediate": MOE_INTER,
    "n_routed_experts": NROUTED, "top_k": TOPK, "n_shared_experts": NSHARED,
    "rms_eps": 1e-6, "route_scale": 2.0, "logits_width_multiplier": 2.0,
    "log_scaling_n_floor": 2, "log_scaling_alpha": 0.2,
    "local_layer_ids": [0],
}


def fuse(gate, up):
    inter, hid = gate.shape
    out = torch.empty(2 * inter, hid, dtype=gate.dtype)
    out[0::2] = gate
    out[1::2] = up
    return out


def make_tensors(layer, seed=11):
    torch.manual_seed(seed + layer)
    is_local = layer in C_CONFIG["local_layer_ids"]
    extent = SW if is_local else RELEXT
    t = {
        "input_norm": torch.rand(H) + 0.5,
        "post_attention_norm": torch.rand(H) + 0.5,
        "wq": torch.randn(HEADS * HD, H) * 0.15,
        "wk": torch.randn(KV * HD, H) * 0.15,
        "wv": torch.randn(KV * HD, H) * 0.15,
        "wr": torch.randn(HEADS * DREL, H) * 0.15,
        "wo": torch.randn(H, HEADS * HD) * 0.15,
        "q_norm": torch.rand(HD) + 0.5,
        "k_norm": torch.rand(HD) + 0.5,
        "relative_proj": torch.randn(DREL, extent) * 0.2,
        "k_sconv": torch.randn(KV * HD, K) * 0.2,
        "v_sconv": torch.randn(KV * HD, K) * 0.2,
        "attn_sconv": torch.randn(H, K) * 0.2,
        "mlp_sconv": torch.randn(H, K) * 0.2,
    }
    if layer < C_CONFIG["dense_layers"]:
        t["dense_gate"] = torch.randn(DENSE_INTER, H) * 0.15
        t["dense_up"] = torch.randn(DENSE_INTER, H) * 0.15
        t["dense_down"] = torch.randn(H, DENSE_INTER) * 0.15
        t["dense_global_scale"] = torch.tensor([1.25])
    else:
        t["router_weight"] = torch.randn(NROUTED + NSHARED, H) * 0.2
        t["router_bias"] = torch.randn(NROUTED) * 0.1
        t["router_global_scale"] = torch.tensor([0.75])
        t["routed_gate"] = torch.randn(NROUTED, MOE_INTER, H) * 0.15
        t["routed_up"] = torch.randn(NROUTED, MOE_INTER, H) * 0.15
        t["routed_down"] = torch.randn(NROUTED, H, MOE_INTER) * 0.15
        t["shared_gate"] = torch.randn(NSHARED, MOE_INTER, H) * 0.15
        t["shared_up"] = torch.randn(NSHARED, MOE_INTER, H) * 0.15
        t["shared_down"] = torch.randn(NSHARED, H, MOE_INTER) * 0.15
    return t


def write_fixture(root, layer, t):
    entries = []

    def add(name, tensor, axis0=None):
        payload = tensor.detach().float().contiguous().view(torch.uint8).numpy().tobytes()
        rel = f"e{len(entries)}.bin"
        (root / rel).write_bytes(payload)
        e = {"name": name, "kind": "tensor" if axis0 is None else "axis0-slice",
             "dtype": "F32", "shape": list(tensor.shape), "bytes": len(payload),
             "crc32": zlib.crc32(payload) & 0xFFFFFFFF, "path": rel}
        if axis0 is not None:
            e["axis0"] = axis0
        entries.append(e)

    n = layer_attention_names(layer, "provider_raw")
    for role, key in (("input_norm","input_norm"),("post_attention_norm","post_attention_norm"),
                      ("q","wq"),("k","wk"),("v","wv"),("r","wr"),("o","wo"),
                      ("q_norm","q_norm"),("k_norm","k_norm"),("rel_proj","relative_proj"),
                      ):
        add(n[role], t[key])
    # Release checkpoints store Conv1d weights as [channels, 1, kernel]; our C
    # canonicalizes to [channels, kernel] with identical payload bytes.
    for role, key in (("k_sconv","k_sconv"),("v_sconv","v_sconv"),
                      ("attn_sconv","attn_sconv"),("mlp_sconv","mlp_sconv")):
        add(n[role], t[key].unsqueeze(1))

    experts = {}
    if layer < C_CONFIG["dense_layers"]:
        m = layer_dense_mlp_names(layer, "provider_raw")
        add(m["fused_gate_up"], fuse(t["dense_gate"], t["dense_up"]))
        add(m["down"], t["dense_down"])
        add(m["global_scale"], t["dense_global_scale"])
    else:
        sp = layer_sparse_mlp_names(layer, "provider_raw")
        add(sp["router"]["weight"], t["router_weight"])
        add(sp["router"]["correction_bias"], t["router_bias"])
        add(sp["router"]["global_scale"], t["router_global_scale"])
        add(sp["shared"]["fused_gate_up"],
            torch.stack([fuse(t["shared_gate"][s], t["shared_up"][s]) for s in range(NSHARED)]))
        add(sp["shared"]["down"], t["shared_down"])
        ids = list(range(NROUTED))
        for e in ids:
            add(sp["routed"]["fused_gate_up"], fuse(t["routed_gate"][e], t["routed_up"][e]), axis0=e)
            add(sp["routed"]["down"], t["routed_down"][e], axis0=e)
        experts[str(layer)] = ids

    (root / "fixture.json").write_text(json.dumps({
        "format": "inkling-parity-fixture", "version": 1, "model_id": "synthetic",
        "layers": [layer], "experts": experts,
        "source": {"config_sha256": "0"*64, "index_sha256": "1"*64},
        "total_payload_bytes": sum(e["bytes"] for e in entries), "entries": entries,
    }, indent=2))
    return root




class OfficialLayerParityTest(unittest.TestCase):
    """Float32 epsilon, not a tolerance chosen to make the test pass."""

    TOL = 1e-5

    @classmethod
    def setUpClass(cls):
        cls.td = tempfile.TemporaryDirectory()
        cls.lib = ctypes.CDLL(str(build_library(
            REPO / "inkling" / "src", Path(cls.td.name) / "layer.so")))
        configure_library(cls.lib)
        cls.config = build_transformers_text_config(RELEASE)

    @classmethod
    def tearDownClass(cls):
        cls.td.cleanup()

    def test_release_schema_maps_dense_and_routed(self):
        self.assertEqual(self.config.intermediate_size, DENSE_INTER)
        self.assertEqual(self.config.moe_intermediate_size, MOE_INTER)

    def _compare(self, layer, n_tokens):
        torch.manual_seed(99)
        seq = torch.randn(1, n_tokens, H)
        inputs = [seq[0, i].tolist() for i in range(n_tokens)]

        root = Path(tempfile.mkdtemp(dir=self.td.name))
        tensors = make_tensors(layer)
        write_fixture(root, layer, tensors)
        fixture = load_fixture(root)

        official = run_layer_reference(
            fixture, self.config, layer, inputs,
            device=torch.device("cpu"), dtype=torch.float32)
        ours = run_layer_trace(
            self.lib, C_CONFIG, layer, bind_layer(fixture, C_CONFIG, layer),
            inputs)["outputs"]

        worst = 0.0
        for i in range(n_tokens):
            a = official[f"token.{i}.layer.{layer}.output"].float()
            b = torch.tensor(ours[i], dtype=torch.float32)
            worst = max(worst, (a - b).abs().max().item())
        return worst

    def test_dense_local_layer_matches_official(self):
        for n in (3, 4, 6):
            with self.subTest(tokens=n):
                self.assertLess(self._compare(0, n), self.TOL)

    def test_sparse_global_layer_matches_official(self):
        for n in (3, 4, 6):
            with self.subTest(tokens=n):
                self.assertLess(self._compare(1, n), self.TOL)

    def test_sliding_window_is_actually_applied(self):
        """A local layer past its window must differ from a full-context one.

        Guards the capacity default: if the ring were sized to the token count
        the two would agree and the windowing would be silently gone.
        """
        torch.manual_seed(99)
        n = 6
        seq = torch.randn(1, n, H)
        inputs = [seq[0, i].tolist() for i in range(n)]
        root = Path(tempfile.mkdtemp(dir=self.td.name))
        write_fixture(root, 0, make_tensors(0))
        fixture = load_fixture(root)
        binding = bind_layer(fixture, C_CONFIG, 0)
        windowed = run_layer_trace(self.lib, C_CONFIG, 0, binding, inputs)["outputs"]
        full = run_layer_trace(self.lib, C_CONFIG, 0, binding, inputs,
                               attention_capacity=n)["outputs"]
        last = max(abs(x - y) for x, y in zip(windowed[-1], full[-1]))
        self.assertGreater(last, 1e-4,
                           "windowed and full-context runs agree; the sliding "
                           "window is not being applied")


if __name__ == "__main__":
    unittest.main()
