#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

from inkling_reference import register_reference_hooks
from inkling_trace import TraceCollector


class Identity(nn.Module):
    def forward(self, x):
        return x + 1


class Router(nn.Module):
    def forward(self, x):
        return x[:4], x[:2], torch.tensor([3, 1]), x[:2]


class Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = Identity(); self.k_proj = Identity(); self.v_proj = Identity()
        self.r_proj = Identity(); self.k_sconv = Identity(); self.v_sconv = Identity()
        self.o_proj = Identity()
    def forward(self, x):
        x = self.q_proj(x); self.k_proj(x); self.v_proj(x); self.r_proj(x)
        self.k_sconv(x); self.v_sconv(x)
        return self.o_proj(x + 2), None


class MLP(nn.Module):
    def __init__(self, sparse):
        super().__init__(); self.gate = Router() if sparse else None
    def forward(self, x):
        if self.gate is not None: self.gate(x)
        return x + 3


class Layer(nn.Module):
    def __init__(self, sparse):
        super().__init__()
        self.input_layernorm = Identity(); self.self_attn = Attn()
        self.attn_sconv = Identity(); self.post_attention_layernorm = Identity()
        self.mlp = MLP(sparse); self.mlp_sconv = Identity()
    def forward(self, x):
        x = self.input_layernorm(x)
        x = self.self_attn(x)[0]
        x = self.attn_sconv(x)
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        return self.mlp_sconv(x)


class Text(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = type("Cfg", (), {"dense_mlp_idx": 1})()
        self.embed_norm = Identity(); self.layers = nn.ModuleList([Layer(False), Layer(True)])
        self.norm = Identity()
    def forward(self, x):
        x = self.embed_norm(x)
        for layer in self.layers: x = layer(x)
        return self.norm(x)


class Model(nn.Module):
    def __init__(self):
        super().__init__(); self.model = Text()


class TraceToolsTest(unittest.TestCase):
    def test_c_trace_collector_names_and_copies(self):
        collector = TraceCollector(); collector.position = 4
        f = (torch.arange(3, dtype=torch.float32).numpy().ctypes.data_as(
            __import__("ctypes").POINTER(__import__("ctypes").c_float)))
        self.assertEqual(collector._emit_float(None, 2, b"q_proj", f, 3), 0)
        self.assertTrue(torch.equal(collector.values["token.4.layer.2.q_proj"], torch.arange(3.0)))

    def test_reference_hooks_capture_dense_sparse_and_model_points(self):
        model = Model(); store = {}; pos = [0]
        handles = register_reference_hooks(model, [0, 1], store, lambda: pos[0])
        try:
            model.model(torch.arange(8, dtype=torch.float32))
        finally:
            for h in handles: h.remove()
        expected = {
            "token.0.model.embedding_norm", "token.0.model.final_norm",
            "token.0.layer.0.input_norm", "token.0.layer.0.q_proj",
            "token.0.layer.0.attention_out", "token.0.layer.0.attention_branch",
            "token.0.layer.0.post_attention_residual",
            "token.0.layer.0.post_attention_norm",
            "token.0.layer.0.dense_mlp_out", "token.0.layer.0.layer_out",
            "token.0.layer.1.router_logits", "token.0.layer.1.routed_index",
            "token.0.layer.1.moe_out", "token.0.layer.1.layer_out",
        }
        self.assertTrue(expected.issubset(store))
        self.assertEqual(store["token.0.layer.1.routed_index"].dtype, torch.int32)


if __name__ == "__main__":
    unittest.main()