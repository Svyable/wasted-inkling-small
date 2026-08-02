#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.

import ctypes
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests"))
sys.path.insert(0, str(REPO / "tools"))

from inkling_vq import (
    KIND_NAMES, VQSpec, dequantize_record, quantize_matrix, read_codebooks,
    train_codebooks, write_codebooks, write_expert_record,
)
from test_inkling_attention_c import rmsnorm
from test_inkling_config_c import Args, Config
from test_inkling_layer_c import (
    FP, ExpertCallback, InklingLayerCDifferentialTest, LayerState,
    Scratch as LayerScratch, Weights as LayerWeights, array,
)

RowCallback = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, FP
)

TraceFloatCallback = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p, FP, ctypes.c_size_t
)
TraceIntCallback = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p,
    ctypes.POINTER(ctypes.c_int), ctypes.c_size_t
)

class Trace(ctypes.Structure):
    _fields_ = [("emit_float", TraceFloatCallback),
                ("emit_int", TraceIntCallback),
                ("ctx", ctypes.c_void_p)]


class ModelWeights(ctypes.Structure):
    _fields_ = [
        ("embedding", FP),
        ("embed_norm", FP),
        ("final_norm", FP),
        ("unembedding", FP),
        ("unembedding_rows", ctypes.c_int),
        ("layer", ctypes.POINTER(LayerWeights)),
        ("embedding_get", RowCallback),
        ("embedding_ctx", ctypes.c_void_p),
        ("unembedding_get", RowCallback),
        ("unembedding_ctx", ctypes.c_void_p),
    ]


class ModelState(ctypes.Structure):
    _fields_ = [
        ("layer", LayerState * 128),
        ("buffer", FP),
        ("buffer_floats", ctypes.c_size_t),
        ("context_capacity", ctypes.c_int),
        ("next_position", ctypes.c_int),
    ]


class ModelScratch(ctypes.Structure):
    _fields_ = [
        ("x", FP),
        ("row", FP),
        ("layer", LayerScratch),
        ("buffer", FP),
        ("ibuffer", ctypes.POINTER(ctypes.c_int)),
        ("buffer_floats", ctypes.c_size_t),
        ("ibuffer_ints", ctypes.c_size_t),
    ]


def as_tensor(values, n):
    return torch.tensor(list(values)[:n], dtype=torch.float32)


class WexpBank(ctypes.Structure):
    _fields_ = [
        ("bank_fd", ctypes.c_int), ("layer", ctypes.c_int),
        ("experts", ctypes.c_int), ("hidden", ctypes.c_int),
        ("intermediate", ctypes.c_int), ("stages", ctypes.c_int),
        ("entries", ctypes.c_int), ("vec_dim", ctypes.c_int),
        ("index_block", ctypes.c_int), ("fmt", ctypes.c_int),
        ("codebook_base", ctypes.c_int), ("record_bytes", ctypes.c_uint64),
        ("verify_crc", ctypes.c_int), ("codebooks", FP),
        ("gate", FP), ("up", FP), ("down", FP),
        ("record", ctypes.POINTER(ctypes.c_ubyte)),
        ("matrix_floats", ctypes.c_size_t),
        ("record_capacity", ctypes.c_size_t),
    ]


class InklingModelCDifferentialTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cc = shutil.which("cc")
        if not cc:
            raise unittest.SkipTest("C compiler unavailable")
        cls.td = tempfile.TemporaryDirectory()
        lib_path = Path(cls.td.name) / "libinkling_model.so"
        subprocess.run(
            [
                cc, "-std=c11", "-Wall", "-Wextra", "-Werror", "-shared", "-fPIC",
                f"-I{REPO / 'src'}",
                str(REPO / "src" / "inkling_model.c"),
                str(REPO / "src" / "inkling_layer.c"),
                str(REPO / "src" / "inkling_attention.c"),
                str(REPO / "src" / "inkling_config.c"),
                str(REPO / "src" / "inkling.c"),
                str(REPO / "src" / "inkling_wexp.c"),
                "-lm", "-o", str(lib_path),
            ],
            check=True, capture_output=True,
        )
        cls.lib = ctypes.CDLL(str(lib_path))
        cls.lib.waste_inkling_config_build.argtypes = [ctypes.POINTER(Config), ctypes.POINTER(Args)]
        cls.lib.waste_inkling_config_build.restype = ctypes.c_int
        cls.lib.waste_inkling_model_state_floats.argtypes = [ctypes.POINTER(Config), ctypes.c_int]
        cls.lib.waste_inkling_model_state_floats.restype = ctypes.c_size_t
        cls.lib.waste_inkling_model_scratch_floats.argtypes = [ctypes.POINTER(Config), ctypes.c_int]
        cls.lib.waste_inkling_model_scratch_floats.restype = ctypes.c_size_t
        cls.lib.waste_inkling_model_scratch_ints.argtypes = [ctypes.POINTER(Config)]
        cls.lib.waste_inkling_model_scratch_ints.restype = ctypes.c_size_t
        cls.lib.waste_inkling_model_state_init.argtypes = [
            ctypes.POINTER(ModelState), ctypes.POINTER(Config), ctypes.c_int,
            FP, ctypes.c_size_t,
        ]
        cls.lib.waste_inkling_model_state_init.restype = ctypes.c_int
        cls.lib.waste_inkling_model_scratch_init.argtypes = [
            ctypes.POINTER(ModelScratch), ctypes.POINTER(Config), ctypes.c_int,
            FP, ctypes.c_size_t, ctypes.POINTER(ctypes.c_int), ctypes.c_size_t,
        ]
        cls.lib.waste_inkling_model_scratch_init.restype = ctypes.c_int
        cls.lib.waste_inkling_model_reset.argtypes = [ctypes.POINTER(ModelState), ctypes.POINTER(Config)]
        cls.lib.waste_inkling_model_step.argtypes = [
            ctypes.POINTER(Config), ctypes.POINTER(ModelWeights),
            ctypes.POINTER(ModelState), ctypes.POINTER(ModelScratch),
            ctypes.c_int, ctypes.c_int, FP, ctypes.c_size_t,
            ExpertCallback, ctypes.c_void_p,
        ]
        cls.lib.waste_inkling_model_step.restype = ctypes.c_int
        cls.lib.waste_inkling_model_step_backend_trace.argtypes = [
            ctypes.POINTER(Config), ctypes.POINTER(ModelWeights), ctypes.c_void_p,
            ctypes.POINTER(ModelState), ctypes.POINTER(ModelScratch),
            ctypes.c_int, ctypes.c_int, FP, ctypes.c_size_t,
            ExpertCallback, ctypes.c_void_p, ctypes.POINTER(Trace),
        ]
        cls.lib.waste_inkling_model_step_backend_trace.restype = ctypes.c_int
        cls.lib.waste_inkling_wexp_bank_open.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int,
        ]
        cls.lib.waste_inkling_wexp_bank_open.restype = ctypes.c_int
        cls.lib.waste_inkling_wexp_bank_close.argtypes = [ctypes.c_void_p]
        # The fixture helper builds its Config through the same library. Make
        # the model test independently runnable instead of relying on unittest
        # module ordering to initialize the layer-test class first.
        InklingLayerCDifferentialTest.lib = cls.lib

    @classmethod
    def tearDownClass(cls):
        cls.td.cleanup()

    def build_fixture(self):
        helper = InklingLayerCDifferentialTest(methodName="runTest")
        cfg, local_keep = helper.build_config()
        weights = (LayerWeights * cfg.n_layers)()
        common = []
        extras = []
        keep = [local_keep]
        for layer in range(cfg.n_layers):
            w, tensors, layer_keep = helper.make_common(layer, cfg)
            torch.manual_seed(900 + layer)
            if layer < cfg.dense_layers:
                extra = {
                    "dense_gate": torch.randn(cfg.dense_intermediate, cfg.hidden) * 0.09,
                    "dense_up": torch.randn(cfg.dense_intermediate, cfg.hidden) * 0.09,
                    "dense_down": torch.randn(cfg.hidden, cfg.dense_intermediate) * 0.09,
                    "dense_scale": torch.tensor(0.85),
                }
                for name in ("dense_gate", "dense_up", "dense_down"):
                    layer_keep[name] = array(extra[name])
                    setattr(w, name, layer_keep[name])
                layer_keep["dense_scale"] = array(extra["dense_scale"].reshape(1))
                w.dense_global_scale = layer_keep["dense_scale"]
                w.sparse = 0
            else:
                total = cfg.n_routed_experts + cfg.n_shared_experts
                extra = {
                    "router_weight": torch.randn(total, cfg.hidden) * 0.09,
                    "router_bias": torch.randn(cfg.n_routed_experts) * 0.02,
                    "router_scale": torch.tensor(0.95),
                    "shared_gate": torch.randn(cfg.n_shared_experts, cfg.moe_intermediate, cfg.hidden) * 0.09,
                    "shared_up": torch.randn(cfg.n_shared_experts, cfg.moe_intermediate, cfg.hidden) * 0.09,
                    "shared_down": torch.randn(cfg.n_shared_experts, cfg.hidden, cfg.moe_intermediate) * 0.09,
                    "routed_gate": torch.randn(cfg.n_routed_experts, cfg.moe_intermediate, cfg.hidden) * 0.09,
                    "routed_up": torch.randn(cfg.n_routed_experts, cfg.moe_intermediate, cfg.hidden) * 0.09,
                    "routed_down": torch.randn(cfg.n_routed_experts, cfg.hidden, cfg.moe_intermediate) * 0.09,
                }
                for name, value in extra.items():
                    layer_keep[name] = array(value.reshape(1) if value.ndim == 0 else value)
                    setattr(w, "router_global_scale" if name == "router_scale" else name,
                            layer_keep[name])
                w.sparse = 1
            weights[layer] = w
            common.append(tensors)
            extras.append(extra)
            keep.append(layer_keep)

        torch.manual_seed(1200)
        tables = {
            "embedding": torch.randn(cfg.vocab, cfg.hidden) * 0.1,
            "embed_norm": torch.rand(cfg.hidden) + 0.5,
            "final_norm": torch.rand(cfg.hidden) + 0.5,
            "unembedding": torch.randn(cfg.unpadded_vocab, cfg.hidden) * 0.1,
        }
        carr = {name: array(value) for name, value in tables.items()}
        keep.append(carr)
        mw = ModelWeights()
        mw.embedding = carr["embedding"]
        mw.embed_norm = carr["embed_norm"]
        mw.final_norm = carr["final_norm"]
        mw.unembedding = carr["unembedding"]
        mw.unembedding_rows = cfg.unpadded_vocab
        mw.layer = weights
        keep.append(weights)
        return helper, cfg, mw, common, extras, tables, keep

    def init_runtime(self, cfg, capacity):
        ns = self.lib.waste_inkling_model_state_floats(ctypes.byref(cfg), capacity)
        nf = self.lib.waste_inkling_model_scratch_floats(ctypes.byref(cfg), capacity)
        ni = self.lib.waste_inkling_model_scratch_ints(ctypes.byref(cfg))
        self.assertGreater(ns, 0)
        self.assertGreater(nf, 0)
        sbuf = (ctypes.c_float * ns)()
        fbuf = (ctypes.c_float * nf)()
        ibuf = (ctypes.c_int * ni)()
        state = ModelState()
        scratch = ModelScratch()
        self.assertEqual(self.lib.waste_inkling_model_state_init(
            ctypes.byref(state), ctypes.byref(cfg), capacity, sbuf, ns), 0)
        self.assertEqual(self.lib.waste_inkling_model_scratch_init(
            ctypes.byref(scratch), ctypes.byref(cfg), capacity,
            fbuf, nf, ibuf, ni), 0)
        return state, scratch, (sbuf, fbuf, ibuf)

    def py_states(self, cfg, capacity):
        states = []
        for layer in range(cfg.n_layers):
            l = cfg.layer[layer]
            cap = min(capacity, cfg.sliding_window) if l.is_local else capacity
            kdim = l.num_kv_heads * l.head_dim
            states.append({
                "capacity": cap, "keys": [], "values": [],
                "kconv": torch.zeros(kdim, cfg.conv_kernel),
                "vconv": torch.zeros(kdim, cfg.conv_kernel),
                "aconv": torch.zeros(cfg.hidden, cfg.conv_kernel),
                "mconv": torch.zeros(cfg.hidden, cfg.conv_kernel),
            })
        return states

    def run_sequence(self, use_callbacks=False):
        helper, cfg, mw, common, extras, tables, keep = self.build_fixture()
        capacity = 6
        state, scratch, runtime_keep = self.init_runtime(cfg, capacity)
        callbacks = []
        if use_callbacks:
            embed = tables["embedding"]
            unembed = tables["unembedding"]

            @RowCallback
            def embed_cb(_ctx, row, cols, out):
                if row < 0 or row >= embed.shape[0] or cols != embed.shape[1]:
                    return -1
                ctypes.memmove(out, array(embed[row]), cols * ctypes.sizeof(ctypes.c_float))
                return 0

            @RowCallback
            def unembed_cb(_ctx, row, cols, out):
                if row < 0 or row >= unembed.shape[0] or cols != unembed.shape[1]:
                    return -1
                ctypes.memmove(out, array(unembed[row]), cols * ctypes.sizeof(ctypes.c_float))
                return 0

            mw.embedding = None
            mw.embedding_get = embed_cb
            mw.unembedding = None
            mw.unembedding_get = unembed_cb
            callbacks.extend([embed_cb, unembed_cb])

        py_state = self.py_states(cfg, capacity)
        tokens = [3, 5, 7, 11, 2]
        for position, token in enumerate(tokens):
            x = rmsnorm(tables["embedding"][token], tables["embed_norm"], cfg.rms_eps)
            for layer in range(cfg.n_layers):
                x = helper.reference_layer(
                    cfg, layer, common[layer], extras[layer], x,
                    position, py_state[layer],
                )
            x = rmsnorm(x, tables["final_norm"], cfg.rms_eps)
            want = tables["unembedding"] @ (x / cfg.logits_width_multiplier)
            out = (ctypes.c_float * cfg.unpadded_vocab)()
            rc = self.lib.waste_inkling_model_step(
                ctypes.byref(cfg), ctypes.byref(mw), ctypes.byref(state),
                ctypes.byref(scratch), token, position, out,
                cfg.unpadded_vocab, ExpertCallback(), None,
            )
            self.assertEqual(rc, 0)
            got = as_tensor(out, cfg.unpadded_vocab)
            torch.testing.assert_close(got, want, rtol=3e-4, atol=3e-4)
        self.assertEqual(state.next_position, len(tokens))
        self.assertIsNotNone((keep, runtime_keep, callbacks))

    def test_complete_model_matches_torch(self):
        self.run_sequence(False)

    def test_embedding_and_unembedding_callbacks_match(self):
        self.run_sequence(True)

    def test_reset_restores_first_token_result(self):
        _helper, cfg, mw, _common, _extras, _tables, keep = self.build_fixture()
        state, scratch, runtime_keep = self.init_runtime(cfg, 4)
        out1 = (ctypes.c_float * cfg.unpadded_vocab)()
        out2 = (ctypes.c_float * cfg.unpadded_vocab)()
        self.assertEqual(self.lib.waste_inkling_model_step(
            ctypes.byref(cfg), ctypes.byref(mw), ctypes.byref(state),
            ctypes.byref(scratch), 4, 0, out1, cfg.unpadded_vocab,
            ExpertCallback(), None), 0)
        self.lib.waste_inkling_model_reset(ctypes.byref(state), ctypes.byref(cfg))
        self.assertEqual(self.lib.waste_inkling_model_step(
            ctypes.byref(cfg), ctypes.byref(mw), ctypes.byref(state),
            ctypes.byref(scratch), 4, 0, out2, cfg.unpadded_vocab,
            ExpertCallback(), None), 0)
        torch.testing.assert_close(as_tensor(out1, cfg.unpadded_vocab),
                                   as_tensor(out2, cfg.unpadded_vocab),
                                   rtol=0, atol=0)
        self.assertIsNotNone((keep, runtime_keep))

    def test_final_wexp_callback_matches_dequantized_reference(self):
        helper, cfg, mw, common, extras, tables, keep = self.build_fixture()
        layer = 1
        routed = extras[layer]
        spec = VQSpec(stages=2, vec_dim=2, entries=8, index_block=4)
        books = {}
        for kind, values in zip(
            KIND_NAMES,
            (routed["routed_gate"], routed["routed_up"], routed["routed_down"]),
        ):
            vectors = []
            for matrix in values:
                scale = matrix.abs().amax(1, keepdim=True).clamp_min(1e-8)
                vectors.append((matrix / scale).reshape(-1, spec.vec_dim))
            books[kind] = train_codebooks(
                torch.cat(vectors), spec, iterations=4, assign_chunk=16,
                seed=40 + KIND_NAMES.index(kind),
            )

        stage_td = tempfile.TemporaryDirectory()
        self.addCleanup(stage_td.cleanup)
        root = Path(stage_td.name)
        book_path = root / "codebooks-L1.bin"
        bank_path = root / "experts-L1.bin"
        write_codebooks(book_path, 0, books, spec)
        record_bytes = 0
        with bank_path.open("wb") as f:
            for expert in range(cfg.n_routed_experts):
                matrices = [
                    quantize_matrix(values[expert], books[kind], spec, native=False,
                                    assign_chunk=16)
                    for kind, values in zip(
                        KIND_NAMES,
                        (routed["routed_gate"], routed["routed_up"], routed["routed_down"]),
                    )
                ]
                record_bytes = write_expert_record(f, layer, expert, 0, matrices, spec)

        loaded_books = read_codebooks(book_path, spec)
        qgate, qup, qdown = [], [], []
        with bank_path.open("rb") as f:
            for expert in range(cfg.n_routed_experts):
                decoded = dequantize_record(
                    f.read(record_bytes), loaded_books,
                    ((cfg.moe_intermediate, cfg.hidden),
                     (cfg.moe_intermediate, cfg.hidden),
                     (cfg.hidden, cfg.moe_intermediate)),
                    spec, expected_layer=layer, expected_expert=expert,
                )
                qgate.append(decoded.gate); qup.append(decoded.up); qdown.append(decoded.down)
        extras[layer] = dict(extras[layer])
        extras[layer]["routed_gate"] = torch.stack(qgate)
        extras[layer]["routed_up"] = torch.stack(qup)
        extras[layer]["routed_down"] = torch.stack(qdown)
        mw.layer[layer].routed_gate = None
        mw.layer[layer].routed_up = None
        mw.layer[layer].routed_down = None

        bank = WexpBank()
        self.assertEqual(self.lib.waste_inkling_wexp_bank_open(
            ctypes.byref(bank), str(bank_path).encode(), str(book_path).encode(),
            layer, cfg.n_routed_experts, cfg.hidden, cfg.moe_intermediate,
            spec.stages, spec.entries, spec.vec_dim, spec.index_block, 0, 1,
        ), 0)
        callback = ctypes.cast(self.lib.waste_inkling_wexp_expert_get, ExpertCallback)
        try:
            capacity = 5
            state, scratch, runtime_keep = self.init_runtime(cfg, capacity)
            py_state = self.py_states(cfg, capacity)
            for position, token in enumerate((3, 5, 7)):
                x = rmsnorm(tables["embedding"][token], tables["embed_norm"], cfg.rms_eps)
                for current in range(cfg.n_layers):
                    x = helper.reference_layer(
                        cfg, current, common[current], extras[current], x,
                        position, py_state[current],
                    )
                x = rmsnorm(x, tables["final_norm"], cfg.rms_eps)
                want = tables["unembedding"] @ (x / cfg.logits_width_multiplier)
                out = (ctypes.c_float * cfg.unpadded_vocab)()
                self.assertEqual(self.lib.waste_inkling_model_step(
                    ctypes.byref(cfg), ctypes.byref(mw), ctypes.byref(state),
                    ctypes.byref(scratch), token, position, out,
                    cfg.unpadded_vocab, callback, ctypes.byref(bank),
                ), 0)
                torch.testing.assert_close(
                    as_tensor(out, cfg.unpadded_vocab), want,
                    rtol=3e-4, atol=3e-4,
                )
            self.assertIsNotNone((keep, runtime_keep, callback))
        finally:
            self.lib.waste_inkling_wexp_bank_close(ctypes.byref(bank))

    def test_trace_emits_model_layer_and_router_boundaries(self):
        _helper, cfg, mw, _common, _extras, _tables, keep = self.build_fixture()
        state, scratch, runtime_keep = self.init_runtime(cfg, 3)
        seen_float = {}
        seen_int = {}

        @TraceFloatCallback
        def emit_float(_ctx, layer, point, data, count):
            name = point.decode()
            seen_float[(layer, name)] = torch.tensor(
                [data[i] for i in range(count)], dtype=torch.float32
            )
            return 0

        @TraceIntCallback
        def emit_int(_ctx, layer, point, data, count):
            name = point.decode()
            seen_int[(layer, name)] = [data[i] for i in range(count)]
            return 0

        trace = Trace(emit_float, emit_int, None)
        out = (ctypes.c_float * cfg.unpadded_vocab)()
        self.assertEqual(self.lib.waste_inkling_model_step_backend_trace(
            ctypes.byref(cfg), ctypes.byref(mw), None, ctypes.byref(state),
            ctypes.byref(scratch), 3, 0, out, cfg.unpadded_vocab,
            ExpertCallback(), None, ctypes.byref(trace),
        ), 0)
        for key in [
            (-1, "embedding_norm"), (0, "input_norm"),
            (0, "attention_out"), (0, "dense_mlp_out"),
            (1, "router_logits"), (1, "moe_out"),
            (1, "layer_out"), (-1, "final_norm"), (-1, "logits"),
        ]:
            self.assertIn(key, seen_float)
        self.assertIn((1, "routed_index"), seen_int)
        self.assertEqual(len(seen_int[(1, "routed_index")]), cfg.top_k)
        torch.testing.assert_close(
            seen_float[(-1, "logits")], as_tensor(out, cfg.unpadded_vocab),
            rtol=0, atol=0,
        )
        self.assertIsNotNone((keep, runtime_keep, emit_float, emit_int))

    def test_position_and_output_bounds_fail_closed(self):
        _helper, cfg, mw, _common, _extras, _tables, keep = self.build_fixture()
        state, scratch, runtime_keep = self.init_runtime(cfg, 2)
        out = (ctypes.c_float * cfg.unpadded_vocab)()
        self.assertNotEqual(self.lib.waste_inkling_model_step(
            ctypes.byref(cfg), ctypes.byref(mw), ctypes.byref(state),
            ctypes.byref(scratch), 1, 1, out, cfg.unpadded_vocab,
            ExpertCallback(), None), 0)
        self.assertNotEqual(self.lib.waste_inkling_model_step(
            ctypes.byref(cfg), ctypes.byref(mw), ctypes.byref(state),
            ctypes.byref(scratch), 1, 0, out, cfg.unpadded_vocab - 1,
            ExpertCallback(), None), 0)
        self.assertIsNotNone((keep, runtime_keep))


if __name__ == "__main__":
    unittest.main()
