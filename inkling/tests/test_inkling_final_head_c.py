#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
"""Differential tests for the checked-in Inkling final head.

The head is final RMS normalization, the logits-width completion, and the
vocabulary projection. This file pins both numeric profiles against an
independent Python reference that models the C float and bfloat16 rounding
explicitly, so a change to either policy has to be deliberate.

Dependency-light on purpose: ctypes and the standard library only, so the
cheap validation gate can run it before anything downloads official weights.
Nothing here is a claim about model logits -- the primitive computes the head
of whatever hidden state it is handed.
"""

import ctypes
import math
import random
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_inkling_config_c import Args, Config

FP = ctypes.POINTER(ctypes.c_float)
PROFILE_F32 = 0
PROFILE_BF16 = 1
KIND_UNEMBED = 15

RowCallback = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, FP
)
MatvecCallback = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    FP, FP, ctypes.c_int, ctypes.c_int,
)
TraceFloatCallback = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p, FP,
    ctypes.c_size_t,
)
TraceIntCallback = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p,
    ctypes.POINTER(ctypes.c_int), ctypes.c_size_t,
)


class MatrixBackend(ctypes.Structure):
    _fields_ = [("matvec", MatvecCallback), ("ctx", ctypes.c_void_p)]


class Trace(ctypes.Structure):
    _fields_ = [("emit_float", TraceFloatCallback),
                ("emit_int", TraceIntCallback),
                ("ctx", ctypes.c_void_p)]


class ModelWeights(ctypes.Structure):
    """Mirror of waste_inkling_model_weights.

    Only the head fields are exercised here; `layer` stays NULL because the
    primitive never touches a decoder layer.
    """

    _fields_ = [
        ("embedding", FP),
        ("embed_norm", FP),
        ("final_norm", FP),
        ("unembedding", FP),
        ("unembedding_rows", ctypes.c_int),
        ("layer", ctypes.c_void_p),
        ("embedding_get", RowCallback),
        ("embedding_ctx", ctypes.c_void_p),
        ("unembedding_get", RowCallback),
        ("unembedding_ctx", ctypes.c_void_p),
    ]


def f32(value):
    """Round a Python float to float32, the way every C assignment does."""
    return struct.unpack("f", struct.pack("f", value))[0]


def bf16(value):
    """Nearest-even bfloat16 rounding, mirroring waste_inkling_bf16_round."""
    bits = struct.unpack("I", struct.pack("f", f32(value)))[0]
    if (bits & 0x7F800000) != 0x7F800000:
        bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return struct.unpack("f", struct.pack("I", bits & 0xFFFFFFFF))[0]


def rms_scale(x, eps):
    """The shared scale factor: double accumulation, float32 completion."""
    ss = 0.0
    for value in x:
        ss += float(value) * float(value)
    return f32(1.0 / f32(math.sqrt(f32(f32(ss / len(x)) + eps))))


def reference_head(x, final_norm, unembedding, rows, eps, multiplier, profile):
    """Return (normalized, scaled, logits) for one profile.

    The F32 branch reproduces the expression order of the checked-in C. The
    BF16 branch reproduces the retained evidence ordering: complete after the
    normalization, after the weight multiply, and after the width division --
    a true division, not a reciprocal multiply.
    """
    scale = rms_scale(x, eps)
    if profile == PROFILE_BF16:
        normalized = [bf16(f32(bf16(f32(value * scale)) * weight))
                      for value, weight in zip(x, final_norm)]
        scaled = [bf16(f32(value / multiplier)) for value in normalized]
    else:
        normalized = [f32(f32(value * scale) * weight)
                      for value, weight in zip(x, final_norm)]
        inv = f32(1.0 / multiplier)
        scaled = [f32(value * inv) for value in normalized]
    logits = []
    for row in rows:
        total = 0.0
        for weight, value in zip(unembedding[row], scaled):
            total += float(weight) * float(value)
        logits.append(f32(total))
    return normalized, scaled, logits


def array(values):
    return (ctypes.c_float * len(values))(*[f32(v) for v in values])


class InklingFinalHeadCTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cc = shutil.which("cc")
        if not cc:
            raise unittest.SkipTest("C compiler unavailable")
        cls.td = tempfile.TemporaryDirectory()
        lib_path = Path(cls.td.name) / "libinkling_final_head.so"
        subprocess.run(
            [
                cc, "-std=c11", "-Wall", "-Wextra", "-Werror", "-shared",
                "-fPIC", f"-I{REPO / 'src'}",
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
        cls.lib.waste_inkling_config_build.argtypes = [
            ctypes.POINTER(Config), ctypes.POINTER(Args),
        ]
        cls.lib.waste_inkling_config_build.restype = ctypes.c_int
        cls.lib.waste_inkling_final_head_profile.argtypes = [
            ctypes.POINTER(Config), ctypes.POINTER(ModelWeights),
            ctypes.POINTER(MatrixBackend), FP,
            ctypes.POINTER(ctypes.c_int), ctypes.c_int,
            FP, ctypes.c_size_t, FP, FP, ctypes.c_int,
            ctypes.POINTER(Trace),
        ]
        cls.lib.waste_inkling_final_head_profile.restype = ctypes.c_int

    @classmethod
    def tearDownClass(cls):
        cls.td.cleanup()

    def build_config(self, hidden=32, vocab=64, unpadded=48, multiplier=3.0):
        a = Args()
        a.n_layers = 2
        a.hidden = hidden
        a.vocab = vocab
        a.unpadded_vocab = unpadded
        a.max_context = 128
        a.global_kv_heads = 2
        a.global_heads = 4
        a.global_head_dim = 8
        a.local_kv_heads = 2
        a.local_heads = 4
        a.local_head_dim = 8
        a.sliding_window = 16
        a.d_rel = 4
        a.rel_extent = 32
        a.conv_kernel = 3
        a.dense_layers = 1
        a.dense_intermediate = hidden * 2
        a.moe_intermediate = hidden
        a.n_routed_experts = 4
        a.top_k = 2
        a.n_shared_experts = 1
        a.rms_eps = 1e-6
        a.route_scale = 8.0
        a.logits_width_multiplier = multiplier
        a.log_scaling_n_floor = 128
        a.log_scaling_alpha = 0.1
        a.local_layer_ids = None
        a.n_local_layers = 0
        cfg = Config()
        self.assertEqual(
            self.lib.waste_inkling_config_build(ctypes.byref(cfg),
                                                ctypes.byref(a)), 0)
        return cfg

    def make_tables(self, cfg, seed=11, quantize=False):
        rng = random.Random(seed)
        # Every value the C side sees is float32; BF16 evidence additionally
        # arrives already completed, so the tables model that too.
        prepare = bf16 if quantize else f32
        x = [prepare(rng.uniform(-2.0, 2.0)) for _ in range(cfg.hidden)]
        final_norm = [prepare(rng.uniform(0.2, 1.8)) for _ in range(cfg.hidden)]
        unembedding = [
            [prepare(rng.uniform(-0.5, 0.5)) for _ in range(cfg.hidden)]
            for _ in range(cfg.unpadded_vocab)
        ]
        return x, final_norm, unembedding

    def bind(self, cfg, final_norm, unembedding=None, keep=None):
        keep = keep if keep is not None else []
        w = ModelWeights()
        keep.append(array(final_norm))
        w.final_norm = keep[-1]
        if unembedding is not None:
            flat = [value for row in unembedding for value in row]
            keep.append(array(flat))
            w.unembedding = keep[-1]
            w.unembedding_rows = len(unembedding)
        return w, keep

    def call(self, cfg, w, x, rows, n_rows, profile, backend=None,
             logits_count=None, trace=None, logits=None):
        hidden = cfg.hidden
        xa = array(x)
        normalized = (ctypes.c_float * hidden)()
        row_scratch = (ctypes.c_float * hidden)()
        out = logits if logits is not None else (ctypes.c_float * max(n_rows, 1))()
        selection = None
        if rows is not None:
            selection = (ctypes.c_int * len(rows))(*rows)
        count = logits_count if logits_count is not None else max(n_rows, 1)
        rc = self.lib.waste_inkling_final_head_profile(
            ctypes.byref(cfg), ctypes.byref(w),
            ctypes.byref(backend) if backend else None,
            xa, selection, n_rows, out, count,
            normalized, row_scratch, profile,
            ctypes.byref(trace) if trace else None,
        )
        return rc, list(out[:max(n_rows, 0)]), list(normalized)

    def test_f32_profile_matches_python_reference_bit_for_bit(self):
        cfg = self.build_config()
        x, final_norm, unembedding = self.make_tables(cfg)
        w, keep = self.bind(cfg, final_norm, unembedding)
        rc, logits, scaled = self.call(cfg, w, x, None, cfg.unpadded_vocab,
                                       PROFILE_F32)
        self.assertEqual(rc, 0)
        _, want_scaled, want = reference_head(
            x, final_norm, unembedding, range(cfg.unpadded_vocab),
            cfg.rms_eps, cfg.logits_width_multiplier, PROFILE_F32)
        self.assertEqual(scaled, want_scaled)
        self.assertEqual(logits, want)
        self.assertIsNotNone(keep)

    def test_bf16_profile_matches_python_reference_bit_for_bit(self):
        cfg = self.build_config()
        x, final_norm, unembedding = self.make_tables(cfg, quantize=True)
        w, keep = self.bind(cfg, final_norm)
        rows = [0, 5, 17, 47]
        seen = {}

        def matvec(_ctx, layer, kind, index, vec, out, n_rows, cols):
            seen["layer"] = layer
            seen["kind"] = kind
            seen["index"] = index
            seen["rows"] = n_rows
            seen["cols"] = cols
            seen["input"] = [vec[i] for i in range(cols)]
            for j in range(n_rows):
                total = 0.0
                for c in range(cols):
                    total += float(unembedding[rows[j]][c]) * float(vec[c])
                out[j] = bf16(f32(total))
            return 0

        callback = MatvecCallback(matvec)
        backend = MatrixBackend(matvec=callback, ctx=None)
        rc, logits, scaled = self.call(cfg, w, x, rows, len(rows),
                                       PROFILE_BF16, backend=backend)
        self.assertEqual(rc, 0)
        _, want_scaled, _ = reference_head(
            x, final_norm, unembedding, rows, cfg.rms_eps,
            cfg.logits_width_multiplier, PROFILE_BF16)
        self.assertEqual(scaled, want_scaled)
        self.assertEqual(seen["input"], want_scaled)
        self.assertEqual((seen["layer"], seen["kind"], seen["index"]),
                         (-1, KIND_UNEMBED, 0))
        self.assertEqual((seen["rows"], seen["cols"]), (len(rows), cfg.hidden))
        # Every completed value stays bfloat16-representable.
        for value in scaled + logits:
            self.assertEqual(value, bf16(value))
        self.assertIsNotNone(keep)

    def test_width_completion_is_a_division_not_a_reciprocal_multiply(self):
        """Under BF16 the two differ; the official forward divides."""
        cfg = self.build_config(multiplier=3.0)
        x, final_norm, _ = self.make_tables(cfg, seed=23, quantize=True)
        w, keep = self.bind(cfg, final_norm)
        rows = [0]

        def matvec(_ctx, _layer, _kind, _index, _vec, out, n_rows, _cols):
            for j in range(n_rows):
                out[j] = 0.0
            return 0

        callback = MatvecCallback(matvec)
        backend = MatrixBackend(matvec=callback, ctx=None)
        rc, _, scaled = self.call(cfg, w, x, rows, 1, PROFILE_BF16,
                                  backend=backend)
        self.assertEqual(rc, 0)
        normalized, want, _ = reference_head(
            x, final_norm, [[0.0] * cfg.hidden], rows, cfg.rms_eps,
            cfg.logits_width_multiplier, PROFILE_BF16)
        self.assertEqual(scaled, want)
        inv = bf16(f32(1.0 / cfg.logits_width_multiplier))
        reciprocal = [bf16(f32(value * inv)) for value in normalized]
        self.assertNotEqual(reciprocal, want)
        self.assertIsNotNone(keep)

    def test_row_callback_matches_resident_table(self):
        cfg = self.build_config()
        x, final_norm, unembedding = self.make_tables(cfg, seed=5)
        resident, keep = self.bind(cfg, final_norm, unembedding)
        rc, want, _ = self.call(cfg, resident, x, None, cfg.unpadded_vocab,
                                PROFILE_F32)
        self.assertEqual(rc, 0)

        requested = []

        def row_get(_ctx, row, cols, out):
            if row < 0 or row >= len(unembedding) or cols != cfg.hidden:
                return -1
            requested.append(row)
            for c in range(cols):
                out[c] = unembedding[row][c]
            return 0

        callback = RowCallback(row_get)
        streamed, keep2 = self.bind(cfg, final_norm)
        streamed.unembedding_get = callback
        rc, got, _ = self.call(cfg, streamed, x, None, cfg.unpadded_vocab,
                               PROFILE_F32)
        self.assertEqual(rc, 0)
        self.assertEqual(got, want)
        self.assertEqual(requested, list(range(cfg.unpadded_vocab)))
        self.assertIsNotNone(keep and keep2)

    def test_bounded_selection_equals_the_same_rows_of_the_full_head(self):
        cfg = self.build_config()
        x, final_norm, unembedding = self.make_tables(cfg, seed=9)
        w, keep = self.bind(cfg, final_norm, unembedding)
        rc, full, _ = self.call(cfg, w, x, None, cfg.unpadded_vocab,
                                PROFILE_F32)
        self.assertEqual(rc, 0)
        rows = [47, 0, 12, 12, 31]
        rc, bounded, _ = self.call(cfg, w, x, rows, len(rows), PROFILE_F32)
        self.assertEqual(rc, 0)
        self.assertEqual(bounded, [full[row] for row in rows])
        self.assertIsNotNone(keep)

    def test_trace_reports_every_completion_point(self):
        cfg = self.build_config()
        x, final_norm, unembedding = self.make_tables(cfg, seed=13)
        w, keep = self.bind(cfg, final_norm, unembedding)
        points = []

        def emit_float(_ctx, layer, point, data, count):
            points.append((layer, point.decode(),
                           [data[i] for i in range(count)]))
            return 0

        emit = TraceFloatCallback(emit_float)
        trace = Trace(emit_float=emit, emit_int=TraceIntCallback(0), ctx=None)
        rows = [3, 8]
        rc, logits, _ = self.call(cfg, w, x, rows, len(rows), PROFILE_F32,
                                  trace=trace)
        self.assertEqual(rc, 0)
        self.assertEqual([(layer, name) for layer, name, _ in points],
                         [(-1, "final_norm"), (-1, "final_norm_scaled"),
                          (-1, "logits")])
        want_normalized, want_scaled, _ = reference_head(
            x, final_norm, unembedding, rows, cfg.rms_eps,
            cfg.logits_width_multiplier, PROFILE_F32)
        self.assertEqual(points[0][2], want_normalized)
        self.assertEqual(points[1][2], want_scaled)
        # The logits trace is bounded by the selection, not by the vocabulary.
        self.assertEqual(points[2][2], logits)
        self.assertIsNotNone(keep)

    def test_trace_failure_stops_the_head(self):
        cfg = self.build_config()
        x, final_norm, unembedding = self.make_tables(cfg, seed=17)
        w, keep = self.bind(cfg, final_norm, unembedding)

        def emit_float(_ctx, _layer, _point, _data, _count):
            return -1

        emit = TraceFloatCallback(emit_float)
        trace = Trace(emit_float=emit, emit_int=TraceIntCallback(0), ctx=None)
        rc, _, _ = self.call(cfg, w, x, None, cfg.unpadded_vocab, PROFILE_F32,
                             trace=trace)
        self.assertNotEqual(rc, 0)
        self.assertIsNotNone(keep)

    def test_bf16_profile_refuses_to_use_the_resident_scalar_path(self):
        """BF16 evidence is kernel-sensitive: no backend, no result."""
        cfg = self.build_config()
        x, final_norm, unembedding = self.make_tables(cfg, quantize=True)
        w, keep = self.bind(cfg, final_norm, unembedding)
        rc, _, _ = self.call(cfg, w, x, None, cfg.unpadded_vocab, PROFILE_BF16)
        self.assertNotEqual(rc, 0)
        empty = MatrixBackend(matvec=MatvecCallback(0), ctx=None)
        rc, _, _ = self.call(cfg, w, x, None, cfg.unpadded_vocab,
                             PROFILE_BF16, backend=empty)
        self.assertNotEqual(rc, 0)
        self.assertIsNotNone(keep)

    def test_backend_failure_propagates(self):
        cfg = self.build_config()
        x, final_norm, _ = self.make_tables(cfg, quantize=True)
        w, keep = self.bind(cfg, final_norm)

        def matvec(_ctx, _layer, _kind, _index, _vec, _out, _rows, _cols):
            return -1

        callback = MatvecCallback(matvec)
        backend = MatrixBackend(matvec=callback, ctx=None)
        rc, _, _ = self.call(cfg, w, x, [0], 1, PROFILE_BF16, backend=backend)
        self.assertNotEqual(rc, 0)
        self.assertIsNotNone(keep)

    def test_invalid_arguments_fail_closed(self):
        cfg = self.build_config()
        x, final_norm, unembedding = self.make_tables(cfg, seed=3)
        w, keep = self.bind(cfg, final_norm, unembedding)
        vocab = cfg.unpadded_vocab

        # A selection reaching past the bound table, and a negative row.
        for rows in ([vocab], [-1], [0, vocab + 3]):
            rc, _, _ = self.call(cfg, w, x, rows, len(rows), PROFILE_F32)
            self.assertNotEqual(rc, 0, rows)
        # An empty selection is not a head.
        rc, _, _ = self.call(cfg, w, x, None, 0, PROFILE_F32)
        self.assertNotEqual(rc, 0)
        # More rows than the caller's logits buffer admits.
        rc, _, _ = self.call(cfg, w, x, None, vocab, PROFILE_F32,
                             logits_count=vocab - 1)
        self.assertNotEqual(rc, 0)
        # More rows than the resident table holds.
        short, keep2 = self.bind(cfg, final_norm, unembedding[: vocab - 4])
        rc, _, _ = self.call(cfg, short, x, None, vocab, PROFILE_F32)
        self.assertNotEqual(rc, 0)
        # No final norm, no unembedding source, and an unknown profile.
        headless, keep3 = self.bind(cfg, final_norm, unembedding)
        headless.final_norm = None
        rc, _, _ = self.call(cfg, headless, x, None, vocab, PROFILE_F32)
        self.assertNotEqual(rc, 0)
        unbound, keep4 = self.bind(cfg, final_norm)
        rc, _, _ = self.call(cfg, unbound, x, None, vocab, PROFILE_F32)
        self.assertNotEqual(rc, 0)
        rc, _, _ = self.call(cfg, w, x, None, vocab, 2)
        self.assertNotEqual(rc, 0)
        # A non-positive width multiplier cannot complete anything. The config
        # builder already refuses one, so reach past it to prove the head does
        # not divide by whatever it is handed.
        zero = self.build_config()
        zero.logits_width_multiplier = 0.0
        rc, _, _ = self.call(zero, w, x, None, vocab, PROFILE_F32)
        self.assertNotEqual(rc, 0)
        self.assertIsNotNone(keep and keep2 and keep3 and keep4)


if __name__ == "__main__":
    unittest.main()
