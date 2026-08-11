#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Structural checks for the checked-in final-head evidence runner.

These run in the torch job before any official bytes are downloaded, and they
guard the properties a green report would otherwise be able to hide: that the
runner compiles production sources rather than a rewritten copy, that it
reaches the vocabulary projection through the matrix backend only, that its
report cannot silently drop the hidden-state provenance, and that importing it
rebinds nothing in the shared evidence modules.
"""
from __future__ import annotations

import ctypes
import importlib
import inspect
import unittest

import torch

import measure_inkling_dense_bf16_profile as dense
import run_inkling_checked_bf16_final_head as head


class FinalHeadRunnerStructureTest(unittest.TestCase):
    def test_import_does_not_rebind_shared_execution(self):
        before = (dense.metrics, dense.run_candidate, dense.DenseMatrixProvider)
        importlib.reload(head)
        after = (dense.metrics, dense.run_candidate, dense.DenseMatrixProvider)
        self.assertEqual(after, before)

    def test_compiles_production_sources_unchanged(self):
        self.assertIn("inkling_model.c", head.HEAD_SOURCES)
        self.assertIn("-Werror", inspect.getsource(head.build_head_library))
        whole = inspect.getsource(head)
        for banned in ("read_text().replace", "rewrite", "patched_source"):
            self.assertNotIn(banned, whole)

    def test_backend_kind_matches_the_c_enum_position(self):
        # WASTE_IK_MAT_UNEMBED is appended after the three routed kinds.
        self.assertEqual(head.MAT_UNEMBED, 15)
        self.assertEqual(head.BF16_REFERENCE, 1)

    def call_backend(self, backend, layer, kind, rows, cols):
        vector = (ctypes.c_float * cols)(*[1.0] * cols)
        out = (ctypes.c_float * max(rows, 1))()
        backend.error = None
        return backend._call(None, layer, kind, 0,
                             ctypes.cast(vector, head.FP),
                             ctypes.cast(out, head.FP), rows, cols)

    def test_backend_serves_only_the_head_projection(self):
        backend = head.UnembedBackend(torch.zeros(2, 4))
        # The real request succeeds and is recorded.
        self.assertEqual(self.call_backend(backend, -1, head.MAT_UNEMBED, 2, 4), 0)
        self.assertEqual(len(backend.calls), 1)
        self.assertEqual(backend.calls[0]["mode"], "native-bf16-linear")
        # A layer-scoped request, another matrix kind, and geometry that
        # disagrees with the selected rows all fail closed.
        for layer, kind, rows, cols in ((0, head.MAT_UNEMBED, 2, 4),
                                        (-1, 0, 2, 4),
                                        (-1, head.MAT_UNEMBED, 3, 4),
                                        (-1, head.MAT_UNEMBED, 2, 5)):
            self.assertNotEqual(
                self.call_backend(backend, layer, kind, rows, cols),
                0, (layer, kind, rows, cols))
            self.assertIsNotNone(backend.error)

    def test_required_arguments_include_hidden_state_provenance(self):
        for argv in (
            ["--fixture", "f", "--model-config", "c", "--c-config", "cc",
             "--hidden-state", "h"],
            ["--fixture", "f", "--model-config", "c", "--c-config", "cc",
             "--hidden-state-origin", "somewhere"],
        ):
            with self.assertRaises(SystemExit) as ctx:
                head.main(argv)
            self.assertEqual(ctx.exception.code, 2)

    def test_report_never_claims_final_model_logits(self):
        whole = inspect.getsource(head)
        self.assertIn('"final_model_logits": False', whole)
        self.assertIn("hidden_state_origin", whole)
        self.assertEqual(head.POINTS,
                         ("final_norm", "final_norm_scaled", "logits"))


if __name__ == "__main__":
    unittest.main()
