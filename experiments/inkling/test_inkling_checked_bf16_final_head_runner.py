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

    def test_end_to_end_on_a_synthetic_head_fixture(self):
        """The whole runner, on invented weights and the official RMSNorm.

        The official-weight gate needs network and a CRC-verified fixture; this
        needs neither, so a wiring break -- the ctypes signature, the trace
        names, the row selection, the BF16 completion of the supplied vector --
        fails here first and in seconds.
        """
        import argparse
        import hashlib
        import json
        import tempfile
        import zlib
        from pathlib import Path

        repo = Path(__file__).resolve().parents[2]
        config_raw = (repo / "inkling" / "tests" / "data"
                      / "inkling-small-config.json").read_bytes()
        hidden = int(json.loads(config_raw)["text_config"]["hidden_size"])
        rows = [0, 5, 200057]

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            fixture = root / "fixture"
            fixture.mkdir()
            torch.manual_seed(3)
            entries = []

            def add(name, tensor, axis0=None):
                raw = bytes(tensor.to(torch.bfloat16).contiguous()
                            .view(torch.uint8).reshape(-1).tolist())
                rel = f"{name.replace('.', '_')}_{axis0}.bin"
                (fixture / rel).write_bytes(raw)
                item = {"name": name, "dtype": "BF16",
                        "kind": "tensor" if axis0 is None else "axis0-slice",
                        "shape": list(tensor.shape), "bytes": len(raw),
                        "crc32": zlib.crc32(raw) & 0xFFFFFFFF, "path": rel}
                if axis0 is not None:
                    item["axis0"] = axis0
                entries.append(item)

            add("model.llm.norm.weight", torch.rand(hidden) * 1.5 + 0.2)
            for row in rows:
                add("model.llm.unembed.weight", torch.randn(hidden) * 0.05, row)
            (fixture / "fixture.json").write_text(json.dumps({
                "format": "inkling-parity-fixture", "version": 1,
                "model_id": "synthetic", "layers": [], "experts": {},
                "vocab_rows": rows,
                "source": {"config_sha256": hashlib.sha256(config_raw).hexdigest(),
                           "index_sha256": "b" * 64, "revision": "synthetic"},
                "total_payload_bytes": sum(e["bytes"] for e in entries),
                "entries": entries,
            }))
            (root / "config.json").write_bytes(config_raw)
            (root / "inputs.json").write_text(
                json.dumps([(torch.randn(hidden) * 0.7).tolist()]))

            import inkling_c_config
            c_config = inkling_c_config.normalized_c_layer_config(
                json.loads(config_raw))
            (root / "c-config.json").write_text(json.dumps(c_config))

            result = head.run(argparse.Namespace(
                fixture=str(fixture), model_config=str(root / "config.json"),
                c_config=str(root / "c-config.json"),
                hidden_state=str(root / "inputs.json"),
                hidden_state_origin="synthetic", position=0,
                vocab_rows="", workdir=str(root), out=None))

        self.assertEqual(result["decision"]["first_boundary"], None)
        self.assertEqual(result["decision"]["classification"],
                         "checked_in_bf16_final_head_exact")
        self.assertEqual(result["selection"]["vocab_rows"], rows)
        self.assertEqual(len(result["execution"]["backend_calls"]), 1)
        self.assertIs(result["claims"]["final_model_logits"], False)
        for point in head.POINTS:
            self.assertEqual(result["comparison"][point]["raw_exact_fraction"],
                             1.0, point)

    def test_report_never_claims_final_model_logits(self):
        whole = inspect.getsource(head)
        self.assertIn('"final_model_logits": False', whole)
        self.assertIn("hidden_state_origin", whole)
        self.assertEqual(head.POINTS,
                         ("final_norm", "final_norm_scaled", "logits"))


if __name__ == "__main__":
    unittest.main()
