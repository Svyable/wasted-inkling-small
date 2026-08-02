#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "tests"))

from inkling_runtime_stage import (
    BANK_ENTRY_BYTES, HEADER_BYTES, LAYER_ENTRY_BYTES, MAGIC,
    RuntimeStageError, TENSOR_ENTRY_BYTES, VERSION, VERSION_VQ,
    VERSION_QTRUNK_VQ, TENSOR_FLAG_QUANTIZED,
    publish_runtime_stage, publish_runtime_vq_stage,
    publish_runtime_qtrunk_vq_stage,
)
from inkling_stage import stage_expert_banks
from inkling_trunk import stage_trunk
from inkling_qtrunk import quantize_trunk
from inkling_vq import VQSpec, quantize_expert_banks
from test_inkling_trunk import make_full_checkpoint


class RuntimeStageTest(unittest.TestCase):
    def build(self):
        td, src, _ = make_full_checkpoint()
        self.addCleanup(td.cleanup)
        out_td = tempfile.TemporaryDirectory()
        self.addCleanup(out_td.cleanup)
        out = Path(out_td.name)
        stage_trunk(src, out, chunk_bytes=256)
        stage_expert_banks(src, out)
        return src, out

    def test_publish_fixed_private_index_after_complete_validation(self):
        src, out = self.build()
        meta = publish_runtime_stage(src, out, require_official=False)
        self.assertTrue(meta["status"]["private_runtime_stage_complete"])
        self.assertFalse(meta["status"]["waste_manifest_written"])
        self.assertFalse((out / "manifest.json").exists())
        raw = (out / "runtime-stage.bin").read_bytes()
        magic, version, header = struct.unpack_from("<IHH", raw)
        self.assertEqual((magic, version, header), (MAGIC, VERSION, HEADER_BYTES))
        layers, tensors, banks = struct.unpack_from("<III", raw, 24)
        self.assertEqual((layers, banks), (3, 2))
        self.assertEqual(
            len(raw),
            HEADER_BYTES + layers * LAYER_ENTRY_BYTES
            + tensors * TENSOR_ENTRY_BYTES + banks * BANK_ENTRY_BYTES,
        )
        # Canonical short-convolution tensors have the singleton Conv1d axis removed.
        trunk = json.loads((out / "trunk-stage.json").read_text())
        conv = next(item for item in trunk["tensors"] if item["target"].endswith("k_sconv"))
        self.assertEqual(conv["shape"], [8, 4])

    def test_publish_final_vq_private_index_without_bf16_expert_stage(self):
        td, src, _ = make_full_checkpoint()
        self.addCleanup(td.cleanup)
        out_td = tempfile.TemporaryDirectory()
        self.addCleanup(out_td.cleanup)
        out = Path(out_td.name)
        stage_trunk(src, out, chunk_bytes=256)
        quantize_expert_banks(
            src, out, spec=VQSpec(stages=2, vec_dim=4, entries=4, index_block=4),
            codebook_sample=2, train_vectors=32, kmeans_iterations=1,
            assign_chunk=64, verify=True,
        )
        self.assertFalse((out / "stage.json").exists())
        meta = publish_runtime_vq_stage(src, out, require_official=False)
        self.assertEqual(meta["schema"], "waste.inkling-private-runtime-stage.v2")
        self.assertEqual(meta["expert_format"], "WEXP/VQ")
        self.assertFalse(meta["status"]["bf16_expert_stage_required"])
        raw = (out / "runtime-stage.bin").read_bytes()
        magic, version, header = struct.unpack_from("<IHH", raw)
        self.assertEqual((magic, version, header), (MAGIC, VERSION_VQ, HEADER_BYTES))
        self.assertEqual(tuple(raw[248:252]), (1, 2, 4, 4))
        self.assertEqual(struct.unpack_from("<H", raw, 252)[0], 4)
        layers, tensors, banks = struct.unpack_from("<III", raw, 24)
        bank0 = HEADER_BYTES + layers * LAYER_ENTRY_BYTES + tensors * TENSOR_ENTRY_BYTES
        self.assertEqual(struct.unpack_from("<I", raw, bank0 + 112)[0], 0)
        self.assertEqual(struct.unpack_from("<I", raw, bank0 + BANK_ENTRY_BYTES + 112)[0], 6)

    def test_publish_qtrunk_v3_index(self):
        td, src, _ = make_full_checkpoint()
        self.addCleanup(td.cleanup)
        out_td = tempfile.TemporaryDirectory()
        self.addCleanup(out_td.cleanup)
        out = Path(out_td.name)
        stage_trunk(src, out, chunk_bytes=256)
        quantize_trunk(out, bits=4, group=4, chunk_rows=2)
        quantize_expert_banks(
            src, out, spec=VQSpec(stages=2, vec_dim=4, entries=4, index_block=4),
            codebook_sample=2, train_vectors=32, kmeans_iterations=1,
            assign_chunk=64, verify=True,
        )
        meta = publish_runtime_qtrunk_vq_stage(src, out, require_official=False)
        self.assertEqual(meta["schema"], "waste.inkling-private-runtime-stage.v3")
        self.assertGreater(meta["counts"]["quantized_tensors"], 0)
        raw = (out / "runtime-stage.bin").read_bytes()
        self.assertEqual(struct.unpack_from("<H", raw, 4)[0], VERSION_QTRUNK_VQ)
        layers, tensors, _ = struct.unpack_from("<III", raw, 24)
        off = HEADER_BYTES + layers * LAYER_ENTRY_BYTES
        seen_quantized = False
        for i in range(tensors):
            e = off + i * TENSOR_ENTRY_BYTES
            flags = struct.unpack_from("<H", raw, e + 210)[0]
            if flags & TENSOR_FLAG_QUANTIZED:
                seen_quantized = True
                self.assertIn(raw[e + 208], (4, 5))
        self.assertTrue(seen_quantized)
        self.assertFalse((out / "manifest.json").exists())

    def test_qtrunk_source_identity_prevents_stale_publication(self):
        td, src, _ = make_full_checkpoint()
        self.addCleanup(td.cleanup)
        out_td = tempfile.TemporaryDirectory()
        self.addCleanup(out_td.cleanup)
        out = Path(out_td.name)
        stage_trunk(src, out, chunk_bytes=256)
        quantize_trunk(out, bits=4, group=4, chunk_rows=2)
        quantize_expert_banks(
            src, out, spec=VQSpec(stages=2, vec_dim=4, entries=4, index_block=4),
            codebook_sample=2, train_vectors=32, kmeans_iterations=1,
            assign_chunk=64, verify=True,
        )
        trunk = json.loads((out / "trunk-stage.json").read_text())
        trunk["tampered"] = True
        (out / "trunk-stage.json").write_text(json.dumps(trunk))
        with self.assertRaisesRegex(RuntimeStageError, "not built from this"):
            publish_runtime_qtrunk_vq_stage(src, out, require_official=False)
        self.assertFalse((out / "runtime-stage.bin").exists())

    def test_missing_or_corrupt_artifact_prevents_publication(self):
        src, out = self.build()
        trunk = json.loads((out / "trunk-stage.json").read_text())
        victim = out / "trunk-stage" / trunk["tensors"][0]["file"]
        victim.unlink()
        with self.assertRaises(RuntimeStageError):
            publish_runtime_stage(src, out, require_official=False)
        self.assertFalse((out / "runtime-stage.bin").exists())

    def test_official_mode_rejects_generic_fixture(self):
        src, out = self.build()
        with self.assertRaisesRegex(RuntimeStageError, "official Inkling-Small"):
            publish_runtime_stage(src, out, require_official=True)


if __name__ == "__main__":
    unittest.main()
