#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "tests"))

from inkling_vq import (
    EXPERT_HEADER,
    VQError,
    VQSpec,
    dequantize_record,
    parse_record_header,
    quantize_expert_banks,
    quantize_layer,
    read_codebooks,
    record_geometry,
    verify_layer_outputs,
)
from inkling_weights import ProviderRawInklingWeights
from test_inkling_weights import make_weight_checkpoint


def plan():
    return {"source_dialect": "provider_raw", "official_small": False}


class InklingVQTest(unittest.TestCase):
    def setUp(self):
        self.src_td, self.src, self.expected = make_weight_checkpoint()
        self.out_td = tempfile.TemporaryDirectory()
        self.out = Path(self.out_td.name)
        self.addCleanup(self.src_td.cleanup)
        self.addCleanup(self.out_td.cleanup)
        self.spec = VQSpec(stages=2, entries=16)

    def convert(self):
        source = ProviderRawInklingWeights(self.src)
        return quantize_layer(
            source,
            self.out,
            1,
            spec=self.spec,
            codebook_sample=4,
            train_vectors=64,
            kmeans_iterations=4,
            assign_chunk=32,
            verify=True,
        )

    def test_wexp_codebook_round_trip_and_quality(self):
        meta = self.convert()
        bank = self.out / meta["file"]
        codebooks = self.out / meta["codebooks_file"]
        self.assertEqual(bank.stat().st_size, meta["record_bytes"] * 4)
        self.assertEqual(meta["record_bytes"] % 4096, 0)
        self.assertEqual(meta["codebook_base"], 0)
        verify_layer_outputs(
            bank,
            codebooks,
            layer=1,
            experts=4,
            hidden=16,
            intermediate=8,
            spec=self.spec,
            codebook_base=0,
        )
        books = read_codebooks(codebooks, self.spec)
        with bank.open("rb") as f:
            for expert in range(4):
                record = f.read(meta["record_bytes"])
                header = parse_record_header(record)
                self.assertEqual((header.layer, header.expert), (1, expert))
                got = dequantize_record(
                    record,
                    books,
                    ((8, 16), (8, 16), (16, 8)),
                    self.spec,
                    expected_layer=1,
                    expected_expert=expert,
                )
                for value, reference in zip(
                    (got.gate, got.up, got.down),
                    (
                        self.expected["routed_gate"][expert],
                        self.expected["routed_up"][expert],
                        self.expected["routed_down"][expert],
                    ),
                ):
                    relative = float((value - reference).norm() / reference.norm().clamp_min(1e-8))
                    self.assertLess(relative, 0.2)

    def test_deterministic_conversion(self):
        first = self.convert()
        raw_bank = (self.out / first["file"]).read_bytes()
        raw_books = (self.out / first["codebooks_file"]).read_bytes()
        second_td = tempfile.TemporaryDirectory()
        self.addCleanup(second_td.cleanup)
        source = ProviderRawInklingWeights(self.src)
        second = quantize_layer(
            source,
            Path(second_td.name),
            1,
            spec=self.spec,
            codebook_sample=4,
            train_vectors=64,
            kmeans_iterations=4,
            assign_chunk=32,
        )
        self.assertEqual(raw_bank, (Path(second_td.name) / second["file"]).read_bytes())
        self.assertEqual(raw_books, (Path(second_td.name) / second["codebooks_file"]).read_bytes())

    def test_crc_corruption_fails(self):
        meta = self.convert()
        bank = self.out / meta["file"]
        with bank.open("r+b") as f:
            f.seek(EXPERT_HEADER.size + 3)
            byte = f.read(1)
            f.seek(EXPERT_HEADER.size + 3)
            f.write(bytes([byte[0] ^ 1]))
        with self.assertRaisesRegex(VQError, "CRC"):
            verify_layer_outputs(
                bank,
                self.out / meta["codebooks_file"],
                layer=1,
                experts=4,
                hidden=16,
                intermediate=8,
                spec=self.spec,
                codebook_base=0,
            )

    def test_source_bound_resume(self):
        with patch("inkling_vq.build_plan", return_value=plan()):
            first = quantize_expert_banks(
                self.src,
                self.out,
                layers=[1],
                spec=self.spec,
                codebook_sample=4,
                train_vectors=64,
                kmeans_iterations=2,
                assign_chunk=32,
            )
        self.assertFalse(first["layers"][0]["reused"])
        with patch("inkling_vq.build_plan", return_value=plan()), patch(
            "inkling_vq.quantize_layer", side_effect=AssertionError("rewritten")
        ):
            second = quantize_expert_banks(
                self.src,
                self.out,
                layers=[1],
                spec=self.spec,
                codebook_sample=4,
                train_vectors=64,
                kmeans_iterations=2,
                assign_chunk=32,
            )
        self.assertTrue(second["layers"][0]["reused"])
        stage = json.loads((self.out / "vq-stage.json").read_text())
        self.assertFalse(stage["manifest_published"])

        # Training identity is part of resume safety. Changing the Lloyd
        # iteration count must rebuild instead of silently reusing a bank.
        with patch("inkling_vq.build_plan", return_value=plan()), patch(
            "inkling_vq.quantize_layer", wraps=quantize_layer
        ) as rebuilt:
            third = quantize_expert_banks(
                self.src,
                self.out,
                layers=[1],
                spec=self.spec,
                codebook_sample=4,
                train_vectors=64,
                kmeans_iterations=3,
                assign_chunk=32,
            )
        self.assertFalse(third["layers"][0]["reused"])
        self.assertEqual(rebuilt.call_count, 1)

    def test_official_small_vq3_geometry(self):
        geometry = record_geometry(4096, 2048, VQSpec())
        self.assertEqual(geometry["record_bytes"], 9_457_664)
        self.assertEqual(geometry["record_bytes"] * 256, 2_421_161_984)
        self.assertEqual(geometry["record_bytes"] * 256 * 40, 96_846_479_360)

    def test_invalid_training_geometry_fails_closed(self):
        source = ProviderRawInklingWeights(self.src)
        with self.assertRaisesRegex(VQError, "train_vectors"):
            quantize_layer(
                source,
                self.out,
                1,
                spec=VQSpec(stages=2, entries=32),
                codebook_sample=1,
                train_vectors=16,
            )


if __name__ == "__main__":
    unittest.main()
