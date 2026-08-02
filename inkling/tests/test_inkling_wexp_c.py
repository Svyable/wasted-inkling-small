#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
import ctypes
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "tests"))

from inkling_vq import VQSpec, dequantize_record, quantize_layer, read_codebooks
from inkling_weights import ProviderRawInklingWeights
from test_inkling_weights import make_weight_checkpoint

FP = ctypes.POINTER(ctypes.c_float)


class Expert(ctypes.Structure):
    _fields_ = [("gate", FP), ("up", FP), ("down", FP)]


class Bank(ctypes.Structure):
    _fields_ = [
        ("bank_fd", ctypes.c_int),
        ("layer", ctypes.c_int),
        ("experts", ctypes.c_int),
        ("hidden", ctypes.c_int),
        ("intermediate", ctypes.c_int),
        ("stages", ctypes.c_int),
        ("entries", ctypes.c_int),
        ("vec_dim", ctypes.c_int),
        ("index_block", ctypes.c_int),
        ("fmt", ctypes.c_int),
        ("codebook_base", ctypes.c_int),
        ("record_bytes", ctypes.c_uint64),
        ("verify_crc", ctypes.c_int),
        ("codebooks", FP),
        ("gate", FP),
        ("up", FP),
        ("down", FP),
        ("record", ctypes.POINTER(ctypes.c_ubyte)),
        ("matrix_floats", ctypes.c_size_t),
        ("record_capacity", ctypes.c_size_t),
    ]


class InklingWexpCTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cc = shutil.which("cc")
        if not cc:
            raise unittest.SkipTest("C compiler unavailable")
        cls.build_td = tempfile.TemporaryDirectory()
        so = Path(cls.build_td.name) / "libwexp.so"
        subprocess.run(
            [
                cc,
                "-std=c11",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-shared",
                "-fPIC",
                f"-I{REPO / 'src'}",
                str(REPO / "src" / "inkling_wexp.c"),
                "-o",
                str(so),
            ],
            check=True,
            capture_output=True,
        )
        cls.lib = ctypes.CDLL(str(so))
        cls.lib.waste_inkling_wexp_bank_open.argtypes = [
            ctypes.POINTER(Bank), ctypes.c_char_p, ctypes.c_char_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int,
        ]
        cls.lib.waste_inkling_wexp_bank_open.restype = ctypes.c_int
        cls.lib.waste_inkling_wexp_expert_get.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.POINTER(Expert)
        ]
        cls.lib.waste_inkling_wexp_expert_get.restype = ctypes.c_int
        cls.lib.waste_inkling_wexp_bank_close.argtypes = [ctypes.POINTER(Bank)]

    @classmethod
    def tearDownClass(cls):
        cls.build_td.cleanup()

    def setUp(self):
        self.src_td, self.src, _ = make_weight_checkpoint()
        self.out_td = tempfile.TemporaryDirectory()
        self.addCleanup(self.src_td.cleanup)
        self.addCleanup(self.out_td.cleanup)
        self.out = Path(self.out_td.name)
        self.spec = VQSpec(stages=2, entries=16)
        self.meta = quantize_layer(
            ProviderRawInklingWeights(self.src),
            self.out,
            1,
            spec=self.spec,
            codebook_sample=4,
            train_vectors=64,
            kmeans_iterations=3,
            assign_chunk=32,
        )

    def test_c_dequantization_matches_python(self):
        bank_path = self.out / self.meta["file"]
        book_path = self.out / self.meta["codebooks_file"]
        bank = Bank()
        rc = self.lib.waste_inkling_wexp_bank_open(
            ctypes.byref(bank),
            str(bank_path).encode(),
            str(book_path).encode(),
            1, 4, 16, 8, 2, 16, 8, 64, 0, 1,
        )
        self.assertEqual(rc, 0)
        try:
            expert = Expert()
            self.assertEqual(
                self.lib.waste_inkling_wexp_expert_get(
                    ctypes.byref(bank), 1, 2, ctypes.byref(expert)
                ),
                0,
            )
            record_bytes = self.meta["record_bytes"]
            with bank_path.open("rb") as f:
                f.seek(2 * record_bytes)
                raw = f.read(record_bytes)
            reference = dequantize_record(
                raw,
                read_codebooks(book_path, self.spec),
                ((8, 16), (8, 16), (16, 8)),
                self.spec,
                expected_layer=1,
                expected_expert=2,
            )
            for ptr, shape, expected in (
                (expert.gate, (8, 16), reference.gate),
                (expert.up, (8, 16), reference.up),
                (expert.down, (16, 8), reference.down),
            ):
                got = torch.tensor([ptr[i] for i in range(shape[0] * shape[1])]).reshape(shape)
                torch.testing.assert_close(got, expected, rtol=1e-6, atol=1e-6)
        finally:
            self.lib.waste_inkling_wexp_bank_close(ctypes.byref(bank))

    def test_c_reader_rejects_crc_corruption(self):
        bank_path = self.out / self.meta["file"]
        book_path = self.out / self.meta["codebooks_file"]
        with bank_path.open("r+b") as f:
            f.seek(48 + 5)
            byte = f.read(1)
            f.seek(48 + 5)
            f.write(bytes([byte[0] ^ 1]))
        bank = Bank()
        self.assertEqual(
            self.lib.waste_inkling_wexp_bank_open(
                ctypes.byref(bank), str(bank_path).encode(), str(book_path).encode(),
                1, 4, 16, 8, 2, 16, 8, 64, 0, 1,
            ),
            0,
        )
        try:
            expert = Expert()
            self.assertEqual(
                self.lib.waste_inkling_wexp_expert_get(
                    ctypes.byref(bank), 1, 0, ctypes.byref(expert)
                ),
                -1,
            )
        finally:
            self.lib.waste_inkling_wexp_bank_close(ctypes.byref(bank))


if __name__ == "__main__":
    unittest.main()
