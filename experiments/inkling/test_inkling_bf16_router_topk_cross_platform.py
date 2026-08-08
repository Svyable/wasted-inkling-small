#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch

from audit_inkling_bf16_router_choice_archive import (
    _canonical_bytes,
    audit_layer,
    classify_platform,
    verify_archive_hash,
)
from export_inkling_bf16_router_choice_archive import choice_bits


class RouterTopkCrossPlatformTest(unittest.TestCase):
    def test_choice_bits_round_trip_exactly(self):
        choice = torch.tensor(
            [[1.71875, -0.5, 3.25], [0.0, 1.0, 1.0]],
            dtype=torch.bfloat16,
        )
        bits = choice_bits(choice)
        replay = torch.tensor(bits, dtype=torch.int16).view(torch.bfloat16)
        self.assertTrue(torch.equal(choice, replay))

    def test_archive_hash_excludes_only_hash_field(self):
        archive = {
            "format": "inkling-bfloat16-router-choice-bits",
            "version": 1,
            "layers": {"2": {"x": 1}},
        }
        digest = hashlib.sha256(_canonical_bytes(archive)).hexdigest()
        archive["archive_sha256"] = digest
        self.assertEqual(verify_archive_hash(archive), digest)
        archive["layers"]["2"]["x"] = 2
        with self.assertRaisesRegex(Exception, "choice archive SHA"):
            verify_archive_hash(archive)

    def test_audit_reports_valid_ambiguous_set_change(self):
        choice = torch.tensor([[9.0, 8.0, 7.0, 7.0, 7.0]], dtype=torch.bfloat16)
        current = torch.topk(choice, 3, dim=-1, sorted=False).indices[0].tolist()
        current_set = set(int(value) for value in current)
        tied = {2, 3, 4}
        selected_tie = next(iter(current_set & tied))
        alternative_tie = next(iter(tied - {selected_tie}))
        expected = [0, 1, alternative_tie]
        data = {
            "top_k": 3,
            "n_routed_experts": 5,
            "choice_bits_int16": choice.view(torch.int16).tolist(),
            "official_indices": [expected],
        }
        result = audit_layer("2", data, repeat_topk=4)
        self.assertTrue(result["supported"])
        self.assertEqual(result["ambiguous_rows"], 1)
        self.assertEqual(result["ambiguous_changed_rows"], 1)
        self.assertEqual(result["unambiguous_changed_rows"], 0)
        self.assertTrue(result["rows"][0]["valid_under_cutoff"])

    def test_audit_matches_identical_reference_set(self):
        choice = torch.tensor([[9.0, 8.0, 7.0, 7.0, 7.0]], dtype=torch.bfloat16)
        expected = torch.topk(choice, 3, dim=-1, sorted=False).indices.tolist()
        data = {
            "top_k": 3,
            "n_routed_experts": 5,
            "choice_bits_int16": choice.view(torch.int16).tolist(),
            "official_indices": expected,
        }
        result = audit_layer("2", data, repeat_topk=4)
        self.assertEqual(result["exact_set_rows"], 1)
        self.assertEqual(result["ambiguous_changed_rows"], 0)

    def test_platform_classification_separates_support_from_route_change(self):
        same = {
            "supported": True,
            "ambiguous_changed_rows": 0,
        }
        changed = {
            "supported": True,
            "ambiguous_changed_rows": 2,
        }
        unsupported = {
            "supported": False,
            "ambiguous_changed_rows": 0,
        }
        self.assertEqual(
            classify_platform({"2": same, "5": same}),
            "official_route_sets_match_linux_archive",
        )
        self.assertEqual(
            classify_platform({"2": changed, "5": same}),
            "official_tied_route_sets_differ_from_linux_archive",
        )
        self.assertEqual(
            classify_platform({"2": unsupported, "5": unsupported}),
            "bfloat16_topk_unsupported",
        )
        self.assertEqual(
            classify_platform({"2": unsupported, "5": same}),
            "bfloat16_topk_partially_supported",
        )


if __name__ == "__main__":
    unittest.main()
