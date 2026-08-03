#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import unittest

import torch

from audit_inkling_router_cutoff_ties import (
    RouterTieAuditError,
    cutoff_tie_row,
)


class RouterCutoffTieTest(unittest.TestCase):
    def test_different_members_of_cutoff_tie_are_equivalent(self) -> None:
        choice = torch.tensor(
            [0.9, 0.8, 0.7, 0.7, 0.7, 0.2], dtype=torch.bfloat16
        )
        result = cutoff_tie_row(
            choice,
            torch.tensor([0, 1, 2]),
            torch.tensor([0, 1, 4]),
            3,
        )

        self.assertFalse(result["exact_ids"])
        self.assertTrue(result["ambiguous_cutoff"])
        self.assertTrue(result["official_valid_under_cutoff"])
        self.assertTrue(result["candidate_valid_under_official_cutoff"])
        self.assertEqual(result["above_ids"], [0, 1])
        self.assertEqual(result["cutoff_tie_ids"], [2, 3, 4])
        self.assertEqual(result["slots_from_cutoff_tie"], 1)

    def test_id_below_cutoff_is_not_tie_equivalent(self) -> None:
        choice = torch.tensor(
            [0.9, 0.8, 0.7, 0.7, 0.6], dtype=torch.bfloat16
        )
        result = cutoff_tie_row(
            choice,
            torch.tensor([0, 1, 2]),
            torch.tensor([0, 1, 4]),
            3,
        )

        self.assertFalse(result["candidate_valid_under_official_cutoff"])
        self.assertEqual(result["differing_ids"], [2, 4])

    def test_invalid_row_shape_fails_closed(self) -> None:
        with self.assertRaisesRegex(RouterTieAuditError, "top_k"):
            cutoff_tie_row(
                torch.tensor([0.9, 0.8]),
                torch.tensor([0]),
                torch.tensor([0, 1]),
                2,
            )


if __name__ == "__main__":
    unittest.main()
