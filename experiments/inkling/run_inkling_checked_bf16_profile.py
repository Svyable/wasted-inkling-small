#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run the checked-in BF16 promotion probe without historical import effects."""
from __future__ import annotations

import diagnose_inkling_portable_bf16_composed_moe as composed
from checked_bf16_profile_collector import (
    CheckedExactWeightCollector,
    CheckedPostReductionWeightHelper,
    build_checked_weight_helper,
)

# The promotion probe consumes the arithmetic helper module directly. Install
# only the explicit adapters it needs; do not import the historical composed
# runner, which also rebinds source transforms, official-stage hooks, and global
# stage lists at import time. The helper swap is equally deliberate: checked-in
# C implements the retained #51 post-reduction denominator policy, not the
# earlier #35 lattice helper embedded in the historical composed probe.
composed.ExactWeightCollector = CheckedExactWeightCollector
composed.FixedWeightHelper = CheckedPostReductionWeightHelper
composed.build_helper = build_checked_weight_helper

from run_inkling_checked_bf16_sparse_layer import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
