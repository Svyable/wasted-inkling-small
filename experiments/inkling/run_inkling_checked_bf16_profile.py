#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run the checked-in BF16 promotion probe without historical import effects."""
from __future__ import annotations

import diagnose_inkling_portable_bf16_composed_moe as composed
from checked_bf16_profile_collector import CheckedExactWeightCollector

# The promotion probe consumes the arithmetic helper module directly.  Install
# only the exception-safe collector it needs; do not import the historical
# composed runner, which also rebinds source transforms, official-stage hooks,
# and global stage lists at import time.
composed.ExactWeightCollector = CheckedExactWeightCollector

from run_inkling_checked_bf16_sparse_layer import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
