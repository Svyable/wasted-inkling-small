#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run the side-effect-free checked-in BF16 sparse promotion gate."""
from __future__ import annotations

from run_inkling_checked_bf16_sparse_layer import main


if __name__ == "__main__":
    raise SystemExit(main())