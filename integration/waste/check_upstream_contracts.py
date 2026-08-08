#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Fail closed when the WASTE baseline loses laptop-relevant contracts.

This is deliberately separate from numerical Inkling parity.  It checks the
upstream integration surface we rely on for a future laptop runtime:

* the exact 64-bit waste_cfg ABI introduced by WASTE 0.6.6;
* safe waste_cfg_init defaults, including an unset CPU list;
* VQ4P and opt-in expert parallelism remaining available;
* conversion controls for VQ shape and reclaim-safe staging;
* diskbench compiling with Linux cache-bypass detection and no hidden K3
  bytes-per-token assumption.

It does not benchmark CI storage and it does not enable Inkling inference.
"""
from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path


class ContractError(RuntimeError):
    """An upstream capability or ABI contract is missing."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def read(root: Path, relative: str) -> str:
    path = root / relative
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ContractError(f"cannot read {relative}: {exc}") from exc


def run(command: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        rendered = " ".join(command)
        raise ContractError(
            f"command failed ({result.returncode}): {rendered}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


def check_sources(root: Path) -> None:
    waste_h = read(root, "src/waste.h")
    require("#define WASTE_API_VERSION 1" in waste_h, "WASTE API version is not 1")
    require("#define WASTE_VERSION_PATCH  6" in waste_h, "WASTE patch version is not 0.6.6")
    require(
        re.search(
            r"int\s+n_threads\s*;[\s\S]*?"
            r"const\s+char\s*\*\s*cpu_list\s*;[\s\S]*?"
            r"waste_cache_policy\s+cache_policy\s*;",
            waste_h,
        )
        is not None,
        "waste_cfg no longer orders n_threads, cpu_list, cache_policy as 0.6.6",
    )

    waste_format = read(root, "src/waste_format.h")
    require("WQ_VQ4P = 8" in waste_format, "WQ_VQ4P format 8 is missing")

    model_c = read(root, "src/model.c")
    require("WASTE_XPAR" in model_c, "opt-in expert-parallel execution is missing")
    require("WASTE_XPAR_BATCH" in model_c, "expert-parallel batch bound is missing")

    convert = read(root, "tools/convert.py")
    for option in ("--entries", "--index-bits", "--stages", "--reclaim"):
        require(option in convert, f"converter option {option} is missing")
    require(
        "config_identifies_inkling" in convert,
        "Inkling must still fail closed out of the Kimi converter",
    )

    diskbench = read(root, "tools/diskbench.c")
    for marker in (
        "O_DIRECT",
        "dio_probe",
        "PAGE CACHE, not the disk",
        "argc > 5 ? atof(argv[5]) : 0.0",
    ):
        require(marker in diskbench, f"diskbench contract marker is missing: {marker}")


def check_compiled_contracts(root: Path) -> None:
    run(["make", "-C", str(root), "libwaste.a"])

    with tempfile.TemporaryDirectory(prefix="waste-v066-contract-") as tmp:
        scratch = Path(tmp)
        abi_source = scratch / "waste_cfg_abi.c"
        abi_binary = scratch / "waste_cfg_abi"
        abi_source.write_text(
            r'''#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include "waste.h"

#if UINTPTR_MAX != UINT64_MAX
#error "this contract records the supported 64-bit host ABI"
#endif

_Static_assert(WASTE_API_VERSION == 1, "unexpected API version");
_Static_assert(WASTE_VERSION_NUMBER == 606, "unexpected engine version");
_Static_assert(offsetof(waste_cfg, ram_budget_bytes) == 0, "ram budget offset");
_Static_assert(offsetof(waste_cfg, ctx_tokens) == 8, "ctx offset");
_Static_assert(offsetof(waste_cfg, n_threads) == 12, "thread offset");
_Static_assert(offsetof(waste_cfg, cpu_list) == 16, "cpu list offset");
_Static_assert(offsetof(waste_cfg, cache_policy) == 24, "cache policy offset");
_Static_assert(offsetof(waste_cfg, use_direct_io) == 28, "direct I/O offset");
_Static_assert(offsetof(waste_cfg, vision) == 32, "vision offset");
_Static_assert(offsetof(waste_cfg, verify_records) == 36, "verify offset");
_Static_assert(offsetof(waste_cfg, usage_path) == 40, "usage path offset");
_Static_assert(sizeof(waste_cfg) == 48, "waste_cfg size");

int main(void) {
    waste_cfg cfg;
    waste_cfg_init(&cfg);
    if (cfg.ram_budget_bytes != 0 || cfg.ctx_tokens != 4096 ||
        cfg.n_threads != 0 || cfg.cpu_list != NULL ||
        cfg.cache_policy != WASTE_CACHE_LFRU || cfg.use_direct_io != 1 ||
        cfg.vision != 0 || cfg.verify_records != 0 || cfg.usage_path != NULL) {
        fputs("waste_cfg_init defaults changed\n", stderr);
        return 2;
    }
    printf("WASTE_CFG_ABI version=%s size=%zu cpu_list=%zu cache_policy=%zu usage_path=%zu\n",
           WASTE_VERSION_STRING, sizeof(cfg), offsetof(waste_cfg, cpu_list),
           offsetof(waste_cfg, cache_policy), offsetof(waste_cfg, usage_path));
    return 0;
}
''',
            encoding="utf-8",
        )
        run(
            [
                "cc",
                "-std=gnu11",
                "-Wall",
                "-Wextra",
                "-Werror",
                f"-I{root / 'src'}",
                str(abi_source),
                str(root / "libwaste.a"),
                "-o",
                str(abi_binary),
                "-pthread",
                "-lm",
            ]
        )
        abi = run([str(abi_binary)])
        require("WASTE_CFG_ABI version=0.6.6 size=48" in abi.stdout, "ABI probe output changed")
        print(abi.stdout.strip())

        diskbench = scratch / "diskbench"
        run(
            [
                "cc",
                "-std=gnu11",
                "-Wall",
                "-Wextra",
                "-Werror",
                str(root / "tools/diskbench.c"),
                "-o",
                str(diskbench),
                "-pthread",
            ]
        )
        usage = subprocess.run(
            [str(diskbench)],
            text=True,
            capture_output=True,
            check=False,
        )
        require(usage.returncode != 0, "diskbench without a path must refuse")
        require("[gb_per_token]" in usage.stderr, "diskbench usage hides the model-specific term")
        print("WASTE_DISKBENCH_CONTRACT cache-bypass-probe=present gb-per-token=explicit")


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {Path(sys.argv[0]).name} GENERATED_WASTE_TREE", file=sys.stderr)
        return 2
    root = Path(sys.argv[1]).resolve()
    try:
        check_sources(root)
        check_compiled_contracts(root)
    except ContractError as exc:
        print(f"check_upstream_contracts.py: FAIL {exc}", file=sys.stderr)
        return 1
    print("check_upstream_contracts.py: PASS WASTE 0.6.6 laptop-runtime contracts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

