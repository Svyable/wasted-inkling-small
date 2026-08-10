#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Fixture-free probe for the oneDNN ISA seam the reference matrix localized.

The same-host reference matrix established that Inkling's BF16 divergence is
`onednn_avx512_isa_path_is_profile_sensitive`, first visible at ``q_proj``. It
took an eligible AVX-512 runner, a 985 MB CRC-verified official fixture, and
about four minutes to say so, which is why it had run neutral far more often
than it had run.

`q_proj` is a plain bfloat16 linear. If oneDNN's AVX-512 path accumulates
differently from its AVX2 path, that shows up in one matmul of the same shape
with synthetic weights, in under a second, on any host, with no network. This
probe does exactly that and nothing more.

**What this is not.** It is not Inkling parity evidence and cannot be
substituted for the reference matrix. Synthetic weights are not the
checkpoint's, and agreement here says nothing about whether the port is
correct. What it does provide is a host census: run it anywhere and it reports
which arms this host's oneDNN answers identically, so a neutral matrix run
still yields information about the host it drew instead of nothing.

The arms mirror the matrix's exactly, so the two can be read side by side:

    native            no controls
    aten_avx2         ATEN_CPU_CAPABILITY=avx2
    onednn_avx2       ONEDNN_MAX_CPU_ISA=AVX2
    mkldnn_off        torch.backends.mkldnn.enabled = False

plus the ISA-ladder rungs between the AVX2 cap and native.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from typing import Any, Mapping


class IsaProbeError(RuntimeError):
    """Raised when the probe cannot produce a result it is willing to report."""


# The shape that diverged: eight source-bound positions through q_proj on a
# 4096-wide hidden state.
TOKENS = 8
HIDDEN = 4096
SEED = 19

ARMS: tuple[tuple[str, dict[str, str]], ...] = (
    ("native", {}),
    ("aten_avx2", {"ATEN_CPU_CAPABILITY": "avx2"}),
    ("onednn_avx2", {"ONEDNN_MAX_CPU_ISA": "AVX2"}),
    ("onednn_avx512_core", {"ONEDNN_MAX_CPU_ISA": "AVX512_CORE"}),
    ("onednn_avx512_core_bf16", {"ONEDNN_MAX_CPU_ISA": "AVX512_CORE_BF16"}),
    ("onednn_avx512_core_amx", {"ONEDNN_MAX_CPU_ISA": "AVX512_CORE_AMX"}),
    ("mkldnn_off", {"INKLING_PROBE_DISABLE_MKLDNN": "1"}),
)

# Ascending oneDNN capability. Every rung is a cap, so a host missing a rung
# executes it as the highest ISA it does support.
LADDER = (
    "onednn_avx2",
    "onednn_avx512_core",
    "onednn_avx512_core_bf16",
    "onednn_avx512_core_amx",
    "native",
)


def cpu_flags() -> list[str]:
    try:
        with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                key, separator, value = line.partition(":")
                if separator and key.strip() == "flags":
                    return sorted(set(value.split()))
    except OSError:
        pass
    return []


def measure() -> dict[str, Any]:
    """Run one arm in this process and describe its output bit-exactly."""
    import torch

    torch.set_num_threads(1)
    if os.environ.get("INKLING_PROBE_DISABLE_MKLDNN") == "1":
        torch.backends.mkldnn.enabled = False

    generator = torch.Generator().manual_seed(SEED)
    x = torch.randn(
        TOKENS, HIDDEN, generator=generator, dtype=torch.float32
    ).to(torch.bfloat16)
    w = torch.randn(
        HIDDEN, HIDDEN, generator=generator, dtype=torch.float32
    ).to(torch.bfloat16)

    y = torch.nn.functional.linear(x, w)
    # An fp64 product rounded once at the end is the closest thing to an
    # unrounded answer available here. It is a yardstick for how far an arm
    # drifts, not a claim about which arm is correct for Inkling.
    reference = (x.to(torch.float64) @ w.to(torch.float64).T).to(torch.bfloat16)

    produced = y.view(torch.int16)
    expected = reference.view(torch.int16)
    return {
        "output_sha256": hashlib.sha256(
            produced.numpy().tobytes()
        ).hexdigest(),
        "exact_fraction_vs_fp64": float((produced == expected).float().mean()),
        "max_abs_error_vs_fp64": float(
            (y.to(torch.float64) - reference.to(torch.float64)).abs().max()
        ),
        "torch_version": torch.__version__,
        "cpu_capability": str(torch.backends.cpu.get_cpu_capability()),
        "mkldnn_available": bool(torch.backends.mkldnn.is_available()),
        "mkldnn_enabled": bool(torch.backends.mkldnn.enabled),
        "aten_cpu_capability": os.environ.get("ATEN_CPU_CAPABILITY"),
        "onednn_max_cpu_isa": os.environ.get("ONEDNN_MAX_CPU_ISA"),
    }


def run_arm(name: str, overrides: Mapping[str, str]) -> dict[str, Any]:
    """Run one arm in a fresh process.

    Every control here is read by oneDNN or ATen once, at first use, so arms
    cannot share an interpreter — setting the variable after the first matmul
    changes nothing and would silently produce duplicate results.
    """
    env = dict(os.environ)
    for key in ("ATEN_CPU_CAPABILITY", "ONEDNN_MAX_CPU_ISA",
                "INKLING_PROBE_DISABLE_MKLDNN"):
        env.pop(key, None)
    env.update(overrides)
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    completed = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "--measure-one"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise IsaProbeError(
            f"arm {name} exited {completed.returncode}: "
            f"{completed.stderr.strip()[-400:]}"
        )
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise IsaProbeError(f"arm {name} did not emit JSON: {exc}") from exc


def classify(arms: Mapping[str, Mapping[str, Any]], flags: list[str]) -> dict[str, Any]:
    """Say which arms this host's oneDNN answers identically, and no more."""
    missing = [name for name, _ in ARMS if name not in arms]
    if missing:
        raise IsaProbeError(f"probe is missing arms: {missing}")

    native = arms["native"]["output_sha256"]
    baseline = arms["onednn_avx2"]["output_sha256"]
    same_as_native = {
        name: arms[name]["output_sha256"] == native for name, _ in ARMS
    }
    # A rung "differs" when it stops agreeing with the AVX2-capped baseline.
    differs_from_baseline = [
        rung for rung in LADDER
        if arms[rung]["output_sha256"] != baseline
    ]
    first_differing_rung = differs_from_baseline[0] if differs_from_baseline else None
    # Differing must be a suffix of the ladder, or the cap is not the variable.
    if first_differing_rung is None:
        monotone = True
    else:
        boundary = LADDER.index(first_differing_rung)
        monotone = all(
            arms[rung]["output_sha256"] == baseline for rung in LADDER[:boundary]
        ) and all(
            arms[rung]["output_sha256"] != baseline for rung in LADDER[boundary:]
        )

    dispatch_sensitive = not same_as_native["aten_avx2"]
    isa_sensitive = not same_as_native["onednn_avx2"]
    amx_available = "amx_bf16" in flags
    avx512_bf16_available = "avx512_bf16" in flags

    if not isa_sensitive and not dispatch_sensitive:
        classification = "host_answers_every_arm_identically"
    elif dispatch_sensitive:
        # The matrix found ATen dispatch excluded. A host where it is not
        # excluded is a different phenomenon and must not be folded in.
        classification = "aten_dispatch_changes_the_result_on_this_host"
    elif not monotone:
        classification = "isa_ladder_nonmonotone"
    elif first_differing_rung == "native":
        classification = "only_the_uncapped_path_differs"
    else:
        classification = "onednn_isa_cap_changes_the_result"

    return {
        "format": "inkling-onednn-isa-arithmetic-probe",
        "version": 1,
        "shape": {"tokens": TOKENS, "hidden": HIDDEN, "seed": SEED},
        "cpu_capability": arms["native"]["cpu_capability"],
        "amx_bf16_available": amx_available,
        "avx512_bf16_available": avx512_bf16_available,
        "flags_sha256": hashlib.sha256(" ".join(flags).encode()).hexdigest()
        if flags else None,
        "output_sha256": {name: arms[name]["output_sha256"] for name, _ in ARMS},
        "bitwise_identical_to_native": same_as_native,
        "ladder": list(LADDER),
        "first_rung_differing_from_avx2_baseline": first_differing_rung,
        "monotone": monotone,
        "aten_dispatch_sensitive": dispatch_sensitive,
        "onednn_isa_sensitive": isa_sensitive,
        "classification": classification,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--measure-one",
        action="store_true",
        help="internal: run a single arm in this process and print its JSON",
    )
    parser.add_argument("--out", help="write the classified probe here")
    args = parser.parse_args(argv)

    if args.measure_one:
        print(json.dumps(measure(), sort_keys=True))
        return 0

    flags = cpu_flags()
    arms = {name: run_arm(name, overrides) for name, overrides in ARMS}
    report = classify(arms, flags)
    report["arms"] = arms
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
