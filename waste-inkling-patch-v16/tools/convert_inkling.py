#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
"""Inkling-family conversion entry point.

The command has six safe modes:

* ``--plan-only`` validates metadata and tensor shapes, then writes only
  ``conversion-plan.json``.
* ``--stage-experts`` writes converter-internal aligned BF16 routed-expert
  banks and ``stage.json``.
* ``--stage-trunk`` writes one resumable canonical artifact per resident text
  tensor and ``trunk-stage.json`` using bounded source reads.
* ``--publish-runtime-stage`` validates completed trunk/BF16 expert stages.
* ``--publish-runtime-vq-stage`` validates completed trunk/final WEXP-VQ stages.
  Both publish only the private ``runtime-stage.bin`` index.
* ``--quantize-experts`` converts routed experts directly to final WEXP/VQ
  layer artifacts, but still does not publish ``manifest.json``.
* ``--quantize-trunk`` converts canonical trunk-stage matrices to bounded
  Q8/Q4 artifacts without publishing ``manifest.json``.

Neither staging mode publishes a WASTE manifest.

Normal container conversion remains disabled until trunk staging, integrated
Inkling loader/forward dispatch, and differential parity are complete.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from inspect_inkling import InspectError
from inkling_plan import PlanError, build_plan, write_json_atomic
from inkling_runtime_stage import (
    RuntimeStageError, publish_runtime_stage, publish_runtime_vq_stage,
    publish_runtime_qtrunk_vq_stage,
)
from inkling_stage import StageError, stage_expert_banks
from inkling_trunk import TrunkStageError, stage_trunk
from inkling_qtrunk import QTrunkError, quantize_trunk
from inkling_vq import VQError, VQSpec, quantize_expert_banks


def _csv_ints(value: str) -> list[int]:
    try:
        return [int(item) for item in value.split(",") if item != ""]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a comma-separated integer list") from exc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path, help="Inkling checkpoint directory")
    ap.add_argument("--out", required=True, type=Path, help="staging/future WASTE output directory")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument(
        "--plan-only",
        action="store_true",
        help="validate metadata/shapes and write conversion-plan.json only",
    )
    mode.add_argument(
        "--stage-experts",
        action="store_true",
        help="write aligned BF16 routed-expert staging banks; never writes manifest.json",
    )
    mode.add_argument(
        "--stage-trunk",
        action="store_true",
        help="write resumable resident-trunk staging artifacts; never writes trunk.bin or manifest.json",
    )
    mode.add_argument(
        "--quantize-trunk",
        action="store_true",
        help="quantize canonical trunk-stage matrices to private Q8/Q4 artifacts",
    )
    mode.add_argument(
        "--quantize-experts",
        action="store_true",
        help="write final-format WEXP/VQ layer artifacts; never writes manifest.json",
    )
    mode.add_argument(
        "--publish-runtime-stage",
        action="store_true",
        help="validate completed stages and publish converter-private runtime-stage.bin",
    )
    mode.add_argument(
        "--publish-runtime-vq-stage",
        action="store_true",
        help="validate trunk plus final WEXP/VQ banks and publish private runtime-stage.bin",
    )
    mode.add_argument(
        "--publish-runtime-qtrunk-stage",
        action="store_true",
        help="publish private v3 runtime index from Q8/Q4 trunk plus final WEXP/VQ banks",
    )
    ap.add_argument(
        "--allow-index-only",
        action="store_true",
        help="allow a review plan when shards are not downloaded; valid only with --plan-only",
    )
    ap.add_argument("--layers", type=_csv_ints, help="sparse layers to stage/quantize, comma-separated")
    ap.add_argument("--experts", type=int, help="debug limit per bank; marks the stage incomplete")
    ap.add_argument("--no-verify", action="store_true", help="skip post-write CRC verification")
    ap.add_argument("--no-resume", action="store_true", help="rewrite requested staging artifacts even when sidecars match")
    ap.add_argument("--trunk-bits", type=int, choices=(4, 8), default=8,
                    help="private trunk matrix width for --quantize-trunk (default: 8)")
    ap.add_argument("--trunk-group", type=int, default=128,
                    help="quantized trunk group size (default: 128)")
    ap.add_argument("--trunk-chunk-rows", type=int, default=64,
                    help="maximum staged rows converted at once")
    ap.add_argument("--vq-stages", type=int, choices=(2, 3), default=3, help="VQ residual stages (default: 3)")
    ap.add_argument("--vq-entries", type=int, default=256, help="codebook entries; 256 for production")
    ap.add_argument("--codebook-sample", type=int, default=12, help="experts sampled per layer for codebook training")
    ap.add_argument("--train-vectors", type=int, default=300000, help="maximum normalized vectors per matrix kind")
    ap.add_argument("--kmeans-iters", type=int, default=10, help="Lloyd iterations per residual stage")
    ap.add_argument("--assign-chunk", type=int, default=32768, help="maximum vectors per nearest-centroid chunk")
    ap.add_argument("--device", default="cpu", help="torch device for VQ training/encoding")
    ap.add_argument(
        "--allow-generic-inkling",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    ap.add_argument(
        "--chunk-mib",
        type=int,
        default=8,
        help="maximum raw-copy chunk size for --stage-trunk (default: 8 MiB)",
    )
    args = ap.parse_args(argv)

    if args.allow_index_only and not args.plan_only:
        ap.error("--allow-index-only is valid only with --plan-only")
    if (args.layers is not None or args.experts is not None) and not (args.stage_experts or args.quantize_experts):
        ap.error("--layers and --experts require --stage-experts or --quantize-experts")
    if args.no_resume and not (args.stage_experts or args.stage_trunk or args.quantize_experts or args.quantize_trunk):
        ap.error("--no-resume requires --stage-experts, --stage-trunk, or --quantize-experts")
    if args.no_verify and not (args.stage_experts or args.stage_trunk or args.quantize_experts or args.quantize_trunk or args.publish_runtime_stage or args.publish_runtime_vq_stage or args.publish_runtime_qtrunk_stage):
        ap.error("--no-verify requires a staging or runtime-stage mode")
    if args.allow_generic_inkling and not (args.publish_runtime_stage or args.publish_runtime_vq_stage or args.publish_runtime_qtrunk_stage):
        ap.error("--allow-generic-inkling requires a runtime-stage publication mode")
    vq_tuned = (args.vq_stages != 3 or args.vq_entries != 256 or args.codebook_sample != 12 or
                  args.train_vectors != 300000 or args.kmeans_iters != 10 or
                  args.assign_chunk != 32768 or args.device != "cpu")
    if vq_tuned and not args.quantize_experts:
        ap.error("VQ tuning options require --quantize-experts")
    if args.vq_entries < 2 or args.vq_entries > 256:
        ap.error("--vq-entries must be in 2..256")
    if min(args.codebook_sample, args.train_vectors, args.kmeans_iters, args.assign_chunk) <= 0:
        ap.error("VQ training values must be positive")
    trunk_tuned = args.trunk_bits != 8 or args.trunk_group != 128 or args.trunk_chunk_rows != 64
    if trunk_tuned and not args.quantize_trunk:
        ap.error("trunk quantization options require --quantize-trunk")
    if args.trunk_group < 2 or args.trunk_group % 2 or args.trunk_chunk_rows < 1:
        ap.error("--trunk-group must be positive/even and --trunk-chunk-rows positive")
    if args.chunk_mib != 8 and not args.stage_trunk:
        ap.error("--chunk-mib requires --stage-trunk")
    if args.chunk_mib <= 0:
        ap.error("--chunk-mib must be positive")

    if args.plan_only:
        try:
            plan = build_plan(args.src, require_payloads=not args.allow_index_only)
        except (InspectError, PlanError) as exc:
            print(f"convert_inkling: {exc}", file=sys.stderr)
            return 2
        path = args.out / "conversion-plan.json"
        write_json_atomic(path, plan)
        print(f"wrote {path}")
        print("no WASTE container files were created")
        return 0

    if args.stage_experts:
        try:
            stage = stage_expert_banks(
                args.src,
                args.out,
                layers=args.layers,
                expert_limit=args.experts,
                verify=not args.no_verify,
                resume=not args.no_resume,
            )
        except StageError as exc:
            print(f"convert_inkling: {exc}", file=sys.stderr)
            return 2
        print(f"wrote {len(stage['banks'])} BF16 expert staging bank(s) and {args.out / 'stage.json'}")
        print("manifest.json was not written; this is not a runnable WASTE container")
        return 0

    if args.quantize_trunk:
        try:
            stage = quantize_trunk(
                args.out, bits=args.trunk_bits, group=args.trunk_group,
                chunk_rows=args.trunk_chunk_rows, verify=not args.no_verify,
                resume=not args.no_resume,
            )
        except QTrunkError as exc:
            print(f"convert_inkling: {exc}", file=sys.stderr)
            return 2
        print(f"wrote {stage['totals']['tensors']} private quantized trunk artifact(s) and {args.out / 'qtrunk-stage.json'}")
        print("manifest.json was not written; public WASTE runtime remains disabled")
        return 0

    if args.quantize_experts:
        try:
            stage = quantize_expert_banks(
                args.src,
                args.out,
                layers=args.layers,
                spec=VQSpec(stages=args.vq_stages, entries=args.vq_entries),
                expert_limit=args.experts,
                codebook_sample=args.codebook_sample,
                train_vectors=args.train_vectors,
                kmeans_iterations=args.kmeans_iters,
                device=args.device,
                assign_chunk=args.assign_chunk,
                verify=not args.no_verify,
                resume=not args.no_resume,
            )
        except VQError as exc:
            print(f"convert_inkling: {exc}", file=sys.stderr)
            return 2
        print(f"wrote {len(stage['layers'])} final-format WEXP/VQ layer bank(s) and {args.out / 'vq-stage.json'}")
        print("manifest.json was not written; public WASTE runtime remains disabled")
        return 0

    if args.publish_runtime_qtrunk_stage:
        try:
            meta = publish_runtime_qtrunk_vq_stage(
                args.src, args.out, verify=not args.no_verify,
                require_official=not args.allow_generic_inkling,
            )
        except RuntimeStageError as exc:
            print(f"convert_inkling: {exc}", file=sys.stderr)
            return 2
        print(
            f"published private quantized-trunk runtime stage with "
            f"{meta['counts']['quantized_tensors']} Q8/Q4 tensors and "
            f"{meta['counts']['banks']} WEXP banks"
        )
        print("manifest.json was not written; public WASTE runtime remains disabled")
        return 0

    if args.publish_runtime_vq_stage:
        try:
            meta = publish_runtime_vq_stage(
                args.src, args.out, verify=not args.no_verify,
                require_official=not args.allow_generic_inkling,
            )
        except RuntimeStageError as exc:
            print(f"convert_inkling: {exc}", file=sys.stderr)
            return 2
        print(
            f"published private final-VQ runtime stage with {meta['counts']['tensors']} tensors "
            f"and {meta['counts']['banks']} WEXP banks"
        )
        print("manifest.json was not written; public WASTE runtime remains disabled")
        return 0

    if args.publish_runtime_stage:
        try:
            meta = publish_runtime_stage(
                args.src, args.out, verify=not args.no_verify,
                require_official=not args.allow_generic_inkling,
            )
        except RuntimeStageError as exc:
            print(f"convert_inkling: {exc}", file=sys.stderr)
            return 2
        print(
            f"published private runtime stage with {meta['counts']['tensors']} tensors "
            f"and {meta['counts']['banks']} expert banks"
        )
        print("manifest.json was not written; public WASTE runtime remains disabled")
        return 0

    if args.stage_trunk:
        try:
            stage = stage_trunk(
                args.src,
                args.out,
                verify=not args.no_verify,
                resume=not args.no_resume,
                chunk_bytes=args.chunk_mib << 20,
            )
        except TrunkStageError as exc:
            print(f"convert_inkling: {exc}", file=sys.stderr)
            return 2
        print(
            f"wrote {stage['totals']['tensors']} resident trunk staging artifacts "
            f"and {args.out / 'trunk-stage.json'}"
        )
        print("trunk.bin and manifest.json were not written; this is not a runnable WASTE container")
        return 0

    print(
        "convert_inkling: final container conversion is not implemented; refusing to create "
        "a partial or misleading WASTE container. Use --plan-only, --stage-experts, --stage-trunk, "
        "--quantize-experts, --quantize-trunk, --publish-runtime-stage, --publish-runtime-vq-stage, or --publish-runtime-qtrunk-stage.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
