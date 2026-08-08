#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
inkling_throughput.py — what Inkling-Small costs per token, and what that
means for tok/s on a machine you own.

Every gate in ROADMAP-V19.md asks whether this port computes the right
thing. None of them asks how fast, and throughput is what decides whether
streaming a 276B model off disk is a product or a curiosity. This tool
answers the part that is knowable from geometry plus upstream's own
measurements, without the checkpoint.

Three subcommands, stdlib only:

  geometry   Exact VQ record size, records and bytes per decoded token, the
             expert bank on disk, one token's working set, and the budget
             ladder WASTE's resolver would pick on a given machine.

  project    Decode throughput over a bandwidth x hit-rate grid, using the
             cost decomposition upstream measured for K3.

  compare    Inkling-Small against K3 on the numbers that set tok/s.

WHAT IS MEASURED AND WHAT IS NOT
--------------------------------
Geometry is exact: it is arithmetic over the released config and the WEXP
record layout in `inkling_vq.py`, and `tests/test_inkling_throughput.py`
checks it against a record that file actually writes.

The projection is an ESTIMATE. It is calibrated against real measurements
of a different model (K3) on one machine, and extrapolated to Inkling by
cost model rather than measured. Its inputs are named in CALIBRATION below
so that a wrong answer can be traced to a wrong input. It is labelled in
every line of output that carries it, because a projection dressed as a
measurement is how this decision goes wrong — see docs/EFFICIENCY.md, which
this file deliberately imitates.

Nothing here has been run against the official checkpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

GIB = 1 << 30
MIB = 1 << 20

# ---------------------------------------------------------------- layout ----
#
# WEXP record layout, from `inkling_vq.py`:
#   [48-byte header][gate indices][up indices][down indices][scales]
# padded to a 4 KiB multiple. Index planes are blocked to `index_block` rows;
# scales are one float16 per row, unpadded.

VQ_HEADER_BYTES = 48
VQ_ALIGN = 4096


@dataclass(frozen=True)
class VQSpec:
    """Mirrors inkling_vq.VQSpec without importing it — that module needs
    torch, and this tool must run anywhere. The test suite checks the two
    agree by writing a real record."""

    stages: int = 3
    vec_dim: int = 8
    index_block: int = 64

    def validate(self) -> None:
        if self.stages not in (2, 3):
            raise ValueError("WASTE expert VQ stages must be 2 or 3")
        if self.vec_dim <= 0 or self.vec_dim > 64:
            raise ValueError("invalid VQ vector dimension")
        if self.index_block <= 0 or self.index_block > 4096:
            raise ValueError("invalid VQ index block")


def _align(value: int, alignment: int = VQ_ALIGN) -> int:
    return (value + alignment - 1) // alignment * alignment


def matrix_index_bytes(rows: int, cols: int, spec: VQSpec) -> int:
    """Bytes one quantized matrix contributes to a record's index planes."""
    if rows <= 0 or cols <= 0:
        raise ValueError("matrix dimensions must be positive")
    if cols % spec.vec_dim:
        raise ValueError(f"columns {cols} not a multiple of vec_dim {spec.vec_dim}")
    padded_rows = _align(rows, spec.index_block)
    vectors_per_row = cols // spec.vec_dim
    return padded_rows * vectors_per_row * spec.stages


def expert_record_bytes(hidden: int, intermediate: int, spec: VQSpec) -> int:
    """Size on disk of one routed expert, exactly as `write_expert_record`
    lays it out. gate and up are [intermediate, hidden]; down is
    [hidden, intermediate]."""
    spec.validate()
    shapes = (
        (intermediate, hidden),   # gate
        (intermediate, hidden),   # up
        (hidden, intermediate),   # down
    )
    total = VQ_HEADER_BYTES
    for rows, cols in shapes:
        total += matrix_index_bytes(rows, cols, spec)
    for rows, _cols in shapes:
        total += rows * 2         # float16 scale per row, unpadded
    return _align(total)


# ------------------------------------------------------------ calibration ----
#
# Measured, not assumed. Each constant names where it came from so a wrong
# projection can be traced to a wrong input rather than argued about.


@dataclass(frozen=True)
class Calibration:
    """Upstream's measured K3 decode, from docs/EFFICIENCY.md (2026-07-31),
    on a 64 GB M5 Pro with the container on the internal SSD."""

    # docs/EFFICIENCY.md §1: 92 MoE layers x top-16.
    k3_records_per_token: int = 1472
    # docs/EFFICIENCY.md §2: "11.83 MB records (K3's real expert size)".
    k3_record_bytes: int = int(11.83 * MIB)

    # docs/EFFICIENCY.md §4, decomposed at the margin at the default budget
    # (17.56 GB cache, 13% hit), before read-ahead landed.
    k3_io_seconds: float = 1.41
    k3_expert_matmul_seconds: float = 1.03
    k3_other_seconds: float = 0.50
    k3_hit_rate: float = 0.13

    # docs/EFFICIENCY.md §2, tools/diskbench.c on the internal SSD.
    k3_bandwidth_qd1: float = 10.73          # GiB/s
    k3_bandwidth_qd4: float = 12.89          # GiB/s

    # docs/EFFICIENCY.md, measured after the read-ahead of §4A and the router
    # lookahead of §4F. The projection is checked against this.
    k3_measured_tok_s: tuple[float, float] = (0.56, 0.63)

    # K3's resident trunk, docs/EFFICIENCY.md §4B ("the 27.5 GiB trunk is
    # read in full every token"). Used to scale non-expert compute.
    k3_trunk_gib: float = 27.28

    def k3_bytes_per_token(self) -> int:
        return self.k3_records_per_token * self.k3_record_bytes

    def check(self) -> None:
        """The decomposition must reproduce its own I/O term, or one of the
        numbers above has been transcribed wrong."""
        cold = self.k3_bytes_per_token() / GIB
        predicted = cold * (1.0 - self.k3_hit_rate) / self.k3_bandwidth_qd1
        if abs(predicted - self.k3_io_seconds) > 0.10:
            raise ValueError(
                f"calibration inconsistent: {cold:.2f} GiB/token at "
                f"{self.k3_hit_rate:.0%} hit and {self.k3_bandwidth_qd1} GiB/s "
                f"gives {predicted:.2f}s, but EFFICIENCY.md §4 says "
                f"{self.k3_io_seconds}s"
            )


CAL = Calibration()

# docs/GATES.md Gate 7: the default-budget resolver takes the largest
# `floor + working_set * k` for k in 3..1 that fits under seven eighths of RAM.
BUDGET_STEPS = (3, 2, 1)
BUDGET_CAP_FRACTION = 7 / 8


# -------------------------------------------------------------- geometry ----


@dataclass(frozen=True)
class Geometry:
    layers: int
    dense_layers: int
    sparse_layers: int
    hidden: int
    moe_intermediate: int
    n_routed_experts: int
    top_k: int
    n_shared_experts: int
    record_bytes: int
    spec: VQSpec
    # Published in docs/ROADMAP-V19.md §G3. Unlike everything else on this
    # class these two are NOT derived from the config — sizing the trunk means
    # walking every trunk tensor at its chosen quantization, which is
    # inkling_plan.py's job. They are correct for the released Inkling-Small at
    # the Q4 bulk + Q8 vocab/router operating point and for nothing else, which
    # is why `released_small` below refuses to vouch for another geometry.
    trunk_gib: float = 3.642
    model_state_floor_gib: float = 4.1
    derived_trunk: bool = False

    # The geometry the trunk and floor constants were derived for.
    RELEASED_SMALL = (42, 4096, 2048, 256, 6, 2)

    @property
    def released_small(self) -> bool:
        """True when this is the config those two constants describe."""
        return (
            self.layers,
            self.hidden,
            self.moe_intermediate,
            self.n_routed_experts,
            self.top_k,
            self.n_shared_experts,
        ) == self.RELEASED_SMALL

    @property
    def trunk_is_trustworthy(self) -> bool:
        return self.derived_trunk or self.released_small

    @property
    def records_per_token(self) -> int:
        return self.sparse_layers * self.top_k

    @property
    def bytes_per_token(self) -> int:
        return self.records_per_token * self.record_bytes

    @property
    def working_set_gib(self) -> float:
        """One token touches this many distinct records. docs/GATES.md
        Gate 5: a demand-only cache below this keeps nothing alive from one
        token to the next and its hit rate is not low but zero."""
        return self.bytes_per_token / GIB

    @property
    def expert_bank_bytes(self) -> int:
        return self.sparse_layers * self.n_routed_experts * self.record_bytes

    def budget(self, ram_gib: float) -> tuple[float, int]:
        """Reproduce upstream's default-budget resolver. Returns
        (budget_gib, k); k == 0 means not even one working set fits, which
        is the regime Gate 5 measured at exactly zero hit rate."""
        cap = ram_gib * BUDGET_CAP_FRACTION
        for k in BUDGET_STEPS:
            want = self.model_state_floor_gib + self.working_set_gib * k
            if want <= cap:
                return want, k
        return self.model_state_floor_gib, 0


def _require_int(cfg: dict, key: str) -> int:
    """Fail closed. A missing field is not a zero, and a plausible default
    here would produce a confident wrong byte count — the exact failure
    `inkling_public.c` refuses at the C boundary."""
    if key not in cfg:
        raise KeyError(f"config is missing required field {key!r}")
    value = cfg[key]
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"config field {key!r} must be a positive integer")
    return value


def load_config(path: Path) -> dict:
    raw = json.loads(Path(path).read_text())
    return raw.get("text_config", raw)


def geometry_from_config(
    cfg: dict,
    spec: VQSpec | None = None,
    trunk_gib: float | None = None,
    floor_gib: float | None = None,
) -> Geometry:
    """Build the decode geometry. `trunk_gib`/`floor_gib` override the
    released-Small constants; pass them for any other config, because the
    defaults do not describe it and this tool cannot derive them."""
    spec = spec or VQSpec()
    layers = _require_int(cfg, "num_hidden_layers")
    hidden = _require_int(cfg, "hidden_size")
    # The routed width. The release schema calls it `intermediate_size` and
    # the dense width `dense_intermediate_size`; Transformers 5.14.1 folds the
    # latter into its own `intermediate_size` and calls the routed one
    # `moe_intermediate_size`. Resolved against the authoritative config.json:
    # routed is 2048. See ROADMAP-V19.md's schema note.
    if "moe_intermediate_size" in cfg and "dense_intermediate_size" in cfg:
        raise ValueError(
            "config carries both 'moe_intermediate_size' and "
            "'dense_intermediate_size'; the routed width is ambiguous"
        )
    routed_key = "moe_intermediate_size" if "moe_intermediate_size" in cfg else "intermediate_size"
    moe_intermediate = _require_int(cfg, routed_key)

    if "n_routed_experts" in cfg:
        n_routed = _require_int(cfg, "n_routed_experts")
    else:
        n_routed = _require_int(cfg, "num_experts")
    top_k = _require_int(cfg, "num_experts_per_tok")
    n_shared = _require_int(cfg, "n_shared_experts")
    dense = _require_int(cfg, "dense_mlp_idx")

    if top_k > n_routed:
        raise ValueError("num_experts_per_tok exceeds the routed expert count")
    if dense >= layers:
        raise ValueError("dense_mlp_idx is not inside the layer range")

    if (trunk_gib is not None) != (floor_gib is not None):
        raise ValueError("pass both --trunk-gib and --floor-gib, or neither")
    if trunk_gib is not None and (trunk_gib <= 0 or floor_gib <= 0):
        raise ValueError("trunk and floor sizes must be positive")

    overrides = {}
    if trunk_gib is not None:
        overrides = {
            "trunk_gib": trunk_gib,
            "model_state_floor_gib": floor_gib,
            "derived_trunk": True,
        }

    return Geometry(
        layers=layers,
        dense_layers=dense,
        sparse_layers=layers - dense,
        hidden=hidden,
        moe_intermediate=moe_intermediate,
        n_routed_experts=n_routed,
        top_k=top_k,
        n_shared_experts=n_shared,
        record_bytes=expert_record_bytes(hidden, moe_intermediate, spec),
        spec=spec,
        **overrides,
    )


# ------------------------------------------------------------ projection ----


@dataclass(frozen=True)
class Projection:
    """One point of the estimate. `other_seconds` is a band, so tok/s is too."""

    bandwidth: float
    hit_rate: float
    io_seconds: float
    expert_matmul_seconds: float
    other_low: float
    other_high: float

    @property
    def serial_low(self) -> float:
        return self.io_seconds + self.expert_matmul_seconds + self.other_low

    def pipelined(self, other: float) -> float:
        """docs/EFFICIENCY.md §4A: read-ahead turns the sum of I/O and expert
        compute into a maximum. Everything else stays serial."""
        return max(self.io_seconds, self.expert_matmul_seconds) + other

    @property
    def tok_s_high(self) -> float:
        return 1.0 / self.pipelined(self.other_low)

    @property
    def tok_s_low(self) -> float:
        return 1.0 / self.pipelined(self.other_high)

    @property
    def io_bound(self) -> bool:
        return self.io_seconds > self.expert_matmul_seconds


def project(
    geo: Geometry,
    bandwidth: float,
    hit_rate: float,
    cal: Calibration = CAL,
    other_high_multiple: float = 2.0,
) -> Projection:
    """Estimate one decode step.

    The two compute terms are scaled from K3's measured decomposition:

    * Expert matmul scales with bytes per token. docs/EFFICIENCY.md §1: the
      cost is one `vq_apply` pass per (token, expert) pair and `vq_rows` does
      `stages` gathers per row per vector position — which is the same
      quantity as the record's index bytes. So the ratio of expert compute is
      the ratio of per-token expert bytes. This is a cost model, not a
      measurement, and it is the largest single assumption here.

    * Everything else (trunk matvecs, attention, norms, routing) scales with
      resident trunk bytes, which is what dominates it on K3. Inkling's
      attention machinery differs from MLA/KDA — a real K/V cache and four
      short convolutions per layer — so the upper end of the band multiplies
      that estimate by `other_high_multiple` rather than pretending the shape
      is known.
    """
    if not 0.0 <= hit_rate < 1.0:
        raise ValueError("hit_rate must be in [0, 1)")
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive")
    cal.check()

    cold_gib = geo.bytes_per_token / GIB
    io_seconds = cold_gib * (1.0 - hit_rate) / bandwidth

    expert_ratio = geo.bytes_per_token / cal.k3_bytes_per_token()
    expert_matmul = cal.k3_expert_matmul_seconds * expert_ratio

    other_low = cal.k3_other_seconds * (geo.trunk_gib / cal.k3_trunk_gib)
    return Projection(
        bandwidth=bandwidth,
        hit_rate=hit_rate,
        io_seconds=io_seconds,
        expert_matmul_seconds=expert_matmul,
        other_low=other_low,
        other_high=other_low * other_high_multiple,
    )


def k3_self_check(cal: Calibration = CAL) -> tuple[float, float]:
    """Run the same model on K3's own numbers. If it does not land near the
    measured 0.56-0.63 tok/s, the model is wrong and every Inkling figure it
    produces is decoration."""
    cal.check()
    io = cal.k3_bytes_per_token() / GIB * (1 - cal.k3_hit_rate) / cal.k3_bandwidth_qd4
    step = max(io, cal.k3_expert_matmul_seconds) + cal.k3_other_seconds
    return 1.0 / step, step


# ------------------------------------------------------------------- cli ----

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "tests" / "data" / "inkling-small-config.json"


def _fmt_gib(n: float) -> str:
    return f"{n:,.2f} GiB"


def _geometry(args: argparse.Namespace) -> Geometry:
    """Shared construction, including the one warning this tool owes the
    caller: the trunk and floor are constants, not derivations, and silently
    reusing them for a different model is exactly the confident-wrong-number
    failure the rest of this port refuses."""
    geo = geometry_from_config(
        load_config(args.config),
        trunk_gib=getattr(args, "trunk_gib", None),
        floor_gib=getattr(args, "floor_gib", None),
    )
    if not geo.trunk_is_trustworthy:
        print(
            "WARNING: this config is not the released Inkling-Small, and the\n"
            "  resident trunk and model-state floor below are constants derived\n"
            "  for that model. Every budget and 'other s' figure that depends on\n"
            "  them is wrong here. Pass --trunk-gib and --floor-gib.",
            file=sys.stderr,
        )
    return geo


def cmd_geometry(args: argparse.Namespace) -> int:
    geo = _geometry(args)
    print("Inkling-Small decode geometry  (exact — arithmetic over the config)")
    print(f"  layers                     {geo.layers} ({geo.dense_layers} dense, {geo.sparse_layers} sparse)")
    print(f"  hidden / routed width      {geo.hidden} / {geo.moe_intermediate}")
    print(f"  routed experts, top-k      {geo.n_routed_experts}, {geo.top_k}")
    print(f"  shared experts             {geo.n_shared_experts} (trunk-resident, not streamed)")
    print(f"  VQ{geo.spec.stages}R record             {geo.record_bytes:,} B")
    print()
    print(f"  records per decoded token  {geo.records_per_token}")
    print(f"  bytes per decoded token    {geo.bytes_per_token:,} B  ({_fmt_gib(geo.working_set_gib)})")
    print(f"  expert bank on disk        {_fmt_gib(geo.expert_bank_bytes / GIB)}")
    print(f"  trunk, resident            {_fmt_gib(geo.trunk_gib)}")
    print(f"  model-state floor          {_fmt_gib(geo.model_state_floor_gib)}")
    print()
    print("Default budget the resolver would pick  (docs/GATES.md Gate 7)")
    print("    RAM      budget    cache   working sets")
    for ram in args.ram:
        budget, k = geo.budget(ram)
        cache = budget - geo.model_state_floor_gib
        note = "  <- below the Gate 5 floor: demand hit rate is zero" if k == 0 else ""
        print(f"  {ram:5.0f} GiB  {budget:6.2f}  {cache:6.2f}   {k}x{note}")
    return 0


def cmd_project(args: argparse.Namespace) -> int:
    geo = _geometry(args)
    tok_s, step = k3_self_check()
    print("Calibration self-check: the same model on K3's own inputs gives")
    print(f"  {step:.2f} s/token = {tok_s:.2f} tok/s against a measured "
          f"{CAL.k3_measured_tok_s[0]}-{CAL.k3_measured_tok_s[1]} tok/s "
          "(docs/EFFICIENCY.md)")
    print()
    print("PROJECTION — an estimate, not a measurement. No official weight has")
    print("been executed by this code. Inputs are named in Calibration.")
    print()
    print("  GiB/s   hit    I/O s   expert s   other s      tok/s      bound by")
    for bw in args.bandwidth:
        for hit in args.hit:
            p = project(geo, bw, hit)
            print(
                f"  {bw:5.2f}  {hit:4.0%}  {p.io_seconds:6.3f}   "
                f"{p.expert_matmul_seconds:7.3f}   "
                f"{p.other_low:.3f}-{p.other_high:.3f}   "
                f"{p.tok_s_low:5.2f}-{p.tok_s_high:5.2f}   "
                f"{'I/O' if p.io_bound else 'compute'}"
            )
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    geo = _geometry(args)
    k3_bytes = CAL.k3_bytes_per_token()
    rows = [
        ("records per token", CAL.k3_records_per_token, geo.records_per_token, "{:,.0f}"),
        ("record bytes", CAL.k3_record_bytes, geo.record_bytes, "{:,.0f}"),
        ("bytes per token (GiB)", k3_bytes / GIB, geo.bytes_per_token / GIB, "{:.2f}"),
        ("resident trunk (GiB)", CAL.k3_trunk_gib, geo.trunk_gib, "{:.2f}"),
    ]
    print(f"  {'':24} {'K3':>12} {'Inkling-Small':>14}   K3 / Inkling")
    for label, a, b, fmt in rows:
        print(f"  {label:24} {fmt.format(a):>12} {fmt.format(b):>14}   {a / b:>8.1f}x")
    print()
    print(f"  Inkling reads {k3_bytes / geo.bytes_per_token:.1f}x less per token than K3.")
    print("  (K3's model-state floor is its trunk: MLA caches a latent, so its")
    print("   per-token state is small next to the resident weights.)")
    print()
    print("RAM needed for the resolver's k-th working-set step:")
    print("        k    K3 needs      Inkling needs")
    for k in BUDGET_STEPS:
        k3_need = (CAL.k3_trunk_gib + (k3_bytes / GIB) * k) / BUDGET_CAP_FRACTION
        ink_need = (geo.model_state_floor_gib + geo.working_set_gib * k) / BUDGET_CAP_FRACTION
        print(f"       {k}x   {k3_need:7.1f} GiB   {ink_need:9.1f} GiB")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--trunk-gib", type=float, default=None,
                    help="resident trunk size; required for any config that is "
                         "not the released Inkling-Small")
    ap.add_argument("--floor-gib", type=float, default=None,
                    help="model-state floor; pass with --trunk-gib")
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("geometry", help="exact per-token cost and budget ladder")
    g.add_argument("--ram", type=float, nargs="+", default=[8, 16, 24, 32, 64, 128])
    g.set_defaults(func=cmd_geometry)

    p = sub.add_parser("project", help="estimated decode throughput")
    p.add_argument("--bandwidth", type=float, nargs="+",
                   default=[0.55, 2.08, 3.00, 7.00, 10.73, 12.89],
                   help="GiB/s of cache-bypassed random record reads")
    p.add_argument("--hit", type=float, nargs="+", default=[0.0, 0.29, 0.42])
    p.set_defaults(func=cmd_project)

    c = sub.add_parser("compare", help="Inkling-Small against K3")
    c.set_defaults(func=cmd_compare)

    args = ap.parse_args(argv)
    try:
        return args.func(args)
    except (KeyError, ValueError) as exc:
        print(f"inkling_throughput: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
