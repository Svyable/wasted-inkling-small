#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
inkling_cache_trace.py — generate routing traces for `inkling_cache_sim.c`.

Inkling's router has never been observed: running it needs the checkpoint.
What *has* been observed is the Kimi router family, twice, by upstream:

  docs/GATES.md Gate 2 (Kimi-Linear-48B, 256 experts, top-8, 300 tokens)
    concentration  top 8.7% of slots cover 50% of activations,
                   top 28% cover 80%, top 51% cover 95%
    reuse          next-token reuse 33.6%

So this generator has exactly two free parameters, each fitted to one of those
two measurements:

  concentration   a stretched exponential p_i ~ exp(-b (i/N)^g) over the
                  experts of a layer, fitted to the coverage curve
                  (g = 0.55, b = 5.5; reproduces 49.4 / 81.8 / 93.5 against a
                  measured 50 / 80 / 95).

  stickiness      the probability that an expert selected at one position is
                  selected again at the next. Concentration ALONE gives only
                  11.6% reuse against a measured 33.6%, so routing carries
                  real temporal correlation that an IID draw does not
                  reproduce — this parameter is that gap and nothing more.

Two parameters, two measurements: the model is not predictive yet. What makes
it usable is that it then reproduces a THIRD measurement it was not fitted to
— Gate 5's real-cache hit-rate curve — which `--validate` runs and prints.

WHAT THIS IS NOT. It is not Inkling's routing. It is the Kimi family's
routing statistics applied to Inkling's geometry, on the explicit assumption
that a sigmoid-plus-top-k router with a correction bias behaves similarly at
256 experts and top-6. Upstream's own Gate 2 caveat applies: reuse falls and
concentration rises as experts get finer-grained, and which dominates is
model-specific. Every number produced from this trace inherits that
assumption, and `--concentration`/`--stickiness` exist so a reader can see how
much the answer moves when it is wrong.

One real routing trace from the official model replaces all of this.
"""

from __future__ import annotations

import argparse
import math
import random
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

TRACE_MAGIC = 0x31544349  # 'ICT1'

# Fitted to Gate 2's coverage curve; see module docstring.
DEFAULT_G = 0.55
DEFAULT_B = 5.5
# Fitted to Gate 2's 33.6% next-token reuse given the concentration above.
DEFAULT_STICKINESS = 0.249

# Gate 2's measured concentration, kept here so --validate can restate it.
GATE2_COVERAGE = ((0.087, 0.50), (0.28, 0.80), (0.51, 0.95))
GATE2_REUSE = 0.336

# docs/GATES.md Gate 5, the real cache on Kimi-Linear over 300 decode tokens.
# fraction of the expert set -> measured hit rate
GATE5_HIT_CURVE = ((0.030, 0.132), (0.060, 0.403), (0.121, 0.619),
                   (0.242, 0.848), (0.484, 0.939))
GATE5_GEOMETRY = dict(layers=26, experts=256, top_k=8, steps=300)


def weights(n: int, g: float = DEFAULT_G, b: float = DEFAULT_B) -> list[float]:
    """Per-expert selection probability, most-used first."""
    if n <= 0:
        raise ValueError("expert count must be positive")
    if g <= 0 or b <= 0:
        raise ValueError("concentration parameters must be positive")
    raw = [math.exp(-b * ((i / n) ** g)) for i in range(n)]
    total = sum(raw)
    return [x / total for x in raw]


def coverage(w: list[float], fraction: float) -> float:
    """Share of activations taken by the top `fraction` of experts."""
    k = max(1, round(fraction * len(w)))
    return sum(sorted(w, reverse=True)[:k])


def _weighted_sample(pool: list[int], w: list[float], k: int,
                     rng: random.Random) -> list[int]:
    """Weighted selection without replacement — a router's top-k."""
    pool = list(pool)
    ww = [w[i] for i in pool]
    out: list[int] = []
    for _ in range(min(k, len(pool))):
        total = sum(ww)
        if total <= 0:
            out.append(pool.pop(0))
            ww.pop(0)
            continue
        r = rng.random() * total
        acc = 0.0
        for i, x in enumerate(ww):
            acc += x
            if acc >= r:
                out.append(pool.pop(i))
                ww.pop(i)
                break
    return out


@dataclass
class Router:
    """One layer's routing over a sequence of positions."""

    experts: int
    top_k: int
    stickiness: float
    w: list[float]
    prev: list[int]

    @classmethod
    def make(cls, experts: int, top_k: int, stickiness: float,
             g: float, b: float) -> "Router":
        if not 0.0 <= stickiness < 1.0:
            raise ValueError("stickiness must be in [0, 1)")
        if top_k <= 0 or top_k > experts:
            raise ValueError("top_k must be in 1..experts")
        return cls(experts, top_k, stickiness, weights(experts, g, b), [])

    def step(self, rng: random.Random) -> list[int]:
        """Select this position's experts: carry some of the previous
        selection, fill the rest from the concentration distribution."""
        kept = [e for e in self.prev if rng.random() < self.stickiness]
        remaining = self.top_k - len(kept)
        if remaining > 0:
            pool = [i for i in range(self.experts) if i not in set(kept)]
            kept += _weighted_sample(pool, self.w, remaining, rng)
        else:
            kept = kept[: self.top_k]
        self.prev = kept
        return kept


def generate(layers: int, experts: int, top_k: int, steps: int,
             chunk: int = 1, stickiness: float = DEFAULT_STICKINESS,
             g: float = DEFAULT_G, b: float = DEFAULT_B,
             seed: int = 1) -> list[list[list[int]]]:
    """Return `steps` entries, each a list per layer of the DISTINCT expert
    ids that step needs.

    `chunk` > 1 groups that many decode positions into one step and takes the
    union per layer — which is what upstream's `moe_chunk` does, iterating the
    distinct experts of a chunk. It is the whole batching lever: the positions
    still each cost their own arithmetic, but they share every record."""
    if steps <= 0 or layers <= 0 or chunk <= 0:
        raise ValueError("layers, steps and chunk must be positive")
    rng = random.Random(seed)
    routers = [Router.make(experts, top_k, stickiness, g, b) for _ in range(layers)]
    out = []
    for _ in range(steps):
        per_layer = []
        for r in routers:
            union: set[int] = set()
            for _ in range(chunk):
                union.update(r.step(rng))
            per_layer.append(sorted(union))
        out.append(per_layer)
    return out


def write_trace(path: Path, trace: list[list[list[int]]], experts: int) -> int:
    steps = len(trace)
    layers = len(trace[0])
    buf = bytearray(struct.pack("<IIII", TRACE_MAGIC, steps, layers, experts))
    for step in trace:
        if len(step) != layers:
            raise ValueError("ragged trace")
        for ids in step:
            buf += struct.pack("<I", len(ids))
            for e in ids:
                if not 0 <= e < experts:
                    raise ValueError(f"expert {e} outside 0..{experts - 1}")
                buf += struct.pack("<I", e)
    Path(path).write_bytes(bytes(buf))
    return len(buf)


def measured_reuse(trace: list[list[list[int]]]) -> float:
    """Next-step reuse, the quantity Gate 2 reports. Only meaningful for
    chunk == 1, where a step is one position."""
    total = 0.0
    n = 0
    for a, b in zip(trace, trace[1:]):
        for prev, cur in zip(a, b):
            if cur:
                total += len(set(prev) & set(cur)) / len(cur)
                n += 1
    return total / n if n else 0.0


def cmd_generate(args: argparse.Namespace) -> int:
    trace = generate(args.layers, args.experts, args.top_k, args.steps,
                     chunk=args.chunk, stickiness=args.stickiness,
                     g=args.g, b=args.b, seed=args.seed)
    size = write_trace(args.out, trace, args.experts)
    per_step = sum(len(ids) for step in trace for ids in step) / len(trace)
    print(f"wrote {args.out} ({size:,} B)")
    print(f"  steps {args.steps}, layers {args.layers}, top-k {args.top_k}, "
          f"chunk {args.chunk}")
    print(f"  distinct records per step: {per_step:.1f} "
          f"(a chunk of {args.chunk} costs {args.layers * args.top_k * args.chunk} "
          f"without dedup)")
    if args.chunk == 1:
        print(f"  next-token reuse: {measured_reuse(trace):.1%} "
              f"(Gate 2 measured {GATE2_REUSE:.1%})")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    """Restate the two fitted quantities against what upstream measured."""
    w = weights(args.experts, args.g, args.b)
    print("Concentration, against docs/GATES.md Gate 2:")
    for frac, want in GATE2_COVERAGE:
        got = coverage(w, frac)
        print(f"  top {frac:5.1%} of experts: measured {want:.0%}, model {got:.1%}")
    trace = generate(GATE5_GEOMETRY["layers"], args.experts, 8, 400,
                     stickiness=args.stickiness, g=args.g, b=args.b,
                     seed=args.seed)
    print(f"\nReuse: measured {GATE2_REUSE:.1%}, model {measured_reuse(trace):.1%}")
    print("\nBoth of the above are FITTED, not predicted. The prediction this")
    print("model has to earn is Gate 5's hit-rate curve; run the simulator")
    print("with --validate to see it.")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--experts", type=int, default=256)
    ap.add_argument("--g", type=float, default=DEFAULT_G,
                    help="concentration shape (fitted to Gate 2)")
    ap.add_argument("--b", type=float, default=DEFAULT_B,
                    help="concentration scale (fitted to Gate 2)")
    ap.add_argument("--stickiness", type=float, default=DEFAULT_STICKINESS,
                    help="temporal correlation (fitted to Gate 2's reuse)")
    ap.add_argument("--seed", type=int, default=1)
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="write a trace")
    g.add_argument("--out", type=Path, required=True)
    g.add_argument("--layers", type=int, default=40)
    g.add_argument("--top-k", type=int, default=6)
    g.add_argument("--steps", type=int, default=300)
    g.add_argument("--chunk", type=int, default=1,
                   help="decode positions grouped into one step (batching)")
    g.set_defaults(func=cmd_generate)

    c = sub.add_parser("check", help="restate the fit against Gate 2")
    c.set_defaults(func=cmd_check)

    args = ap.parse_args(argv)
    try:
        return args.func(args)
    except ValueError as exc:
        print(f"inkling_cache_trace: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
