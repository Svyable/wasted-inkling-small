# What Inkling-Small costs per token (2026-08-08)

Every gate in [ROADMAP-V19.md](ROADMAP-V19.md) asks whether this port computes
the right thing. None of them asked how fast. That was the wrong omission to
carry into a launch: correctness decides whether the project is honest, and
throughput decides whether anybody runs it.

This document answers the part that is knowable without the checkpoint. It is
organised the way [upstream's EFFICIENCY.md](https://github.com/sqliteai/waste/blob/main/docs/EFFICIENCY.md)
is, and for the same reason — **measurements and estimates are labelled
separately, and the estimates name their inputs**, so a wrong conclusion can
be traced to a wrong number rather than argued about.

**Nothing here has executed an official weight.** The geometry is exact. The
throughput is a projection.

Reproduce anything below with `tools/inkling_throughput.py`.

---

## 1. The exact part

`tools/inkling_throughput.py geometry` — arithmetic over the released config
and the WEXP record layout, with no measurement in it:

| | |
| --- | --- |
| layers | 42 (2 dense, 40 sparse) |
| routed experts, top-k | 256, 6 |
| VQ3R record | **9,457,664 B** |
| records per decoded token | 40 × 6 = **240** |
| **bytes per decoded token** | **2,269,839,360 B = 2.11 GiB** |
| expert bank on disk | 90.20 GiB |
| trunk, resident | 3.64 GiB |
| model-state floor | 4.10 GiB |

The record size reproduces the 9,457,664 B quoted throughout this repository,
and the bank reproduces the published 90.2 GiB. `tests/test_inkling_throughput.py`
checks the size model against a record `inkling_vq.write_expert_record`
actually writes, so the dependency-free copy cannot drift from the writer.

The two shared experts are **not** in this count. They bind as trunk tensors
(`inkling_bind.c:104-106`), so they are resident and never streamed.

### Against K3, which is the model WASTE was built for

| | K3 | Inkling-Small | K3 / Inkling |
| --- | ---: | ---: | ---: |
| records per token | 1,472 | 240 | 6.1x |
| record bytes | 12,404,654 | 9,457,664 | 1.3x |
| **bytes per token** | **17.01 GiB** | **2.11 GiB** | **8.0x** |
| resident trunk | 27.28 GiB | 3.64 GiB | 7.5x |

**Inkling-Small reads 8x less per token than the model this engine already
runs.** That single ratio is the reason to expect a different product.

---

## 2. A measurement, and a bug it found

`tools/diskbench.c` is upstream's I/O benchmark — "random record reads,
cache-bypassed ← the number that sets tok/s".

**On Linux it never bypassed the page cache.** `nocache()` was
`#ifdef __APPLE__` with no Linux arm, and no `open()` used `O_DIRECT`. Any
Linux run with a file smaller than RAM measured memory bandwidth and printed
it under the label "cache bypassed". Measured here, same host, same 8 GiB
file, same 9,457,664 B record:

| | unpatched | with `O_DIRECT` | inflation |
| --- | ---: | ---: | ---: |
| seq read | 5.66 GiB/s | 2.39 GiB/s | 2.4x |
| rand 1 thread | 5.67 GiB/s | 2.08 GiB/s | 2.7x |
| rand 2 threads | 8.41 GiB/s | 2.41 GiB/s | 3.5x |
| rand 4 threads | **14.49 GiB/s** | **3.00 GiB/s** | **4.8x** |

The unpatched run reports 14.49 GiB/s — *faster than the M5 Pro internal SSD
upstream measured at 12.89* — on a cloud VM that really does 3.00. This is
precisely the fiction [GATES.md](https://github.com/sqliteai/waste/blob/main/docs/GATES.md)
Gate 5 exists to prevent ("on a 64 GB machine the kernel was quietly holding
all 17 GB … the measured I/O cost was fiction"), defeated on one platform by
a missing `#else`.

Fixed in `integration/waste/overlay/tools_diskbench.c.diff`: reads open
`O_DIRECT`, a refused `O_DIRECT` is reported rather than silently downgraded,
and the trailing tok/s column takes the model's GB/token instead of hardcoding
K3's 12.5.

**Caveat on the numbers themselves.** This host is a cloud VM on virtio, not a
laptop. 2.08 GiB/s at queue depth 1 is a *slow* modern NVMe. It is a real
lower-ish anchor, not the target hardware.

---

## 3. The projection — an estimate

`tools/inkling_throughput.py project`. The cost model comes from upstream's
measured K3 decomposition (EFFICIENCY.md §4: I/O 1.41 s, expert matmul
1.03 s, everything else 0.50 s), scaled to Inkling:

- **Expert matmul scales with bytes per token.** EFFICIENCY.md §1 states the
  cost is one `vq_apply` pass per (token, expert) pair and `vq_rows` does
  `stages` gathers per row per vector position — which is the same quantity as
  the record's index bytes. **This is the largest single assumption here.**
- **Everything else scales with resident trunk bytes**, which is what
  dominates it on K3. Inkling's attention differs from MLA/KDA — a real K/V
  cache and four short convolutions per layer — so the upper end of the band
  doubles that estimate rather than pretending the shape is known. That is why
  tok/s is reported as a range.
- **Read-ahead turns the sum of I/O and expert compute into a maximum**
  (EFFICIENCY.md §4A, shipped and measured at ~1.6x).

### Calibration self-check

Run the same model on K3's own inputs and it must reproduce K3's own measured
decode, or every Inkling number below is decoration:

```
1.65 s/token = 0.61 tok/s   against a measured 0.56-0.63 tok/s
```

### Projected Inkling-Small decode

| GiB/s | hit | I/O s | expert s | other s | **tok/s** | bound by |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.55 (SATA SSD) | 0% | 3.844 | 0.128 | 0.067-0.134 | 0.25-0.26 | I/O |
| 2.08 (this VM, QD1) | 0% | 1.016 | 0.128 | 0.067-0.134 | 0.87-0.92 | I/O |
| 2.08 | 29% | 0.722 | 0.128 | 0.067-0.134 | 1.17-1.27 | I/O |
| 3.00 (this VM, QD4) | 29% | 0.500 | 0.128 | 0.067-0.134 | 1.58-1.76 | I/O |
| 7.00 (typical NVMe) | 0% | 0.302 | 0.128 | 0.067-0.134 | 2.30-2.71 | I/O |
| 7.00 | 29% | 0.214 | 0.128 | 0.067-0.134 | **2.87-3.56** | I/O |
| 10.73 (M5 Pro, QD1) | 29% | 0.140 | 0.128 | 0.067-0.134 | 3.66-4.84 | I/O |
| 12.89 (M5 Pro, QD4) | 42% | 0.095 | 0.128 | 0.067-0.134 | **3.82-5.13** | compute |

The 29% hit rate is upstream's LEARNED.md §39 measurement at a fifth of a
working set with the router lookahead on; 42% is the Gate 2 figure at a
larger cache.

**Read the "bound by" column.** I expected Inkling to flip the engine from
I/O-bound to compute-bound, and it does not — it stays I/O-bound everywhere
except a top-tier SSD with a warm cache. **Cache and prefetch work continues
to pay on Inkling**, which is the opposite of what I assumed before running
this and is the practical reason to keep investing in `ecache` rather than the
VQ kernels.

---

## 4. The finding that matters most

WASTE's default-budget resolver (Gate 7) takes the largest
`floor + working_set × k`, for k in 3..1, that fits under seven eighths of RAM.
The quantum is one token's working set, because Gate 5 measured that a
demand-only cache below it hits **exactly zero**.

That quantum is 17.01 GiB on K3 and **2.11 GiB on Inkling-Small**:

| step | K3 needs | Inkling-Small needs |
| --- | ---: | ---: |
| 3x working set | 89.5 GiB | **11.9 GiB** |
| 2x | 70.0 GiB | 9.5 GiB |
| 1x | 50.6 GiB | **7.1 GiB** |

So the resolver on a real machine:

| RAM | budget | cache | working sets |
| ---: | ---: | ---: | --- |
| 8 GiB | 6.21 | 2.11 | 1x |
| 16 GiB | 10.44 | 6.34 | **3x — the resolver's maximum** |
| 64 GiB | 10.44 | 6.34 | 3x |

**An 8 GiB machine clears a full working set. A 16 GiB machine reaches the
largest cache the resolver will ever ask for.** K3 needs 50 GiB to reach the
first rung and 90 GiB for the third.

This is the argument. It is not "we ported another model": the streaming-MoE
design needed a 64 GB workstation to demonstrate at 0.5 tok/s, and on
Inkling-Small the same engine, unchanged, fits a 276B frontier model onto the
laptop people already own — with the cache in its best regime rather than its
worst.

---

## 5. What would falsify this

Named now, before the measurement, so the record is honest either way:

1. **VQ3R quality on Inkling's experts is untested.** Every number here — the
   9,457,664 B record, the 2.11 GiB working set, the 90.2 GiB bank, the whole
   budget ladder — assumes 3-bit VQ is an acceptable operating point for this
   model. Upstream chose it for K3 after Gate 3. Inkling activates 6 of 256
   against K3's 16 of 896 (Gate 2) — a larger *fraction* of a smaller pool, on
   a much narrower routed width (2048 against K3's 3584 latent projection), so
   there is no reason to assume the two quantize alike. **If G2 forces 4 bits,
   every figure above moves by ~1.33x and the ladder needs re-deriving.**
   This is the single largest risk to the claim and it is a measurement nobody
   has taken.
2. **The compute scaling is a cost model, not a measurement.** If Inkling's
   per-expert compute does not scale with record bytes, the `expert s` column
   is wrong and the compute-bound corner moves.
3. **"Everything else" is scaled from K3's trunk ratio.** Inkling's four short
   convolutions per layer and its real K/V cache have no counterpart in
   MLA/KDA. The band's width is an admission, not a confidence interval.
4. **The hit rates are K3's**, measured on a different router with a different
   expert count. Inkling's routing distribution is unobserved. Upstream's own
   Gate 2 caveat applies verbatim: reuse falls and concentration rises as
   experts get finer-grained, and which dominates is model-specific.

Item 4 is cheap to close and does not need the checkpoint's weights — only a
routing trace, which `tools/routing_stats.py simulate` already consumes and
which any instrumented run of the official model can emit.

---

## 6. What this changes about the plan

- **G2 (quantized tolerance) is promoted.** It was fourth; it is now the gate
  that decides whether the headline survives, because every geometry figure in
  this document is downstream of VQ3R being usable.
- **A routing trace is worth more than it looks.** It closes falsifier 4 for
  the cost of one instrumented run, and it feeds a tool that already exists.
- **Cache and prefetch work still pays**, contrary to the compute-bound guess.
- **The demo is the deliverable.** A parity report is evidence; a terminal
  recording of a 276B model generating text on a 16 GiB laptop, with a command
  anyone can paste, is the thing that travels.
