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

## 2. A measurement, and a bug someone else had already found

`tools/diskbench.c` is upstream's I/O benchmark — "random record reads,
cache-bypassed ← the number that sets tok/s". On Linux it was neither:
`nocache()` had an `#ifdef __APPLE__` body and nothing else, and `O_DIRECT`
appeared nowhere in the file, so any run with a file smaller than RAM measured
memory bandwidth and printed it under the label "cache bypassed".

**This repository found that independently on 2026-08-08 and it was four days
late.** Upstream fixed it on 2026-08-04 in `def83ef`, reported and diagnosed by
fab2s in [sqliteai/waste#22](https://github.com/sqliteai/waste/pull/22), and
recorded it as LEARNED §49. A standalone patch prepared here was withdrawn
rather than sent: it would have been a duplicate, and it was worse in two ways
that are worth keeping on the record because both were my reasoning failing,
not my code.

1. **Upstream bypasses the write; I deliberately did not.** I left it buffered
   on the argument that a buffered write plus `fsync` models the conversion
   landing. Upstream measured that argument wrong: `F_NOCACHE` stops new pages
   being cached but does not evict resident ones, so a buffered write leaves
   the file in the cache and *every read row below it reports RAM* — 8.07 GB/s
   sequential with the write bypassed against 26.04 GB/s with it buffered, on
   an M5 Pro. The original `nocache()` on the write fd was load-bearing and I
   removed its equivalent.
2. **Upstream probes `O_DIRECT` with a real aligned transfer; I trusted the
   open.** `O_DIRECT` can be accepted at open and refused at transfer — tmpfs
   does exactly this — so my version would turn a refusing filesystem into
   short reads and a table of zeroes with no cause given. Theirs falls back to
   a plain open plus `POSIX_FADV_RANDOM` and labels every row.

### The numbers, from upstream's version

Re-measured with WASTE 0.6.6's `diskbench`, same host, same 8 GiB file, same
9,457,664 B record:

| | unpatched (0.6.3) | upstream 0.6.6 | inflation |
| --- | ---: | ---: | ---: |
| seq read | 5.66 GiB/s | 2.19 GiB/s | 2.6x |
| rand 1 thread | 5.67 GiB/s | 2.11 GiB/s | 2.7x |
| rand 2 threads | 8.41 GiB/s | 2.36 GiB/s | 3.6x |
| rand 4 threads | **14.49 GiB/s** | **2.45 GiB/s** | **5.9x** |

The unpatched run reports 14.49 GiB/s — faster than the 12.89 GiB/s upstream
measured on an M5 Pro internal SSD — on a cloud VM that really does 2.45.

**An earlier revision of this document quoted 3.00 GiB/s at four threads**,
from the local fix rather than upstream's. That figure was 22% optimistic, in
exactly the direction the buffered write predicts. Every projection below uses
upstream's numbers.

**Caveat on the numbers themselves.** This host is a cloud VM on virtio, not a
laptop. 2.11 GiB/s at queue depth 1 is a *slow* modern NVMe. It is a real
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
| 0.55 | 29% | 2.729 | 0.128 | 0.067-0.134 | 0.35-0.36 | I/O |
| 0.55 | 42% | 2.229 | 0.128 | 0.067-0.134 | 0.42-0.44 | I/O |
| 2.11 (this VM, QD1) | 0% | 1.002 | 0.128 | 0.067-0.134 | 0.88-0.94 | I/O |
| 2.11 | 29% | 0.711 | 0.128 | 0.067-0.134 | 1.18-1.29 | I/O |
| 2.11 | 42% | 0.581 | 0.128 | 0.067-0.134 | 1.40-1.54 | I/O |
| 2.45 (this VM, QD4) | 0% | 0.863 | 0.128 | 0.067-0.134 | 1.00-1.08 | I/O |
| 2.45 | 29% | 0.613 | 0.128 | 0.067-0.134 | 1.34-1.47 | I/O |
| 2.45 | 42% | 0.500 | 0.128 | 0.067-0.134 | 1.58-1.76 | I/O |
| 7.00 (typical NVMe) | 0% | 0.302 | 0.128 | 0.067-0.134 | 2.30-2.71 | I/O |
| 7.00 | 29% | 0.214 | 0.128 | 0.067-0.134 | 2.87-3.56 | I/O |
| 7.00 | 42% | 0.175 | 0.128 | 0.067-0.134 | 3.24-4.13 | I/O |
| 10.73 (M5 Pro, QD1) | 0% | 0.197 | 0.128 | 0.067-0.134 | 3.03-3.79 | I/O |
| 10.73 | 29% | 0.140 | 0.128 | 0.067-0.134 | 3.66-4.84 | I/O |
| 10.73 | 42% | 0.114 | 0.128 | 0.067-0.134 | 3.82-5.13 | compute |
| 12.89 (M5 Pro, QD4) | 0% | 0.164 | 0.128 | 0.067-0.134 | 3.36-4.33 | I/O |
| 12.89 | 29% | 0.116 | 0.128 | 0.067-0.134 | 3.82-5.13 | compute |
| 12.89 | 42% | 0.095 | 0.128 | 0.067-0.134 | 3.82-5.13 | compute |

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

## 6. Where an order of magnitude is, and is not (2026-08-08)

§3 projected the cache's contribution from K3's measured hit rates. That was
the weakest input in the document, and it is now measured for Inkling's own
geometry — against upstream's **real** `ecache.c`, not a second Python
reimplementation of the policy, because Gate 5 already established that a
Python model of this cache is wrong in a known direction.

`tools/inkling_cache_sim.c` links `src/ecache.c` and runs the shipping LFRU
implementation — sampled eviction, frequency-first ordering, hash table and
all — over traces from `tools/inkling_cache_trace.py`.

### The trace is a model, and it earns its keep or it does not

Inkling's router has never been observed. The Kimi family's has, twice:
Gate 2 measured its concentration (top 8.7% of experts cover 50% of
activations) and its next-token reuse (33.6%). The generator has exactly two
free parameters, one fitted to each.

Concentration alone reproduces only **11.6%** reuse against a measured 33.6%,
so routing carries real temporal correlation that an IID draw does not
produce. That gap *is* the second parameter, and nothing more.

Two parameters, two measurements — so the model predicts nothing yet. What it
has to earn is Gate 5's **real-cache hit curve**, which it was not fitted to:

| frac of expert set | Gate 5 measured | model | delta |
| ---: | ---: | ---: | ---: |
| 3.0% | 13.2% | 5.2% | −8.0pp |
| 6.0% | 40.3% | 24.3% | −16.0pp |
| 12.1% | 61.9% | 53.1% | −8.8pp |
| 24.2% | 84.8% | 74.6% | −10.2pp |
| 48.4% | 93.9% | 87.6% | −6.3pp |

It reproduces the shape and the knee and **under-predicts the level at every
budget**, mean −9.9pp. The cause is structural: the trace's correlation is
first-order, so reuse decays geometrically, while real routing also has
session-scale structure — a prompt's domain keeps a subset of experts warm for
hundreds of tokens, which is exactly what a cache converts into hits.

**So every cache figure below is a lower bound on hit rate and an upper bound
on bytes read.** That is the safe direction for a throughput claim, and it is
why the model was not tuned until the gap closed: a third parameter fitted to
Gate 5 would have made the model fit everything and predict nothing.

### Measured: the cache on Inkling's geometry

300 decode tokens, 40 sparse layers, top-6 of 256, 9,457,664 B records.

| cache | slots | % of bank | LFRU hit | GiB/token | LRU hit |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1.00 GiB | 113 | 1.1% | **0.0%** | 2.114 | 0.0% |
| 2.11 GiB | 239 | 2.3% | 4.6% | 2.017 | 15.9% |
| 4.00 GiB | 454 | 4.4% | 18.0% | 1.734 | 34.2% |
| **6.34 GiB** | 719 | 7.0% | **34.7%** | **1.355** | 40.7% |
| 8.00 GiB | 908 | 8.9% | 43.7% | 1.190 | 45.1% |
| 16.00 GiB | 1816 | 17.7% | 66.7% | 0.704 | 62.5% |
| 32.00 GiB | 3633 | 35.5% | 82.2% | 0.376 | 80.4% |

The first row reproduces upstream's most predictive measurement independently:
**a cache below one token's 2.11 GiB working set hits exactly zero.**

The 16 GiB-laptop budget of §4 — 6.34 GiB of cache — gives **34.7%**, so
1.355 GiB/token rather than 2.11.

One anomaly, flagged rather than acted on: **LRU beats LFRU below ~8 GiB
here**, which is the reverse of Gate 2's measurement on a real trace (LRU
collapsing to 5.1% where LFRU held 29.4%). A first-order trace is structurally
recency-friendly, so this is most likely an artifact of the generator and
**not** a reason to change policy. It is a reason to distrust the small-cache
rows specifically.

### Measured: the batching lever, isolated

Positions routed together share every record but keep all their arithmetic.
EFFICIENCY.md §1 measured 0.24x reads/token at N=32 on K3.

| chunk | no cache | with 6.34 GiB cache |
| ---: | ---: | ---: |
| 1 | 2.114 GiB (1.00x) | 1.355 GiB (1.00x) |
| 4 | 1.564 (0.74x) | 1.335 (0.99x) |
| 16 | 1.095 (0.52x) | 1.095 (0.81x) |
| 32 | 0.834 (0.39x) | 0.834 (0.62x) |
| 64 | 0.591 (0.28x) | 0.591 (0.44x) |

Two findings, neither of them the expected one:

1. **Batching is worth less on Inkling than on K3** — 0.39x at chunk 32 against
   K3's 0.24x. Inkling already reads 8x less per token, so there is less
   redundancy left to remove. The lever shrinks as the model gets leaner.
2. **The cache and the chunk are substitutes, not complements.** At chunk ≥ 16
   the two columns are *identical* — a 6.34 GiB cache buys exactly nothing that
   chunking has not already taken. (Caveat: with the generator's short memory,
   cross-chunk reuse is understated, so a real trace would leave the cache
   more to do. The convergence is directionally right and quantitatively soft.)

### The ceiling nobody can move

`tools/inkling_throughput.py ceiling`. Exact, and nothing above touches it:

| | GFLOP per decoded position |
| --- | ---: |
| routed experts (240) | 12.08 |
| shared experts (80) | 4.03 |
| dense MLP (2) | 0.81 |
| **total** | **16.91** |

Put the best I/O case against it. At 7 GiB/s with chunk 64 — every lever in
this document stacked — I/O is 0.084 s/token, while the expert matmul alone is
~0.128 s. **The step is compute-bound, and it was I/O-bound before any of this.**

| what | s/token | tok/s |
| --- | ---: | ---: |
| §3 baseline, chunk 1, modelled cache | 0.294 | 3.4 |
| measured cache, chunk 1 | 0.294 | 3.4 |
| measured cache, chunk 64 | 0.228 | 4.4 |
| **I/O made free (0 s)** | **0.228** | **4.4** |

**Every remaining I/O lever combined is worth about 1.3x, and the asymptote of
free I/O is 4.4 tok/s.** An order of magnitude on top of §3's 3.4 tok/s means
34 tok/s, which demands **575 GFLOP/s sustained** before attention, norms,
routing or dequantization. There is no cache policy, chunk size or prefetch
depth that produces it.

### The order of magnitude was in our own code

The paragraph above stood for about an hour. Looking for what the expert
matmul actually executes turned up something better than a GPU, and it is not
subtle.

**This port expands every expert to F32 before multiplying it.**
`decode_matrix` in `inkling_wexp.c` reconstructs the whole quantized matrix —
3 x 2048 x 4096 floats, **100.7 MB per expert** — into a workspace, and
`inkling_layer.c` then runs an ordinary dense matvec over it.

Upstream's `vq_matvec` (`src/model.c`) never expands an expert at all. It
builds a table from the *activation* once per matrix,
`lut[v][stage][entry] = sum_d x[v*dim+d] * book[stage][entry][d]`, and each
output row becomes `nv * stages` table lookups and adds. Upstream's own README
makes the point in passing — "WASTE is ~100x faster and never expands an
expert at all" — and `docs/WASTE-CONSTRAINTS.md` §2 already said the scalar
path "should **not** become the hot path". Nobody had measured what that
sentence was worth.

`tools/inkling_expert_bench.c` implements both and **checks they agree before
it will print a timing** — a speed comparison between two implementations of
different functions measures nothing. They agree to a relative 2.4e-06, so the
LUT formulation is a correct drop-in.

| | expand-then-dense | LUT gather | ratio |
| --- | ---: | ---: | ---: |
| one matrix, `-O2` | 29.4 ms | 6.6 ms | **4.5x** |
| one matrix, `-O3` | 22.4 ms | 4.9 ms | 4.6x |
| one matrix, `-O3 -march=native` | 22.9 ms | 3.9 ms | **5.9x** |
| DRAM traffic per expert | 201.3 MB | 11.0 MB | **18.3x** |
| **DRAM traffic per token** (320 experts) | **60.0 GiB** | **3.28 GiB** | 18.3x |

Three things make this robust rather than a microbenchmark artifact:

1. **The ratio grows with optimization** (4.5x → 5.9x). The expand path is
   memory-bound and cannot be vectorized out of it; the LUT build is a dense
   dot product that vectorizes well. Better compilers widen the gap.
2. **The traffic ratio is implementation-independent.** Expanding writes
   100.7 MB and reads it straight back, per expert, per token. No amount of
   SIMD or threading changes that 60 GiB.
3. **60 GiB/token of DRAM traffic is 28x what the model reads from disk.**
   Every §6 measurement above optimises a 2.11 GiB disk read while the same
   token pushes sixty gigabytes through memory. The I/O analysis was
   optimising the wrong side of the machine.

The LUT path's 3.28 GiB/token is *mostly the index planes themselves* — bytes
the engine had to read from disk anyway. It adds a 1.5 MB table per matrix,
which is L2-resident.

**Caveat, stated plainly:** both implementations here are scalar and
single-threaded, so the absolute times are not the engine's. Upstream's real
`vq_apply` is threaded and tiled, with interleaved rows to hide the
gather's load-address-load dependency — `model.c`'s own comments record that
tiling as the change that finally moved it. The **ratio** is the finding; the
absolute numbers are this machine's.

### And upstream has already measured how much of a ratio like this survives

This is the part that should temper the section above, and it comes from
WASTE 0.6.6's `c10b2fb`, which added `WQ_VQ4P` and benchmarked a table-lookup
kernel against the VQ3R gather:

> The kernel delivers. **In isolation it is 3.88x** … In the engine it is
> **1.18x on Kimi-Linear and 1.09x on K3** … The gap between 3.88x on a bench
> and 1.17x in place **is the finding, and it is not explained.**

So a 3.88x isolation win bought 1.1-1.2x in place, and upstream could not
account for the gap. That is a direct warning about the 4.5-5.9x above, which
is an isolation number of exactly the kind that did not survive.

**Two reasons to expect this one to survive better, and neither is proof:**

- Their comparison is *kernel against kernel* — VQ4P's byte shuffle against
  VQ3R's gather, both LUT formulations, both already avoiding expansion. Ours
  is *structural*: it deletes a 100.7 MB materialization and the 201.3 MB of
  DRAM traffic that goes with it, per expert, per token. A gather-rate
  improvement can be swallowed by whatever bounded the step already; 60 GiB of
  removed memory traffic has to go somewhere.
- The traffic ratio is arithmetic over the record layout, not a timing.

**And one reason to expect it to be harder than it looks.** The same commit
records that an int8 runtime table made the engine *discontinuous* — "a logit
moved 0.68 by a 1e-8 FMA difference … so paths must be bit-identical rather
than equivalent." This benchmark's agreement check is `rel < 1e-5`, and it
measures 2.4e-06. **That is equivalence, not bit-identity**, and upstream's
experience says equivalence is not the bar for swapping a path in this engine.
Landing the LUT formulation means a bit-identity requirement this benchmark
does not currently test, which is a real cost and belongs in the estimate.

### So the first order of magnitude is a port fix, and the next is a backend

In order, largest first:

1. **Stop expanding experts** — a measured 4.5-5.9x *in isolation* on expert
   compute and 18.3x less DRAM traffic, in code this repository owns, against
   a reference implementation upstream already ships and this port already
   links. It is the largest item in this document, and upstream's VQ4P result
   (3.88x on a bench, 1.09-1.18x in place) is the reason not to call it the
   most certain one. Expect a range, not the headline.
   `docs/ROADMAP-V19.md` G6 step 3 already says to reuse `ecache` and the
   optimized VQ kernels "rather than the scalar `inkling_wexp.c` path"; this
   measures the price of not having done it.
2. **The remaining I/O levers**, worth about 1.3x combined (§6 above), and
   worth less once (1) lands because the step gets shorter on the compute side
   without the reads changing.
3. **A GPU backend**, for whatever is left. WASTE is already shaped for one:
   `ecache.h` aligns slots to 16 KiB specifically so that "Metal's
   `newBufferWithBytesNoCopy`" can read a record the CPU already holds, so a
   streamed expert lands in a buffer the GPU can use without a copy. The hard
   part of GPU MoE offload is feeding it, and feeding is what this engine
   already does. Batching belongs here rather than in (2): chunk 32-64 turns
   the expert GEMV into a GEMM, which is the shape a GPU can saturate.

**Not claimed:** that (1) has been landed in the runtime — the benchmark is
standalone, and wiring it means routing the layer through a quantized backend
rather than a resident-F32 expert, which is real work with real parity risk
and belongs behind G1. Nor that a GPU backend exists, nor that the
dequantization pipeline survives the move to one.

The claim is narrower and more useful than the one this section opened with:
**the I/O work that looked like the frontier is nearly finished, the biggest
remaining win is a known defect in this port rather than a missing feature,
and only after that is it a hardware question.**

## 7. What this changes about the plan

- **G2 (quantized tolerance) is promoted.** It was fourth; it is now the gate
  that decides whether the headline survives, because every geometry figure in
  this document is downstream of VQ3R being usable.
- **A routing trace is worth more than it looks.** It closes falsifier 4 for
  the cost of one instrumented run, and it replaces the fitted generator in §6
  with the real thing, which is the largest uncertainty in every cache number
  here.
- **The I/O frontier is nearly closed.** §6 measures every remaining I/O lever
  at about 1.3x combined, with free I/O asymptoting at 4.4 tok/s. Further cache
  and prefetch work is no longer the highest-value engineering.

  *(This supersedes the line that stood here before §6 was measured. §3's
  bandwidth grid put Inkling I/O-bound at every realistic disk and I concluded
  that cache work still paid. That was right about the grid and wrong about the
  consequence: with the cache measured rather than assumed, the step is
  compute-bound at the operating point, and it is the arithmetic that binds.)*
- **The largest single win is a defect in this port, not a missing feature.**
  `inkling_wexp.c` expands every expert to 100.7 MB of F32 before multiplying
  it, where upstream's `vq_matvec` never expands one: a measured 4.5-5.9x on
  expert compute and 18.3x less DRAM traffic, against a reference
  implementation already linked into the same binary. §6 has the numbers.
- **A GPU backend is the win after that**, with batching as its enabler rather
  than a lever in its own right, because chunk 32-64 is what turns the expert
  GEMV into a GEMM.
- **The demo is the deliverable.** A parity report is evidence; a terminal
  recording of a 276B model generating text on a 16 GiB laptop, with a command
  anyone can paste, is the thing that travels.
