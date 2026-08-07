# MVP readiness handoff

## Public boundary remains unchanged

The public memory planner is live for valid Inkling manifests. Public loading,
step execution, generation, chat, and serving remain deliberately refused.
Nothing in the current BF16 evidence stack relaxes that boundary.

The next hard gate is no longer “obtain the first official fixture.” The
repository now has bounded official-weight evidence deep enough to separate
three different questions that previously looked like one parity problem:

1. sparse-layer arithmetic with routed expert IDs held equal;
2. router tie semantics at exact BF16 top-k cutoffs;
3. stateful, multi-position decoder behavior.

Those questions must remain separate in both CI and readiness claims.

## Current evidence frontier

The stacked evidence through PRs #39–#42 establishes the following bounded
results. These are evidence-branch results and do not enable the public runtime.

### Position-zero sparse-layer arithmetic

PR #39 composes the proven BF16 attention, residual, router-weight,
expert-local, aggregation, MLP, and final-residual policies in a copied
temporary source. With canonical routed IDs supplied, both sparse layer classes
used in the bounded probe reproduce the official position-zero decoder-layer
output exactly:

- layer 2: final `layer_out` raw exact and BF16 exact;
- layer 5: final `layer_out` raw exact and BF16 exact;
- every traced stage from `post_attention_norm` through `layer_out` is BF16
  exact;
- the result reproduced on two hosted AVX2 executions of the exact same head.

The hidden FP32 `mlp_branch` need not be raw-exact. Its BF16-completed value is
exact, and the proven final residual boundary prevents the hidden FP32
remainder from affecting `layer_out`.

This is **position-zero, same-route arithmetic parity**. It is not full-model
parity.

### Exact BF16 router cutoff ties are not a portable official ID rule

PR #40 reconstructs the official BF16 router choice values and audits every
committed eight-token route row. Unchanged `torch.topk(sorted=False)` is stable
on one pinned host, but value-preserving permutations change which expert IDs
are selected from exact BF16 cutoff ties while remaining inside the same valid
cutoff equivalence class.

The observed official tied subset does not follow a consistent lowest-ID or
highest-ID rule. Therefore the C router must not be changed merely to imitate
one observed PyTorch partition result.

### Cutoff-equivalent routes are not output-equivalent

PR #41 enumerates every valid position-zero cutoff-equivalent route for the two
sparse layer classes and executes the complete official layer for each route.
The alternatives materially change final `layer_out`:

- layer 2: worst observed absolute change `90.4375`;
- layer 5: worst observed absolute change `112`.

The deterministic C low-ID route, however, is raw-exact to an official
counterfactual using that same low-ID route for both layers. This isolates the
large divergence to route choice rather than sparse-layer arithmetic.

Consequently, the strict evidence gate must **not** treat “same BF16 cutoff
class” as activation or generation parity.

### Official tied route IDs are cross-platform variant

PR #42 exports the exact source-bound 8×256 BF16 router-choice bit patterns and
runs only Torch 2.13 CPU `topk` over those identical bits on Linux, Windows,
and macOS.

The source profile is Linux x86_64, Torch 2.13.0, Transformers 5.14.1, forced
AVX2. The exact archive SHA-256 is:

`686877d38f441df16ba6f89ef0dcbf5a1c84a2f65c3389aac4b3a86c7cafa766`

The result reproduced end-to-end on the exact same head:

- Linux reference: all 16 route sets match the source archive;
- Windows: tied subsets change on 5/5 ambiguous layer-2 rows and 4/6
  ambiguous layer-5 rows;
- macOS: tied subsets change on 2/5 ambiguous layer-2 rows and 5/6 ambiguous
  layer-5 rows;
- no platform changes any unambiguous route row.

This means a platform-independent claim of exact official tied-route IDs is not
well-defined. Exact official claims at ambiguous BF16 cutoffs must be attached
to a named reference profile.

See `docs/INKLING-REFERENCE-PROFILES.md` and
`docs/OFFICIAL-REFERENCE-PROFILE-LINUX-AVX2.json`.

## Routing contracts from here

Two contracts now coexist and must not be silently substituted for one another.

### Portable deterministic WASTE routing

The current C router uses an explicit deterministic lower-expert-ID tie policy
for equal choice scores. That is the portable WASTE behavior unless a future
review changes it.

A WASTE route selected under that rule may differ from a pinned official
reference at an ambiguous BF16 cutoff. Such a difference must be reported as a
routing-semantic difference, not hidden behind an activation tolerance.

### Pinned official-reference profile

An exact official route/output claim must name enough environment information
to reproduce the official selection primitive: model hashes, input hash, Torch
and Transformers versions, OS/runtime, CPU dispatch, thread controls, route
archive, and the BF16 choice-archive identity where applicable.

The first machine-readable profile is
`docs/OFFICIAL-REFERENCE-PROFILE-LINUX-AVX2.json`.

This profile is evidence metadata only. It is not a runtime configuration and
must not be used to auto-enable loading or inference.

## Existing bounded fixture infrastructure

The remote extractor still provides the larger evidence-selected fixture plan
for layers 0, 2, and 5 at immutable model revision
`21152b5312c653be115f33a8342759064144e281`:

- 175 entries;
- 3,842,395,658 payload bytes, approximately 3.58 GiB;
- 179,397 metadata bytes and 52 requests to plan;
- 25 of 32 checkpoint shards touched by header ranges.

The per-position official route selection, weights, deterministic BF16 input
SHA-256, release revision, config hash, and index hash remain committed in
`docs/OFFICIAL-ROUTER-SELECTION.json`.

The remote extractor fails closed on mutable revisions, unsafe paths, invalid
safetensors geometry, missing tensors, incomplete requested expert selections,
and servers that ignore HTTP `Range`. Every evidence workflow CRC-verifies the
payload it executes.

## Next numerical gate: multi-position and stateful parity

The next hard numerical work should not chase a platform-specific `topk`
artifact. It should validate stateful execution while keeping routing semantics
explicit.

Recommended sequence:

1. Extend the current complete temporary BF16 candidate from position zero to a
   bounded multi-position sequence.
2. Preserve real attention and short-convolution state across positions rather
   than rebuilding fresh state per token.
3. At each audited position, run the deterministic WASTE route policy and force
   the official counterfactual to the same routed expert IDs.
4. Recompute official routed/shared weights from those fixed IDs using the
   pinned BF16 reference arithmetic.
5. Require exact or explicitly justified BF16 parity at each state boundary and
   final `layer_out` before advancing to the next position.
6. In parallel, retain a separate profile-bound official-reference check that
   records exact official route IDs for a named profile. Never mix this with
   the portable deterministic-route result.
7. Expand expert coverage only from source-bound observed routes; never silently
   widen fixtures after a missing expert.

The first multi-position experiment should remain evidence-only in
`experiments/inkling/`. Do not promote the temporary BF16 source into production
as part of that probe.

## Remaining gates before an MVP

Even successful multi-position layer evidence is not sufficient for public
inference. The remaining sequence is:

1. Establish stateful multi-position decoder parity across the relevant layer
   classes under explicit routing semantics.
2. Extend from bounded layer evidence to a full decoder-stack/reference pass
   with a named official profile and deterministic WASTE contract kept
   separate.
3. Establish tokenizer and chat-template parity.
4. Validate the intended quantized container rather than extrapolating from the
   approximately 532 GB BF16 checkpoint.
5. Measure RAM, mmap/cache behavior, direct-I/O/storage throughput, expert-cache
   behavior, and token latency on the intended laptop class.
6. Regenerate and review the WASTE integration bundle only when production
   arithmetic/runtime code is intentionally promoted.
7. Perform an explicit public-boundary review before enabling loading, stepping,
   generation, chat, or serving.

Unsupported variants must continue to fail closed.

## Definition of an MVP

The first usable MVP remains a normal WASTE text flow, not a special parity
binary: `waste plan`, `waste run`, and the OpenAI-compatible server against one
verified Inkling-Small container and one documented runtime contract.

That milestone requires more than the current position-zero arithmetic result.
It requires stateful decoder evidence, tokenizer/chat parity, a validated
container and resource envelope, and a deliberate decision about which routing
contract public execution promises.

Multimodal execution and speculative decoding remain outside that first MVP
boundary.
