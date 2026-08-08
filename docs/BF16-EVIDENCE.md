# Inkling BF16 evidence consolidation

## Scope

This document replaces the narrative spread across pull requests #19–#36,
#38, and #44–#51 with one reviewable evidence record. The retained code is
evidence-only: it copies production sources into temporary directories,
applies fail-closed experimental transforms, and compares the result with the
pinned official Inkling-Small reference.

Nothing here changes production C arithmetic, the public WASTE ABI, the
generated integration bundle, or the fail-closed public inference boundary.

The source lineage is preserved by the original pull requests. This branch
retains the final harnesses and their unit contracts as one commit; it does not
replay the 99-commit research history.

## Durable conclusions

| Boundary | Evidence | Conclusion |
| --- | --- | --- |
| RMSNorm and projections | #19, #21, #22 | Official RMSNorm completes the normalized value to BF16 before multiplying its BF16 scale. Prefill-versus-token-step shape does not explain projection drift; the existing matrix-backend seam can supply native BF16 projections exactly. |
| Attention | #23, #26–#28 | Q/K head normalization is real but insufficient. The complete arithmetic lattice identifies a portable BF16 attention policy that closes the tested attention boundary. The narrower #24/#25 decomposition was superseded by this lattice. |
| Residual and expert path | #29–#36 | BF16 completion closes the post-attention residual, SiLU/product, expert weighting, aggregation, and fixed-ID router-weight boundaries. Composing the proven policies makes the tested sparse MoE path exact. |
| Final sparse-layer residual | #38, #44, #52 | Completing both final-residual operands and the result to BF16 produced raw-exact source-bound position-zero sparse layers 2 and 5 in the authoritative #44 run and four of five serial executions at the merged #52 head. One unchanged #52 execution failed the final exactness classification. The result is measured Ubuntu-hosted evidence with an unidentified runner/CPU-class variable, not a platform-independent theorem. #44 is the retained runner. |
| Stateful execution | #45, #46 | Eight-token stateful execution first develops a large mismatch in layer-2 fixed-ID router weights at positions 3 and 4; attention state and expert arithmetic are not the first large boundary. |
| Router projection and normalization | #48, #49 | Selected router logits are raw-exact across all eight positions for both batched and row-wise projection shapes. Position 3 first loses parity inside `logsumexp`, not in the projection or `logsigmoid`. |
| Denominator reduction | #50, #51 | The exact BF16 exponential terms must be accumulated in float32 and the completed denominator exposed to BF16 once after the full reduction. With that boundary, the expected post-reduction policy is raw-exact for complete routed/shared weights at all eight tested positions. |
| Stateful sparse-layer composition | #57 | Supplying the fixed `0x7f` post-reduction policy through the retained trace adapter closes the previously failing layer-2 stateful ladder at all eight positions on the recorded AVX2 profile. Routed/shared weights, `moe_out`, `mlp_branch`, and `layer_out` are BF16-exact; no first mismatch is present. |

## Authoritative final runs

- Portable attention: workflow `30917634111` on #28.
- Composed sparse MoE: workflow `31221607257` on #36.
- Final residual classification: workflow `31222793235` on #38.
- Complete position-zero sparse layer: workflow `31236655888` on #44.
- Stateful sparse layer and first large boundary: workflows `31237163304`
  and `31237633169` on #45/#46.
- Router projection, normalization, and reduction: workflows `31238203344`,
  `31238511338`, and `31239168785` on #48–#50.
- Post-reduction denominator: workflow `31239470605` on #51. Its terminal
  classification was `expected_post_reduction_policy_is_raw_exact`, with no
  failure position and exact official routed/shared weights at positions 0–7.
- Stateful post-reduction composition: workflow `31277195747`, attempt 2, on
  #57. Its terminal classification was
  `tested_stateful_sparse_mlp_ladders_exact`; all eight positions passed and
  the profiled result was preserved as artifact `9027735369` with ZIP SHA-256
  `b3b7d066d42068156bc2d32455703cff7a359b3f8bc7d224ab15d5311b9f87c5`.
- Retained contract validation at the same #57 head: workflow `31277195712`,
  attempt 3. Every evidence module compiled and all 25 isolated BF16 contract
  files passed.

Every run used immutable model revision
`21152b5312c653be115f33a8342759064144e281`, source-bound fixture metadata,
CRC-verified payloads, Torch 2.13.0, and explicit thread controls. Individual
PR descriptions retain the exact fixture geometry and per-stage metrics.

## Reproducibility qualification

The fresh complete-layer workflow on #52 was executed serially five times at
`ade4324ace74d084f4da69b0afb810cfa91b32a0` with unchanged inputs. Four runs
passed and one failed the exact final classification: workflow run
`31261401762` failed on attempt 1 and passed unchanged on attempt 2; dispatch
runs `31262503047`, `31262827121`, and `31263073077` passed. The BLAS/intra-op
controls were pinned to one, the 64-entry 985,959,432-byte fixture was
CRC-verified each time, and the comparison is exact rather than
tolerance-based. Those facts exclude variable configured thread count,
differing fixture bytes, and a threshold knife edge, but they do not identify
the cause.

The complete-layer claim therefore remains scoped to repeated measurement on
GitHub-hosted `ubuntu-24.04`; the 4/5 result is not a platform-independent
theorem. The result JSON records the workflow/run attempt, runner ID, hosted
image, first `/proc/cpuinfo` model block, Torch CPU dispatch, Torch thread
counts, and thread-control environment. It also records a stable host-class SHA
over those class-defining fields while excluding ephemeral run and runner IDs.
The workflow preserves that JSON even when its exactness assertion fails.

The successful #57 stateful composition is a separate, narrower observation on
the named Linux AVX2 profile: Torch `2.13.0+cu130`, one compute thread, AMD EPYC
7763, and host class
`67b28be7a887a766b02e9896d80a16dfa47321243117592abdd92ce047d8322b`.
It closes the layer-2 eight-position sparse-MLP seam under that profile. It does
not explain the earlier position-zero failure or establish the same result for
another dispatch/backend profile.

## What is retained

- the exact Python harness and test blobs at the #51 tip, which contains the
  requested #19–#36/#38/#44–#51 lineage;
- one dependency-light workflow that compiles every retained module and runs
  every BF16 unit contract in a fresh process;
- the complete position-zero sparse-layer and eight-position router-denominator
  workflows, plus a separate bounded workflow that composes those policies
  statefully without rewriting the retained historical harnesses.

The 23 per-checkpoint workflows are intentionally omitted. They were useful
while locating the boundary, but keeping all of them would turn historical
search steps into permanent CI surface.

## Current frontier

The #57 hosted run satisfies the deliberately narrow acceptance rule for the
previously failing local sparse layer 2: the position-zero anchors held, all
eight routed/shared weights and downstream ladder stages satisfied their
established raw/BF16 contracts, and the terminal classification was
`tested_stateful_sparse_mlp_ladders_exact`. The exact 33-expert, 87-entry,
1,851,934,212-byte fixture was CRC-verified before execution.

That result must be described as **profile-bound stateful layer-2 evidence**.
It is not full stateful-layer, full-decoder, generation, tokenizer, container,
or public-runtime parity. In particular:

- the policy is still supplied by evidence-only transforms/adapters rather than
  checked-in production C;
- the unexplained 4/5 position-zero observation remains open across hosted
  profiles;
- stateful dense layer 0 and global sparse layer 5 are not closed by #57;
- cross-layer decoder continuity, final normalization, logits, tokenization,
  and generation have not been tested under the composed policy.

The next evidence action is the predeclared same-host backend matrix: native
AVX512, forced ATen AVX2, oneDNN AVX2, and MKLDNN disabled against one bound
fixture. Once that classifies the profile-sensitive seam, the smallest useful
implementation change is to promote the proven arithmetic policy into a
private, fail-closed C execution profile and rerun the representative
dense/local-sparse/global-sparse fixtures without rewriting production source
at test time. Public dispatch remains out of scope for that promotion.

## Safety boundary

All transforms remain temporary and source-integrity checked. Official data
acquisition remains immutable, bounded, and fail-closed. Public Inkling
loading, stepping, generation, and serving remain unsupported.
