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

The durable claim is therefore scoped to repeated measurement on GitHub-hosted
`ubuntu-24.04`, with host class still unknown. The complete-layer result JSON
records the workflow/run attempt, runner ID, hosted image, first `/proc/cpuinfo`
model block, Torch CPU dispatch, Torch thread counts, and thread-control
environment. It also records a stable host-class SHA over those class-defining
fields while excluding the ephemeral run and runner IDs. The workflow preserves
that JSON even when its exactness assertion fails. A named official-reference
profile should replace this provisional scope only after the metadata either
correlates a second failure with a host class or rules host class out.

## What is retained

- the exact Python harness and test blobs at the #51 tip, which contains the
  requested #19–#36/#38/#44–#51 lineage;
- one dependency-light workflow that compiles every retained module and runs
  every BF16 unit contract in a fresh process;
- the two final official-weight workflows: complete position-zero sparse-layer
  parity and eight-position router-denominator parity.

The 23 per-checkpoint workflows are intentionally omitted. They were useful
while locating the boundary, but keeping all of them would turn historical
search steps into permanent CI surface.

## Current frontier

The current arithmetic policy is stronger than the #46 stateful candidate
because #51 closes its eight-position router-weight defect. However, #51 is a
router-only proof: it does not rerun the complete stateful sparse layer with
the corrected denominator policy.

The next justified experiment is therefore one bounded stateful sparse-layer
run that composes the #51 denominator rule with the retained #44–#46 candidate.
Until that passes, this evidence must not be described as full stateful-layer,
full-decoder, generation, tokenizer, container, or public-runtime parity.

Separately, position-zero exactness must not become a settled G1 premise until
the 4/5 observation is resolved. Repeat the profiled complete-layer run until a
second failure is captured. If failures cluster by CPU/runner profile, bind the
claim to that named profile; otherwise investigate the `logsumexp` denominator
reduction as an unresolved arithmetic defect.

## Safety boundary

All transforms remain temporary and source-integrity checked. Official data
acquisition remains immutable, bounded, and fail-closed. Public Inkling
loading, stepping, generation, and serving remain unsupported.
