# Inkling reference profiles

> Consolidated from PRs #40–#43. The executable tie-sensitivity harness is
> ported onto the retained #44 complete-sparse-layer runner; the original
> observations and profile hashes are unchanged.

## Why this contract exists

Inkling-Small exposes a real reproducibility boundary at BF16 router cutoffs.
The current evidence stack separates three facts that must not be conflated:

1. With the routed expert IDs held equal, the position-zero sparse-layer
   arithmetic can reproduce the official layer exactly under the pinned native
   BF16 backend (#39, independently retained by the #44 lineage).
2. When several routed experts have exactly equal BF16 choice scores at the
   top-k cutoff, the pinned official CPU `torch.topk` does not define a portable
   expert-ID tie rule (#40).
3. Different valid tied subsets can materially change the complete decoder
   layer output (#41), and the exact tied subset selected from identical BF16
   choice bits differs across Linux, Windows, and macOS Torch 2.13 CPU builds
   (#42).

Therefore the phrase **official parity** is incomplete unless it names the
reference environment that resolves ambiguous tied routes.

This document defines two distinct contracts. Neither contract enables public
Inkling execution.

## 1. Portable deterministic WASTE routing contract

The portable WASTE contract is the implementation-defined behavior intended to
be reproducible across supported WASTE platforms.

For the current C router:

- routed choice scores are ordered by value;
- experts with exactly equal choice scores are ordered by lower expert ID;
- the tie policy is explicit and deterministic rather than inherited from a
  platform standard-library partition primitive;
- routed/shared normalization is a separate arithmetic contract;
- changing this tie policy requires an explicit review and new evidence.

The implementation currently lives in `inkling/src/inkling.c` and must remain
fail-closed for unsupported execution paths.

A portable WASTE route is **not** automatically an exact official-reference
route at an ambiguous BF16 cutoff. PR #41 proves that this distinction is
numerically material, so a cutoff-equivalent route must never be described as
output-equivalent without direct evidence.

## 2. Pinned official-reference profile

An official-reference profile is a complete description of the environment in
which an exact official route/output claim was observed. At minimum it binds:

- model ID and immutable model revision;
- model `config.json` SHA-256;
- safetensors index SHA-256;
- deterministic input SHA-256 and input dtype;
- Torch version, including the wheel/build identity when relevant;
- Transformers version when model-side arithmetic is executed;
- operating system, architecture, and relevant runtime/toolchain identity;
- canonical visible CPU feature flags as well as the CPU model identity;
- CPU dispatch policy, including an `ATEN_CPU_CAPABILITY` override when used;
- oneDNN version/build fingerprint and `ONEDNN_MAX_CPU_ISA` when used;
- whether `torch.backends.mkldnn.enabled` was true for the reference process;
- thread/BLAS controls that can affect the native reference backend;
- route-selection artifact identity;
- BF16 router-choice archive SHA-256 when the route primitive is isolated;
- scope of the claim and the evidence run that established it.

The first committed profile is
`docs/OFFICIAL-REFERENCE-PROFILE-LINUX-AVX2.json`. It identifies the Linux
x86_64 / Torch 2.13.0 / Transformers 5.14.1 / forced-AVX2 reference used by the
current source-bound router archive. It is evidence metadata, not a runtime
configuration switch.

## Exact-route claim rules

An exact route claim must follow these rules:

1. **Unambiguous cutoff:** exact routed expert IDs are required. PR #42 observed
   zero unambiguous route-set changes across Linux, Windows, and macOS for the
   audited archive.
2. **Ambiguous BF16 cutoff:** exact tied expert IDs are meaningful only against
   a named official-reference profile. They are not a platform-independent
   model-value property.
3. **Portable WASTE claim:** report the deterministic WASTE tie policy and do
   not relabel its tied subset as universal official behavior.
4. **Counterfactual arithmetic claim:** if the official reference is forced to
   the same route as WASTE, state that route explicitly. #41 shows the current
   deterministic low-ID route is arithmetic-exact for the tested position-zero
   sparse layers when both sides use the same experts.
5. **Cutoff-equivalent is not output-equivalent:** do not relax activation or
   generation parity merely because two routes are valid members of the same
   BF16 cutoff class. The measured layer-output changes are large.

## Same-host reference-matrix classification

The position-zero exactness instability is tested by a paired experiment, not
by separate workflow dispatches. An eligible job must:

1. check `/proc/cpuinfo` for `avx512f` before checkout or dependency install;
2. install the pinned stack and continue only if native Torch reports AVX512;
3. acquire and CRC-verify one source-bound fixture;
4. execute four fresh processes on that runner: native, forced ATen AVX2,
   `ONEDNN_MAX_CPU_ISA=AVX2`, and
   `torch.backends.mkldnn.enabled=False`;
5. require one schema-version-4 `host_class_sha256`, including canonical CPU
   feature flags, while retaining four distinct
   `reference_profile_sha256` values;
6. retain float32 and BF16 payload bits and hashes at every ordered pre-router
   point from `input_norm` through `post_attention_norm`; and
7. retain oneDNN dispatch logs, backend enabled state, oneDNN version, and the
   Torch build-configuration hash for every arm.

Ineligible hosts are neutral observations. They must skip the fixture download
and must not count as exactness passes or failures.

The outcome meanings are fixed before the next measurement:

| Native / ATen AVX2 | oneDNN AVX2 | MKLDNN off | Earliest actual compared boundary | Classification | Consequence |
| --- | --- | --- | --- | --- | --- |
| exact / exact | exact | exact | none | `all_reference_profiles_exact` | No failure reproduced; no profile is cleared globally. |
| nonexact / nonexact | exact | any | any | `onednn_avx512_isa_path_is_profile_sensitive` | Scope the observation to the oneDNN-AVX2 profile, then isolate its operator. |
| nonexact / nonexact | nonexact | exact | any | `onednn_backend_is_profile_sensitive` | Scope the observation and isolate the oneDNN-backed operator. |
| any | any | any | projection first and payload changes under a backend control | `projection_backend_path_is_profile_sensitive` | Isolate the first profile-variant projection. |
| nonexact / nonexact | nonexact | nonexact | after all projections in every failing arm | `post_projection_reduction_or_attention_seam_remains_live` | Localize the first post-projection operation. |
| any other combination | any | any | any | `reference_matrix_mismatch_remains_unlocalized` | Preserve the first-stage evidence and do not broaden the claim. |

The controls follow the upstream
[oneDNN CPU dispatcher](https://uxlfoundation.github.io/oneDNN/dev_guide_cpu_dispatcher_control.html),
[oneDNN verbose](https://uxlfoundation.github.io/oneDNN/dev_guide_verbose.html),
and [PyTorch MKLDNN backend](https://docs.pytorch.org/docs/stable/notes/mkldnn.html)
contracts.

The earlier #55 paired artifact remains valid but narrower: both native AVX512
and forced ATen AVX2 were nonexact, all official and candidate `layer_out`
payloads were bitwise invariant across those two processes, and routed/shared
weights matched the official values. At that head, `post_attention_norm` was
the only pre-router point included in the composed-stage order. It was the
first **compared** pre-router mismatch, not proof that the preceding ten trace
points were exact. The schema-version-4 matrix closes that observability gap.

This table classifies evidence; it does not weaken any parity assertion or
enable public execution.

## Current evidence frontier

The stacked evidence through #42 establishes the following bounded facts:

- sparse layer classes 2 and 5, position zero, are exact through final
  `layer_out` when canonical routed IDs are supplied and the proven BF16
  arithmetic boundaries are composed (#39 and the retained #44 lineage);
- exact official tied IDs at ambiguous BF16 cutoffs are order-sensitive inside
  the pinned CPU `topk` implementation (#40);
- valid alternative tied subsets materially change final layer output (#41);
- identical BF16 choice bits select different tied expert sets across Linux,
  Windows, and macOS Torch 2.13 CPU builds, while unambiguous rows remain fixed
  (#42).

This is **not** full-model parity. It does not cover multi-position state,
end-to-end token generation, tokenizer/chat-template parity, a quantized public
container, or public loader/server execution.

## Promotion gates after this contract

The next numerical work should keep portable semantics and official-reference
semantics separate. The BF16 evidence consolidation carries the stateful
investigation through #51; see `docs/BF16-EVIDENCE.md`.

1. Rerun the bounded stateful sparse layer with #51's proven mixed-precision
   router denominator while forcing the official counterfactual to the same
   route at each audited position.
2. In parallel, retain profile-bound official-reference checks for exact route
   and output reproduction where the profile is fully specified.
3. Extend the evidence across the remaining relevant layer/state classes before
   claiming full decoder parity.
4. Establish tokenizer and chat-template parity.
5. Validate the intended quantized container, resource budgets, and laptop
   storage/I/O behavior.
6. Only after those gates are green and explicitly reviewed may public loading,
   stepping, generation, chat, or serving be considered.

Until then, the public Inkling runtime remains deliberately unsupported.

## Evidence provenance

- #40 / workflow `31224122201`: exact BF16 cutoff ties do not imply a
  portable official expert-ID rule.
- #41 / workflow `31224845163`: valid tied subsets materially change complete
  layer output; the deterministic low-ID route remains exact against the
  same-route official counterfactual.
- #42 / workflow `31226304158`: identical BF16 choice bits select different
  tied subsets across Linux, Windows, and macOS Torch 2.13 CPU builds, while
  unambiguous rows remain stable.
- #43 / workflow `31227218253`: the portable WASTE and pinned official
  reference profiles are distinct, explicit contracts.

## What this contract does not change

This contract does not:

- alter `waste_inkling_route`;
- emulate platform-specific `std::nth_element`, MSVC STL, or libc++ behavior;
- weaken the existing strict official evidence gate;
- promote the temporary BF16 experiment source into production;
- enable public Inkling loading, generation, chat, or serving.

It only makes the reference target explicit enough that future parity results
cannot silently mix deterministic WASTE semantics with platform-specific
official behavior.
