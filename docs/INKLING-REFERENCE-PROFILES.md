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
- CPU dispatch policy, including an `ATEN_CPU_CAPABILITY` override when used;
- **oneDNN ISA policy: `ONEDNN_MAX_CPU_ISA` when set, and whether
  `torch.backends.mkldnn` was enabled.** The reference matrix showed this is
  the control that decides exactness on an AVX-512 host, and that forcing ATen
  does not — see "What makes a profile a profile" below. A profile missing it
  is incomplete;
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

## Current evidence frontier

The stacked evidence through #57 establishes the following bounded facts:

- sparse layer classes 2 and 5, position zero, are exact through final
  `layer_out` when canonical routed IDs are supplied and the proven BF16
  arithmetic boundaries are composed (#39 and the retained #44 lineage);
- exact official tied IDs at ambiguous BF16 cutoffs are order-sensitive inside
  the pinned CPU `topk` implementation (#40);
- valid alternative tied subsets materially change final layer output (#41);
- identical BF16 choice bits select different tied expert sets across Linux,
  Windows, and macOS Torch 2.13 CPU builds, while unambiguous rows remain fixed
  (#42).
- on the recorded Linux AVX2 profile, the #51 post-reduction denominator policy
  composes through all eight source-bound positions of local sparse layer 2:
  routed/shared weights, `moe_out`, `mlp_branch`, and `layer_out` are BF16-exact
  and no first mismatch is present (#57).

### What makes a profile a profile: the oneDNN ISA cap

Until #59 the profile contract listed `ATEN_CPU_CAPABILITY` as the CPU dispatch
control and said nothing about oneDNN's ISA cap. That ordering was backwards.
The same-host reference matrix
([run 31280408153](https://github.com/Svyable/wasted-inkling-small/actions/runs/31280408153))
ran all four arms on one AVX-512 host and found:

| Arm | Exact | First pre-router divergence |
| --- | --- | --- |
| native (AVX512) | no | `q_proj` |
| `ATEN_CPU_CAPABILITY=avx2` | no | `q_proj` |
| `ONEDNN_MAX_CPU_ISA=AVX2` | **yes** | — |
| `torch.backends.mkldnn.enabled=False` | **yes** | — |

Forcing ATen changed nothing — same exactness, same stage, both layers. The
control the contract already named is not the one that moves the arithmetic;
`ONEDNN_MAX_CPU_ISA` is. **A profile that records `ATEN_CPU_CAPABILITY` and
omits `ONEDNN_MAX_CPU_ISA` does not pin the thing that decides the result.**

`experiments/inkling/probe_onednn_isa_arithmetic.py` reproduces that partition
arm-for-arm on a bare bfloat16 linear of `q_proj`'s shape, with no fixture and
no Inkling code in the process. On an AVX-512 host advertising neither
`amx_bf16` nor `avx512_bf16`, the ladder is monotone and the first rung
differing from the AVX2 baseline is `AVX512_CORE`: neither AMX nor
AVX-512-BF16 kernels are necessary for the divergence.

Two consequences for claim scope:

- The `LINUX-AVX2` profile's exactness is bound to the oneDNN AVX2 cap, not to
  the ATen override in its name. Restating an AVX2-profile result on an
  uncapped AVX-512 host is not a restatement, it is a different measurement.
- An earlier claim that "AMX is excluded by the reproduced-failure profiles" is
  withdrawn. It was read off CPU flag sets on hosts that did not advertise AMX.
  The ISA setting is a cap rather than a request, so on such a host the AMX
  path is never selected and never tested — and untested is not excluded.

The synthetic probe is a host census, **not** parity evidence: its weights are
not the checkpoint's, and agreement there says nothing about whether the port
is correct. It localizes the mechanism; the reference matrix remains what
establishes what the mechanism does to Inkling.

This is **not** full-model parity. The stateful result covers one local sparse
layer on one named reference profile. It does not cover stateful dense/global
layer classes, cross-layer decoder continuity, logits, end-to-end token
generation, tokenizer/chat-template parity, a quantized public container, or
public loader/server execution.

## Promotion gates after this contract

The next numerical work must keep portable semantics and official-reference
semantics separate. The BF16 evidence consolidation now carries the stateful
investigation through #57; see `docs/BF16-EVIDENCE.md`.

1. Complete the predeclared same-host backend matrix for the unresolved
   position-zero sensitivity: native AVX512, forced ATen AVX2, oneDNN AVX2, and
   MKLDNN disabled against one CRC-bound fixture.
2. Retain profile-bound official-reference checks for exact route and output
   reproduction, while keeping deterministic low-ID WASTE routing as the
   portable contract.
3. Promote the proven BF16 boundaries into a private, fail-closed C execution
   profile; do not enable public dispatch as part of that change.
4. Rerun representative stateful dense, local-sparse, and global-sparse layers
   without temporary source rewriting, then extend through decoder continuity
   and logits before claiming full decoder parity.
5. Establish tokenizer and chat-template parity, and validate the intended
   quantized container and measured resource budgets.
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
- #57 / workflow `31277195747`, attempt 2: the named Linux AVX2 profile is
  exact through the eight-position layer-2 stateful sparse-MLP ladder; artifact
  `9027735369` preserves the profiled result.

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
