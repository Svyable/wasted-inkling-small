# Inkling reference profiles

## Why this contract exists

Inkling-Small exposes a real reproducibility boundary at BF16 router cutoffs.
The current evidence stack separates three facts that must not be conflated:

1. With the routed expert IDs held equal, the position-zero sparse-layer
   arithmetic can reproduce the official layer exactly under the pinned native
   BF16 backend (#39).
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

The stacked evidence through #42 establishes the following bounded facts:

- sparse layer classes 2 and 5, position zero, are exact through final
  `layer_out` when canonical routed IDs are supplied and the proven BF16
  arithmetic boundaries are composed (#39);
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
semantics separate.

1. Validate multi-position/stateful sparse layers under the deterministic WASTE
   route policy by forcing the official counterfactual to the same route at
   each audited position.
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
