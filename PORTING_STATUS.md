# Inkling-Small porting status

> Canonical re-entry point for active porting work. Update this file when a gate
> is promoted, invalidated, or superseded. Historical investigation remains in
> `docs/`, but this file describes what `main` can claim after the current gate
> is merged.

## Baseline

- Main checkpoint: `thinkingmachines/Inkling-Small`
- Pinned revision: `21152b5312c653be115f33a8342759064144e281`
- Exact BF16 reference profile: Linux CPU with `ONEDNN_MAX_CPU_ISA=AVX2`
- Merged foundation: final-head production primitive + immutable real-byte oracle;
  this gate adds routed-expert storage through native WASTE cache machinery.
- Generated WASTE tree: `7fafa7942ee43abe6196481bd88c5621e40e6843`
- Generated patch SHA-256: `21fc0c90bd985a582c5e36b5c08bd59ed75f7c8d6cfb601543718bdc337f4ea2`

## Proven on `main` after this gate

### Public WASTE boundary

- `waste plan`, normal container open/binding, and `waste info` work for Inkling.
- Trunk matrices use the normal WASTE backend; row-backed embedding/unembedding
  tables use row callbacks.
- Expert-bank metadata/record geometry is validated at load time.
- Sparse-layer WEXP banks are opened through WASTE's existing architecture-neutral
  bank opener and bounded `ecache`; no Inkling-specific cache or transcoding
  format exists.
- Raw routed records are checked for layer/expert identity, format, codebook,
  offsets, 4 KiB record geometry and, when verification is enabled, payload CRC.
- Cached get, hold/release, routing hints and an explicit zero-cache aligned-read
  fallback are available at the Inkling architecture boundary.
- Public step, prefill, eval, generation, chat execution, and serving remain
  deliberately refused. Routed expert **compute** is not promoted yet.

### Checked BF16 arithmetic

Official-weight gates compile checked-in C unchanged (`source_rewriting=false`).

- dense layer 0: exact across eight source-bound positions;
- local sparse layer 2: position-zero and eight-token stateful ladder exact;
- global sparse layer 5: position-zero and eight-token stateful ladder exact;
- consecutive dense 0 -> dense 1 -> sparse 2: exact across eight positions;
- consecutive local sparse 4 -> global sparse 5: exact across eight positions,
  including independently reproduced routes and a matching BF16 chained trace.

### Frozen final-head primitive

`waste_inkling_final_head_profile()` in `inkling/src/inkling_model.c` is the one
checked-in definition of the head: final RMS normalization, the logits-width
completion, and the vocabulary projection over an explicit row selection. The
public F32 step calls it instead of carrying its own copy, and the head shares
`waste_inkling_rmsnorm_profile()` with decoder layers.

Both profiles are pinned by `inkling/tests/test_inkling_final_head_c.py` against
an independent bit-level Python reference. The real-byte oracle in
`tests/fixtures/inkling/final_head_primitive_real` is independently regenerated
from the pinned official config/checkpoint and replayed offline byte-for-byte.

This is a head primitive with a validated policy, not a final-logits claim. It
computes the head of whatever hidden state it is handed; see open gate 1.

Routing and arithmetic are separate contracts: raw WASTE routes are retained;
reference routes are injected only for downstream arithmetic when the official
stack selects a different valid tied top-k subset.

### Expert storage quality

The routed storage promotion deliberately reuses upstream WASTE instead of
building a parallel Inkling engine:

- WEXP records are already byte-layout compatible with WASTE format v0, so no
  record transcoding is introduced;
- the existing WASTE direct-I/O opener provides platform probe/fallback behavior;
- the existing bounded `ecache` provides LFRU/LRU policy, hits/misses, hold/release
  and hinting;
- `test_inkling_expert_store.c` proves first-read miss, repeat-hit/no-I/O,
  hold/release, out-of-range refusal, identity refusal, CRC refusal, and honest
  zero-cache misses;
- the same storage contract is compiled with warnings-as-errors for Windows and
  executed under Wine, in addition to Linux strict, sanitizer/fuzz and full
  differential gates.

This is intentionally better factored than expanding each expert into temporary
F32 weights: the next compute gate can hand the cached native record directly to
WASTE's existing VQ/SIMD kernels.

### Evidence hygiene

- semantic-producer freshness fails closed;
- exact fixture entry/byte/request counts are asserted before download;
- bounded fixtures carry only selected vocabulary rows where appropriate;
- route archives and important output hashes are source-controlled;
- expensive official-weight jobs run only after cheap provenance/import/build
  gates;
- generated WASTE bundles are content-addressed and independently re-applied to
  pinned upstream before publication.

## Open gates

1. **True final hidden state and logits.** Representative consecutive decoder
   transitions are proven, and the final-head primitive plus real-byte oracle are
   frozen. What remains open is the real 42-layer final hidden state feeding it;
   nothing here is yet final-model logits.
2. **Native routed-expert compute / public step.** Storage and bounded cache
   dispatch are promoted. The next runtime gate is selected expert ID -> cached
   native WEXP record -> WASTE's existing VQ/SIMD gate/up/down kernels, followed
   by private checked stepping. Public stepping remains fail-closed until both
   runtime and numerical endpoint evidence close.
3. **Tokenizer/chat-template release parity.** Not yet a promoted public gate.
4. **Quantized quality.** Official VQ/Q8/Q4 reconstruction/logit quality and the
   release format choice remain open.
5. **Full conversion/operational measurement.** Full conversion, resume/cancel,
   wall time, peak RSS/disk, and real decode throughput remain open.
6. **Production official-checkpoint converter.** Format/synthetic writing exist;
   production conversion is not promoted.

## Next load-bearing work

### Numerical path

1. ~~Freeze and validate final-norm/unembedding completion semantics on a bounded
   real hidden-state vector.~~ Done and independently replayable.
2. Extend chained decoder/localizer evidence toward the true final-layer endpoint
   with evidence-driven routed-expert acquisition.
3. Feed that true final hidden state into the frozen head primitive and pin logits
   before public token execution.

### Runtime path in parallel

1. ~~Open routed expert records through WASTE's existing expert cache.~~ Done:
   native WEXP records are cached without transcoding or F32 expansion.
2. Bind cached WEXP records to WASTE's existing native VQ/SIMD expert matvecs and
   prove routed gate/up/down parity under checked BF16 execution.
3. Bind checked BF16 stepping privately against a public-container load while
   retaining public refusal.
4. Promote normal WASTE step/prefill only after decoder/logit evidence closes.

### Release path in parallel

- tokenizer/chat-template parity;
- quantized-quality measurement and format decision;
- one real full conversion with restart/resume and resource measurements.

## Merge discipline

Keep the active PR stack short. Merge an earned gate before opening multiple
children. Never repair stale evidence by editing hashes: reacquire/regenerate
from the current semantic parent. Preserve merge ancestry for stacked evidence.
