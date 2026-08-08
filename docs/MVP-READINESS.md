# MVP readiness handoff

## Current handoff

The public memory planner is live for valid Inkling manifests. Public loading,
step execution, chat, and serving remain deliberately refused. The bounded
official-weight loop is now real and reproducible; the next hard gate is moving
its proven BF16 policy into a private checked-in runtime path without confusing
that promotion with public model support.

The repository has bounded checkpoint fixtures, a fixture-backed C decoder
layer runner, and an official `InklingDecoderLayer` harness that avoids
materializing the complete checkpoint or 256-expert bank.

The real release `config.json` confirms the recorded Small geometry: dense
width 16384 and routed-expert width 2048. The apparent 3072 width came from
passing the release schema directly into Transformers 5.14.1, whose field names
have different meanings. See `docs/CONFIG-SCHEMA-RESOLUTION.md`.

PR #13 established an independent synthetic oracle: the C decoder layer and
official Transformers layer compute the same dense/local and sparse/global
functions to float32 epsilon. The later evidence stack then ran immutable,
CRC-verified official tensors through representative layers and isolated the
required BF16 completion boundaries.

The strongest current result is PR #57 / workflow `31277195747`, attempt 2.
On the recorded Linux AVX2 reference profile, local sparse layer 2 is BF16-exact
through routed/shared weights, `moe_out`, `mlp_branch`, and `layer_out` for all
eight source-bound positions. The run used 33 experts, 87 fixture entries,
1,851,934,212 verified payload bytes, and recorded 240 backend calls. That is a
profile-bound layer result, not full decoder or generation parity.

Three boundaries remain between this result and a production implementation:

1. one unchanged position-zero complete-layer run failed among five serial
   runs, so the same-host backend/dispatch matrix must classify the remaining
   profile sensitivity;
2. the proven `0x7f` arithmetic policy is still injected by evidence-only
   transforms and adapters rather than implemented in checked-in production C;
3. stateful dense layer 0, global sparse layer 5, cross-layer continuity, final
   normalization, and logits remain unproven under the composed policy.

## Evidence harness

`mvp/inkling_fixture_reference.py` contains the bounded official layer
implementation. It:

- verifies `config.json` against the SHA-256 recorded in `fixture.json`;
- instantiates the official `InklingDecoderLayer` on the meta device;
- removes the full routed-expert bank before materializing the layer;
- loads provider-raw attention, convolution, dense/shared MLP, and router
  tensors into the official module;
- materializes only expert slices selected by the bounded fixture;
- fails closed, naming any expert selected by the router but absent from the
  fixture;
- emits the existing CRC-protected activation archive format.

Use `mvp/run_inkling_fixture_reference.py` as the entry point. It applies the
release-to-Transformers schema translation before constructing the layer and
canonicalizes routed `(expert_id, weight)` pairs. Invoking the implementation
file directly bypasses that compatibility boundary.

`mvp/compare_inkling_layer_archives.py` canonicalizes router pairs in both
archives before calling the existing activation comparator. This matters
because the official router uses `topk(sorted=False)` while the C side emits a
different slot order.

The historical official-reference and diagnostic harnesses intentionally remain
outside the generated WASTE patch. They have now run against official fixtures;
their purpose is to specify and audit the production policy, not to become a
second runtime. Promotion should move only reviewed arithmetic behavior and the
minimum private test seam into `inkling/`, then regenerate and verify the
distribution bundle.

## Routed expert coverage is now evidence-driven

The original remote plan used six speculative routed experts per sparse layer.
Before downloading that fixture, the official router was probed over eight
deterministic BF16 positions.

The probe constructs each official sparse layer from a bounded one-expert seed,
replaces the expert bank before forward, captures the exact unsorted expert IDs
and attached weights, and stops before expert computation. The seed expert
values cannot influence routing.

The immutable release selected:

- layer 2: 33 unique experts;
- layer 5: 26 unique experts.

The full per-position choices, weights, input SHA-256, release revision, config
hash, and index hash are committed in
`docs/OFFICIAL-ROUTER-SELECTION.json`. CI reproduces that JSON exactly.

## Acquire the bounded official fixture

A local 532 GB checkpoint mount is no longer required. The remote extractor
reads only release metadata, safetensors headers, selected layer tensors, and
selected expert slices from immutable release commit
`21152b5312c653be115f33a8342759064144e281`.

The evidence-selected fixture plan is committed in
`docs/OFFICIAL-FIXTURE-PLAN.json`:

- 175 entries;
- 3,842,395,658 payload bytes, approximately 3.58 GiB;
- 179,397 metadata bytes and 52 requests to plan;
- 25 of 32 shards touched by header ranges.

Plan first; this downloads no tensor payloads. Use the exact command stored in
`docs/OFFICIAL-FIXTURE-PLAN.json`, which contains the 59 discovered expert IDs.
Then run the same command without `--plan-only` and add:

```sh
--out /parity/fixture-L0-L2-L5
```

Verify every payload and CRC:

```sh
python inkling/tools/inkling_fixture.py \
  --fixture /parity/fixture-L0-L2-L5 --verify
```

The extractor refuses mutable revisions, unsafe index paths, missing tensors,
invalid safetensors geometry, incomplete sparse expert selections, and any
server that ignores HTTP `Range`. See `docs/REMOTE-FIXTURES.md`.

## Reproduce the bounded official loop

Generate the same deterministic BF16 input sequence recorded by the router
selection artifact:

```sh
PYTHONPATH=mvp python - <<'PY'
import json
from pathlib import Path
from discover_inkling_router_experts import deterministic_hidden_states

values = deterministic_hidden_states(8, 4096, 19)
Path('/parity/inputs.json').write_text(json.dumps(values.tolist()) + '\n')
PY
```

Then run both sides:

```sh
python -m pip install 'transformers==5.14.1' torch

python mvp/run_inkling_fixture_reference.py \
  --fixture /parity/fixture-L0-L2-L5 \
  --model-config /models/Inkling-Small/config.json \
  --inputs /parity/inputs.json \
  --layers 0,2,5 \
  --dtype bfloat16 \
  --out /parity/python

python inkling/tools/inkling_layer_parity.py \
  --fixture /parity/fixture-L0-L2-L5 \
  --config /parity/normalized-config.json \
  --inputs /parity/inputs.json \
  --layers 0,2,5 \
  --out /parity/waste

python mvp/compare_inkling_layer_archives.py \
  --compare-reference /parity/python \
  --compare-candidate /parity/waste \
  --atol 1e-3 --rtol 1e-3 \
  --report /parity/report.json
```

The committed expert set covers this exact eight-position input sequence.
Different hidden states may route differently and require a separately recorded
selection artifact rather than silent fixture expansion.

## Immediate next enhancement

Finish the same-host reference matrix before changing production arithmetic.
Run native AVX512, forced `ATEN_CPU_CAPABILITY=avx2`,
`ONEDNN_MAX_CPU_ISA=AVX2`, and MKLDNN-disabled arms as fresh processes on one
eligible host and one CRC-verified fixture. Preserve every arm even when it is
nonexact. This decides whether the unresolved position-zero failure follows
oneDNN ISA selection, MKLDNN, a projection backend, or a later reduction.

After that measurement, make one private-runtime promotion PR with this
acceptance contract:

1. introduce an internal Inkling BF16 execution profile; do not change the
   public ABI, public loader dispatch, or unsupported behavior;
2. port only the proven normalization, attention, residual, expert, router, and
   post-reduction denominator boundaries from the evidence transforms;
3. keep deterministic WASTE routing and profile-bound official routing as
   distinct contracts at ambiguous BF16 cutoffs;
4. exercise the checked-in path against representative stateful dense layer 0,
   local sparse layer 2, and global sparse layer 5 fixtures without rewriting C
   source at test time;
5. preserve the first mismatch and execution profile rather than relaxing an
   assertion or comparison threshold;
6. regenerate the WASTE bundle, verify its tree/contracts, and leave public
   Inkling loading, stepping, generation, chat, and serving fail-closed.

Then extend the private path through cross-layer decoder continuity, final
normalization, and logits. Tokenizer/chat-template parity can proceed in
parallel because it does not depend on arithmetic promotion. Quantized
tolerances and public dispatch remain downstream gates.

## Definition of an MVP

The first usable MVP is a normal WASTE text flow, not a special parity binary:
`waste plan`, `waste run`, and the OpenAI-compatible server against one verified
Inkling-Small container. Chat can follow once template parity is recorded.
Multimodal execution and speculative decoding remain outside that boundary.
