# MVP readiness handoff

## Where the previous work stopped

The public memory planner is live for valid Inkling manifests. Public loading,
step execution, chat, and serving remain deliberately refused. The next hard
gate is not another loader branch: it is official-weight layer parity.

The repository already has both bounded checkpoint fixtures and a fixture-backed
C decoder-layer runner. What was missing was the official Transformers side
that can consume the same fixture without allocating the complete checkpoint or
the complete 256-expert bank.

## This change

`mvp/inkling_fixture_reference.py` is an evidence harness for that missing side.
It:

- verifies `config.json` against the SHA-256 recorded in `fixture.json`;
- instantiates the official `InklingDecoderLayer` on the meta device;
- removes the full routed-expert bank before materializing the layer;
- loads provider-raw attention, convolution, dense/shared MLP, and router
  tensors into the official module;
- materializes only the expert slices selected by the bounded fixture;
- fails closed, naming any expert ID selected by the router but absent from the
  fixture;
- emits the existing CRC-protected activation archive format and names, so the
  existing comparator can compare it with `inkling_layer_parity.py` unchanged.

It is intentionally not copied into the generated WASTE patch yet. The patch is
a reviewed runtime artifact; this is an evidence-producing tool that still
needs its first official-fixture run. Promotion into `inkling/tools/` should be
the commit that also records that run and regenerates the bundle.

## Run the G0 loop

```sh
python -m pip install 'transformers==5.14.*' torch

python mvp/inkling_fixture_reference.py \
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

python inkling/tools/inkling_parity.py \
  --compare-reference /parity/python \
  --compare-candidate /parity/waste \
  --atol 1e-5 --rtol 1e-5 \
  --report /parity/report.json
```

A sparse run may fail because the router selects an expert not extracted into
the bounded fixture. That is useful evidence, not a test failure: re-extract
with the named IDs and repeat until the chosen token inputs are covered.

## MVP sequence from here

1. Run the command above against official layer 0 (dense), layer 2 (sparse,
   local attention), and layer 5 (sparse, global attention).
2. Fix the first activation mismatch and commit the comparison report.
3. Promote the harness into `inkling/tools/`, add its tests to the differential
   suite, regenerate the distribution patch, and update the applied-tree hash.
4. Establish tokenizer/chat-template parity in parallel.
5. Only after BF16 and text-interface evidence is green, promote public loading
   and stepping. Keep unsupported behavior for every unverified Inkling variant.

## Definition of an MVP

The first usable MVP is a normal WASTE text flow, not a special parity binary:
`waste plan`, `waste run`, and the OpenAI-compatible server against one verified
Inkling-Small container. Chat can follow immediately once template parity is
recorded. Multimodal and speculative decoding remain outside that boundary.
