# Inkling-Small config schema resolution

Date resolved: 2026-08-03

## Authoritative artifact

The released model repository now publishes the real file:

```text
thinkingmachines/Inkling-Small/config.json
Hub commit shown on the file page: 21152b5
```

Its text geometry matches `inkling/tests/data/inkling-small-config.json`:

```json
{
  "hidden_size": 4096,
  "num_hidden_layers": 42,
  "dense_intermediate_size": 16384,
  "intermediate_size": 2048,
  "n_routed_experts": 256,
  "num_experts_per_tok": 6,
  "n_shared_experts": 2
}
```

There is no `moe_intermediate_size` key in the release JSON.

## Resolution

For the released checkpoint schema:

- `dense_intermediate_size` is the dense MLP width: **16384**;
- `intermediate_size` is the routed-expert width: **2048**.

The 2048-based VQ3R record geometry, expert-bank estimate, and minimum expert
cache calculation are therefore not invalidated by the release artifact. They
still need normal implementation and measurement validation, but the config
width they use is the released width.

## Transformers 5.14.x incompatibility

`InklingTextConfig` names those two runtime fields differently:

- `intermediate_size` means the dense width;
- `moe_intermediate_size` means the routed width and defaults to 3072.

Passing the release JSON through unchanged produces dense 16384 but leaves the
routed width at 3072. That is a schema-ingestion mismatch, not evidence that the
release experts are 3072 wide.

The MVP evidence harness now translates explicitly:

```text
release dense_intermediate_size -> Transformers intermediate_size
release intermediate_size       -> Transformers moe_intermediate_size
```

The adapter refuses missing fields and conflicting explicit
`moe_intermediate_size` values. It then verifies the constructed Transformers
config still contains dense 16384 and routed 2048.

## Router comparison rule

The official router uses `topk(sorted=False)`. Expert slots are therefore not a
stable ordering. Differential comparison must keep each expert ID attached to
its weight and canonicalize `(expert_id, weight)` pairs before comparing.
Sorting indices and gathering the corresponding weights is valid; sorting the
index and weight arrays independently is not.

## Commands

```sh
python -m pip install 'transformers==5.14.1' torch

PYTHONPATH=mvp python -m unittest \
  mvp/test_inkling_fixture_reference.py \
  mvp/test_inkling_release_config.py -v
```

The G0 execution commands are in `docs/MVP-READINESS.md`.

## Remaining evidence boundary

This resolves the config geometry and unblocks construction of correctly sized
official decoder layers. It does not establish activation parity, quantized
quality, conversion performance, tokenizer parity, or public inference safety.
Those gates remain unchanged.
