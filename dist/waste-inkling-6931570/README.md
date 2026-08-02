# Inkling-Small on WASTE 0.6.3 — generated bundle

This directory is a **build product**. It is regenerated from
[`inkling/`](../../inkling) and [`integration/waste/`](../../integration/waste)
by `integration/waste/generate.sh`; nothing here is edited by hand.

If you want to read the code, read `inkling/src/*.c`. If you want to apply it
to WASTE, use the patch below.

## Apply

```sh
git clone https://github.com/sqliteai/waste.git
cd waste
git checkout 69315701f634648f7a790915a0a525ed8aabf218
git am /path/to/dist/waste-inkling-6931570/patches/0001-Add-the-Inkling-Small-runtime-foundation-to-WASTE.patch
PATH=/usr/bin:/bin make check
```

Verify it first:

```sh
cd /path/to/dist/waste-inkling-6931570
sha256sum -c SHA256SUMS
```

The applied Git tree must be `bd7c8750100d393c13cd36062489cc4fce27ad69`.

## Regenerate and verify from source

```sh
integration/waste/verify.sh            # tree hash, -Werror compile, make check
integration/waste/generate.sh WORKDIR dist/waste-inkling-6931570/patches
```

The generator pins the commit timestamp and passes `--no-signature`, so
repeated runs over identical sources produce identical patch bytes rather than
recording when the build ran or which Git emitted it.

The authoritative check is nonetheless the **applied tree hash**, not the patch
bytes. `format-patch` output is a function of the local Git as well as the
content, so CI verifies that the committed patch *applies to*
`bd7c8750100d393c13cd36062489cc4fce27ad69` — which is what a consumer actually
depends on, and is immune to toolchain drift.

## What changed against `waste-inkling-patch-v18`

- the same eleven Inkling translation units, unchanged;
- `tools/inkling_fixture.py` — a dependency-free reader for the bounded parity
  fixtures `inkling_parity.py` has always been able to extract and nothing
  could consume;
- `tools/inkling_layer_parity.py` — the candidate side of layer-level parity:
  binds one layer's weights from a fixture and runs the traced C decoder layer;
- `inkling_plan.py` now exposes its source-name tables, so the harness resolves
  tensor names through the planner instead of a second copy;
- `tests/test_inkling_fixture.py` and `tests/test_inkling_layer_parity.py` —
  53 new tests.

## Evidence

See [`TEST-RESULTS.txt`](./TEST-RESULTS.txt). Summary: 29 passed / 0 failed /
13 skipped in the WASTE suite, 168 server checks, 11 units compiled with
`-Werror`, and 152 Python tests passing.

Public Inkling inference remains disabled. This bundle changes how the port is
built and reviewed, not what the public loader will run.
