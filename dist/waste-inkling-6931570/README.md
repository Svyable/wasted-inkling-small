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

The applied Git tree must be `e372f1ef2b92c4bcc94f5c2474d6597d068f5c84`.

## Regenerate and verify from source

```sh
integration/waste/verify.sh            # tree hash, -Werror compile, make check
integration/waste/generate.sh WORKDIR dist/waste-inkling-6931570/patches
```

The patch is byte-deterministic: identical sources produce an identical patch
file, which is what makes `SHA256SUMS` meaningful rather than a record of when
the build ran.

## What changed against `waste-inkling-patch-v18`

- the same eleven Inkling translation units, unchanged;
- `tools/inkling_fixture.py` — a dependency-free reader for the bounded parity
  fixtures `inkling_parity.py` has always been able to extract and nothing
  could consume;
- `tests/test_inkling_fixture.py` — 41 tests, no torch, CI-runnable.

## Evidence

See [`TEST-RESULTS.txt`](./TEST-RESULTS.txt). Summary: 29 passed / 0 failed /
13 skipped in the WASTE suite, 168 server checks, 11 units compiled with
`-Werror`, and 140 Python tests passing — the first time this repository has
run the full Inkling Python suite rather than quoting it.

Public Inkling inference remains disabled. This bundle changes how the port is
built and reviewed, not what the public loader will run.
