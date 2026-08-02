# WASTE Inkling-Small Patch Bundle

**Frontier model. Laptop-shaped RAM budget. An SSD about to earn its keep.**

This repository contains the **Patch 16 handoff bundle** for experimental
Inkling-Small support in WASTE. It combines an ordered patch series, an applied
source snapshot, conversion and parity tools, tests, and technical notes for the
released `thinkingmachines/Inkling-Small` checkpoint.

> [!IMPORTANT]
> This is a development and parity bundle, not a production-ready WASTE release.
> The private conversion/runtime path is implemented and covered with synthetic
> tests, but public loader integration, tokenizer/chat execution, and official
> 532 GB checkpoint parity remain completion gates.

The finish line is intentionally boring in the best possible way: Inkling-Small
should become a normal WASTE container that works through `waste plan`,
`waste run`, `waste chat`, and the existing OpenAI-compatible server. No special
Inkling-only executable, mystery memory budget, or "works on my 532 GB model"
energy.

## Start here

The project lives in [`waste-inkling-patch-v16/`](waste-inkling-patch-v16/).

| Document | Purpose |
| --- | --- |
| [Inkling implementation notes](waste-inkling-patch-v16/docs/INKLING.md) | Architecture, formats, converter pipeline, correctness evidence, and remaining gates |
| [Upstream alignment](waste-inkling-patch-v16/docs/UPSTREAM-ALIGNMENT.md) | Comparison with `sqliteai/waste` v0.6.2 and the production integration gates |
| [Patch 16 handoff](waste-inkling-patch-v16/PATCH16-HANDOFF.md) | Activation-tracing APIs and the Python-vs-C comparison workflow |
| [Known errata](waste-inkling-patch-v16/KNOWN-ERRATA.md) | Superseded provenance claims and mandatory patch guidance |
| [Patch 16 test results](waste-inkling-patch-v16/TEST-RESULTS-P16.txt) | Recorded Python and strict C validation results |
| [SHA-256 checksums](waste-inkling-patch-v16/SHA256SUMS) | Integrity hashes for the bundle |

## What is included

- `patches/` — patches 0001 through 0016, intended to be applied in numeric
  order to the corresponding WASTE source tree.
- `src/` — the applied private C implementation for configuration, attention,
  decoder layers, staged artifacts, quantized tensors, WEXP experts, model
  execution, and activation tracing.
- `tools/` — checkpoint inspection, conversion planning, bounded staging,
  Q8/Q4 trunk quantization, routed-expert VQ, runtime-index publication, and
  Python/C parity tooling.
- `tests/` — Python tests plus C fixtures covering format validation, conversion,
  runtime binding, quantized execution, corruption rejection, and trace output.
- `docs/` — the detailed technical design and current readiness assessment.

## Current status

Implemented and tested:

- fail-closed recognition of the official Inkling-Small package;
- bounded and resumable checkpoint inspection, planning, and staging;
- Q8/Q4 trunk artifacts and final WEXP/WCBK routed-expert artifacts;
- versioned private runtime-stage indexes, including the quantized v3 path;
- synthetic multi-token byte-to-logits validation;
- named activation capture from the private C runtime and the official
  Transformers implementation;
- CRC-protected parity archives and tolerance-based comparison reports.

Still gated:

- official-checkpoint layerwise and logits parity;
- public WASTE manifest, loader, cache, and serving integration;
- tokenizer, chat-template, image, and audio execution;
- native Windows validation and measured conversion/runtime performance.

The bundle records **99 passing Python tests** and strict C11 compilation for the
traced runtime sources. This upstream-alignment round adds four model-free unit
tests for the baseline checker. None of those results constitutes official-weight
parity.

## Upstream foundation

This bundle is now compared against
[`sqliteai/waste` at `c7cb640`](https://github.com/sqliteai/waste/commit/c7cb64022963871506908317a661338f1794f70e),
which identifies itself as WASTE 0.6.2 with public API version 1 and container
format version 0.

The overlap is promising: both sides use 4 KiB-aligned WEXP expert records,
WCBK codebooks, Q8/Q4 trunk formats, bounded reads, resumable conversion, and
oracle-based validation. The important gap is architectural: this bundle still
publishes a private `runtime-stage.bin` and executes a parallel private runtime,
while upstream serves normal `.waste` containers through `waste_open`, the hard
RAM planner, LFRU expert cache, tokenizer, CLI, and server.

Check a local upstream checkout before rebasing or adding integration patches:

```sh
cd waste-inkling-patch-v16
python tools/check_waste_baseline.py /path/to/sqliteai-waste
```

The checker fails when the public API, container format, alignment contract, key
symbols, or expected production seams are missing. A newer upstream version is
reported as a warning so it triggers a human comparison instead of quietly
becoming "probably fine."

## Requirements

- A recent Python 3 environment.
- PyTorch for conversion, quantization, and parity tools.
- `transformers` with Inkling support for official reference capture.
- A C11 compiler for the private runtime and C validation.
- Local access to the official checkpoint for real conversion or parity work.

The released BF16 package is approximately 532 GB. Plan storage and memory for
the selected workflow before downloading or converting it. The tools support an
index-only planning mode when checkpoint shards are not present.

## Quick validation

Run commands from the bundle directory:

```sh
cd waste-inkling-patch-v16
python -m unittest discover -s tests -p 'test_*.py'
```

Strict-compile the runtime files changed by Patch 16:

```sh
cc -std=c11 -Wall -Wextra -Werror -Isrc -c src/inkling_layer.c
cc -std=c11 -Wall -Wextra -Werror -Isrc -c src/inkling_model.c
cc -std=c11 -Wall -Wextra -Werror -Isrc -c src/inkling_private.c
```

## Conversion workflow

Start with a metadata and tensor-shape plan:

```sh
python tools/convert_inkling.py \
  --src /path/to/Inkling-Small \
  --out /path/to/Inkling-Small.stage \
  --plan-only
```

Use `--allow-index-only` with `--plan-only` to review a checkpoint index before
all shards are available. A private quantized v3 stage is then built in separate,
resumable steps:

```sh
python tools/convert_inkling.py --src /path/to/Inkling-Small --out /path/to/Inkling-Small.stage --stage-trunk
python tools/convert_inkling.py --src /path/to/Inkling-Small --out /path/to/Inkling-Small.stage --quantize-trunk
python tools/convert_inkling.py --src /path/to/Inkling-Small --out /path/to/Inkling-Small.stage --quantize-experts
python tools/convert_inkling.py --src /path/to/Inkling-Small --out /path/to/Inkling-Small.stage --publish-runtime-qtrunk-stage
```

These commands publish the converter-private `runtime-stage.bin`; they do not
publish a public WASTE `manifest.json`.

## Official-weight parity workflow

Capture the official Transformers reference, trace the private C runtime with
the same tokens, and compare the resulting activation archives:

```sh
python tools/inkling_reference.py \
  --src /path/to/Inkling-Small \
  --tokens 200006,1234,5678 \
  --layers 0,1,2 \
  --out /path/to/parity/python

python tools/inkling_trace.py \
  --library /path/to/libwaste.so \
  --stage /path/to/Inkling-Small.stage \
  --tokens 200006,1234,5678 \
  --out /path/to/parity/waste

python tools/inkling_parity.py \
  --compare-reference /path/to/parity/python \
  --compare-candidate /path/to/parity/waste \
  --atol 1e-5 --rtol 1e-5 \
  --report /path/to/parity/report.json
```

A comparison mismatch exits with status 2. Establish BF16 semantic parity before
setting separate tolerances for Q8/Q4/VQ execution.

## Patch usage

If you are integrating this work into its target WASTE source tree, apply all
patches from 0001 through 0016 in numeric order. Patch 10 contains the provenance
correction for the released Inkling-Small package and must not be skipped. Review
the handoff notes and checksums before applying the series.

For new production work, prefer a refreshed patch stack based on the pinned
upstream commit over extending the parallel private loader. The private runtime
remains valuable as an independent parity oracle; it should not become a second
public engine that has to rediscover WASTE's cache, budget, platform, and serving
lessons the expensive way.

## License and attribution

The official Inkling-Small model is identified in the bundle as Apache-2.0, and
source files carry their own SPDX and copyright headers. This repository does not
currently include a top-level license file; review the file-level notices and the
upstream WASTE and model terms before redistribution.
