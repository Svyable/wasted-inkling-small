# Inkling-Small on current WASTE — v17

This round turns the advanced v16 Inkling work into a sensible foundation on
the **current** upstream WASTE tree.

The comparison found a very specific problem: historical patches 1–3 still
apply to WASTE c7cb640, but patch 4's Makefile, model-loader, public API, and
test-runner contexts predate today's upstream layout. The later private runtime
work is valuable; its entry ramp had simply become a small archaeological site.

v17 replaces that stack with one consolidated, replay-tested patch targeting:

- [sqliteai/waste](https://github.com/sqliteai/waste)
- commit c7cb64022963871506908317a661338f1794f70e
- WASTE 0.6.2, API 1, container format 0

## What the patch contains

- The complete v16 source inspector, conversion planner, bounded staging,
  Q8/Q4 trunk, WEXP/VQ expert, private runtime, and activation-tracing surface.
- Integration of every private Inkling C source into current libwaste.
- A current test_inkling build/test target.
- Case-insensitive Inkling recognition that fails closed before Kimi memory
  planning, public model opening, or direct internal loading.
- Current-upstream regression tests for waste plan, waste info, and the direct
  loader path.
- Corrected status documentation: quantized private execution exists; public
  dispatch, official-weight parity, tokenizer/chat, and serving remain gated.

This is deliberately not public Inkling execution. The private engine is now
attached to a maintained foundation, while the public loader continues to
return WASTE_E_UNSUPPORTED until parity earns promotion. The model may be
frontier; the error handling has a mortgage.

## Apply

~~~sh
git clone https://github.com/sqliteai/waste.git
cd waste
git checkout c7cb64022963871506908317a661338f1794f70e
git am /path/to/waste-inkling-patch-v17/patches/0017-waste-c7cb640-consolidated-inkling-runtime.patch
PATH=/usr/bin:/bin make check
~~~

Do **not** apply historical patches 1–16 first. Patch 17 is their consolidated
current-upstream replacement.

Verify the artifact from this directory:

~~~sh
sha256sum -c SHA256SUMS
~~~

## Evidence

- current WASTE suite: 28 passed, 0 failed, 13 skipped
- ASan + UBSan: 27 passed, 0 failed, 14 skipped
- sanitizer fuzzing: 200 cases, 0 crashes, 0 hangs
- strict -Werror compile: 11 Inkling/architecture translation units
- dependency-light Python selection: 17 passed
- Python source compilation: passed

See [TEST-RESULTS-P17.txt](./TEST-RESULTS-P17.txt) for limitations and exact
commands.

## Next production gates

1. Run the existing Patch 15/16 trace workflow against official weights and
   resolve every BF16 semantic mismatch.
2. Set measured Q8/Q4/VQ tolerances, conversion time, memory floor, and
   throughput from real hardware.
3. Add official tokenizer and chat-template parity.
4. Design the public manifest extension and dispatch only after those gates.
5. Exercise native Windows, cancellation/resume, observability, and release
   packaging.
