# Inkling-Small Patch 15 handoff

This bundle supersedes v14 and adds the bounded official-weight parity harness.

Apply patches 0001 through 0015 in numeric order to the current WASTE source tree.
Patch 15 adds only Python conversion/validation tooling, tests, and documentation;
it does not alter the public runtime, manifest schema, or loader guard.

Validation in this environment:

- 96 Python tests passed.
- Python source compilation passed.
- Patch 15 applies cleanly to the v14 source-only bundle.
- No official 532 GB checkpoint payload was locally available, so real-weight
  activation comparisons are not claimed.

The next gate is to run the extractor against the official BF16 package, generate
Python and WASTE activation archives for layers 0, 1, and the first sparse layer,
and resolve every mismatch before public loader integration.
