# Known errata and supersession notice

The v5-v9 artifact series stated that the official Inkling-Small checkpoint was
not yet released. That was true during an earlier audit pass but is no longer
true as of 2026-08-01.

Thinking Machines has now released `thinkingmachines/Inkling-Small` under
Apache-2.0 with an official config, 32-shard BF16 checkpoint index, tokenizer,
processor configuration, special-token map, and chat template.

Patch 10 is therefore mandatory for Small work. It:

- replaces the synthetic candidate profile with the exact released profile;
- validates the official package identity before labelling it Inkling-Small;
- preserves generic Inkling support without silently borrowing Small values;
- adds the private runtime-stage publisher and staged-directory opener;
- proves synthetic staged-directory open-to-logits parity;
- leaves the public WASTE manifest and loader guard unchanged.

Do not rely on the provenance conclusions in a v9-or-earlier handoff. Patch 10
is the provenance correction; Patch 11 is the first direct final-format expert quantizer. Patch 12 connects
those final WEXP/WCBK files to the complete private staged-directory runtime,
removing the BF16 expert-stage dependency from the byte-to-logits path. Apply
all patches through Patch 12 or use the complete v12 bundle.
