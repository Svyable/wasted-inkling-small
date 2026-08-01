# Patch 16 handoff

This patch adds optional activation tracing to the private Inkling C runtime and
an official Transformers reference capture tool.

## C APIs

- `waste_inkling_layer_step_backend_trace`
- `waste_inkling_model_step_backend_trace`
- `waste_inkling_private_step_trace`

The original APIs remain unchanged and call the traced variants with `NULL`.
Trace callbacks must copy data before returning; scratch buffers are reused.
Returning nonzero aborts the step without advancing `next_position`.

## Reference command

```powershell
python tools/inkling_reference.py `
  --src D:\models\Inkling-Small `
  --tokens 200006,1234,5678 `
  --layers 0,1,2 `
  --out D:\parity\python
```

## C trace command

```powershell
python tools/inkling_trace.py `
  --library .\libwaste.dll `
  --stage D:\models\Inkling-Small.stage `
  --tokens 200006,1234,5678 `
  --out D:\parity\waste
```

The library must export the private Inkling symbols; this is still a parity
surface, not the public WASTE API.

## Compare

```powershell
python tools/inkling_parity.py `
  --layers 0 `
  --compare-reference D:\parity\python `
  --compare-candidate D:\parity\waste `
  --atol 1e-5 --rtol 1e-5 `
  --report D:\parity\report.json
```

No official real-weight comparison was run in the packaging environment.
