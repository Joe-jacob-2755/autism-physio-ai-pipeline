# Rule: Clinical Data Integrity

## Scope
All code that reads, transforms, or writes physiological signal data.

## Rules

### R1.1 — Validate signal ranges at every boundary
Every function that transforms signal data MUST validate outputs against physiological ranges defined in `config.py`. Out-of-range values after transformation indicate a bug — raise an error, do not silently clamp.

```python
# CORRECT
if filtered_eda.max() > SIGNAL_RANGES['EDA']['max']:
    raise ValueError(f"EDA post-filter exceeds physiological max: {filtered_eda.max():.2f} µS")

# WRONG — silent corruption
filtered_eda = np.clip(filtered_eda, 0.01, 30.0)  # hides the bug
```

### R1.2 — Never silently drop data
If rows, samples, or entire channels are removed, the operation MUST:
1. Log what was removed and why
2. Record the count in the cleaning report or metadata
3. Return the count to the caller

### R1.3 — Preserve timestamp alignment
The `timestamp_s` column is the temporal backbone. Any operation that reindexes, resamples, or merges DataFrames MUST verify that `timestamp_s` remains monotonically increasing and correctly aligned.

### R1.4 — NaN is a signal, not noise
NaN in physiological data means "we don't know" — it must propagate honestly or be handled explicitly. Never fill NaN with 0 (that's a real physiological value). Use median imputation only at the feature level, never at the signal level.

### R1.5 — Annotation integrity
Columns `target_label`, `event_id`, and `category` must survive all transformations unchanged. If a transformation changes row count (resampling, windowing), annotations must be re-aligned, not dropped.

## Rationale
This pipeline processes data from non-verbal autistic children who cannot report errors. Silent data corruption propagates through every downstream module and ultimately affects clinical predictions. Every signal value must be trustworthy.
