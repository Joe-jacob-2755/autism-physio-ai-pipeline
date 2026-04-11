# Rule: Reproducibility

## Scope
All code that involves randomness, output generation, or numerical computation.

## Rules

### R2.1 — Seeded randomness only
Every random operation MUST use an explicitly passed `np.random.Generator`. Never use `np.random.rand()`, `random.random()`, or any global random state.

```python
# CORRECT
rng = np.random.default_rng(seed=42)
noise = rng.normal(0, 0.1, size=n_samples)

# WRONG — non-reproducible
noise = np.random.normal(0, 0.1, size=n_samples)
```

### R2.2 — Never overwrite previous runs
Output folders use auto-numbered naming: `M*_v*_run_NNN/`. The code MUST find the next available number, never reuse or overwrite an existing run folder.

### R2.3 — Record full parameter state
Every module's `metadata.json` MUST include:
- All parameters that affect output (filter type, cutoffs, window size, overlap, scaler type)
- Software versions (module version, numpy, scipy, scikit-learn)
- Input source path or session ID
- Seed value
- Timestamp of execution

### R2.4 — Deterministic output given identical input
Same input + same seed + same dependency versions = bit-identical output. If this invariant breaks, it is a bug.

### R2.5 — Pin dependency versions
`requirements.txt` must specify exact versions (`numpy==1.26.4`) not ranges (`numpy>=1.24`). Numerical behaviour changes across versions.

## Rationale
Regulatory auditors, ethics boards, and peer reviewers must be able to reproduce any result from this pipeline months or years later. Non-reproducibility undermines the scientific and clinical validity of the entire project.
