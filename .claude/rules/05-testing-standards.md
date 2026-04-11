# Rule: Testing Standards

## Scope
All test files and the testing practices for this pipeline.

## Rules

### R5.1 — Every module must have tests
When a new module is built, tests MUST be added to `tests/` before the module is considered complete. Minimum coverage:
- All public functions called with valid input
- Edge cases: empty input, single-sample input, all-NaN input
- Boundary values: minimum/maximum physiological ranges
- Reproducibility: same seed produces same output

### R5.2 — Tests must be deterministic
All tests MUST use fixed seeds. A test that passes 99% of the time is not a passing test.

### R5.3 — Test physiological plausibility
For signal processing modules, tests MUST verify physiological plausibility:
- Known emotion → expected physiological direction (Fear → EDA increase)
- Feature values within published ranges (RMSSD: 10-100 ms typical)
- Transformations preserve signal energy or explain why not

### R5.4 — Test the contract, not the implementation
Tests should verify that module outputs match the documented contract (correct columns, correct types, correct ranges) rather than testing internal implementation details.

### R5.5 — No test interdependence
Each test must be independently runnable. No test may depend on output from a previous test or on a specific execution order.

### R5.6 — Test naming convention
```
test_<module>_<function>_<scenario>
```
Example: `test_feature_extractor_eda_features_empty_signal`

### R5.7 — Run tests before every commit
```powershell
python -m pytest tests/ -v --tb=short
```
All tests must pass. No skips without documented justification.

## Rationale
This pipeline will eventually process real clinical data. Untested code in a clinical pipeline is a liability. Tests are the first line of defence against regressions that could silently corrupt clinical predictions.
