# Rule: Numerical Safety

## Scope
All signal processing, feature extraction, and mathematical operations.

## Rules

### R4.1 — NumPy 2.x compatibility
- Use `np.trapezoid()` not `np.trapz()` (removed in NumPy 2.0)
- Use `arr.max() - arr.min()` not `arr.ptp()` (removed in NumPy 2.0)
- Test with the pinned NumPy version before committing

### R4.2 — Guard every division
Any division where the denominator could be zero MUST be guarded. Common cases:
- HRV features with 0 or 1 IBI values
- Normalisation with zero IQR or zero std
- Spectral power ratios with zero total power
- Slope calculations with constant signals

```python
# CORRECT
lf_hf_ratio = lf_power / hf_power if hf_power > 1e-10 else np.nan

# WRONG — silent inf
lf_hf_ratio = lf_power / hf_power
```

### R4.3 — Filter stability verification
Butterworth filters MUST have cutoff frequencies strictly below the Nyquist frequency. Filter order must not cause numerical instability for the signal length. Use `scipy.signal.sosfilt` (second-order sections) for orders > 4.

### R4.4 — Window edge handling
Feature extraction windows at signal boundaries must either:
- Be excluded if they contain < 50% valid data, OR
- Be explicitly flagged as partial windows in the output

Never pad with zeros and silently include — zeros are real physiological values.

### R4.5 — Float comparison tolerance
Never compare floating-point physiological values with `==`. Use `np.isclose()` or explicit tolerance:

```python
# CORRECT
if np.isclose(eda_value, baseline, atol=0.01): ...

# WRONG
if eda_value == baseline: ...
```

### R4.6 — Overflow protection for index arithmetic
When computing sample indices from time values (e.g., `int(time_s * sampling_rate)`), verify the result fits in the array bounds. Off-by-one on a 64 Hz signal over 5 minutes is a 19,200-element array — index errors are real.

## Rationale
Numerical bugs in signal processing are insidious — they produce plausible-looking but wrong results that silently degrade every downstream model. A Butterworth filter with cutoff above Nyquist doesn't crash; it produces aliased garbage that looks like a valid signal.
