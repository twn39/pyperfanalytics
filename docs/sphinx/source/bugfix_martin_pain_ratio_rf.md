# Bug Fix: Martin Ratio & Pain Ratio — Rf Annualization

## Summary

`martin_ratio` and `pain_ratio` previously subtracted the **periodic** (per-period) risk-free rate `Rf` from the **annualized** portfolio return `Rp`, producing a numerator with mismatched time units. This has been corrected in **v0.x** to follow the original Peter Martin / Zephyr Associates definition.

---

## The Problem

Both ratios share the same structure:

$$\text{Ratio} = \frac{R_{p,ann} - R_f}{\text{DenominatorIndex}}$$

The old code passed `Rf` straight through:

```python
# BEFORE (incorrect when Rf ≠ 0)
ann_ret = return_annualized(R, scale=scale)   # e.g. 13.75 %/yr
rf_val  = Rf                                  # e.g.  0.32 %/month
return (ann_ret - rf_val) / ui               # 13.75% − 0.32% — unit mismatch!
```

- `R` package `PerformanceAnalytics::MartinRatio` / `PainRatio` has the same defect (confirmed from source code).
- When `Rf = 0` (default) the bug has no effect.
- When `Rf ≠ 0`, the subtracted value is ~12× smaller than it should be, inflating the ratio.

### Numerical example (HAM1, monthly data, rf_monthly ≈ 0.32 %/mo)

| | Before (incorrect) | After (correct) |
|-|-------------------|-----------------|
| `Rf` used in numerator | 0.32 %/mo | 3.94 %/yr |
| Martin Ratio | **3.710** | **2.710** |
| Pain Ratio | **8.573** | **6.263** |

---

## The Fix

Rf is annualized before subtraction using compound interest:

$$R_{f,ann} = (1 + R_{f,\text{periodic}})^{scale} - 1$$

```python
# AFTER (correct)
ann_ret = return_annualized(R, scale=scale)
rf_ann  = (1 + rf_val) ** scale - 1          # periodic → annual
return (ann_ret - rf_ann) / ui               # both in %/yr ✓
```

This matches the definition used by:
- **Peter Martin** (Ulcer Performance Index creator): *"annualized rates of return are used for both the total return and the risk-free return"*
- **Zephyr Associates** (Pain Ratio naming): `Pain Ratio = (AnnRtn(r) - AnnRtn(cash)) / PainIndex`

---

## API Impact

### Input convention (unchanged)

`Rf` must be passed as a **per-period rate** matching the periodicity of `R`:

```python
# Monthly returns → monthly Rf
martin_ratio(monthly_returns, Rf=0.004)   # 0.4 %/month ✓

# Daily returns → daily Rf
martin_ratio(daily_returns, Rf=0.00015)   # ~3.8 %/yr daily ✓
```

The function always annualizes `Rf` internally; you do **not** need to pre-annualize it.

### Deviating from R

If you need to reproduce R's (incorrect) output exactly, compute the ratio manually:

```python
from pyperfanalytics.returns import return_annualized
from pyperfanalytics.risk import ulcer_index

ann_ret = return_annualized(R)
ui = ulcer_index(R)
r_compatible = (ann_ret - rf_monthly) / ui   # matches R, but units mismatch
```

---

## References

- Peter G. Martin & Byron McCann (1989). *The Investor's Guide to Fidelity Funds*. See also: [Ulcer Index — Wikipedia](https://en.wikipedia.org/wiki/Ulcer_index)
- Zephyr Associates. *The Pain Ratio*. Swan Global Investments white paper.
- Carl Bacon. *Practical Portfolio Performance Measurement and Attribution*, 2nd ed. (2008), p. 91.
- PerformanceAnalytics source: `R/MartinRatio.R`, `R/PainRatio.R` — confirmed to use periodic `Rf`.
