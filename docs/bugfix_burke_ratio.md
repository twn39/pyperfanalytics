# Bugfix: Burke Ratio Denominator (paper-correct RMS form)

## Problem

The original Burke Ratio formula, as published in Burke (1994) *Futures* magazine, defines the denominator as:

$$BR = \frac{R_{ann} - R_f}{\sqrt{\dfrac{\sum_{t=1}^{d} D_t^2}{d}}}$$

The denominator $\sqrt{\sum D_t^2 / d}$ is the **root-mean-square (RMS)** of the $d$ drawdown events.

### Previous (incorrect) implementation

Both R's `PerformanceAnalytics::BurkeRatio` and the earlier `pyperfanalytics` implementation used:

$$\text{denominator} = \sqrt{\sum_{t=1}^d D_t^2}$$

This form **does not divide by $d$**, making it equal to the paper's form multiplied by $\sqrt{d}$.
Consequently the ratio was deflated by $\sqrt{d}$ relative to the paper definition.

For the test portfolio (`portfolio_bacon` column 1, $d = 7$ drawdown events):

| Quantity | Old (sqrt-sum) | New (RMS / paper) |
|----------|---------------|-------------------|
| Denominator | 0.137101 | 0.051819 |
| Burke Ratio | 0.756221 | **2.000773** |
| Modified Burke Ratio | 3.704711 | **9.801745** |

The factor is exactly $\sqrt{7} \approx 2.646$: `2.000773 / 0.756221 ≈ 2.646`.

## Why It Matters

The RMS form has an important **scaling invariance property**: a portfolio with $d$ identical small drawdowns gets the *same* ratio as a portfolio with 1 drawdown of the same magnitude. The old sum form grows with $\sqrt{d}$, penalizing portfolios that have many small recovery events — inconsistent with the paper's intent.

## Fix Applied

In `src/pyperfanalytics/returns.py`, the `_calc` inner function of `burke_ratio` was changed from:

```python
# OLD
denom_sq = np.sum(drawdowns**2)
ratio = (rp - rf_val) / np.sqrt(denom_sq)
```

to:

```python
# NEW (paper-correct)
d = len(drawdowns)
denom = np.sqrt(np.sum(drawdowns**2) / d)   # RMS of drawdowns
ratio = (rp - rf_val) / denom
```

The `modified=True` path (`ratio *= sqrt(n)`) is unchanged.

## Two Deviations from R Corrected in Total

This implementation deviates from R's `PerformanceAnalytics::BurkeRatio` in **two** ways, both corrections:

| Issue | R Behaviour | pyperfanalytics |
|-------|-------------|-----------------|
| Input scale | Expects % inputs (e.g., 5.0 for 5%), multiplies drawdowns by `0.01` | Expects decimal inputs (e.g., 0.05 for 5%) — no scaling |
| Denominator | `sqrt(sum(D²))` — deviates from original paper | `sqrt(sum(D²)/d)` — matches Burke (1994) paper exactly |

## Test Update

`tests/test_downside.py :: test_burke_ratio_corrected` updated with new expected values:
- `burke_ratio(col1, scale=12)` → `2.000773` (was `0.756221`)
- `burke_ratio(col1, modified=True, scale=12)` → `9.801745` (was `3.704711`)

## References

- Burke, G. (1994). *A Sharper Sharpe Ratio.* Futures Magazine.
- Bacon, C. (2008). *Practical Portfolio Performance Measurement and Attribution*, 2nd ed. p. 90.
- Wikipedia: [Burke Ratio](https://en.wikipedia.org/wiki/Burke_ratio)
