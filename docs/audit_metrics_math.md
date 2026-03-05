# Mathematical Audit: All Performance Metrics

Cross-referenced Python implementations against **both**:
1. R `PerformanceAnalytics` source code (GitHub `braverock/PerformanceAnalytics`)
2. Original academic papers and authoritative definitions

---

## Summary Table

| Metric | vs R | vs Academic | Notes |
|--------|--------|-------|
| `var_historical` | ✅ | ✅ | Empirical quantile |
| `var_gaussian` | ✅ | ✅ | $-\mu - z_\alpha \sigma$; standard parametric VaR |
| `var_modified` | ✅ | ✅ | Cornish-Fisher; Favre & Galeano (2002) |
| `es_historical` | ✅ | ✅ | Mean of tail returns |
| `es_gaussian` | ✅ | ✅ | $-\mu + \sigma\phi(z_\alpha)/\alpha$ |
| `es_modified` | ✅ | ✅ | Boudt et al. (2008); see [analysis_es_modified.md](./analysis_es_modified.md) |
| `ulcer_index` | ✅ | ✅ | Peter Martin & McCann (1989) |
| `pain_index` | ✅ | ✅ | Zephyr Associates |
| `tracking_error` | ✅ | ✅ | $\sigma(R_a - R_b)\sqrt{scale}$; Sharpe (1994) |
| `capm_beta` | ✅ | ✅ | $\text{Cov}(xR_a,xR_b)/\text{Var}(xR_b)$; Sharpe (1964) |
| `sharpe_ratio` | ✅ | ✅ | Sharpe (1994) |
| `sortino_ratio` | ✅ | ✅ | Sortino & Price (1994) |
| `information_ratio` | ✅ | ✅ | ActivePremium / TrackingError; Sharpe (1994) |
| `treynor_ratio` | ✅ | ✅ | Treynor (1965/1966) |
| `calmar_ratio` | ✅ | ✅ | Young (1991), no Rf in original |
| `kelly_ratio` | ✅ | ✅ | Kelly (1956); $f^* = (\mu-R_f)/\sigma^2$ |
| `omega_ratio` | ✅ | ✅ | Keating & Shadwick (2002) |
| `upside_potential_ratio` | ✅ | ✅ | Sortino et al. (1999) |
| `martin_ratio` | ✅ | ✅ Fixed | Rf annualized — see [bugfix_martin_pain_ratio_rf.md](./bugfix_martin_pain_ratio_rf.md) |
| `pain_ratio` | ✅ | ✅ Fixed | Rf annualized — see [bugfix_martin_pain_ratio_rf.md](./bugfix_martin_pain_ratio_rf.md) |
| `burke_ratio` | ⚠️ R-fix | ⚠️ Denominator differs from paper | See Burke Ratio section below |
| `modigliani` | ✅ | ✅ | $SR_a \cdot \sigma_b + R_f$; Modigliani & Modigliani (1997) |
| `kappa` | ✅ | ✅ | Kaplan & Knowles (2004); LPM divisor = full $n$ |
| `bernardo_ledoit_ratio` | ✅ | ✅ | $E[R^+]/E[\|R^-\|]$; Bernardo & Ledoit (1996) |
| `d_ratio` | ✅ | ✅ | Inverse of Bernardo-Ledoit |
| `volatility_skewness` | ✅ | ✅ | UpsideVar / DownsideVar |
| `omega_sharpe_ratio` | ✅ | ✅ | $(U_p - D_p)/D_p$ |
| `downside_sharpe_ratio` | ✅ | ✅ | $(\bar{R}-R_f)/(\sqrt{2}\cdot\text{SemiSD})$ |
| `mean_absolute_deviation` | ✅ | ✅ | $\frac{1}{n}\sum|R_i - \bar{R}|$ |
| `downside_frequency` | ✅ | ✅ | Count(R < MAR) / n |
| `annualized_excess_return` | ✅ | ✅ | Geometric: $(1+R_{pa})/(1+R_{ba})-1$ |

---

## Detailed Notes by Metric

### Risk Metrics (risk.py)

#### VaR Variants
All three variants correctly return **positive values** representing the loss amount:
- **Historical**: `−quantile(R, α)` ✅
- **Gaussian**: `−μ − z_α σ` ✅ — standard parametric VaR (Jorion 2007)
- **Modified (CF)**: `−μ − h σ` where `h` is the Cornish-Fisher expansion of `z_α` ✅ — Favre & Galeano (2002)

#### ES Variants
- **Historical**: average of returns in the tail ✅
- **Gaussian**: $-\mu + \sigma \phi(z_\alpha)/\alpha$ ✅ — Rockafellar & Uryasev (2000)
- **Modified**: Boudt et al. (2008) Edgeworth expansion using `h` as upper integration limit ✅ — see [analysis_es_modified.md](./analysis_es_modified.md)

#### CAPM Beta
`capm_beta` correctly computes excess returns for both Ra and Rb before computing
`Cov(xRa, xRb) / Var(xRb)`, consistent with both R's `CAPM.beta` and the CAPM definition (Sharpe 1964). ✅

---

### Performance Metrics (returns.py)

#### Treynor Ratio
**Academic formula** (Treynor 1965/1966): $T = (r_i - r_f)/\beta_i$

**Python**: `AnnReturn(xRa) / capm_beta(Ra, Rb, Rf)` — uses annualized excess return in numerator, beta on excess returns ✅

> Note: Treynor's original formula used average (not annualized) excess return / beta. Both R and Python use annualized excess return, which is the standard modern convention per Bacon (2008) p.77.

#### Calmar Ratio
**Academic formula** (Young 1991, *Futures* magazine): $Calmar = R_{ann}/|MaxDD|$

The original definition **does not include a risk-free rate** in the numerator. Python implementation correctly uses `return_annualized(R) / abs(max_drawdown(R))` matching the original. ✅

#### Kelly Ratio
**Academic formula** (Kelly 1956): $f^* = (\mu - r_f)/\sigma^2$

**Python**: `mean(R - Rf) / std(R)²` — uses total stddev `std(R)`, not excess return stddev, consistent with R's `StdDev(R)`. Half-Kelly = divide by 2. ✅

#### Omega Ratio
**Academic formula** (Keating & Shadwick 2002):
$$\Omega(L) = \frac{\int_L^\infty [1-F(r)]dr}{\int_{-\infty}^L F(r)dr} = \frac{E[\max(R-L,0)]}{E[\max(L-R,0)]}$$

Python uses the discrete empirical estimator. The `exp(-Rf)` factor in some formulations cancels between numerator and denominator and is correctly omitted. ✅

#### Kappa Ratio
**Academic formula** (Kaplan & Knowles 2004):
$$K_n(\tau) = \frac{\mu - \tau}{LPM_n(\tau)^{1/n}}$$
where $LPM_n(\tau) = \frac{1}{T}\sum_{t=1}^{T}\max(\tau - R_t, 0)^n$

Python implementation: denominator uses full $n$ (all observations) as divisor, consistent with the Kaplan-Knowles definition of LPM. ✅

> Note: $K_1 \propto \Omega$, $K_2 =$ Sortino (when $\tau=MAR$), $K_3$ is Kappa 3.

#### Burke Ratio — Two Known Deviations

**Academic formula** (Burke 1994, *Futures* magazine):
$$BR = \frac{r_P - r_F}{\sqrt{\sum_{t=1}^d D_t^2 / d}}$$

where the denominator is $\sqrt{\text{mean}(D^2)}$ — **root-mean-square** of drawdowns over $d$ events.

**Deviation 1 — vs R**: R's implementation multiplies drawdown segments by `0.01` (expects percentage inputs). Python fixes this for decimal returns. ✅ (documented)

**Deviation 2 — vs Paper**: Python uses `√(Σ D²)` (sum, not mean), R also uses `√(sum)`. The original 1994 Burke paper uses `√(ΣD²/d)` = `√(mean(D²))`. The two forms differ by a factor of `√d`.

> The paper denominator scales down with more drawdowns (averaging effect), while sum-form grows with number of drawdowns. This affects strategies with many small drawdowns vs. few large ones differently. **For consistency with the original paper**, the denominator should divide by `d` (number of drawdown events). However, both R and Python use the sum form — this is a shared convention deviation from the original.

#### Sortino Ratio
**Academic formula** (Sortino & Price 1994):
$$S = \frac{\bar{R} - MAR}{DD_{MAR}}$$
where $DD_{MAR} = \sqrt{\frac{1}{n}\sum_{t=1}^{n}\min(R_t - MAR,\, 0)^2}$ (full $n$ in denominator)

Python: `mean(R - MAR) / downside_deviation(R, MAR, method="full")` — `method="full"` correctly uses all $n$ observations in denominator. ✅

#### Modigliani M² (`modigliani`)
**Academic formula** (Modigliani & Modigliani 1997):
$$M^2 = SR_a \cdot \sigma_b + R_f$$

Python uses the **non-annualized** Sharpe ratio and **per-period** $\sigma_b$. The original paper uses periodic data throughout, so this is correct. ✅

#### Bernardo-Ledoit Ratio
**Academic formula** (Bernardo & Ledoit 1996):
$$GLR = \frac{E[x^+]}{E[x^-]}$$
where $x^+ = \max(0, R)$, $x^- = \max(0, -R)$

Python: `sum(R[R>0]) / abs(sum(R[R<0]))` = sample estimate of $E[R^+]/E[R^-]$ (1/n cancels). ✅

---

## Previously Fixed Issues

| Issue | File | Fix |
|-------|------|-----|
| `martin_ratio`: periodic Rf vs annualized Rp | `returns.py` | Rf annualized: `(1+rf)^scale - 1` |
| `pain_ratio`: periodic Rf vs annualized Rp | `returns.py` | Rf annualized: `(1+rf)^scale - 1` |

---

## References

- R `PerformanceAnalytics` source: [github.com/braverock/PerformanceAnalytics](https://github.com/braverock/PerformanceAnalytics)
- Sharpe, W.F. (1994). *The Sharpe Ratio.* Journal of Portfolio Management.
- Sortino, F. & Price, L. (1994). *Performance Measurement in a Downside Risk Framework.*
- Boudt, K., Peterson, B., & Croux, C. (2008). *Estimation and Decomposition of Downside Risk.*
- Burke, G. (1994). *A Sharper Sharpe Ratio.* Futures Magazine.
- Modigliani, F. & Modigliani, L. (1997). *Risk-Adjusted Performance.* Journal of Portfolio Management.
