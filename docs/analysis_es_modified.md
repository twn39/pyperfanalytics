# Analysis: Modified Expected Shortfall (`es_modified`) — Formula Verification

## Conclusion

**No bug found.** The use of `h` (Cornish-Fisher corrected quantile) in the polynomial adjustment terms of `es_modified` is **mathematically correct** and consistent with both R's `PerformanceAnalytics` source and the original Boudt et al. (2008) derivation.

---

## Background

`es_modified` implements the Modified Expected Shortfall from:

> Boudt, K., Peterson, B., & Croux, C. (2008). *Estimation and Decomposition of Downside Risk for Portfolios with Non-Normal Returns.* Journal of Risk, 11(2), 79–103.

A concern was raised that the polynomial terms should use the standard normal quantile $z_\alpha$ instead of the Cornish-Fisher corrected quantile $h$.

---

## The Two Quantiles

| Symbol | Definition | Role |
|--------|-----------|------|
| $z_\alpha = \Phi^{-1}(\alpha)$ | Standard normal quantile | Base of the CF expansion |
| $h = g_\alpha = G^{-1}_2(\alpha)$ | CF corrected quantile (code: `h`) | Integration upper limit for MES |

The CF expansion is:

$$h = z_\alpha + \frac{1}{6}(z_\alpha^2-1)s + \frac{1}{24}(z_\alpha^3-3z_\alpha)\kappa - \frac{1}{36}(2z_\alpha^3-5z_\alpha)s^2$$

---

## Why `h` Is Correct in the Polynomial Terms

Modified ES is defined as:

$$\text{mES}(\alpha) = -\mu - \sqrt{m_2}\, E_G[z \mid z \leq g_\alpha]$$

The conditional expectation $E_G[z \mid z \leq g_\alpha]$ is computed via an Edgeworth expansion with **integration upper limit $g_\alpha = h$**. After expansion (Boudt et al., Appendix A):

$$E_G[z \mid z \leq g_\alpha] = \frac{-1}{\alpha}\,\phi(h)\Bigl[1 + \frac{s}{6}h^3 + \frac{\kappa}{24}(h^4 - 2h^2 - 1) + \frac{s^2}{72}(h^6 - 9h^4 + 9h^2 + 3)\Bigr]$$

The polynomial arguments are $h$ because **$h$ is the integration boundary**, not by convention.

---

## Code vs R Source Agreement

**Python (`risk.py`):**
```python
h = z + (1/6)*(z**2-1)*skew + (1/24)*(z**3-3*z)*exkurt - (1/36)*(2*z**3-5*z)*skew**2
E = norm.pdf(h) * (
    1.0 + h**3 * skew / 6.0
    + (h**6 - 9*h**4 + 9*h**2 + 3) * skew**2 / 72.0
    + (h**4 - 2*h**2 - 1) * exkurt / 24.0
) / alpha
```

**R (`PortfolioRisk.R`, `ES.CornishFisher`):**
```r
h = z + (1/6)*(z^2-1)*skew + (1/24)*(z^3-3*z)*exkurt - (1/36)*(2*z^3-5*z)*skew^2
MES <- dnorm(h) * (1 + h^3*skew/6.0 +
                   (h^6 - 9*h^4 + 9*h^2 + 3)*skew^2/72 +
                   (h^4 - 2*h^2 - 1)*exkurt/24)
```

**Both implementations are identical and mathematically justified.**

---

## Common Misconception

The confusion arises from comparing with Gaussian ES:

$$\text{ES}_{Gauss} = -\mu + \sigma \cdot \frac{\phi(z_\alpha)}{\alpha}$$

Here $z_\alpha$ appears because it is the integration upper limit under the normal distribution. Modified ES replaces that limit with $h = g_\alpha$, so all polynomial and PDF terms naturally shift to use $h$.

---

## References

- Boudt, K., Peterson, B., & Croux, C. (2008). *Estimation and Decomposition of Downside Risk for Portfolios with Non-Normal Returns.* Journal of Risk, 11(2). Appendix A.
- PerformanceAnalytics source: `R/PortfolioRisk.R` — `ES.CornishFisher` and `operES.CornishFisher` functions (confirmed via GitHub).
