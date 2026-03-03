# PyPerfAnalytics

PyPerfAnalytics is a Python port of the popular R package `PerformanceAnalytics`. It provides a collection of functions for performance and risk analysis of financial portfolios.

The library ensures algorithmic consistency with the original R implementation, validated against R benchmarks with high precision.

## Key Features

- **Returns Calculation**: Cumulative, annualized, and excess returns.
- **Risk Metrics**: VaR (Historical, Gaussian, Modified), Expected Shortfall, Tracking Error, etc.
- **Performance Ratios**: Sharpe, Sortino, Information, Treynor, Calmar, Omega, Burke, and more.
- **Advanced Metrics**: Hurst Index, Smoothing Index (Getmansky), CDaR (Conditional Drawdown at Risk).
- **Summary Tables**: Comprehensive reporting tables for distributions, correlations, downside risk, and drawdowns.
- **Attribution**: Aggregation of asset contributions to portfolio returns.

## Installation

```bash
# Using uv (recommended)
uv pip install -e .

# Using pip
pip install -e .
```

## Quick Start

```python
import pandas as pd
import pyperfanalytics as pa

# Load your return data
returns = pd.read_csv("your_returns.csv", index_col=0, parse_dates=True)

# Calculate Annualized Sharpe Ratio
sharpe = pa.sharpe_ratio(returns, Rf=0, annualize=True)

# Generate a Downside Risk Summary Table
risk_table = pa.table_downside_risk_ratio(returns)
print(risk_table)
```

## API Overview

### Returns & Data Adjustments (`pyperfanalytics.returns`)
- `return_calculate`: Calculate returns from prices.
- `return_portfolio`: Aggregate portfolio returns with periodic rebalancing and geometric/arithmetic compounding.
- `return_clean`: Robust data cleaning using MCD (Minimum Covariance Determinant).
- `return_geltner`: Liquidity adjustment using autocorrelation unsmoothing.
- `return_annualized` / `return_cumulative`: Annualize or compound returns.
- `return_excess`: Calculate excess returns over a risk-free rate.

### Risk & Drawdowns (`pyperfanalytics.risk`, `pyperfanalytics.drawdowns`)
- `var_historical` / `var_gaussian` / `var_modified`: Value at Risk.
- `es_historical` / `es_gaussian` / `es_modified`: Expected Shortfall.
- `tracking_error` / `specific_risk` / `systematic_risk`: Active management risk metrics.
- `capm_beta` / `capm_alpha` / `jensen_alpha` / `fama_beta`: Asset pricing model metrics.
- `max_drawdown` / `find_drawdowns`: Identify and measure historical drawdowns.
- `cdar` / `cdar_beta` / `cdar_alpha`: Conditional Drawdown at Risk.

### Performance Ratios (`pyperfanalytics.returns`)
- **Risk-Adjusted**: `sharpe_ratio`, `sortino_ratio`, `calmar_ratio`, `omega_ratio`.
- **Advanced**: `appraisal_ratio`, `kappa`, `prospect_ratio`, `sterling_ratio`, `m_squared`, `m2_excess`.
- **Active / Drawdown**: `information_ratio`, `burke_ratio`, `martin_ratio`, `pain_ratio`, `net_selectivity`.
- **Statistical**: `hurst_index`, `smoothing_index`, `mean_absolute_deviation`.

### Summary Tables (`pyperfanalytics.tables`)
- `table_stats`: Comprehensive statistical summary.
- `table_annualized_returns`: Core annualized performance facts.
- `table_distributions`: Summary of moments (Skewness, Kurtosis).
- `table_correlation` / `table_autocorrelation`: Pearson correlations and lag dependencies.
- `table_downside_risk_ratio` / `table_drawdowns_ratio`: Risk and drawdown tracking.
- `table_capture_ratios` / `table_up_down_ratios`: Market capture evaluations.
- `table_prob_outperformance` / `table_rolling_periods`: Outperformance probability and trailing metrics.

## Verification

Every function is verified against the R `PerformanceAnalytics` implementation using the `managers` and `edhec` standard datasets. Tests are located in the `tests/` directory and can be run using `pytest`.

```bash
uv run pytest
```
