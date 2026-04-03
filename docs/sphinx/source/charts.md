# Visualization Module: Interactive Charts

The `pyperfanalytics.charts` module provides interactive visualizations built on top of **Plotly**. These functions are designed to match the visual output and logic of the R `PerformanceAnalytics` package while leveraging modern web-based interactivity.

---

## Core Performance Charts

### `chart_cum_returns`
Creates a cumulative returns chart. Matches R's `chart.CumReturns`.

- **Parameters:**
  - `R`: `pd.Series` or `pd.DataFrame` of returns.
  - `wealth_index` (bool, default `False`): If `True`, starts at $1. If `False`, starts at $0.
  - `geometric` (bool, default `True`): If `True`, uses geometric compounding: $\prod(1+R)-1$.
  - `title` (str): Chart title.
  - `colorset` (list[str]): Optional list of hex colors.
- **Returns:** `plotly.graph_objects.Figure`

### `chart_bar_returns`
Creates a bar chart of period returns. Matches R's `chart.Bar`.

- **Parameters:**
  - `R`: `pd.Series` or `pd.DataFrame` of returns.
  - `title` (str): Chart title.
  - `colorset` (list[str]): Optional list of hex colors.
- **Returns:** `plotly.graph_objects.Figure` (uses `barmode='overlay'`)

### `chart_drawdown`
Creates a drawdown (underwater) chart showing losses from peak. Matches R's `chart.Drawdown`.

- **Parameters:**
  - `R`: `pd.Series` or `pd.DataFrame` of returns.
  - `geometric` (bool, default `True`): Use geometric returns for drawdown calculation.
  - `title` (str): Chart title.
- **Returns:** `plotly.graph_objects.Figure` with area fill to zero.

### `charts_performance_summary`
A combined dashboard containing Cumulative Returns, Period Returns, and Drawdowns. Matches R's `charts.PerformanceSummary`.

- **Parameters:**
  - `R`: `pd.Series` or `pd.DataFrame` of returns.
  - `geometric` (bool, default `True`): Use geometric compounding.
  - `wealth_index` (bool, default `False`): Start cumulative returns at $1.
- **Returns:** `plotly.graph_objects.Figure` with 3 subplots.

---

## Risk & Distribution Charts

### `chart_bar_var`
Plots periodic returns as a bar chart with interactive risk metric overlays (VaR/ES). Matches R's `chart.BarVaR`.

- **Parameters:**
  - `methods` (list[str]): List of risk metrics (e.g., `["ModifiedVaR", "GaussianVaR"]`).
  - `width` (int, default `0`): Window width for rolling calculation. `0` uses expanding window.
  - `p` (float, default `0.95`): Confidence level.
  - `show_symmetric` (bool): If `True`, plots upper-tail threshold (e.g., for standard deviation).
- **Returns:** Bar chart with line overlays.

### `chart_histogram`
Histogram of returns with optional density curves and risk markers. Matches R's `chart.Histogram`.

- **Parameters:**
  - `breaks` (int, default `30`): Number of bins.
  - `methods` (list[str]): Overlay types: `"add.density"`, `"add.normal"`, `"add.risk"`.
- **Returns:** Histogram with Scatter (line) overlays and vertical risk lines.

### `chart_boxplot`
Horizontal box and whiskers plot for comparing return distributions across assets. Matches R's `chart.Boxplot`.

- **Parameters:**
  - `sort_by` (str): Sort assets by `"mean"`, `"median"`, or `"variance"`.
  - `sort_ascending` (bool): Sorting direction.
- **Returns:** Horizontal box plot with outlier markers.

### `chart_qqplot`
Quantile-Quantile plot with theoretical normal reference and confidence bands. Matches R's `chart.QQPlot`.

- **Parameters:**
  - `main` (str): Chart title.
- **Returns:** Scatter plot with a reference line and a shaded confidence area.

### `chart_var_sensitivity`
Visualizes how risk estimates (VaR/ES) change across a range of confidence levels (89% to 99%). Matches R's `chart.VaRSensitivity`.

- **Returns:** Multi-line chart showing the slope of risk metrics.

---

## Rolling Metrics & Regression

### `chart_rolling_performance`
Generic wrapper to chart any performance metric over a rolling window. Matches R's `chart.RollingPerformance`.

- **Parameters:**
  - `width` (int): Rolling window size.
  - `FUN` (str): Name of the function to apply (e.g., `"return_annualized"`, `"sharpe_ratio"`).
- **Returns:** Time-series line chart.

### `chart_rolling_correlation`
Plots the rolling correlation between two sets of assets. Matches R's `chart.RollingCorrelation`.

- **Parameters:**
  - `Ra`, `Rb`: Two sets of returns to correlate.
  - `width` (int): Rolling window size.
- **Returns:** Multi-line chart of pairwise correlations.

### `charts_rolling_regression`
Dashboard showing Rolling Alpha, Rolling Beta, and Rolling R-Squared against a benchmark. Matches R's `charts.RollingRegression`.

- **Parameters:**
  - `Ra`: Asset returns.
  - `Rb`: Benchmark returns.
  - `width` (int): Rolling window size.
  - `Rf`: Risk-free rate.
- **Returns:** 3-row subplot figure.

---

## Specialized Diagnostic Charts

### `chart_correlation`
A visualization of a Correlation Matrix. Matches R's `chart.Correlation`.
- **Diagonal**: Histogram + Density.
- **Lower Triangle**: Scatter plots with linear regression lines.
- **Upper Triangle**: Correlation coefficients with significance stars (`***` for p < 0.001).

### `chart_acf_plus`
Time series diagnostics showing both Autocorrelation (ACF) and Partial Autocorrelation (PACF). Matches R's `chart.ACFplus`.

- **Parameters:**
  - `maxlag` (int): Maximum number of lags to display.
- **Returns:** 2-row subplot with confidence interval dashed lines.

### `chart_snail_trail`
Plots rolling risk vs. return over time as a "trail" in a scatter plot. Matches R's `chart.SnailTrail`.

- **Parameters:**
  - `stepsize` (int): Periods between markers in the trail.
  - `add_sharpe` (list[float]): Values for Sharpe ratio reference lines.
- **Returns:** Scatter plot with lines connecting historical risk/return points.

---

## Portfolio & Relative Performance

### `chart_risk_return_scatter`
Standard risk-return scatter plot with Sharpe ratio indifference lines. Matches R's `chart.RiskReturnScatter`.

- **Parameters:**
  - `Rf`: Risk-free rate for the intercept.
  - `add_sharpe` (list[float]): Sharpe ratio lines to draw (default `[1, 2, 3]`).
- **Returns:** Labeled scatter plot.

### `chart_relative_performance`
Plots the ratio of cumulative performance between two assets over time. Matches R's `chart.RelativePerformance`.

- **Returns:** Ratio chart (Values > 1 indicate Outperformance).

### `chart_capture_ratios`
Scatter plot of Upside Capture versus Downside Capture against a benchmark. Matches R's `chart.CaptureRatios`.

- **Returns:** Scatter plot with (1,1) benchmark crosshairs.

### `chart_component_returns`
Stacked bar chart showing the contribution of each asset to the total portfolio return. Matches R's `chart.ComponentReturns`.

- **Parameters:**
  - `weights`: Portfolio weights (default: equal weight).
- **Returns:** Stacked bar chart where sum equals total portfolio return.

---

## Utility Plots

- `chart_ecdf`: Empirical Cumulative Distribution Function vs. Normal CDF.
- `chart_events`: Event study plot aligning returns around specific dates.
- `chart_scatter`: Generic scatter plot with marginal rug/histograms and OLS trendline.
- `chart_stacked_bar`: General purpose stacked bar chart.
