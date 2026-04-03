import numpy as np
import pandas as pd
from scipy.stats import pearsonr, t
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf

from pyperfanalytics.drawdowns import (
    drawdowns,
    find_drawdowns,
    max_drawdown,
    sort_drawdowns,
)
from pyperfanalytics.returns import (
    _get_scale,
    active_premium,
    burke_ratio,
    calmar_ratio,
    capm_alpha,
    down_capture,
    downside_deviation,
    downside_potential,
    gain_deviation,
    information_ratio,
    loss_deviation,
    martin_ratio,
    mean_absolute_deviation,
    omega_ratio,
    omega_sharpe_ratio,
    pain_ratio,
    prob_sharpe_ratio,
    return_annualized,
    return_excess,
    semi_deviation,
    sharpe_ratio,
    sortino_ratio,
    std_dev_annualized,
    sterling_ratio,
    treynor_ratio,
    up_capture,
    up_down_ratios,
    upside_potential,
    upside_potential_ratio,
)
from pyperfanalytics.risk import (
    capm_beta,
    es_historical,
    es_modified,
    pain_index,
    specific_risk,
    systematic_risk,
    total_risk,
    tracking_error,
    ulcer_index,
    var_historical,
    var_modified,
)
from pyperfanalytics.utils import _get_scale as _get_utils_scale
from pyperfanalytics.utils import (
    beta_co_kurtosis,
    beta_co_skewness,
    beta_co_variance,
    co_kurtosis,
    co_skewness,
    kurtosis,
    skewness,
)


def table_annualized_returns(
    R: pd.Series | pd.DataFrame,
    scale: int | None = None,
    Rf: float | pd.Series | pd.DataFrame = 0.0,
    geometric: bool = True,
    digits: int = 6,
) -> pd.DataFrame:
    r"""
    Annualized Returns Summary: Statistics and Stylized Facts

    Creates a table of Annualized Return, Annualized Standard Deviation,
    and Annualized Sharpe Ratio.

    Formulas:

    - Annualized Return: :math:`(1 + R_{cum})^{scale / n} - 1`
    - Annualized Std Dev: :math:`\sigma \cdot \sqrt{scale}`
    - Annualized Sharpe: :math:`\frac{R_{ann} - R_{f, ann}}{\sigma_{ann}}`

    Parameters
    ----------
    R : pd.Series or pd.DataFrame
        Asset returns.
    scale : int, optional
        Number of periods in a year.
    Rf : float, pd.Series, or pd.DataFrame, optional
        Risk-free rate. Default is 0.0.
    geometric : bool, optional
        Generate geometric (True) or arithmetic (False) annualized returns. Default is True.
    digits : int, optional
        Number of decimal places to round to.

    Returns
    -------
    pd.DataFrame
        A table containing the summary statistics.
    r"""
    if scale is None:
        scale = _get_scale(R)

    # Calculate components
    ann_ret = return_annualized(R, scale=scale, geometric=geometric)
    ann_std = std_dev_annualized(R, scale=scale)

    # Match R's label rounding: round(mean(Rf)*scale, 4)*100
    rf_mean_ann = Rf.mean() * scale if isinstance(Rf, (pd.Series, pd.DataFrame)) else Rf * scale
    # R uses base::round(mean(Rf)*scale, 4) * 100
    rf_label_val = float(np.round(rf_mean_ann, 4) * 100)
    # Format as string, stripping trailing zeros if they are after decimal
    rf_label_str = f"{rf_label_val:g}"

    sharpe_label = f"Annualized Sharpe (Rf={rf_label_str}%)"

    ann_sharpe = sharpe_ratio(R, Rf=Rf, annualize=True, scale=scale)

    # Combine into a table
    if isinstance(R, pd.Series):
        res = pd.DataFrame({R.name or "Value": [ann_ret, ann_std, ann_sharpe]})
    else:
        res = pd.DataFrame([ann_ret, ann_std, ann_sharpe])

    res.index = ["Annualized Return", "Annualized Std Dev", sharpe_label]

    return res.round(digits)


def table_capm(
    Ra: pd.Series | pd.DataFrame,
    Rb: pd.Series | pd.DataFrame,
    scale: int | None = None,
    Rf: float | pd.Series | pd.DataFrame = 0.0,
    digits: int = 4,
) -> pd.DataFrame:
    r"""
    Single Factor Asset-Pricing Model (CAPM) Summary Table.

    Creates a summary table containing multiple CAPM and relative risk/return metrics:
    Alpha, Beta, Beta+ (Bull Beta), Beta- (Bear Beta), R-squared, Annualized Alpha,
    Correlation, Correlation p-value, Tracking Error, Active Premium, Information Ratio,
    and Treynor Ratio.

    Parameters
    ----------
    Ra : pd.Series or pd.DataFrame
        Asset returns.
    Rb : pd.Series or pd.DataFrame
        Benchmark returns.
    scale : int, optional
        Number of periods in a year.
    Rf : float, pd.Series, or pd.DataFrame, optional
        Risk-free rate. Default is 0.0.
    digits : int, optional
        Number of decimal places.

    Returns
    -------
    pd.DataFrame
        CAPM summary table comparing each asset in Ra to each benchmark in Rb.
    r"""
    if scale is None:
        scale = _get_scale(Ra)

    # Standardize inputs
    if isinstance(Ra, pd.Series):
        ra_df = Ra.to_frame()
    else:
        ra_df = Ra

    if isinstance(Rb, pd.Series):
        rb_df = Rb.to_frame()
    else:
        rb_df = Rb

    ra_cols = ra_df.columns
    rb_cols = rb_df.columns

    results = []
    for rb_col in rb_cols:
        for ra_col in ra_cols:
            # Align
            merged = pd.concat([ra_df[ra_col], rb_df[rb_col]], axis=1).dropna()
            if merged.empty:
                results.append([np.nan] * 13)
                continue

            xRa = return_excess(merged.iloc[:, 0], Rf)
            xRb = return_excess(merged.iloc[:, 1], Rf)

            # Regression components
            alpha = capm_alpha(merged.iloc[:, 0], merged.iloc[:, 1], Rf)
            beta = capm_beta(merged.iloc[:, 0], merged.iloc[:, 1], Rf)

            # Use excess returns for correlation and R-squared
            corr, p_val = pearsonr(xRa, xRb)
            r2 = corr**2

            # Ann Alpha
            ann_alpha = (1 + alpha) ** scale - 1

            # Beta+ / Beta- (Bull/Bear Beta)
            # R's CAPM.beta.bull: returns from Ra where excess benchmark return > 0
            # Reference: CAPM.beta.R calls getResults with xRa, xRb and subsets based on xRb
            beta_plus = capm_beta(merged.iloc[:, 0][xRb > 0], merged.iloc[:, 1][xRb > 0], Rf)
            beta_minus = capm_beta(merged.iloc[:, 0][xRb <= 0], merged.iloc[:, 1][xRb <= 0], Rf)

            te = tracking_error(merged.iloc[:, 0], merged.iloc[:, 1], scale=scale)
            ap = active_premium(merged.iloc[:, 0], merged.iloc[:, 1], scale=scale)
            ir = information_ratio(merged.iloc[:, 0], merged.iloc[:, 1], scale=scale)
            tr = treynor_ratio(merged.iloc[:, 0], merged.iloc[:, 1], Rf=Rf, scale=scale)

            row = [alpha, beta, beta_plus, beta_minus, r2, ann_alpha, corr, p_val, te, ap, ir, tr]
            results.append(row)

    znames = [
        "Alpha",
        "Beta",
        "Beta+",
        "Beta-",
        "R-squared",
        "Annualized Alpha",
        "Correlation",
        "Correlation p-value",
        "Tracking Error",
        "Active Premium",
        "Information Ratio",
        "Treynor Ratio",
    ]

    res_df = pd.DataFrame(results).T
    res_df.index = znames

    # Column names: asset to benchmark
    col_names = []
    for rb_col in rb_cols:
        for ra_col in ra_cols:
            col_names.append(f"{ra_col} to {rb_col}")
    res_df.columns = col_names

    return res_df.round(digits)


def table_downside_risk(
    R: pd.Series | pd.DataFrame,
    scale: int | None = None,
    Rf: float | pd.Series | pd.DataFrame = 0.0,
    MAR: float = 0.1 / 12.0,
    p: float = 0.95,
    digits: int = 4,
) -> pd.DataFrame:
    r"""
    Downside Risk Summary: Statistics and Stylized Facts.

    Creates a table containing multiple downside-focused risk metrics:
    Semi Deviation, Gain/Loss Deviation, Downside Deviation (with different MARs),
    Maximum Drawdown, Historical VaR/ES, and Modified VaR/ES.

    Parameters
    ----------
    R : pd.Series or pd.DataFrame
        Asset returns.
    scale : int, optional
        Number of periods in a year.
    Rf : float, pd.Series, or pd.DataFrame, optional
        Risk-free rate. Default is 0.0.
    MAR : float, optional
        Minimum Acceptable Return. Default is 0.1/12.0.
    p : float, optional
        Confidence level for VaR/ES, default is 0.95.
    digits : int, optional
        Number of decimal places to round to.

    Returns
    -------
    pd.DataFrame
        Downside risk summary table comparing all columns in R.
    r"""
    if scale is None:
        scale = _get_scale(R)

    if isinstance(R, pd.Series):
        r_df = R.to_frame()
    else:
        r_df = R

    columns = r_df.columns

    # Fetch Rf mean for label
    if isinstance(Rf, (pd.Series, pd.DataFrame)):
        rf_val = Rf.mean()
    else:
        rf_val = Rf

    results = []
    for col in columns:
        x = r_df[col].dropna()
        if x.empty:
            results.append([np.nan] * 11)
            continue

        z = [
            semi_deviation(x),
            gain_deviation(x),
            loss_deviation(x),
            downside_deviation(x, MAR=MAR),
            downside_deviation(x, MAR=float(rf_val)), # type: ignore
            downside_deviation(x, MAR=0),
            max_drawdown(x),
            -var_historical(x, p=p),
            -es_historical(x, p=p),
            -var_modified(x, p=p),
            -es_modified(x, p=p),
        ]
        results.append(z)

    znames = [
        "Semi Deviation",
        "Gain Deviation",
        "Loss Deviation",
        f"Downside Deviation (MAR={MAR * scale * 100:.1f}%)",
        f"Downside Deviation (Rf={rf_val * scale * 100:.1f}%)",
        "Downside Deviation (0%)",
        "Maximum Drawdown",
        f"Historical VaR ({p * 100:.0f}%)",
        f"Historical ES ({p * 100:.0f}%)",
        f"Modified VaR ({p * 100:.0f}%)",
        f"Modified ES ({p * 100:.0f}%)",
    ]

    res_df = pd.DataFrame(results).T
    res_df.index = znames
    res_df.columns = columns

    return res_df.round(digits)


def table_capture_ratios(Ra: pd.Series | pd.DataFrame, Rb: pd.Series | pd.DataFrame, digits: int = 4) -> pd.DataFrame:
    r"""
    Up and Down Market Capture Ratio Table.

    Calculate and display a table of capture ratio and related statistics.
    Up Capture indicates how much of the benchmark's positive returns the asset captured,
    while Down Capture measures the same for negative returns.

    Formula:

    .. math::

        Capture = \frac{R_{a, ann}}{R_{b, ann}} subsetted by positive/negative benchmark returns.

    Parameters
    ----------
    Ra : pd.Series or pd.DataFrame
        Asset returns.
    Rb : pd.Series or pd.DataFrame
        Benchmark returns.
    digits : int, optional
        Number of decimal precision digits.

    Returns
    -------
    pd.DataFrame
        Table of Up and Down Capture Ratios.
    r"""
    # Standardize inputs
    if isinstance(Ra, pd.Series):
        ra_df = Ra.to_frame()
    else:
        ra_df = Ra

    if isinstance(Rb, pd.Series):
        rb_df = Rb.to_frame()
    else:
        rb_df = Rb

    ra_cols = ra_df.columns
    rb_cols = rb_df.columns

    # R implementation of table.CaptureRatios seems to only use the FIRST Rb column
    # if Rb is a matrix. "merged.assets = merge(Ra[,column.a,drop=FALSE], Rb[,1,drop=FALSE])"
    # However, we can make it more flexible and supporting multiple Rb if needed.
    # For now, let's stick to R's behavior (first Rb if many).
    benchmark_col = rb_cols[0]

    results = []
    for ra_col in ra_cols:
        uc = up_capture(ra_df[ra_col], rb_df[benchmark_col])
        dc = down_capture(ra_df[ra_col], rb_df[benchmark_col])
        results.append([uc, dc])

    res_df = pd.DataFrame(results, index=ra_cols, columns=["Up Capture", "Down Capture"])
    return res_df.round(digits)


def table_up_down_ratios(Ra: pd.Series | pd.DataFrame, Rb: pd.Series | pd.DataFrame, digits: int = 4) -> pd.DataFrame:
    r"""
    Up and Down Market Ratios Table.

    Calculate and display a table of up and down market statistics including
    capture ratios, number of up/down periods, and percent of up/down periods.

    Parameters
    ----------
    Ra : pd.Series or pd.DataFrame
        Asset returns.
    Rb : pd.Series or pd.DataFrame
        Benchmark returns.
    digits : int, optional
        Number of decimal precision digits.

    Returns
    -------
    pd.DataFrame
        Table of Up/Down Capture, Number, and Percent.
    r"""
    # Standardize inputs
    if isinstance(Ra, pd.Series):
        ra_df = Ra.to_frame()
    else:
        ra_df = Ra

    if isinstance(Rb, pd.Series):
        rb_df = Rb.to_frame()
    else:
        rb_df = Rb

    ra_cols = ra_df.columns
    rb_cols = rb_df.columns

    # Again, R implementation uses first Rb
    benchmark_col = rb_cols[0]

    results = []
    for ra_col in ra_cols:
        metrics = []
        for method in ["Capture", "Number", "Percent"]:
            for side in ["Up", "Down"]:
                val = up_down_ratios(ra_df[ra_col], rb_df[benchmark_col], method=method, side=side)
                metrics.append(val)
        results.append(metrics)

    znames = ["Up Capture", "Down Capture", "Up Number", "Down Number", "Up Percent", "Down Percent"]

    res_df = pd.DataFrame(results, index=ra_cols, columns=znames)

    # R implementation of table.UpDownRatios appends " to [Bench]" to rownames
    bench_name = rb_df.columns[0]
    res_df.index = [f"{asset} to {bench_name}" for asset in res_df.index]

    return res_df.round(digits)


def table_calendar_returns(
    R: pd.Series | pd.DataFrame, digits: int = 1, as_perc: bool = True, geometric: bool = True
) -> pd.DataFrame:
    r"""
    Monthly and Calendar Year Return Table.

    Transforms a time series of returns into a calendar-formatted table where rows
    are years, columns are months, and an additional column holds the YTD/annual return.

    Parameters
    ----------
    R : pd.Series or pd.DataFrame
        Asset returns. If a DataFrame with multiple columns is provided, only the first is used.
    digits : int, optional
        Number of decimal precision digits. Default is 1.
    as_perc : bool, optional
        If True, multiplies returns by 100 for display.
    geometric : bool, optional
        If True, compound returns geometrically for the annual total. Default is True.

    Returns
    -------
    pd.DataFrame
        Calendar returns matrix.
    r"""
    if isinstance(R, pd.DataFrame):
        # PerformanceAnalytics defaults to first column if many
        s = R.iloc[:, 0]
    else:
        s = R

    s = s.dropna()
    if s.empty:
        return pd.DataFrame()

    # Create Year x Month pivot
    df = s.to_frame()
    if not isinstance(df.index, pd.DatetimeIndex):
        dt_index = pd.to_datetime(df.index)
    else:
        dt_index = df.index
        
    df["Year"] = dt_index.year # type: ignore
    df["Month"] = dt_index.month # type: ignore

    # Map month numbers to short names
    month_map = {
        1: "Jan",
        2: "Feb",
        3: "Mar",
        4: "Apr",
        5: "May",
        6: "Jun",
        7: "Jul",
        8: "Aug",
        9: "Sep",
        10: "Oct",
        11: "Nov",
        12: "Dec",
    }
    df["MonthName"] = df["Month"].map(month_map)

    # Pivot
    pivot = df.pivot(index="Year", columns="MonthName", values=s.name or "Returns")

    # Reorder columns
    ordered_months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    # Ensure all months exist in columns for reindexing
    pivot = pivot.reindex(columns=ordered_months)

    # Calculate Year totals
    def _calc_year(row: pd.Series, geom: bool) -> float:
        vals = row.dropna()
        if vals.empty:
            return np.nan
        if geom:
            return (1 + vals).prod() - 1
        else:
            return vals.sum()

    year_col = pivot.apply(_calc_year, axis=1, geom=geometric)
    pivot[s.name or "Returns"] = year_col

    multiplier = 100.0 if as_perc else 1.0
    return (pivot * multiplier).round(digits)


def table_higher_moments(Ra: pd.Series | pd.DataFrame, Rb: pd.Series | pd.DataFrame, digits: int = 4) -> pd.DataFrame:
    r"""
    Higher Moments Summary: Statistics and Stylized Facts (Co-Moments).

    Creates a table of CoSkewness, CoKurtosis, Beta CoVariance, Beta CoSkewness,
    and Beta CoKurtosis of asset returns against benchmark returns.

    Parameters
    ----------
    Ra : pd.Series or pd.DataFrame
        Asset returns.
    Rb : pd.Series or pd.DataFrame
        Benchmark returns.
    digits : int, optional
        Number of decimal precision digits. Default is 4.

    Returns
    -------
    pd.DataFrame
        Higher moments summary table.
    r"""
    if isinstance(Ra, pd.Series):
        ra_df = Ra.to_frame()
    else:
        ra_df = Ra

    if isinstance(Rb, pd.Series):
        rb_df = Rb.to_frame()
    else:
        rb_df = Rb

    ra_cols = ra_df.columns
    rb_cols = rb_df.columns

    results = []
    col_names = []

    for rb_col in rb_cols:
        for ra_col in ra_cols:
            a = ra_df[ra_col]
            b = rb_df[rb_col]

            row = [
                co_skewness(a, b),
                co_kurtosis(a, b),
                beta_co_variance(a, b),
                beta_co_skewness(a, b),
                beta_co_kurtosis(a, b),
            ]
            results.append(row)
            col_names.append(f"{ra_col} to {rb_col}")

    znames = ["CoSkewness", "CoKurtosis", "Beta CoVariance", "Beta CoSkewness", "Beta CoKurtosis"]

    res_df = pd.DataFrame(results).T
    res_df.index = znames
    res_df.columns = col_names

    return res_df.round(digits)


def table_prob_skewness_kurtosis(R: pd.Series | pd.DataFrame, digits: int = 4) -> pd.DataFrame:
    r"""
    Summary table for univariate skewness and kurtosis methods.

    Generates a table showing different estimates of skewness and kurtosis
    (moment, fisher, sample, excess).

    Parameters
    ----------
    R : pd.Series or pd.DataFrame
        Asset returns.
    digits : int, optional
        Number of decimal precision digits. Default is 4.

    Returns
    -------
    pd.DataFrame
        Skewness and kurtosis metrics table.
    r"""
    if isinstance(R, pd.Series):
        r_df = R.to_frame()
    else:
        r_df = R

    columns = r_df.columns
    results = []

    for col in columns:
        x = r_df[col].dropna()
        if x.empty:
            results.append([np.nan] * 8)
            continue

        row = [
            skewness(x, method="moment"),
            skewness(x, method="fisher"),
            skewness(x, method="sample"),
            kurtosis(x, method="moment"),
            kurtosis(x, method="fisher"),
            kurtosis(x, method="sample"),
            kurtosis(x, method="excess"),
            kurtosis(x, method="sample_excess"),
        ]
        results.append(row)

    znames = [
        "Skewness (moment)",
        "Skewness (fisher)",
        "Skewness (sample)",
        "Kurtosis (moment)",
        "Kurtosis (fisher)",
        "Kurtosis (sample)",
        "Kurtosis (excess)",
        "Kurtosis (sample_excess)",
    ]

    res_df = pd.DataFrame(results).T
    res_df.index = znames
    res_df.columns = columns

    return res_df.round(digits)


def table_variability(
    R: pd.Series | pd.DataFrame, scale: int | None = None, geometric: bool = True, digits: int = 4
) -> pd.DataFrame:
    r"""
    Variability Summary: Statistics and Stylized Facts.

    Creates a table of Mean Absolute Deviation, Period Standard Deviation,
    and Annualized Standard Deviation.

    Parameters
    ----------
    R : pd.Series or pd.DataFrame
        Asset returns.
    scale : int, optional
        Number of periods in a year.
    geometric : bool, optional
        Geometric aggregation flag. Default is True.
    digits : int, optional
        Number of decimal precision digits. Default is 4.

    Returns
    -------
    pd.DataFrame
        Variability metrics table.
    r"""
    if scale is None:
        scale = _get_scale(R)

    if isinstance(R, pd.Series):
        r_df = R.to_frame()
    else:
        r_df = R

    columns = r_df.columns
    results = []

    for col in columns:
        x = r_df[col].dropna()
        if x.empty:
            results.append([np.nan] * 3)
            continue

        mad = mean_absolute_deviation(x)

        ann_std = std_dev_annualized(x, scale=scale)
        period_std = ann_std / np.sqrt(scale)

        results.append([mad, period_std, ann_std])

    znames = ["Mean Absolute deviation", "Period Std Dev", "Annualized Std Dev"]

    res_df = pd.DataFrame(results).T
    res_df.index = znames
    res_df.columns = columns

    return res_df.round(digits)


def table_drawdowns(R: pd.Series, top: int = 5, digits: int = 4, geometric: bool = True) -> pd.DataFrame:
    r"""
    Worst Drawdowns Summary: Statistics and Stylized Facts.

    Finds the largest drawdowns in the return series and formats them into a table
    containing their Start Date (From), Trough Date, End Date (To), Depth, Length,
    Time to Trough, and Recovery Time.

    Parameters
    ----------
    R : pd.Series
        Asset returns (only single series supported).
    top : int, optional
        Number of worst drawdowns to display. Default is 5.
    digits : int, optional
        Rounding digits for depth. Default is 4.
    geometric : bool, optional
        Use geometric compounding. Default is True.

    Returns
    -------
    pd.DataFrame
        Worst drawdowns table.
    r"""
    # PerformanceAnalytics' table.Drawdowns only works for single column
    if isinstance(R, pd.DataFrame):
        if R.shape[1] > 1:
            R = R.iloc[:, 0]
        else:
            R = R.iloc[:, 0]

    R_clean = R.dropna()
    runs = find_drawdowns(R_clean, geometric=geometric)
    sorted_runs = sort_drawdowns(runs)

    n_drawdowns = sum(sorted_runs["return"] < 0)
    if n_drawdowns < top:
        top = n_drawdowns

    if top == 0:
        return pd.DataFrame(columns=["From", "Trough", "To", "Depth", "Length", "To Trough", "Recovery"])

    # Extract dates
    idx = R_clean.index

    # from, trough, to are 1-based indices from find_drawdowns
    from_dates = idx[sorted_runs["from"][:top] - 1]
    trough_dates = idx[sorted_runs["trough"][:top] - 1]

    # Recovery check: in PA, if the last drawdown is not recovered, 'To' is NA
    to_indices = sorted_runs["to"][:top]
    to_dates = []
    recovery_periods = []

    drawdowns(R_clean, geometric=geometric)

    for i in range(top):
        end_idx_1 = to_indices[i]
        end_idx_0 = end_idx_1 - 1

        # In findDrawdowns.R, to_idx = i_1 + 1 happens after val is processed.
        # If the series ends at n, the last sequence has to_idx = n + 1.
        if end_idx_1 > len(R_clean):
            to_dates.append(np.nan)
            recovery_periods.append(np.nan)
        else:
            to_dates.append(idx[end_idx_0])
            recovery_periods.append(sorted_runs["recovery"][i])

    result = pd.DataFrame(
        {
            "From": from_dates,
            "Trough": trough_dates,
            "To": to_dates,
            "Depth": np.round(sorted_runs["return"][:top], digits),
            "Length": sorted_runs["length"][:top],
            "To Trough": sorted_runs["peaktotrough"][:top],
            "Recovery": recovery_periods,
        }
    )

    return result


def table_information_ratio(
    Ra: pd.Series | pd.DataFrame, Rb: pd.Series | pd.DataFrame, scale: int | None = None, digits: int = 4
) -> pd.DataFrame:
    r"""
    Information Ratio Summary: Statistics and Stylized Facts.

    Creates a table containing calculated metrics relating to the tracking error and
    information ratio of returns against a benchmark.

    Parameters
    ----------
    Ra : pd.Series or pd.DataFrame
        Asset returns.
    Rb : pd.Series or pd.DataFrame
        Benchmark returns.
    scale : int, optional
        Number of periods in a year.
    digits : int, optional
        Number of decimal precision digits. Default is 4.

    Returns
    -------
    pd.DataFrame
        A table showing Periodic Tracking Error, Annualized Tracking Error, and Information Ratio.
    r"""
    if scale is None:
        scale = _get_utils_scale(Ra)

    # Standardize inputs
    if isinstance(Ra, pd.Series):
        ra_df = Ra.to_frame()
    else:
        ra_df = Ra

    ra_cols = ra_df.columns

    results = []
    for col in ra_cols:
        # periodic tracking error
        # R: TrackingError(y[,column,drop=FALSE])/sqrt(scale)
        te_periodic = tracking_error(ra_df[col], Rb, scale=1)  # scale=1 for periodic
        te_annual = tracking_error(ra_df[col], Rb, scale=scale)
        ir = information_ratio(ra_df[col], Rb, scale=scale)

        results.append([te_periodic, te_annual, ir])

    znames = ["Tracking Error", "Annualised Tracking Error", "Information Ratio"]
    res_df = pd.DataFrame(results, index=ra_cols, columns=znames).T
    return res_df.round(digits)


def table_specific_risk(
    Ra: pd.Series | pd.DataFrame,
    Rb: pd.Series | pd.DataFrame,
    Rf: float | pd.Series | pd.DataFrame = 0,
    digits: int = 4,
) -> pd.DataFrame:
    r"""
    Specific risk Summary: Statistics and Stylized Facts
    Table of specific risk, systematic risk and total risk.
    r"""
    # Standardize inputs
    if isinstance(Ra, pd.Series):
        ra_df = Ra.to_frame()
    else:
        ra_df = Ra

    ra_cols = ra_df.columns

    results = []
    for col in ra_cols:
        spec = specific_risk(ra_df[col], Rb, Rf=Rf)
        sys = systematic_risk(ra_df[col], Rb, Rf=Rf)
        tot = total_risk(ra_df[col], Rb, Rf=Rf)
        results.append([spec, sys, tot])

    znames = ["Specific Risk", "Systematic Risk", "Total Risk"]
    res_df = pd.DataFrame(results, index=ra_cols, columns=znames).T
    return res_df.round(digits)


def table_prob_sharpe_ratio(
    R: pd.Series | pd.DataFrame, refSR: float | list | np.ndarray = 0.0, Rf: float = 0.0, digits: int = 4
) -> pd.DataFrame:
    r"""
    Summary table for Probabilistic Sharpe Ratio across different thresholds.

    Calculates the statistical significance of the Sharpe Ratio against a given set
    of reference Sharpe Ratios, yielding the Probability Sharpe Ratio (PSR),
    significance flag, etc.

    Parameters
    ----------
    R : pd.Series or pd.DataFrame
        Asset returns.
    refSR : float, list, or np.ndarray, optional
        Reference Sharpe Ratio(s). Default is 0.0.
    Rf : float, optional
        Risk-free rate. Default is 0.0.
    digits : int, optional
        Number of decimal precision digits. Default is 4.

    Returns
    -------
    pd.DataFrame
        A table of Probabilistic Sharpe Ratio values for each threshold.
    r"""
    if isinstance(refSR, (float, int, np.float64, np.int64)):
        refSR = [refSR]

    if isinstance(R, pd.Series):
        ra_df = R.to_frame()
    else:
        ra_df = R

    final_results = []
    if isinstance(refSR, (list, tuple, np.ndarray, pd.Series)):
        refSR_list = list(refSR)
    else:
        refSR_list = [refSR]
        
    for rsr in refSR_list:
        row = prob_sharpe_ratio(ra_df, refSR=float(rsr), Rf=float(Rf))
        final_results.append(row)

    res_df = pd.DataFrame(final_results, index=[f"PSR (refSR={round(float(r), 4)})" for r in refSR_list])
    return res_df.round(digits)


def table_autocorrelation(R: pd.Series | pd.DataFrame, digits: int = 4, max_lag: int = 6) -> pd.DataFrame:
    r"""
    Table for calculating the first six (default) autocorrelation coefficients and significance.
    Produces data table of autocorrelation coefficients rho and corresponding Q(max_lag)-statistic.
    r"""
    if isinstance(R, pd.Series):
        r_df = R.to_frame()
    else:
        r_df = R

    columns = r_df.columns
    results = []

    for col in columns:
        x = r_df[col].dropna()
        if x.empty:
            results.append([np.nan] * (max_lag + 1))
            continue

        # acf returns lags [0, 1, ..., max_lag]. R's table starts from lag 1.
        # statsmodels.acf uses fft by default, PerformanceAnalytics uses acf() which defaults to correlation.
        # acf(y, plot=FALSE, lag.max=max.lag)[[1]] in R gives autocorrelation.
        rho = acf(x, nlags=max_lag, fft=False)[1:]

        # Ljung-Box test
        lb = acorr_ljungbox(x, lags=[max_lag], return_df=True)
        p_val = lb["lb_pvalue"].iloc[0]

        row = list(rho) + [p_val]
        results.append(row)

    znames = [f"rho {i + 1}" for i in range(max_lag)] + [f"Q({max_lag}) p-value"]
    res_df = pd.DataFrame(results, index=columns, columns=znames).T

    return res_df.round(digits)


def table_correlation(
    Ra: pd.Series | pd.DataFrame, Rb: pd.Series | pd.DataFrame, digits: int = 4, conf_level: float = 0.95
) -> pd.DataFrame:
    r"""
    Calculate correlations and significance of multicolumn data.

    Computes the Pearson correlation coefficient between assets and benchmarks,
    along with the p-value and a confidence interval (default 95%) leveraging the Fisher Z-transform.

    Parameters
    ----------
    Ra : pd.Series or pd.DataFrame
        Asset returns.
    Rb : pd.Series or pd.DataFrame
        Benchmark returns.
    digits : int, optional
        Number of decimal precision digits. Default is 4.
    conf_level : float, optional
        Confidence level for the interval (e.g., 0.95 for 95% CI).

    Returns
    -------
    pd.DataFrame
        A table showing Correlation, p-value, Lower CI, and Upper CI.
    r"""
    if isinstance(Ra, pd.Series):
        ra_df = Ra.to_frame()
    else:
        ra_df = Ra

    if isinstance(Rb, pd.Series):
        rb_df = Rb.to_frame()
    else:
        rb_df = Rb

    ra_cols = ra_df.columns
    rb_cols = rb_df.columns

    results = []
    row_names = []

    for ra_col in ra_cols:
        for rb_col in rb_cols:
            merged = pd.concat([ra_df[ra_col], rb_df[rb_col]], axis=1).dropna()
            if merged.empty:
                results.append([np.nan] * 4)
                row_names.append(f"{ra_col} to {rb_col}")
                continue

            x = merged.iloc[:, 0]
            y = merged.iloc[:, 1]
            n = len(x)

            corr, p_val = pearsonr(x, y)

            # Confidence interval for Pearson correlation
            # stderr = 1 / sqrt(n - 3)
            # z = 0.5 * log((1 + r) / (1 - r))
            # ucl = tanh(z + qnorm(1 - alpha/2) * stderr)
            if n > 3:
                z = 0.5 * np.log((1 + corr) / (1 - corr))
                t.ppf((1 + conf_level) / 2, n - 2)  # R uses normal distribution for large n, but t is safer
                # Actually R's cor.test uses t for Pearson
                # R: r * sqrt(df / (1 - r^2)) ~ t(df)
                # For CI: Fisher's Z-transform
                stderr = 1.0 / np.sqrt(n - 3)
                z_crit_norm = -t.ppf((1 - conf_level) / 2, float("inf"))  # Using normal for Z-transform
                lower_z = z - z_crit_norm * stderr
                upper_z = z + z_crit_norm * stderr
                lower_r = np.tanh(lower_z)
                upper_r = np.tanh(upper_z)
            else:
                lower_r = upper_r = np.nan

            results.append([corr, p_val, lower_r, upper_r])
            row_names.append(f"{ra_col} to {rb_col}")

    res_df = pd.DataFrame(results, index=row_names, columns=["Correlation", "p-value", "Lower CI", "Upper CI"])
    return res_df.round(digits)


def table_distributions(R: pd.Series | pd.DataFrame, scale: int | None = None, digits: int = 4) -> pd.DataFrame:
    r"""
    Distributions Summary: Statistics and Stylized Facts.
    Table of standard deviation, Skewness, Kurtosis, etc.
    r"""
    if scale is None:
        scale = _get_utils_scale(R)

    if isinstance(R, pd.Series):
        r_df = R.to_frame()
    else:
        r_df = R

    columns = r_df.columns
    # Get scale string (e.g., 'monthly')
    freq_map = {252: "Daily", 52: "Weekly", 12: "Monthly", 4: "Quarterly", 1: "Yearly"}
    freq_str = freq_map.get(scale, "Period")

    results = []
    for col in columns:
        x = r_df[col].dropna()
        if x.empty:
            results.append([np.nan] * 6)
            continue

        # R logic: StdDev.annualized(y)/sqrt(scale)
        period_std = std_dev_annualized(x, scale=scale) / np.sqrt(scale)

        row = [
            period_std,
            skewness(x, method="moment"),
            kurtosis(x, method="moment"),
            kurtosis(x, method="excess"),
            skewness(x, method="sample"),
            kurtosis(x, method="sample_excess"),
        ]
        results.append(row)

    znames = [
        f"{freq_str} Std Dev",
        "Skewness",
        "Kurtosis",
        "Excess kurtosis",
        "Sample skewness",
        "Sample excess kurtosis",
    ]
    res_df = pd.DataFrame(results, index=columns, columns=znames).T

    return res_df.round(digits)


def table_downside_risk_ratio(
    R: pd.Series | pd.DataFrame, MAR: float = 0, scale: int | None = None, digits: int = 4
) -> pd.DataFrame:
    r"""
    Downside Risk Summary: Ratios and Metrics.

    Creates a comprehensive table of downside-related metrics, contrasting
    variations of downside deviation against downside potential and upside potential.

    Incorporates: Downside Deviation, Downside Potential, Omega Ratio, Sortino Ratio,
    Upside Potential, Upside Potential Ratio, and Omega-Sharpe Ratio.

    Parameters
    ----------
    R : pd.Series or pd.DataFrame
        Asset returns.
    MAR : float, optional
        Minimum Acceptable Return. Default is 0.
    scale : int, optional
        Number of periods in a year.
    digits : int, optional
        Number of decimal precision digits. Default is 4.

    Returns
    -------
    pd.DataFrame
        Downside ratios and statistics matrix.
    r"""
    if scale is None:
        scale = _get_utils_scale(R)

    if isinstance(R, pd.Series):
        r_df = R.to_frame()
    else:
        r_df = R

    columns = r_df.columns
    # Get scale string (e.g., 'monthly')
    freq_map = {252: "Daily", 52: "Weekly", 12: "Monthly", 4: "Quarterly", 1: "Yearly"}
    freq_str = freq_map.get(scale, "Period")

    results = []
    for col in columns:
        x = r_df[col].dropna()
        if x.empty:
            results.append([np.nan] * 8)
            continue

        dd = downside_deviation(x, MAR=MAR)
        dp = downside_potential(x, MAR=MAR)
        # Omega = UpsidePotential / DownsidePotential
        om = omega_ratio(x, L=MAR)
        sr = sortino_ratio(x, MAR=MAR)
        up = upside_potential(x, MAR=MAR)
        upr = upside_potential_ratio(x, MAR=MAR)
        osr = omega_sharpe_ratio(x, MAR=MAR)

        row = [dd, dd * np.sqrt(scale), dp, om, sr, up, upr, osr]
        results.append(row)

    znames = [
        f"{freq_str} downside risk",
        "Annualised downside risk",
        "Downside potential",
        "Omega",
        "Sortino ratio",
        "Upside potential",
        "Upside potential ratio",
        "Omega-sharpe ratio",
    ]
    res_df = pd.DataFrame(results, index=columns, columns=znames).T

    return res_df.round(digits)


def table_drawdowns_ratio(
    R: pd.Series | pd.DataFrame, Rf: float | pd.Series | pd.DataFrame = 0, scale: int | None = None, digits: int = 4
) -> pd.DataFrame:
    r"""
    Drawdowns Summary: Statistics and ratios.
    Table of Sterling ratio, Calmar ratio, Burke ratio, etc.
    r"""
    if scale is None:
        scale = _get_utils_scale(R)

    if isinstance(R, pd.Series):
        r_df = R.to_frame()
    else:
        r_df = R

    columns = r_df.columns
    results = []

    for col in columns:
        x = r_df[col].dropna()
        if x.empty:
            results.append([np.nan] * 7)
            continue

        ster = sterling_ratio(x, scale=scale)
        calm = calmar_ratio(x, scale=scale)
        burk = burke_ratio(x, Rf=Rf)
        pain_idx = pain_index(x)
        ulcer = ulcer_index(x)
        pr = pain_ratio(x, Rf=Rf)
        mr = martin_ratio(x, Rf=Rf)

        row = [ster, calm, burk, pain_idx, ulcer, pr, mr]
        results.append(row)

    znames = [
        "Sterling ratio",
        "Calmar ratio",
        "Burke ratio",
        "Pain index",
        "Ulcer index",
        "Pain ratio",
        "Martin ratio",
    ]
    res_df = pd.DataFrame(results, index=columns, columns=znames).T

    return res_df.round(digits)


def table_stats(R: pd.Series | pd.DataFrame, ci: float = 0.95, digits: int = 4) -> pd.DataFrame:
    r"""
    Returns Summary: Statistics and Stylized Facts.
    R equivalent: table.Stats or table.MonthlyReturns.
    r"""
    if isinstance(R, pd.Series):
        r_df = R.to_frame()
    else:
        r_df = R

    columns = r_df.columns
    results = []

    for col in columns:
        x_orig = r_df[col]
        x = x_orig.dropna()
        n = len(x)
        n_na = len(x_orig) - n

        if n == 0:
            results.append([0, n_na] + [np.nan] * 14)
            continue

        mn = x.mean()
        sd = x.std(ddof=1)
        var = x.var(ddof=1)

        # Confidence Level
        se_mean = sd / np.sqrt(n)
        if n > 1:
            # -t.ppf is because we want the positive value of t
            t_crit = -t.ppf((1 - ci) / 2, n - 1)
            lcl = mn - se_mean * t_crit
            ucl = mn + se_mean * t_crit
        else:
            lcl = ucl = np.nan

        # Geometric Mean
        if (x <= -1).any():
            geom_mean = np.nan
        else:
            geom_mean = np.exp(np.log(1 + x).mean()) - 1

        row = [
            n,
            n_na,
            x.min(),
            x.quantile(0.25),
            x.median(),
            mn,
            geom_mean,
            x.quantile(0.75),
            x.max(),
            se_mean,
            lcl,
            ucl,
            var,
            sd,
            skewness(x),
            kurtosis(x),
        ]
        results.append(row)

    znames = [
        "Observations",
        "NAs",
        "Minimum",
        "Quartile 1",
        "Median",
        "Arithmetic Mean",
        "Geometric Mean",
        "Quartile 3",
        "Maximum",
        "SE Mean",
        f"LCL Mean ({ci})",
        f"UCL Mean ({ci})",
        "Variance",
        "Stdev",
        "Skewness",
        "Kurtosis",
    ]

    res_df = pd.DataFrame(results, index=columns, columns=znames).T
    return res_df.round(digits)


def table_prob_outperformance(
    R: pd.Series | pd.DataFrame,
    Rb: pd.Series | pd.DataFrame,
    period_lengths: list[int] | None = None,
) -> pd.DataFrame:
    r"""
    Outperformance Report of Asset vs Benchmark.
    Returns a table that contains the counts and probabilities
    of outperformance relative to benchmark for the various period_lengths.
    r"""
    if period_lengths is None:
        period_lengths = [1, 3, 6, 9, 12, 18, 36]
    if isinstance(R, pd.Series):
        r_df = R.to_frame()
    else:
        r_df = R

    if isinstance(Rb, pd.Series):
        rb_df = Rb.to_frame()
    else:
        rb_df = Rb

    ra_col = r_df.columns[0]
    rb_col = rb_df.columns[0]

    merged = pd.concat([r_df[ra_col], rb_df[rb_col]], axis=1).dropna()
    a = merged.iloc[:, 0]
    b = merged.iloc[:, 1]

    results = []
    for p_len in period_lengths:
        # Trailing cumulative returns
        a_cum = (1 + a).rolling(p_len).apply(np.prod) - 1
        b_cum = (1 + b).rolling(p_len).apply(np.prod) - 1

        diff = a_cum - b_cum
        out = (diff > 0).sum()
        under = (diff < 0).sum()
        total = out + under

        prob_out = float(out) / total if total > 0 else np.nan
        prob_under = float(under) / total if total > 0 else np.nan

        results.append([p_len, out, under, total, prob_out, prob_under])

    cols = [
        "period_lengths",
        ra_col,
        rb_col,
        "total periods",
        f"prob_{ra_col}_outperformance",
        f"prob_{rb_col}_outperformance",
    ]
    return pd.DataFrame(results, columns=cols)


def table_rolling_periods(
    R: pd.Series | pd.DataFrame,
    periods: list[int] | None = None,
    funcs: list[str] | None = None,
    funcs_names: list[str] | None = None,
    digits: int = 4,
) -> pd.DataFrame:
    r"""
    Rolling Periods Summary: Statistics and Stylized Facts.
    r"""
    if periods is None:
        periods = [12, 36, 60]
    if funcs is None:
        funcs = ["mean", "sd"]
    if funcs_names is None:
        funcs_names = ["Average", "Std Dev"]

    if isinstance(R, pd.Series):
        r_df = R.to_frame()
    else:
        r_df = R

    from pyperfanalytics.returns import (
        return_annualized,
        sharpe_ratio,
        std_dev_annualized,
    )

    # Map function names to actual functions or series methods
    func_map = {
        "mean": lambda s: s.mean(),
        "sd": lambda s: s.std(),
        "return_annualized": lambda s: return_annualized(s),
        "std_dev_annualized": lambda s: std_dev_annualized(s),
        "sharpe_ratio": lambda s: sharpe_ratio(s),
    }

    results = []
    row_names = []

    columns = r_df.columns
    scale = _get_scale(r_df)
    scale_map = {252: "day", 52: "week", 12: "month", 4: "quarter", 1: "year"}
    scale_str = scale_map.get(scale, "period")

    for i, func_str in enumerate(funcs):
        curr_f_name = funcs_names[i] if i < len(funcs_names) else func_str
        f = func_map.get(func_str)

        for p in periods:
            row_results = []
            row_names.append(f"Last {p} {scale_str} {curr_f_name}")
            for col in columns:
                # PerformanceAnalytics: last(column.data, period)
                # In R, last(x, n) gives the last n entries.
                s_full = r_df[col].dropna()
                if len(s_full) < p:
                    row_results.append(np.nan)
                else:
                    s = s_full.tail(p)
                    if f:
                        try:
                            val = f(s)
                            row_results.append(val)
                        except Exception:
                            row_results.append(np.nan)
                    else:
                        try:
                            val = getattr(s, func_str)()
                            row_results.append(val)
                        except Exception:
                            row_results.append(np.nan)
            results.append(row_results)

    res_df = pd.DataFrame(results, index=row_names, columns=columns)
    return res_df.round(digits)


# Alias for compatibility with implementation plan
table_monthly_returns = table_stats
