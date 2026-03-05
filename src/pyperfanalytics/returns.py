from typing import Optional, Union

import numpy as np
import pandas as pd

from pyperfanalytics.utils import _get_scale


def return_calculate(
    prices: Union[pd.Series, pd.DataFrame],
    method: str = "discrete"
) -> Union[pd.Series, pd.DataFrame]:
    """
    Calculate returns from a price stream.

    Determines the period-over-period returns based on a pricing series, supporting
    both discrete (simple) and continuous (log) methods.

    Formula:
    - discrete: $R_t = \frac{P_t}{P_{t-1}} - 1$
    - continuous: $r_t = \\ln(P_t) - \\ln(P_{t-1})$
    - difference: $D_t = P_t - P_{t-1}$

    Parameters
    ----------
    prices : pd.Series or pd.DataFrame
        Price levels of assets.
    method : str, optional
        Type of return calculation: "discrete" (default), "log", or "diff".

    Returns
    -------
    pd.Series or pd.DataFrame
        Returns series.
    """
    if method in ["discrete", "simple", "arithmetic"]:
        return prices.pct_change()
    elif method in ["log", "compound", "continuous"]:
        return np.log(prices).diff()
    elif method in ["diff", "difference"]:
        return prices.diff()
    else:
        raise ValueError(f"Unknown method: {method}")

def return_excess(
    R: Union[pd.Series, pd.DataFrame],
    Rf: Union[float, pd.Series, pd.DataFrame] = 0.0
) -> Union[pd.Series, pd.DataFrame]:
    """
    Calculate excess returns by subtracting the risk-free rate.
    """
    if isinstance(Rf, (pd.Series, pd.DataFrame)):
        # R implementation uses na.locf on Rf and aligns it to R
        # We merge them to ensure alignment
        combined = pd.concat([R, Rf], axis=1, sort=False)
        # Assuming the last column(s) are Rf
        rf_cols = Rf.columns if isinstance(Rf, pd.DataFrame) else [Rf.name]
        # Forward fill Rf
        combined[rf_cols] = combined[rf_cols].ffill()
        # Subtract Rf from R columns
        # If Rf is a single series, subtract it from all R columns
        if len(rf_cols) == 1:
            res = R.sub(combined[rf_cols[0]], axis=0)
        else:
            # Match columns if possible, or assume 1-to-1 mapping
            res = R.sub(combined[rf_cols].values, axis=0)

        return res.loc[R.index] if isinstance(R, pd.DataFrame) else res[R.index]

    return R - Rf

def return_annualized(
    R: Union[pd.Series, pd.DataFrame],
    scale: Optional[int] = None,
    geometric: bool = True
) -> Union[float, pd.Series]:
    """
    Calculate annualized return.

    Aggregates period returns into an annualized equivalent, assuming continuous
    compounding (geometric) or simple arithmetic averaging.

    Formula (Geometric):
    $$ R_{ann} = \\left[ \\prod_{i=1}^n (1+R_i) \right]^{\frac{scale}{n}} - 1 $$

    Parameters
    ----------
    R : pd.Series or pd.DataFrame
        Asset returns.
    scale : int, optional
        Number of periods in a year.
    geometric : bool, optional
        Whether to compound returns. Default is True.

    Returns
    -------
    float or pd.Series
        Annualized return.
    """
    if scale is None:
        scale = _get_scale(R)

    # R code: n = length(na.omit(R))
    # For DataFrame, we need to handle per-column
    def _calc(s: pd.Series, sc: int, geom: bool) -> float:
        s = s.dropna()
        n = len(s)
        if n == 0: return np.nan
        if geom:
            # Result matches R: (prod(1+R)^(scale/n)) - 1
            return (1 + s).prod() ** (float(sc) / n) - 1
        else:
            return s.mean() * sc

    if isinstance(R, pd.DataFrame):
        return R.apply(_calc, sc=scale, geom=geometric)
    else:
        return _calc(R, scale, geometric)

def std_dev_annualized(R: Union[pd.Series, pd.DataFrame], scale: Optional[int] = None) -> Union[float, pd.Series]:
    """
    Calculate annualized standard deviation.
    """
    if scale is None:
        scale = _get_scale(R)

    # R uses sd(x, na.rm=TRUE) which is ddof=1
    if isinstance(R, pd.DataFrame):
        return R.std(ddof=1) * np.sqrt(scale)
    else:
        return R.dropna().std(ddof=1) * np.sqrt(scale)

def downside_deviation(
    R: Union[pd.Series, pd.DataFrame],
    MAR: float = 0.0,
    method: str = "full",
    potential: bool = False
) -> Union[float, pd.Series]:
    """
    Calculate downside deviation or potential.

    Downside deviation measures the volatility of returns below a minimum acceptable
    return (MAR). Different from standard deviation, it only penalizes losses.

    Formula:
    $$ \\delta_{MAR} = \\sqrt{\frac{1}{n} \\sum_{t=1}^n \\min(R_t - MAR, 0)^2} $$

    Parameters
    ----------
    R : pd.Series or pd.DataFrame
        Asset returns.
    MAR : float, optional
        Minimum acceptable return. Default is 0.0.
    method : str, optional
        "full" (divide by total $n$) or "subset" (divide by number of downside periods).
    potential : bool, optional
        If True, calculates Downside Potential (first lower partial moment).

    Returns
    -------
    float or pd.Series
        Downside deviation.
    """
    def _calc(s: pd.Series, mar: float, meth: str, pot: bool) -> float:
        s = s.dropna()
        # R code: r = subset(R, R < MAR)
        under = s[s < mar]

        if meth == "full":
            n = len(s)
        else:
            n = len(under)

        if n == 0:
            return 0.0

        # R code: (MAR - r)
        diff = mar - under

        if pot:
            return diff.sum() / n
        else:
            return np.sqrt((diff**2).sum() / n)

    if isinstance(R, pd.DataFrame):
        return R.apply(_calc, mar=MAR, meth=method, pot=potential)
    else:
        return _calc(R, MAR, method, potential)

def downside_potential(R: Union[pd.Series, pd.DataFrame], MAR: float = 0.0) -> Union[float, pd.Series]:
    """
    Calculate downside potential.
    """
    return downside_deviation(R, MAR=MAR, method="full", potential=True)

def semi_deviation(R: Union[pd.Series, pd.DataFrame]) -> Union[float, pd.Series]:
    """
    Calculate semi-deviation.

    Semi-deviation is the downside deviation where MAR is the mean return.
    Formula corresponds to the square root of the second lower partial moment.

    Parameters
    ----------
    R : pd.Series or pd.DataFrame
        Asset returns.

    Returns
    -------
    float or pd.Series
        Semi-deviation.
    """
    if isinstance(R, pd.DataFrame):
        return R.apply(lambda x: downside_deviation(x, MAR=x.mean(), method="full"))
    else:
        return downside_deviation(R, MAR=R.mean(), method="full")

def semi_variance(R: Union[pd.Series, pd.DataFrame]) -> Union[float, pd.Series]:
    """
    Calculate semi-variance (MAR = mean, method = subset).
    """
    if isinstance(R, pd.DataFrame):
        return R.apply(lambda x: downside_deviation(x, MAR=x.mean(), method="subset")**2)
    else:
        return downside_deviation(R, MAR=R.mean(), method="subset")**2

def gain_deviation(R: Union[pd.Series, pd.DataFrame]) -> Union[float, pd.Series]:
    """
    Standard deviation of the positive returns.
    """
    def _calc(s: pd.Series) -> float:
        subset = s[s > 0]
        if len(subset) < 2: return np.nan
        return subset.std(ddof=1)

    if isinstance(R, pd.DataFrame):
        return R.apply(_calc)
    else:
        return _calc(R)

def loss_deviation(R: Union[pd.Series, pd.DataFrame]) -> Union[float, pd.Series]:
    def _calc(s: pd.Series) -> float:
        subset = s[s < 0]
        if len(subset) < 2: return np.nan
        return subset.std(ddof=1)

    if isinstance(R, pd.DataFrame):
        return R.apply(_calc)
    else:
        return _calc(R)

def sortino_ratio(
    R: Union[pd.Series, pd.DataFrame],
    MAR: float = 0.0
) -> Union[float, pd.Series]:
    """
    Calculate the Sortino Ratio.
    """
    # R code: mean(Return.excess(R, MAR), na.rm=TRUE)/DownsideDeviation(R, MAR)
    excess_return = (R - MAR).mean()
    downside_risk = downside_deviation(R, MAR=MAR, method="full")
    return excess_return / downside_risk

def sharpe_ratio(
    R: Union[pd.Series, pd.DataFrame],
    Rf: Union[float, pd.Series, pd.DataFrame] = 0.0,
    p: float = 0.95,
    FUN: str = "StdDev",
    annualize: bool = False,
    scale: Optional[int] = None
) -> Union[float, pd.Series, pd.DataFrame]:
    """
    Calculate the Sharpe Ratio.

    Measures excess return per unit of risk. The risk metric can be defined as
    Standard Deviation, Value at Risk (VaR), Expected Shortfall (ES), or SemiSD.

    Formula:
    $$ SR = \frac{\\overline{R_a - R_f}}{Risk} $$

    Parameters
    ----------
    R : pd.Series or pd.DataFrame
        Asset returns.
    Rf : float, pd.Series, or pd.DataFrame, optional
        Risk-free rate. Default is 0.0.
    p : float, optional
        Confidence level for VaR/ES. Default is 0.95.
    FUN : str, optional
        Risk measure to use: "StdDev", "VaR", "ES", "SemiSD".
    annualize : bool, optional
        If True, annualizes the return and risk inputs. Default is False.
    scale : int, optional
        Number of periods in a year.

    Returns
    -------
    float or pd.Series
        The Sharpe Ratio.
    """
    from pyperfanalytics.risk import es_modified, var_modified

    if scale is None and annualize:
        scale = _get_scale(R)

    # Calculate excess returns
    xR = return_excess(R, Rf)

    if FUN == "StdDev":
        if annualize:
            # R is used for risk calculation, xR for return
            ann_return = return_annualized(xR, scale=scale)
            ann_risk = std_dev_annualized(R, scale=scale)
            return ann_return / ann_risk
        else:
            return xR.mean() / R.std()
    elif FUN == "VaR":
        # Default VaR in PA's SharpeRatio is 'modified'
        risk = var_modified(R, p=p)
        if annualize:
            return return_annualized(xR, scale=scale) / risk # Note: PA's annualization for VaR-Sharpe is a bit simple
        else:
            return xR.mean() / risk
    elif FUN == "ES":
        risk = es_modified(R, p=p)
        if annualize:
            return return_annualized(xR, scale=scale) / risk
        else:
            return xR.mean() / risk
    elif FUN == "SemiSD":
        # R's SharpeRatio(FUN="SemiSD") calls DownsideSharpeRatio.
        # DSR uses SemiDeviation (MAR=mean(R), method="full") as the risk measure.
        # Note: SharpeRatio does NOT annualize when FUN="SemiSD" in R.
        # Reference: PerformanceAnalytics/R/SharpeRatio.R line 219
        if isinstance(R, pd.DataFrame):
            risk = R.apply(lambda s: downside_deviation(s, MAR=s.mean(), method="full"))
        else:
            risk = downside_deviation(R, MAR=R.mean(), method="full")

        mu = xR.mean()
        return mu / (risk * np.sqrt(2))
    else:
        raise NotImplementedError(f"Function {FUN} not yet implemented.")

def active_premium(
    R: Union[pd.Series, pd.DataFrame],
    Rb: Union[pd.Series, pd.DataFrame],
    scale: Optional[int] = None,
    geometric: bool = True
) -> Union[float, pd.Series, pd.DataFrame]:
    """
    Calculate Active Premium or Active Return.

    Active Premium = Investment's annualized return - Benchmark's annualized return
    """
    if scale is None:
        scale = _get_scale(R)

    # Standardize inputs to DataFrame
    if isinstance(R, pd.Series):
        r_df = R.to_frame()
    else:
        r_df = R

    if isinstance(Rb, pd.Series):
        rb_df = Rb.to_frame()
    else:
        rb_df = Rb

    ra_cols = r_df.columns
    rb_cols = rb_df.columns

    results = []
    for rb_col in rb_cols:
        col_results = []
        for ra_col in ra_cols:
            # Align Ra and Rb
            merged = pd.concat([r_df[ra_col], rb_df[rb_col]], axis=1).dropna()
            if merged.empty:
                col_results.append(np.nan)
                continue

            ap = (return_annualized(merged.iloc[:, 0], scale=scale, geometric=geometric) -
                  return_annualized(merged.iloc[:, 1], scale=scale, geometric=geometric))
            col_results.append(ap)
        results.append(col_results)

    res_df = pd.DataFrame(results, index=rb_cols, columns=ra_cols)

    if len(ra_cols) == 1 and len(rb_cols) == 1:
        return res_df.iloc[0, 0]
    elif len(rb_cols) == 1:
        return res_df.iloc[0]
    else:
        return res_df

def information_ratio(
    R: Union[pd.Series, pd.DataFrame],
    Rb: Union[pd.Series, pd.DataFrame],
    scale: Optional[int] = None,
    geometric: bool = True
) -> Union[float, pd.Series, pd.DataFrame]:
    """
    Calculate the Information Ratio.

    The Information Ratio measures the active return of an investment divided
    by its tracking error (active risk).

    Formula:
    $$ IR = \frac{ActivePremium}{TrackingError} $$

    Parameters
    ----------
    R : pd.Series or pd.DataFrame
        Asset returns.
    Rb : pd.Series or pd.DataFrame
        Benchmark returns.
    scale : int, optional
        Number of periods in a year.
    geometric : bool, optional
        Use geometric compounding for active premium. Default is True.

    Returns
    -------
    float, pd.Series, or pd.DataFrame
        Information Ratio.
    """
    from pyperfanalytics.risk import tracking_error

    ap = active_premium(R, Rb, scale=scale, geometric=geometric)
    te = tracking_error(R, Rb, scale=scale)

    # Handle the case where tracking_error might return a scalar, Series, or DataFrame
    if isinstance(te, (pd.Series, pd.DataFrame)):
        return ap.divide(te.replace(0, np.nan))
    else:
        return ap / te if te != 0 else np.nan

def capm_alpha(
    Ra: Union[pd.Series, pd.DataFrame],
    Rb: Union[pd.Series, pd.DataFrame],
    Rf: Union[float, pd.Series, pd.DataFrame] = 0
) -> Union[float, pd.Series, pd.DataFrame]:
    """
    Calculate CAPM alpha of returns against a benchmark.

    Alpha represents the excess return of an investment relative to the return of
    a benchmark index, adjusted for the systemic risk (beta) of the investment.

    Formula:
    $$ \alpha = \\overline{R_a - R_f} - \beta \\cdot \\overline{R_b - R_f} $$

    Parameters
    ----------
    Ra : pd.Series or pd.DataFrame
        Asset returns.
    Rb : pd.Series or pd.DataFrame
        Benchmark returns.
    Rf : float, pd.Series, or pd.DataFrame, optional
        Risk-free rate. Default is 0.

    Returns
    -------
    float, pd.Series, or pd.DataFrame
        CAPM Alpha value(s).
    """
    from pyperfanalytics.risk import capm_beta

    # Standardize inputs
    if isinstance(Ra, pd.Series):
        ra_df = Ra.to_frame()
    else:
        ra_df = Ra

    xRa = return_excess(ra_df, Rf)
    # capm_beta expects Ra, Rb, Rf
    beta = capm_beta(Ra, Rb, Rf)

    # xRa is a DataFrame. xRb handled by capm_beta grid-style
    # R's CAPM.alpha: xRa = Return.excess(Ra, Rf) and then alpha = mean(xRa) - beta * mean(xRb)
    # Our capm_beta returns a float, Series, or DataFrame

    if isinstance(Rb, pd.Series):
        rb_df = Rb.to_frame()
    else:
        rb_df = Rb

    xRb = return_excess(rb_df, Rf)

    # Alignment and calculation (matching capm_beta logic for pairs)
    ra_cols = xRa.columns
    rb_cols = xRb.columns

    results = []
    for rb_col in rb_cols:
        col_results = []
        for ra_col in ra_cols:
            merged = pd.concat([xRa[ra_col], xRb[rb_col]], axis=1).dropna()
            if merged.empty:
                col_results.append(np.nan)
                continue

            # alpha = mean(xRa) - beta * mean(xRb)
            # Fetch beta for this specific pair
            if isinstance(beta, pd.DataFrame):
                b = beta.loc[rb_col, ra_col]
            elif isinstance(beta, pd.Series):
                b = beta[ra_col]
            else:
                b = beta

            alpha = merged.iloc[:, 0].mean() - b * merged.iloc[:, 1].mean()
            col_results.append(alpha)
        results.append(col_results)

    res_df = pd.DataFrame(results, index=rb_cols, columns=ra_cols)

    if len(ra_cols) == 1 and len(rb_cols) == 1:
        return res_df.iloc[0, 0]
    elif len(rb_cols) == 1:
        return res_df.iloc[0]
    else:
        return res_df

def treynor_ratio(
    Ra: Union[pd.Series, pd.DataFrame],
    Rb: Union[pd.Series, pd.DataFrame],
    Rf: Union[float, pd.Series, pd.DataFrame] = 0,
    scale: Optional[int] = None
) -> Union[float, pd.Series, pd.DataFrame]:
    """
    Calculate Treynor Ratio.

    The Treynor ratio measures returns earned in excess of that which could have
    been earned on a riskless investment per each unit of market risk (beta).

    Formula:
    $$ TR = \frac{R_{a, ann} - R_{f, ann}}{\beta} $$

    Parameters
    ----------
    Ra : pd.Series or pd.DataFrame
        Asset returns.
    Rb : pd.Series or pd.DataFrame
        Benchmark returns.
    Rf : float, pd.Series, or pd.DataFrame, optional
        Risk-free rate. Default is 0.
    scale : int, optional
        Number of periods in a year.

    Returns
    -------
    float, pd.Series, or pd.DataFrame
        Treynor Ratio.
    """
    from pyperfanalytics.risk import capm_beta

    if scale is None:
        scale = _get_scale(Ra)

    xRa = return_excess(Ra, Rf)
    ann_return = return_annualized(xRa, scale=scale)
    beta = capm_beta(Ra, Rb, Rf)

    # Division will handle Series/DataFrame alignment
    return ann_return / beta

def calmar_ratio(
    R: Union[pd.Series, pd.DataFrame],
    scale: Optional[int] = None,
    geometric: bool = True
) -> Union[float, pd.Series]:
    """
    Calculate Calmar Ratio.

    Calculate the ratio of annualized return over the absolute value of the
    maximum drawdown. It is a risk-adjusted measure that balances performance
    with tail risk.

    Formula:
    $$ Calmar = \frac{R_{ann}}{\\max(|D_t|)} $$

    Parameters
    ----------
    R : pd.Series or pd.DataFrame
        Asset returns.
    scale : int, optional
        Number of periods in a year.
    geometric : bool, optional
        Use geometric compounding. Default is True.

    Returns
    -------
    float or pd.Series
        Calmar Ratio.
    """
    from pyperfanalytics.drawdowns import max_drawdown

    if scale is None:
        scale = _get_scale(R)

    ann_return = return_annualized(R, scale=scale, geometric=geometric)
    mdd = np.abs(max_drawdown(R, geometric=geometric))

    return ann_return / mdd


def up_down_ratios(
    Ra: Union[pd.Series, pd.DataFrame],
    Rb: Union[pd.Series, pd.DataFrame],
    method: str = "Capture",
    side: str = "Up"
) -> Union[float, pd.Series, pd.DataFrame]:
    """
    Calculate metrics on up and down markets for the benchmark asset.
    """
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
        col_results = []
        for ra_col in ra_cols:
            merged = pd.concat([ra_df[ra_col], rb_df[rb_col]], axis=1).dropna()
            if merged.empty:
                col_results.append(np.nan)
                continue

            s_ra = merged.iloc[:, 0]
            s_rb = merged.iloc[:, 1]

            if method == "Capture":
                if side == "Up":
                    mask = s_rb > 0
                else:
                    mask = s_rb <= 0

                if not mask.any():
                    col_results.append(np.nan)
                else:
                    # R code: sum(UpRa)/sum(UpRb)
                    col_results.append(s_ra[mask].sum() / s_rb[mask].sum())

            elif method == "Number":
                if side == "Up":
                    num = ((s_ra > 0) & (s_rb > 0)).sum()
                    den = (s_rb > 0).sum()
                else:
                    num = ((s_ra < 0) & (s_rb < 0)).sum()
                    den = (s_rb < 0).sum()

                col_results.append(num / den if den != 0 else np.nan)

            elif method == "Percent":
                if side == "Up":
                    num = ((s_ra > s_rb) & (s_rb > 0)).sum()
                    den = (s_rb > 0).sum()
                else:
                    num = ((s_ra > s_rb) & (s_rb < 0)).sum()
                    den = (s_rb < 0).sum()

                col_results.append(num / den if den != 0 else np.nan)
            else:
                raise ValueError(f"Unknown method: {method}")

        results.append(col_results)

    res_df = pd.DataFrame(results, index=rb_cols, columns=ra_cols)

    if len(ra_cols) == 1 and len(rb_cols) == 1:
        return res_df.iloc[0, 0]
    elif len(rb_cols) == 1:
        return res_df.iloc[0]
    else:
        return res_df

def up_capture(
    Ra: Union[pd.Series, pd.DataFrame],
    Rb: Union[pd.Series, pd.DataFrame]
) -> Union[float, pd.Series, pd.DataFrame]:
    """
    Calculate Up Capture Ratio.
    """
    return up_down_ratios(Ra, Rb, method="Capture", side="Up")

def down_capture(
    Ra: Union[pd.Series, pd.DataFrame],
    Rb: Union[pd.Series, pd.DataFrame]
) -> Union[float, pd.Series, pd.DataFrame]:
    """
    Calculate Down Capture Ratio.
    """
    return up_down_ratios(Ra, Rb, method="Capture", side="Down")
def kelly_ratio(
    R: Union[pd.Series, pd.DataFrame],
    Rf: Union[float, pd.Series, pd.DataFrame] = 0,
    method: str = "half"
) -> Union[float, pd.Series]:
    """
    Calculate Kelly criterion ratio (leverage or bet size) for a strategy.

    The Kelly criterion is a formula used to determine the optimal size of a series of bets.

    Formula:
    $$ KellyRatio = \frac{\\overline{R - R_f}}{\\sigma^2_R} $$

    Parameters
    ----------
    R : pd.Series or pd.DataFrame
        Asset returns.
    Rf : float, pd.Series, or pd.DataFrame, optional
        Risk-free rate. Default is 0.
    method : str, optional
        "half" to return half-Kelly, "full" for full Kelly. Default is "half".

    Returns
    -------
    float or pd.Series
        Kelly Ratio.
    """
    xR = return_excess(R, Rf)

    def _calc(s: pd.Series, sr: pd.Series, meth: str) -> float:
        s = s.dropna()
        sr = sr.dropna()
        if len(s) == 0 or len(sr) == 0: return np.nan

        # R code: mean(xR) / StdDev(R)^2
        # StdDev(R) in PA is sample SD (ddof=1)
        kr = s.mean() / (sr.std(ddof=1)**2)
        if meth == "half":
            kr = kr / 2
        return kr

    if isinstance(R, pd.DataFrame):
        # We need to apply per column
        results = []
        for col in R.columns:
            results.append(_calc(xR[col], R[col], method))
        return pd.Series(results, index=R.columns)
    else:
        return _calc(xR, R, method)

def upside_potential_ratio(
    R: Union[pd.Series, pd.DataFrame],
    MAR: float = 0,
    method: str = "subset"
) -> Union[float, pd.Series]:
    """
    Calculate Upside Potential Ratio of upside performance over downside risk.

    A performance measure dividing upside potential by downside deviation, similar
    to the Sortino ratio.

    Formula:
    $$ UPR = \frac{\frac{1}{n} \\sum_{R>MAR} (R - MAR)}{DownsideDeviation(MAR)} $$

    Parameters
    ----------
    R : pd.Series or pd.DataFrame
        Asset returns.
    MAR : float, optional
        Minimum Acceptable Return. Default is 0.
    method : str, optional
        "subset" or "full". Default is "subset".

    Returns
    -------
    float or pd.Series
        Upside Potential Ratio.
    """
    def _calc(s: pd.Series, mar: float, meth: str) -> float:
        s = s.dropna()
        upside = s[s > mar]

        if meth == "full":
            n = len(s)
        else:
            n = len(upside)

        if n == 0: return np.nan

        # Numerator: sum(R - MAR) / n
        numerator = (upside - mar).sum() / n
        # Denominator: DownsideDeviation(R, MAR, method=meth)
        denominator = downside_deviation(s, MAR=mar, method=meth)

        if denominator == 0: return np.nan
        return numerator / denominator

    if isinstance(R, pd.DataFrame):
        return R.apply(_calc, mar=MAR, meth=method)
    else:
        return _calc(R, MAR, method)

def martin_ratio(
    R: Union[pd.Series, pd.DataFrame],
    Rf: Union[float, pd.Series, pd.DataFrame] = 0,
    scale: Optional[int] = None,
    geometric: bool = True
) -> Union[float, pd.Series]:
    """
    Calculate Martin ratio of the return distribution.

    The Martin ratio is the annualized return minus the annualized risk-free rate,
    divided by the Ulcer Index. It evaluates performance similar to the Sharpe ratio
    but penalizes drawdowns rather than volatility.

    Formula:
    $$ MartinRatio = \frac{R_{ann} - R_{f,ann}}{UlcerIndex} $$

    where $R_{f,ann} = (1 + R_f)^{scale} - 1$ is the annualized risk-free rate.

    Notes
    -----
    **Deviation from R:** R's ``PerformanceAnalytics::MartinRatio`` subtracts the
    periodic (per-period) ``Rf`` directly from the annualized portfolio return,
    mixing units.  This implementation follows the original Peter Martin / Zephyr
    Associates definition and annualizes ``Rf`` before subtraction, ensuring both
    numerator terms share the same time unit (annualized return).

    Parameters
    ----------
    R : pd.Series or pd.DataFrame
        Asset returns (periodic, e.g. monthly).
    Rf : float, pd.Series, or pd.DataFrame, optional
        Risk-free rate **per period** (same periodicity as ``R``). Default is 0.
        The function annualizes it internally via ``(1 + Rf)^scale - 1``.
    scale : int, optional
        Number of periods in a year (e.g. 12 for monthly data).
        Auto-detected from the index frequency when not supplied.
    geometric : bool, optional
        Use geometric compounding for the annualized return. Default is True.

    Returns
    -------
    float or pd.Series
        Martin ratio.
    """
    from pyperfanalytics.risk import ulcer_index

    if scale is None:
        scale = _get_scale(R)

    ann_ret = return_annualized(R, scale=scale, geometric=geometric)
    ui = ulcer_index(R)

    if isinstance(Rf, (pd.Series, pd.DataFrame)):
        rf_val = Rf.mean()
    else:
        rf_val = Rf

    # Annualize the periodic Rf so both terms in the numerator share the same
    # time unit.  (1 + rf_periodic)^scale - 1 gives the equivalent annual rate.
    rf_ann = (1 + rf_val) ** scale - 1

    return (ann_ret - rf_ann) / ui

def pain_ratio(
    R: Union[pd.Series, pd.DataFrame],
    Rf: Union[float, pd.Series, pd.DataFrame] = 0,
    scale: Optional[int] = None,
    geometric: bool = True
) -> Union[float, pd.Series]:
    """
    Calculate Pain ratio of the return distribution.

    The Pain ratio is the annualized return minus the annualized risk-free rate,
    divided by the Pain Index.

    Formula:
    $$ PainRatio = \frac{R_{ann} - R_{f,ann}}{PainIndex} $$

    where $R_{f,ann} = (1 + R_f)^{scale} - 1$ is the annualized risk-free rate.

    Notes
    -----
    **Deviation from R:** R's ``PerformanceAnalytics::PainRatio`` subtracts the
    periodic (per-period) ``Rf`` directly from the annualized portfolio return,
    mixing units.  This implementation follows the original Zephyr Associates
    definition and annualizes ``Rf`` before subtraction.

    Parameters
    ----------
    R : pd.Series or pd.DataFrame
        Asset returns (periodic, e.g. monthly).
    Rf : float, pd.Series, or pd.DataFrame, optional
        Risk-free rate **per period** (same periodicity as ``R``). Default is 0.
        The function annualizes it internally via ``(1 + Rf)^scale - 1``.
    scale : int, optional
        Number of periods in a year (e.g. 12 for monthly data).
        Auto-detected from the index frequency when not supplied.
    geometric : bool, optional
        Use geometric compounding for the annualized return. Default is True.

    Returns
    -------
    float or pd.Series
        Pain ratio.
    """
    from pyperfanalytics.risk import pain_index

    if scale is None:
        scale = _get_scale(R)

    ann_ret = return_annualized(R, scale=scale, geometric=geometric)
    pi = pain_index(R)

    if isinstance(Rf, (pd.Series, pd.DataFrame)):
        rf_val = Rf.mean()
    else:
        rf_val = Rf

    # Annualize the periodic Rf so both terms in the numerator share the same
    # time unit.  (1 + rf_periodic)^scale - 1 gives the equivalent annual rate.
    rf_ann = (1 + rf_val) ** scale - 1

    return (ann_ret - rf_ann) / pi

def upside_risk(
    R: Union[pd.Series, pd.DataFrame],
    MAR: float = 0.0,
    method: str = "full",
    stat: str = "risk"
) -> Union[float, pd.Series]:
    """
    Calculate upside risk, variance, or potential.

    Depending on the selected stat, measures the variability or sum of returns
    above a specified Minimum Acceptable Return (MAR).

    Formula (Variance):
    $$ UV = \frac{1}{n} \\sum_{R>MAR} (R - MAR)^2 $$

    Parameters
    ----------
    R : pd.Series or pd.DataFrame
        Asset returns.
    MAR : float, optional
        Minimum Acceptable Return. Default is 0.0.
    method : str, optional
        "full" or "subset". Default is "full".
    stat : str, optional
        Type of return metric: "risk", "variance", or "potential". Default is "risk".

    Returns
    -------
    float or pd.Series
        Upside risk, variance, or potential.
    """
    def _calc(s: pd.Series, mar: float, meth: str, st: str) -> float:
        s = s.dropna()
        # R code: r = subset(R, R > MAR)
        above = s[s > mar]

        if meth == "full":
            n = len(s)
        else:
            n = len(above)

        if n == 0: return 0.0

        diff = above - mar

        if st == "risk":
            return np.sqrt((diff**2).sum() / n)
        elif st == "variance":
            return (diff**2).sum() / n
        elif st == "potential":
            return diff.sum() / n
        else:
            raise ValueError(f"Unknown stat: {st}")

    if isinstance(R, pd.DataFrame):
        return R.apply(_calc, mar=MAR, meth=method, st=stat)
    else:
        return _calc(R, MAR, method, stat)

def upside_potential(R: Union[pd.Series, pd.DataFrame], MAR: float = 0.0) -> Union[float, pd.Series]:
    """
    Calculate upside potential.
    """
    return upside_risk(R, MAR=MAR, method="full", stat="potential")

def volatility_skewness(
    R: Union[pd.Series, pd.DataFrame],
    MAR: float = 0.0,
    stat: str = "volatility"
) -> Union[float, pd.Series]:
    """
    Calculate Volatility or Variability Skewness.

    A ratio comparing upside variability to downside variability.

    Formula (Volatility Skewness):
    $$ VS = \frac{UpsideVariance}{DownsideVariance} $$

    Parameters
    ----------
    R : pd.Series or pd.DataFrame
        Asset returns.
    MAR : float, optional
        Minimum Acceptable Return. Default is 0.0.
    stat : str, optional
        "volatility" (variance based) or "variability" (risk based). Default is "volatility".

    Returns
    -------
    float or pd.Series
        Skewness metric.
    """
    if stat == "volatility":
        uv = upside_risk(R, MAR=MAR, method="full", stat="variance")
        dv = downside_deviation(R, MAR=MAR, method="full")**2
        return uv / dv
    elif stat == "variability":
        ur = upside_risk(R, MAR=MAR, method="full", stat="risk")
        dr = downside_deviation(R, MAR=MAR, method="full")
        return ur / dr
    else:
        raise ValueError(f"Unknown stat: {stat}")

def omega_ratio(
    R: Union[pd.Series, pd.DataFrame],
    L: float = 0.0,
    Rf: float = 0.0,
    method: str = "simple"
) -> Union[float, pd.Series]:
    """
    Calculate Omega Ratio.

    The Omega Ratio divides the average return above a threshold by the average
    return below that threshold. Values above 1 signify desirable upside relative to downside risk.

    Formula:
    $$ \\Omega = \frac{\\int_L^\\infty (1 - F(r)) dr}{\\int_{-\\infty}^L F(r) dr} $$
    $$ \\Omega = \frac{\frac{1}{n} \\sum \\max(R-L, 0)}{\frac{1}{n} \\sum \\max(L-R, 0)} $$

    Parameters
    ----------
    R : pd.Series or pd.DataFrame
        Asset returns.
    L : float, optional
        Return threshold (L). Default is 0.0.
    Rf : float, optional
        Risk-free rate. Default is 0.0.
    method : str, optional
        Calculation method. Only "simple" is supported currently.

    Returns
    -------
    float or pd.Series
        Omega Ratio.
    """
    if method != "simple":
        raise NotImplementedError("Only 'simple' method is implemented.")

    def _calc(s: pd.Series, l: float, rf: float) -> float:
        s = s.dropna()
        if len(s) == 0: return np.nan
        # R code: numerator = exp(-Rf) * mean(pmax(x - L, 0))
        # R code: denominator = exp(-Rf) * mean(pmax(L - x, 0))
        # exp(-rf) cancels out
        num = np.maximum(s - l, 0).mean()
        den = np.maximum(l - s, 0).mean()
        if den == 0:
            return np.inf
        return num / den

    if isinstance(R, pd.DataFrame):
        return R.apply(_calc, l=L, rf=Rf)
    else:
        return _calc(R, L, Rf)


def burke_ratio(
    R: Union[pd.Series, pd.DataFrame],
    Rf: Union[float, pd.Series, pd.DataFrame] = 0,
    modified: bool = False,
    scale: Optional[int] = None
) -> Union[float, pd.Series]:
    """
    Calculate Burke Ratio or Modified Burke Ratio.

    The Burke Ratio evaluates a portfolio's return against its drawdown risk,
    squaring drawdowns to penalize larger single losses over multiple smaller ones.

    Formula (original Burke 1994 definition):
    $$ BurkeRatio = \\frac{R_{ann} - R_f}{\\sqrt{\\frac{\\sum_{t=1}^d D_t^2}{d}}} $$

    where $d$ is the number of distinct drawdown events. This form uses the
    **root-mean-square (RMS)** of drawdowns, so the denominator does not
    grow merely because there are many small drawdowns.

    **Notes on deviations from other implementations:**

    1. **Fixed vs paper**: The original Burke (1994) denominator is
       $\\sqrt{\\Sigma D^2 / d}$ (mean of squared drawdowns). Some implementations
       (including R's `PerformanceAnalytics::BurkeRatio`) use $\\sqrt{\\Sigma D^2}$
       (sum, not mean), which differs by a factor of $\\sqrt{d}$.
       This implementation uses the mathematically correct paper definition.
    2. **Fixed vs R**: R's `BurkeRatio` multiplies each drawdown segment by `0.01`
       (expecting percentage-scale inputs, e.g. 5.0 for 5%). This implementation
       works correctly with standard decimal returns (e.g. 0.05 for 5%).

    Parameters
    ----------
    R : pd.Series or pd.DataFrame
        Asset returns (decimal format, e.g. 0.05 for 5%).
    Rf : float, pd.Series, or pd.DataFrame, optional
        Risk-free rate (same periodicity as R). Default is 0.0.
    modified : bool, optional
        If True, multiplies the ratio by $\\sqrt{n}$ where $n$ is the number of periods.
    scale : int, optional
        Number of periods in a year.

    Returns
    -------
    float or pd.Series
        Burke Ratio.
    """
    if scale is None:
        scale = _get_scale(R)

    def _get_burke_drawdowns(s: pd.Series) -> np.ndarray:
        s_vals = s.values
        drawdowns = []
        in_drawdown = False
        start_idx = 0

        # R's implementation starts from index 2 (1-based), which is index 1 (0-based)
        for i in range(1, len(s_vals)):
            if s_vals[i] < 0:
                if not in_drawdown:
                    start_idx = i
                    in_drawdown = True
            else:
                if in_drawdown:
                    # Corrected: no *0.01 scaling bug for decimal returns
                    segment = s_vals[start_idx:i]
                    dd = np.prod(1 + segment) - 1
                    drawdowns.append(dd)
                    in_drawdown = False

        if in_drawdown:
            segment = s_vals[start_idx:]
            dd = np.prod(1 + segment) - 1
            drawdowns.append(dd)

        return np.array(drawdowns)

    def _calc(s: pd.Series, rf_val: float) -> float:
        s = s.dropna()
        n = len(s)
        if n == 0:
            return np.nan

        # Annualized return
        rp = return_annualized(s, scale=scale)

        # Burke drawdown events
        drawdowns = _get_burke_drawdowns(s)
        d = len(drawdowns)

        if d == 0:
            return np.nan

        # Paper-correct denominator: sqrt(sum(D²) / d) = sqrt(mean(D²)) = RMS of drawdowns
        # This matches Burke (1994) original definition.
        # Note: R uses sqrt(sum(D²)) without dividing by d, which deviates from the paper.
        denom = np.sqrt(np.sum(drawdowns**2) / d)
        if denom == 0:
            return np.nan

        ratio = (rp - rf_val) / denom
        if modified:
            ratio *= np.sqrt(n)
        return float(ratio)

    if isinstance(Rf, (pd.Series, pd.DataFrame)):
        rf_mean = Rf.mean()
    else:
        rf_mean = Rf

    if isinstance(R, pd.DataFrame):
        return R.apply(_calc, rf_val=rf_mean)
    else:
        return _calc(R, rf_mean)

def modigliani(
    Ra: Union[pd.Series, pd.DataFrame],
    Rb: Union[pd.Series, pd.DataFrame],
    Rf: Union[float, pd.Series, pd.DataFrame] = 0
) -> Union[float, pd.Series, pd.DataFrame]:
    r"""
    Calculate Modigliani-Modigliani (M-squared) measure.

    M-Squared (M2) adjusts the portfolio's return to match the benchmark's risk
    level, expressing the Sharpe ratio in percentage terms.

    Formula:
    $$ M^2 = SR_a \cdot \sigma_b + \overline{R_f} $$

    Parameters
    ----------
    Ra : pd.Series or pd.DataFrame
        Asset returns.
    Rb : pd.Series or pd.DataFrame
        Benchmark returns.
    Rf : float, pd.Series, or pd.DataFrame, optional
        Risk-free rate. Default is 0.

    Returns
    -------
    float, pd.Series, or pd.DataFrame
        M-Squared measure.
    """
    if isinstance(Ra, pd.Series):
        ra_df = Ra.to_frame()
    else:
        ra_df = Ra

    if isinstance(Rb, pd.Series):
        rb_df = Rb.to_frame()
    else:
        rb_df = Rb

    if isinstance(Rf, (pd.Series, pd.DataFrame)):
        rf_mean = Rf.mean()
    else:
        rf_mean = Rf

    ra_cols = ra_df.columns
    rb_cols = rb_df.columns

    results = []
    for rb_col in rb_cols:
        col_results = []
        for ra_col in ra_cols:
            merged = pd.concat([ra_df[ra_col], rb_df[rb_col]], axis=1).dropna()
            if merged.empty:
                col_results.append(np.nan)
                continue

            a = merged.iloc[:, 0]
            b = merged.iloc[:, 1]

            sr_a = sharpe_ratio(a, Rf=rf_mean, annualize=False)
            std_b = b.std(ddof=1)

            m2 = sr_a * std_b + rf_mean
            col_results.append(float(m2))
        results.append(col_results)

    res_df = pd.DataFrame(results, index=rb_cols, columns=ra_cols)

    if len(rb_cols) == 1 and len(ra_cols) == 1:
        return res_df.iloc[0, 0]
    elif len(rb_cols) == 1:
        return res_df.iloc[0, :]
    elif len(ra_cols) == 1:
        return res_df.iloc[:, 0]
    else:
        return res_df

def mean_absolute_deviation(R: Union[pd.Series, pd.DataFrame]) -> Union[float, pd.Series]:
    """
    Calculate Mean Absolute Deviation (MAD).

    An alternative measure of dispersion around the mean, often more robust to
    outliers than standard deviation.

    Formula:
    $$ MAD = \frac{1}{n} \\sum_{i=1}^n |R_i - \bar{R}| $$

    Parameters
    ----------
    R : pd.Series or pd.DataFrame
        Asset returns.

    Returns
    -------
    float or pd.Series
        Mean absolute deviation.
    """
    def _calc(s: pd.Series) -> float:
        s = s.dropna()
        if len(s) == 0:
            return np.nan
        return float((s - s.mean()).abs().mean())

    if isinstance(R, pd.DataFrame):
        return R.apply(_calc)
    else:
        return _calc(R)

def downside_frequency(R: Union[pd.Series, pd.DataFrame], MAR: float = 0.0) -> Union[float, pd.Series]:
    """
    Calculate Downside Frequency.
    Number of returns below MAR divided by total number of returns.
    """
    def _calc(s: pd.Series, mar: float) -> float:
        s = s.dropna()
        if len(s) == 0:
            return np.nan
        return float(len(s[s < mar]) / len(s))

    if isinstance(R, pd.DataFrame):
        return R.apply(_calc, mar=MAR)
    else:
        return _calc(R, MAR)

def m2_sortino(
    Ra: Union[pd.Series, pd.DataFrame],
    Rb: Union[pd.Series, pd.DataFrame],
    MAR: float = 0.0,
    scale: Optional[int] = None
) -> Union[float, pd.Series, pd.DataFrame]:
    """
    Calculate M squared for Sortino.
    M2S = Rp + SortinoRatio * (DownsideRiskBenchmark - DownsideRiskPortfolio)
    All components are annualized.
    """
    if scale is None:
        scale = _get_scale(Ra)

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
        col_results = []
        for ra_col in ra_cols:
            merged = pd.concat([ra_df[ra_col], rb_df[rb_col]], axis=1).dropna()
            if merged.empty:
                col_results.append(np.nan)
                continue

            a = merged.iloc[:, 0]
            b = merged.iloc[:, 1]

            rp = return_annualized(a, scale=scale)
            sigma_d = downside_deviation(a, MAR=MAR) * np.sqrt(scale)
            sigma_dm = downside_deviation(b, MAR=MAR) * np.sqrt(scale)
            sr = sortino_ratio(a, MAR=MAR)

            m2s = rp + sr * (sigma_dm - sigma_d)
            col_results.append(float(m2s))
        results.append(col_results)

    res_df = pd.DataFrame(results, index=rb_cols, columns=ra_cols)

    if len(rb_cols) == 1 and len(ra_cols) == 1:
        return float(res_df.iloc[0, 0])
    elif len(rb_cols) == 1:
        return res_df.iloc[0, :]
    elif len(ra_cols) == 1:
        return res_df.iloc[:, 0]
    else:
        return res_df

def m_squared(
    Ra: Union[pd.Series, pd.DataFrame],
    Rb: Union[pd.Series, pd.DataFrame],
    Rf: Union[float, pd.Series, pd.DataFrame] = 0.0,
    scale: Optional[int] = None
) -> Union[float, pd.Series, pd.DataFrame]:
    """
    Calculate M squared.
    M2 = (Rp - Rf) * (MarketRisk / PortfolioRisk) + Rf
    Uses population standard deviation for risk (matches PerformanceAnalytics).
    """
    if scale is None:
        scale = _get_scale(Ra)

    if isinstance(Rf, (pd.Series, pd.DataFrame)):
        rf_mean = Rf.mean()
        # Ensure rf_mean is a float if it was a single series/df
        if isinstance(rf_mean, pd.Series):
            rf_mean = rf_mean.iloc[0]
    else:
        rf_mean = Rf
        # R's MSquared uses Rf directly (not annualized). Formula: (Rp - Rf)*σm/σp + Rf.
        # Users should pass Rf at the same scale as Rp (i.e., annualized or 0).

    rf_val = float(rf_mean)

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
        col_results = []
        for ra_col in ra_cols:
            a = ra_df[ra_col]
            b = rb_df[rb_col]

            # Match R's strict NA handling: if any NA, return NA
            if a.isna().any() or b.isna().any():
                col_results.append(np.nan)
                continue

            merged = pd.concat([a, b], axis=1).dropna()
            if merged.empty:
                col_results.append(np.nan)
                continue

            a_clean = merged.iloc[:, 0]
            b_clean = merged.iloc[:, 1]

            rp = return_annualized(a_clean, scale=scale)

            # Population standard deviation
            sigp = a_clean.std(ddof=0) * np.sqrt(scale)
            sigm = b_clean.std(ddof=0) * np.sqrt(scale)

            if sigp == 0:
                m2 = np.nan
            else:
                m2 = (rp - rf_val) * sigm / sigp + rf_val
            col_results.append(float(m2))
        results.append(col_results)

    res_df = pd.DataFrame(results, index=rb_cols, columns=ra_cols)

    if len(rb_cols) == 1 and len(ra_cols) == 1:
        return float(res_df.iloc[0, 0])
    elif len(rb_cols) == 1:
        return res_df.iloc[0, :]
    elif len(ra_cols) == 1:
        return res_df.iloc[:, 0]
    else:
        return res_df

def m_squared_excess(
    Ra: Union[pd.Series, pd.DataFrame],
    Rb: Union[pd.Series, pd.DataFrame],
    Rf: Union[float, pd.Series, pd.DataFrame] = 0.0,
    method: str = "geometric",
    scale: Optional[int] = None
) -> Union[float, pd.Series, pd.DataFrame]:
    """
    Calculate M squared excess.
    Geometric: (1 + M2) / (1 + Rbp) - 1
    Arithmetic: M2 - Rbp
    """
    if scale is None:
        scale = _get_scale(Ra)

    m2 = m_squared(Ra, Rb, Rf=Rf, scale=scale)

    if isinstance(Rb, pd.Series):
        rb_df = Rb.to_frame()
    else:
        rb_df = Rb

    # Calculate Rbp for each benchmark
    rbp = rb_df.apply(lambda x: return_annualized(x.dropna(), scale=scale))

    if isinstance(m2, float):
        m2_val = m2
        rbp_val = float(rbp.iloc[0])
        if method == "geometric":
            return (1 + m2_val) / (1 + rbp_val) - 1
        else:
            return m2_val - rbp_val

    # Complex case: align M2 (which is DataFrame or Series) with Rbp
    if isinstance(m2, pd.Series):
        if len(rb_df.columns) == 1:
            rbp_val = float(rbp.iloc[0])
            if method == "geometric":
                return (1 + m2) / (1 + rbp_val) - 1
            else:
                return m2 - rbp_val
        else:
            # m2 is series where index is assets, but rb had multiple? Wait, if len(ra)=1, index is rb.
            raise NotImplementedError("Complex alignments for M2 Excess not fully supported yet.")

    if isinstance(m2, pd.DataFrame):
        res = m2.copy()
        for rb_col in res.index:
            rbp_val = float(rbp[rb_col])
            if method == "geometric":
                res.loc[rb_col] = (1 + res.loc[rb_col]) / (1 + rbp_val) - 1
            else:
                res.loc[rb_col] = res.loc[rb_col] - rbp_val
        return res

def net_selectivity(
    Ra: Union[pd.Series, pd.DataFrame],
    Rb: Union[pd.Series, pd.DataFrame],
    Rf: Union[float, pd.Series, pd.DataFrame] = 0.0,
    scale: Optional[int] = None
) -> Union[float, pd.Series, pd.DataFrame]:
    """
    Net selectivity = Selectivity - diversification
    Selectivity is Jensen's alpha.
    diversification = (FamaBeta(Ra,Rb) - CAPM.beta(Ra,Rb)) * (Annualized_Rb - Rf)
    """
    if scale is None:
        scale = _get_scale(Ra)

    if isinstance(Rf, (pd.Series, pd.DataFrame)):
        rf_val = float(Rf.mean())
    else:
        rf_val = float(Rf)

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

    from pyperfanalytics.risk import capm_beta, fama_beta

    for rb_col in rb_cols:
        col_results = []
        for ra_col in ra_cols:
            a = ra_df[ra_col]
            b = rb_df[rb_col]

            # Match R's strict NA handling
            if a.isna().any() or b.isna().any():
                col_results.append(np.nan)
                continue

            merged = pd.concat([a, b], axis=1).dropna()
            if merged.empty:
                col_results.append(np.nan)
                continue

            a = merged.iloc[:, 0]
            b = merged.iloc[:, 1]

            sel = jensen_alpha(a, b, Rf=rf_val, scale=scale)
            f_beta = fama_beta(a, b, scale=scale)
            c_beta = capm_beta(a, b, Rf=rf_val)
            rb_ann = return_annualized(b, scale=scale)

            d = (f_beta - c_beta) * (rb_ann - rf_val)
            col_results.append(float(sel - d))

        results.append(col_results)

    res_df = pd.DataFrame(results, index=rb_cols, columns=ra_cols)

    if len(rb_cols) == 1 and len(ra_cols) == 1:
        return float(res_df.iloc[0, 0])
    elif len(rb_cols) == 1:
        return res_df.iloc[0, :]
    elif len(ra_cols) == 1:
        return res_df.iloc[:, 0]
    else:
        return res_df

def omega_excess_return(
    Ra: Union[pd.Series, pd.DataFrame],
    Rb: Union[pd.Series, pd.DataFrame],
    MAR: float = 0.0,
    scale: Optional[int] = None
) -> Union[float, pd.Series, pd.DataFrame]:
    """
    Omega excess return = Rp - 3 * SigmaD * SigmaDM
    """
    if scale is None:
        scale = _get_scale(Ra)

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
        col_results = []
        for ra_col in ra_cols:
            merged = pd.concat([ra_df[ra_col], rb_df[rb_col]], axis=1).dropna()
            if merged.empty:
                col_results.append(np.nan)
                continue

            a = merged.iloc[:, 0]
            b = merged.iloc[:, 1]

            rp = return_annualized(a, scale=scale)
            sigma_d = downside_deviation(a, MAR=MAR) * np.sqrt(scale)
            sigma_dm = downside_deviation(b, MAR=MAR) * np.sqrt(scale)

            val = rp - 3 * sigma_d * sigma_dm
            col_results.append(float(val))

        results.append(col_results)

    res_df = pd.DataFrame(results, index=rb_cols, columns=ra_cols)

    if len(rb_cols) == 1 and len(ra_cols) == 1:
        return float(res_df.iloc[0, 0])
    elif len(rb_cols) == 1:
        return res_df.iloc[0, :]
    elif len(ra_cols) == 1:
        return res_df.iloc[:, 0]
    else:
        return res_df

def omega_sharpe_ratio(R: Union[pd.Series, pd.DataFrame], MAR: float = 0.0) -> Union[float, pd.Series]:
    """
    Omega-Sharpe Ratio = (UpsidePotential - DownsidePotential) / DownsidePotential
    """
    def _calc(s: pd.Series, mar: float) -> float:
        s = s.dropna()
        if len(s) == 0:
            return np.nan
        up_pot = upside_potential(s, MAR=mar)
        down_pot = downside_potential(s, MAR=mar)
        if down_pot == 0:
            return np.nan
        return float((up_pot - down_pot) / down_pot)

    if isinstance(R, pd.DataFrame):
        return R.apply(_calc, mar=MAR)
    else:
        return _calc(R, MAR)

def downside_sharpe_ratio(
    R: Union[pd.Series, pd.DataFrame],
    Rf: Union[float, pd.Series, pd.DataFrame] = 0.0
) -> Union[float, pd.Series]:
    """
    Downside Sharpe Ratio = mean(R - Rf) / (sqrt(2) * SemiSD(R))
    SemiSD is calculated wrt mean of returns (not Rf or MAR).
    """
    if isinstance(Rf, (pd.Series, pd.DataFrame)):
        rf_val = float(Rf.mean())
    else:
        rf_val = float(Rf)

    def _calc(s: pd.Series, rf: float) -> float:
        s = s.dropna()
        if len(s) == 0:
            return np.nan
        mu_hat = s.mean()
        # semisd = sqrt(mean((returns - mu_hat)^2 * (returns <= mu_hat)))
        semisd = np.sqrt(np.mean(((s - mu_hat)**2) * (s <= mu_hat)))
        if semisd == 0:
            return np.nan
        return float((mu_hat - rf) / (semisd * np.sqrt(2)))

    if isinstance(R, pd.DataFrame):
        return R.apply(_calc, rf=rf_val)
    else:
        return _calc(R, rf_val)

def return_cumulative(R: Union[pd.Series, pd.DataFrame], geometric: bool = True) -> Union[float, pd.Series]:
    """
    Calculate a compounded (geometric) or simple cumulative return.
    """
    def _calc(s: pd.Series, geom: bool) -> float:
        s = s.dropna()
        if len(s) == 0:
            return np.nan
        if geom:
            return float((1 + s).prod() - 1)
        else:
            return float(s.sum())

    if isinstance(R, pd.DataFrame):
        return R.apply(_calc, geom=geometric)
    else:
        return _calc(R, geometric)

def kappa(R: Union[pd.Series, pd.DataFrame], MAR: float = 0.0, l: int = 2) -> Union[float, pd.Series]:
    """
    Calculate Kappa.
    Kappa = (mean(R) - MAR) / (mean(max(MAR - R, 0)^l))^(1/l)
    """
    def _calc(s: pd.Series, mar: float, L: int) -> float:
        s = s.dropna()
        if len(s) == 0:
            return np.nan

        m = s.mean()
        r_down = s[s < mar]
        denom = ((1 / len(s)) * np.sum((mar - r_down)**L))**(1 / L)
        if denom == 0:
            return np.nan
        return float((m - mar) / denom)

    if isinstance(R, pd.DataFrame):
        return R.apply(_calc, mar=MAR, L=l)
    else:
        return _calc(R, MAR, l)

def annualized_excess_return(
    Ra: Union[pd.Series, pd.DataFrame],
    Rb: Union[pd.Series, pd.DataFrame],
    scale: Optional[int] = None,
    geometric: bool = True
) -> Union[float, pd.Series, pd.DataFrame]:
    """
    Annualized excess return.
    Geometric: (1 + Rpa) / (1 + Rba) - 1
    Arithmetic: Rpa - Rba
    where Rpa and Rba are annualized returns.
    """
    if scale is None:
        scale = _get_scale(Ra)

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
        col_results = []
        for ra_col in ra_cols:
            merged = pd.concat([ra_df[ra_col], rb_df[rb_col]], axis=1).dropna()
            if merged.empty:
                col_results.append(np.nan)
                continue

            a = merged.iloc[:, 0]
            b = merged.iloc[:, 1]

            rpa = return_annualized(a, scale=scale, geometric=geometric)
            rba = return_annualized(b, scale=scale, geometric=geometric)

            if geometric:
                val = (1 + rpa) / (1 + rba) - 1
            else:
                val = rpa - rba

            col_results.append(float(val))
        results.append(col_results)

    res_df = pd.DataFrame(results, index=rb_cols, columns=ra_cols)

    if len(rb_cols) == 1 and len(ra_cols) == 1:
        return float(res_df.iloc[0, 0])
    elif len(rb_cols) == 1:
        return res_df.iloc[0, :]
    elif len(ra_cols) == 1:
        return res_df.iloc[:, 0]
    else:
        return res_df

def bernardo_ledoit_ratio(R: Union[pd.Series, pd.DataFrame]) -> Union[float, pd.Series]:
    """
    Calculate Bernardo and Ledoit Ratio.
    Sum of positive returns / Absolute sum of negative returns.
    """
    def _calc(s: pd.Series) -> float:
        s = s.dropna()
        pos = s[s > 0].sum()
        neg = s[s < 0].sum()
        if neg == 0: return np.inf
        return pos / abs(neg)

    if isinstance(R, pd.DataFrame):
        return R.apply(_calc)
    else:
        return _calc(R)

def d_ratio(R: Union[pd.Series, pd.DataFrame]) -> Union[float, pd.Series]:
    """
    Calculate D Ratio.
    Absolute sum of negative returns / Sum of positive returns (Inverse of Bernardo-Ledoit).
    """
    return 1.0 / bernardo_ledoit_ratio(R)

def rachev_ratio(
    R: Union[pd.Series, pd.DataFrame],
    alpha: float = 0.1,
    beta: float = 0.1,
    rf: float = 0.0
) -> Union[float, pd.Series]:
    """
    Calculate Rachev Ratio.
    ETL(upper, beta) / ETL(lower, alpha).
    """

    def _calc(s: pd.Series, a: float, b: float) -> float:
        s = s.dropna()
        n = len(s)
        if n == 0: return np.nan

        # Lower ETL (ES)
        # R code: quantile.lower <- quantile(data, alpha)
        # ES.lower <- -mean(data[data <= quantile.lower])
        q_lower = np.percentile(s, a * 100)
        es_lower = -s[s <= q_lower].mean()

        # Upper ETL
        # R code: n.upper <- floor((1-beta)*length(data))
        # quantile.upper <- sorted.returns[n.upper]
        # ES.upper <- mean(data[data >= quantile.upper])
        n_upper = int(np.floor((1 - b) * n))
        sorted_s = np.sort(s.values)
        q_upper = sorted_s[n_upper - 1]
        es_upper = s[s >= q_upper].mean()

        return es_upper / es_lower

    if isinstance(R, pd.DataFrame):
        return R.apply(_calc, a=alpha, b=beta)
    else:
        return _calc(R, alpha, beta)

def prospect_ratio(R: Union[pd.Series, pd.DataFrame], MAR: float = 0.0) -> Union[float, pd.Series]:
    """
    Calculate Prospect Ratio.
    (Sum(pos) + 2.25 * Sum(neg) - MAR) / (DownsideDeviation * n).
    """
    def _calc(s: pd.Series, mar: float) -> float:
        n_total = len(s)
        s = s.dropna()
        if len(s) == 0: return np.nan

        pos = s[s > 0].sum()
        neg = s[s < 0].sum()

        dd = downside_deviation(s, MAR=mar, method="full")
        if dd == 0: return np.nan

        # R code: (sum(r1)+2.25*sum(r2)-MAR)/(SigD*n)
        # where n is length(R) including NAs
        return (pos + 2.25 * neg - mar) / (dd * n_total)

    if isinstance(R, pd.DataFrame):
        return R.apply(_calc, mar=MAR)
    else:
        return _calc(R, MAR)

def adjusted_sharpe_ratio(
    R: Union[pd.Series, pd.DataFrame],
    Rf: float = 0.0,
    scale: Optional[int] = None
) -> Union[float, pd.Series]:
    """
    Calculate Adjusted Sharpe Ratio.
    Incorporates skewness and kurtosis penalty.
    """
    from pyperfanalytics.utils import kurtosis, skewness

    if scale is None:
        scale = _get_scale(R)

    def _calc(s: pd.Series, rf: float, sc: int) -> float:
        s = s.dropna()
        if len(s) < 2: return np.nan

        sr = sharpe_ratio(s, Rf=rf, annualize=True, scale=sc)
        s_val = skewness(s, method="moment")
        k_val = kurtosis(s, method="moment") # Absolute kurtosis

        return sr * (1 + (s_val / 6.0) * sr - ((k_val - 3.0) / 24.0) * sr**2)

    if isinstance(R, pd.DataFrame):
        return R.apply(_calc, rf=Rf, sc=scale)
    else:
        return _calc(R, Rf, scale)

def jensen_alpha(
    Ra: Union[pd.Series, pd.DataFrame],
    Rb: Union[pd.Series, pd.DataFrame],
    Rf: Union[float, pd.Series, pd.DataFrame] = 0,
    scale: Optional[int] = None
) -> Union[float, pd.Series, pd.DataFrame]:
    """
    Calculate Jensen's Alpha.

    Alpha = Rp - Rf - Beta * (Rpb - Rf)
    where Rp and Rpb are annualized returns.
    """
    from pyperfanalytics.risk import capm_beta

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
        col_results = []
        for ra_col in ra_cols:
            a = ra_df[ra_col]
            b = rb_df[rb_col]

            # Match R's strict NA handling
            if a.isna().any() or b.isna().any():
                col_results.append(np.nan)
                continue

            # Align
            merged = pd.concat([a, b], axis=1).dropna()
            if merged.empty:
                col_results.append(np.nan)
                continue

            a = merged.iloc[:, 0]
            b = merged.iloc[:, 1]

            # Annualized geometric returns
            rp = (1 + a).prod()**(scale / len(a)) - 1
            rpb = (1 + b).prod()**(scale / len(b)) - 1

            beta = capm_beta(a, b, Rf=Rf)

            # Use mean Rf if Rf is a series
            # R's CAPM.jensenAlpha formula: result = Rp - Rf - beta * (Rpb - Rf)
            # where Rp and Rpb are annualized but Rf is used as-is (NOT annualized).
            # This is R's convention: users pass Rf at the same rate as Rp (e.g., annualized 0.035)
            # or pass 0 as the default. The previous Rf.mean()*scale was incorrect.
            if isinstance(Rf, (pd.Series, pd.DataFrame)):
                rf_val = float(Rf.mean())
            else:
                rf_val = float(Rf)

            j_alpha = rp - rf_val - beta * (rpb - rf_val)
            col_results.append(j_alpha)
        results.append(col_results)

    res_df = pd.DataFrame(results, index=rb_cols, columns=ra_cols)

    if len(ra_cols) == 1 and len(rb_cols) == 1:
        return res_df.iloc[0, 0]
    elif len(rb_cols) == 1:
        return res_df.iloc[0]
    else:
        return res_df

def appraisal_ratio(
    Ra: Union[pd.Series, pd.DataFrame],
    Rb: Union[pd.Series, pd.DataFrame],
    Rf: Union[float, pd.Series, pd.DataFrame] = 0,
    scale: Optional[int] = None
) -> Union[float, pd.Series, pd.DataFrame]:
    """
    Calculate Appraisal Ratio.

    Appraisal ratio = Jensen's Alpha / Specific Risk
    """
    from pyperfanalytics.risk import specific_risk

    j_alpha = jensen_alpha(Ra, Rb, Rf=Rf, scale=scale)
    spec_risk = specific_risk(Ra, Rb, Rf=Rf, scale=scale)

    return j_alpha / spec_risk

def market_timing(
    Ra: Union[pd.Series, pd.DataFrame],
    Rb: Union[pd.Series, pd.DataFrame],
    Rf: Union[float, pd.Series, pd.DataFrame] = 0,
    method: str = "TM"
) -> pd.DataFrame:
    """
    Estimate Market Timing models (Treynor-Mazuy or Henriksson-Merton).

    Method:
        "TM" - Treynor-Mazuy model
        "HM" - Henriksson-Merton model
    """
    import statsmodels.api as sm

    # Standardize inputs
    if isinstance(Ra, pd.Series):
        ra_df = Ra.to_frame()
    else:
        ra_df = Ra

    if isinstance(Rb, pd.Series):
        rb_df = Rb.to_frame()
    else:
        rb_df = Rb

    xRa = return_excess(ra_df, Rf)
    xRb = return_excess(rb_df, Rf)

    ra_cols = xRa.columns
    rb_cols = xRb.columns

    results = []
    index_names = []

    for rb_col in rb_cols:
        for ra_col in ra_cols:
            # Align
            merged = pd.concat([xRa[ra_col], xRb[rb_col]], axis=1).dropna()
            if merged.empty:
                results.append([np.nan, np.nan, np.nan])
            else:
                y = merged.iloc[:, 0]
                x1 = merged.iloc[:, 1]

                if method == "TM":
                    # TM: Rp-Rf = alpha + beta(Rb-Rf) + gamma(Rb-Rf)^2
                    x2 = x1**2
                elif method == "HM":
                    # HM: Rp-Rf = alpha + beta(Rb-Rf) + gamma * max(0, Rf-Rb)
                    # R implementation uses xRb * (-xRb > 0), i.e., min(0, xRb)
                    x2 = x1.clip(upper=0)
                else:
                    raise ValueError("Method must be 'TM' or 'HM'")

                X = sm.add_constant(np.column_stack([x1, x2]))
                model = sm.OLS(y, X).fit()
                results.append(model.params.values)

            index_names.append(f"{ra_col} to {rb_col}")

    res_df = pd.DataFrame(results, index=index_names, columns=["Alpha", "Beta", "Gamma"])
    return res_df

def prob_sharpe_ratio(
    R: Union[pd.Series, pd.DataFrame],
    refSR: Union[float, pd.Series, pd.DataFrame],
    Rf: float = 0.0,
    ignore_skewness: bool = False,
    ignore_kurtosis: bool = True
) -> Union[float, pd.Series]:
    """
    Calculate Probabilistic Sharpe Ratio (PSR).
    Probability that the observed Sharpe Ratio is higher than the reference one.
    Adjusts for the inflationary effect of short series with skewness and kurtosis.
    Reference: Marcos Lopez de Prado. 2018. Advances in Financial Machine Learning.
    """
    from scipy.stats import norm

    from pyperfanalytics.utils import kurtosis, skewness

    def _calc(s: pd.Series, rsr: float, rf: float) -> float:
        s = s.dropna()
        n = len(s)
        if n <= 2:
            return np.nan

        # Periodic SR (non-annualized)
        sr = sharpe_ratio(s, Rf=rf, annualize=False)

        # Moments
        sk = skewness(s) if not ignore_skewness else 0.0
        # R's PerformanceAnalytics uses kurtosis(method='moment') which is non-excess
        kr = kurtosis(s, method='moment') if not ignore_kurtosis else 3.0

        # PSR formula
        # sr_prob = pnorm(((sr - refSR)*((n-1)^(0.5)))/(1-sr*sk+(sr^2)*(kr-1)/4)^(0.5))
        numerator = (sr - rsr) * np.sqrt(n - 1)
        # Avoid division by zero or negative in square root
        denom_sq = 1.0 - sr * sk + (sr**2) * (kr - 1.0) / 4.0
        if denom_sq <= 0:
            return np.nan

        denominator = np.sqrt(denom_sq)
        return float(norm.cdf(numerator / denominator))

    if isinstance(R, pd.DataFrame):
        if isinstance(refSR, (pd.Series, pd.DataFrame, list, np.ndarray)):
            # Normalize refSR to a Series matching R's columns
            if not isinstance(refSR, (pd.Series, pd.DataFrame)):
                ref_s = pd.Series(refSR, index=R.columns)
            elif isinstance(refSR, pd.DataFrame):
                ref_s = refSR.iloc[0]
            else:
                ref_s = refSR

            results = {}
            for col in R.columns:
                rsr_val = ref_s[col]
                results[col] = _calc(R[col], rsr_val, Rf)
            return pd.Series(results)
        else:
            return R.apply(_calc, rsr=refSR, rf=Rf)
    else:
        return _calc(R, refSR, Rf)


def sterling_ratio(

    R: Union[pd.Series, pd.DataFrame],
    scale: Optional[int] = None,
    excess: float = 0.1
) -> Union[float, pd.Series]:
    """
    Calculate Sterling Ratio.

    Sterling Ratio = Annualized Return / (Absolute Maximum Drawdown + Excess)
    """
    if scale is None:
        scale = _get_scale(R)

    def _calc(s: pd.Series, sc: int, ex: float) -> float:
        from pyperfanalytics.drawdowns import max_drawdown
        s_clean = s.dropna()
        if s_clean.empty:
            return np.nan
        ann_ret = return_annualized(s_clean, scale=sc)
        mdd = abs(max_drawdown(s_clean))
        return ann_ret / (mdd + ex)

    if isinstance(R, pd.DataFrame):
        return R.apply(_calc, sc=scale, ex=excess)
    else:
        return _calc(R, scale, excess)


def hurst_index(R: Union[pd.Series, pd.DataFrame]) -> Union[float, pd.Series]:
    """
    Calculate the Hurst Index (Simplified Rescaled Range analysis).
    H = log(m)/log(n)
    where m = [max(r_i) - min(r_i)]/sigma_p and n = number of observations
    """
    def _calc(s: pd.Series) -> float:
        s = s.dropna()
        n = len(s)
        if n < 2:
            return np.nan
        m = (s.max() - s.min()) / s.std()
        if m <= 0:
            return np.nan
        return np.log(m) / np.log(n)

    if isinstance(R, pd.DataFrame):
        return R.apply(_calc)
    else:
        return _calc(R)


def to_period_contributions(
    Contributions: Union[pd.Series, pd.DataFrame],
    period: str = "years"
) -> pd.DataFrame:
    """
    Aggregate high-frequency contributions to lower frequency.
    Valid periods: 'weeks', 'months', 'quarters', 'years', 'all'.
    """
    if isinstance(Contributions, pd.Series):
        C = Contributions.to_frame()
    else:
        C = Contributions.copy()

    if not isinstance(C.index, pd.DatetimeIndex):
        raise ValueError("Contributions index must be a DatetimeIndex.")

    # Calculate portfolio return from contributions
    pret = C.sum(axis=1)

    # Lagged cumulative wealth index
    # R code: lag.cum.ret = na.fill(xts::lag.xts(cumprod(1+pret),1),1)
    cum_ret = (1 + pret).cumprod()
    lag_cum_ret = cum_ret.shift(1).fillna(1.0)

    # Weighted contributions
    # R code: wgt.contrib = C * rep(lag.cum.ret, NCOL(C))
    wgt_contrib = C.multiply(lag_cum_ret, axis=0)

    if period == "all":
        # Sum everything
        res = wgt_contrib.sum().to_frame().T
        res.index = [C.index[-1]]
    else:
        # Aggregate by period
        freq_map = {
            "weeks": "W",
            "months": "ME",
            "quarters": "QE",
            "years": "YE"
        }
        freq = freq_map.get(period)
        if freq is None:
            raise ValueError(f"Invalid period: {period}")

        # R code: sum(wgt.contrib[span] / rep(head(lag.cum.ret[span],1), NCOL(wgt.contrib)))
        # For each period, we divide by the wealth index at the start of that period.

        def _aggregate(group):
            first_lag = lag_cum_ret.loc[group.index[0]]
            return group.sum() / first_lag

        res = wgt_contrib.groupby(pd.Grouper(freq=freq)).apply(_aggregate)

    # Add Portfolio Return column
    res["Portfolio Return"] = res.sum(axis=1)
    return res

def return_geltner(R: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
    """
    Calculate Geltner liquidity-adjusted return series.
    Geltner.returns = (R(t) - R(t-1) * rho) / (1 - rho)
    where rho is the first-order autocorrelation of the returns.
    """
    def _calc(s: pd.Series) -> pd.Series:
        s_clean = s.dropna()
        if len(s_clean) < 2:
            return s

        # compute first order autocorrelation
        from statsmodels.tsa.stattools import acf
        rho = acf(s_clean, nlags=1, fft=False)[1]

        # calculate geltner series
        lag_s = s.shift(1)
        res = (s - lag_s * rho) / (1 - rho)

        # Keep original indices/NAs for unshiftable portions
        return res

    if isinstance(R, pd.DataFrame):
        return R.apply(_calc)
    else:
        return _calc(R)

def return_clean(
    R: Union[pd.Series, pd.DataFrame],
    method: str = "boudt",
    alpha: float = 0.01,
    trim: float = 0.001
) -> Union[pd.Series, pd.DataFrame]:
    """
    Clean extreme observations in a time series to provide robust risk estimates.
    Robustly clean a time series to reduce the magnitude of observations that exceed
    the 1-alpha risk threshold using Minimum Covariance Determinant (MCD).

    method: "none", "boudt", or "geltner".
    """
    if method == "none":
        return R.copy()
    elif method == "geltner":
        return return_geltner(R)
    elif method == "boudt":
        from scipy.stats import chi2
        try:
            from sklearn.covariance import MinCovDet
        except ImportError as e:
            raise ImportError(
                "return_clean(method='boudt') requires scikit-learn. "
                "Install it with: pip install 'pyperfanalytics[boudt]' or pip install scikit-learn"
            ) from e

        if isinstance(R, pd.Series):
            df = R.to_frame()
            is_series = True
        else:
            df = R
            is_series = False

        df_clean = df.dropna()
        n_obs, n_vars = df_clean.shape

        if n_obs < 2:
            return R.copy()

        try:
            mcd = MinCovDet(support_fraction=1-alpha).fit(df_clean.values)
            mu = mcd.raw_location_
            sigma = mcd.raw_covariance_
            invSigma = np.linalg.inv(sigma)
        except Exception:
            import warnings
            warnings.warn("Covariance matrix is singular, returning original data.")
            return R.copy()

        # 1. Sort the data in function of their extremeness (Mahalanobis distance)
        diff = df_clean.values - mu
        d2 = np.sum((diff.dot(invSigma)) * diff, axis=1)

        # 2. Outlier detection
        # empirical 1-alpha quantile
        sorted_d2 = np.sort(d2)
        idx = int(np.floor((1 - alpha) * n_obs)) - 1 # 0-based index correction
        if idx < 0:
            idx = 0

        empirical_threshold = sorted_d2[idx]
        chi2_thresh = chi2.ppf(1 - trim, n_vars)

        # 2.2 Multivariate winsorization
        res = df.copy()
        threshold = max(empirical_threshold, chi2_thresh)
        needs_cleaning = (d2 > empirical_threshold) & (d2 > chi2_thresh)

        scale_factors = np.sqrt(threshold / d2[needs_cleaning])

        clean_values = df_clean.values.copy()
        clean_values[needs_cleaning, :] = clean_values[needs_cleaning, :] * scale_factors[:, None]

        res.update(pd.DataFrame(clean_values, index=df_clean.index, columns=df_clean.columns))

        if is_series:
            return res.iloc[:, 0]
        else:
            return res
    else:
        raise ValueError(f"Unknown method {method}")

def return_portfolio(
    R: Union[pd.Series, pd.DataFrame],
    weights: Optional[Union[pd.Series, pd.DataFrame, list, np.ndarray]] = None,
    geometric: bool = True,
    rebalance_on: str = "none"
) -> pd.Series:
    """
    Calculate weighted returns for a portfolio of assets.
    Supports geometric or arithmetic compounding with optional periodic rebalancing.
    """
    if isinstance(R, pd.Series):
        r_df = R.to_frame()
    else:
        r_df = R.copy()

    if r_df.isna().any().any():
        import warnings
        warnings.warn("NAs detected: filling NAs with zeros")
        r_df = r_df.fillna(0)

    n_obs, n_cols = r_df.shape
    r_idx = r_df.index

    if weights is None:
        w_arr = np.ones(n_cols) / n_cols
    else:
        w_arr = np.array(weights).flatten()
        if len(w_arr) != n_cols:
            raise ValueError(f"weights must have {n_cols} elements")

    ret_arr = r_df.values
    out_ret = np.zeros(n_obs)

    # Identify rebalance endpoints
    if rebalance_on == "none":
        endpoints = [0, n_obs]
    else:
        freq_map = {
            "years": "YE",
            "quarters": "QE",
            "months": "ME",
            "weeks": "W",
            "days": "D"
        }
        resamp_rule = freq_map.get(rebalance_on)
        if resamp_rule is None:
            raise ValueError(f"unsupported rebalance_on: {rebalance_on}")

        grp = r_df.groupby(pd.Grouper(freq=resamp_rule))

        endpoints = []
        for name, g in grp:
            if not g.empty:
                start_idx = r_df.index.get_loc(g.index[0])
                endpoints.append(start_idx)
        endpoints.append(n_obs)
        if endpoints[0] != 0:
            endpoints.insert(0, 0)

    endpoints = sorted(list(set(endpoints)))

    if geometric:
        end_value = 1.0
        bop_value = np.zeros(n_cols)
        eop_value = np.zeros(n_cols)

        for i in range(len(endpoints) - 1):
            start = endpoints[i]
            end = endpoints[i+1]
            if start == end: continue

            for j in range(start, end):
                if j == start:
                    bop_value = end_value * w_arr
                else:
                    bop_value = eop_value.copy()

                eop_value = bop_value * (1 + ret_arr[j, :])
                eop_value_total = np.sum(eop_value)

                out_ret[j] = eop_value_total / end_value - 1
                end_value = eop_value_total

    else:
        bop_weights = np.zeros(n_cols)
        eop_weights = np.zeros(n_cols)

        for i in range(len(endpoints) - 1):
            start = endpoints[i]
            end = endpoints[i+1]
            if start == end: continue

            for j in range(start, end):
                if j == start:
                    bop_weights = w_arr.copy()
                else:
                    bop_weights = eop_weights.copy()

                period_contrib = ret_arr[j, :] * bop_weights

                eop_weights = (period_contrib + bop_weights) / np.sum(period_contrib + bop_weights)
                out_ret[j] = np.sum(period_contrib)

    res = pd.Series(out_ret, index=r_idx, name="portfolio.returns")
    return res

