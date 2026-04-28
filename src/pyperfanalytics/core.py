"""
pyperfanalytics.core
====================

Foundational return and risk statistics with zero intra-package dependencies
(beyond :mod:`pyperfanalytics.utils`).

All symbols are re-exported from :mod:`pyperfanalytics.returns` for backward
compatibility, so existing code using::

    from pyperfanalytics.returns import return_annualized

continues to work without any changes.

Functions
---------
return_calculate, return_excess, return_annualized, return_cumulative,
std_dev_annualized, downside_deviation, downside_potential, semi_deviation,
semi_variance, gain_deviation, loss_deviation, mean_absolute_deviation,
upside_potential, upside_risk, downside_frequency, volatility_skewness
"""

import numpy as np
import pandas as pd

from pyperfanalytics.utils import _get_scale


# ---------------------------------------------------------------------------
# Price → Returns
# ---------------------------------------------------------------------------


def return_calculate(prices: pd.Series | pd.DataFrame, method: str = "discrete") -> pd.Series | pd.DataFrame:
    r"""
    Calculate returns from a price stream.

    Determines the period-over-period returns based on a pricing series, supporting
    both discrete (simple) and continuous (log) methods.

    Formula:

    - discrete: :math:`R_t = \frac{P_t}{P_{t-1}} - 1`
    - continuous: :math:`r_t = \ln(P_t) - \ln(P_{t-1})`
    - difference: :math:`D_t = P_t - P_{t-1}`

    Parameters
    ----------
    prices : pd.Series or pd.DataFrame
        Price levels of assets.
    method : str, optional
        Type of return calculation: ``"discrete"`` (default), ``"log"``, or
        ``"diff"``.

    Returns
    -------
    pd.Series or pd.DataFrame
        Returns series.
    r"""
    if method in ["discrete", "simple", "arithmetic"]:
        return prices.pct_change()
    elif method in ["log", "compound", "continuous"]:
        return np.log(prices.astype(float)).diff()  # type: ignore[return-value]
    elif method in ["diff", "difference"]:
        return prices.diff()
    else:
        raise ValueError(f"Unknown method: {method!r}")


# ---------------------------------------------------------------------------
# Excess returns (Rf subtraction)
# ---------------------------------------------------------------------------


def return_excess(
    R: pd.Series | pd.DataFrame,
    Rf: float | pd.Series | pd.DataFrame = 0.0,
) -> pd.Series | pd.DataFrame:
    r"""
    Calculate excess returns by subtracting the risk-free rate.

    When *Rf* is a :class:`~pandas.Series` or :class:`~pandas.DataFrame` it is
    forward-filled and aligned to *R*'s index before subtraction.  Both
    single- and multi-column *Rf* inputs are handled safely even when *Rf*
    spans a wider date range than *R*.
    r"""
    if isinstance(Rf, (pd.Series, pd.DataFrame)):
        # Merge to get a common index; Rf dates outside R are ignored after alignment.
        combined = pd.concat([R, Rf], axis=1, sort=False)
        rf_cols = Rf.columns if isinstance(Rf, pd.DataFrame) else [Rf.name]
        # Forward-fill Rf gaps (matches R's na.locf behaviour)
        combined[rf_cols] = combined[rf_cols].ffill()

        if len(rf_cols) == 1:
            # Single Rf column: pandas aligns by index automatically
            res = R.sub(combined[rf_cols[0]], axis=0)
        else:
            # Multiple Rf columns: align to R's index BEFORE converting to
            # a plain NumPy array so row counts always match, even when Rf
            # extends beyond R's date range.
            rf_aligned = combined.loc[R.index, rf_cols]
            if isinstance(R, pd.DataFrame):
                res = R.sub(rf_aligned.values, axis=0)
            else:
                res = R - rf_aligned.iloc[:, 0]

        # Return only rows present in the original R
        return res.loc[R.index] if isinstance(R, pd.DataFrame) else res[R.index]

    return R - Rf


# ---------------------------------------------------------------------------
# Annualisation
# ---------------------------------------------------------------------------


def return_annualized(
    R: pd.Series | pd.DataFrame,
    scale: int | None = None,
    geometric: bool = True,
) -> float | pd.Series:
    r"""
    Calculate annualized return.

    Aggregates period returns into an annualized equivalent, assuming continuous
    compounding (geometric) or simple arithmetic averaging.

    Formula (Geometric):

    .. math::

        R_{ann} = \left[ \prod_{i=1}^n (1+R_i) \right]^{\frac{scale}{n}} - 1

    Parameters
    ----------
    R : pd.Series or pd.DataFrame
        Asset returns.
    scale : int, optional
        Number of periods in a year.  Inferred from the index frequency when
        not supplied.
    geometric : bool, optional
        Whether to compound returns geometrically.  Default is ``True``.

    Returns
    -------
    float or pd.Series
        Annualized return.
    r"""
    if scale is None:
        scale = _get_scale(R)

    def _calc(s: pd.Series, sc: int, geom: bool) -> float:
        s = s.dropna()
        n = len(s)
        if n == 0:
            return np.nan
        if geom:
            return (1 + s).prod() ** (float(sc) / n) - 1
        else:
            return s.mean() * sc

    if isinstance(R, pd.DataFrame):
        return R.apply(_calc, sc=scale, geom=geometric)
    else:
        return _calc(R, scale, geometric)


def return_cumulative(R: pd.Series | pd.DataFrame, geometric: bool = True) -> float | pd.Series:
    r"""
    Calculate the total cumulative return.

    Parameters
    ----------
    R : pd.Series or pd.DataFrame
        Asset returns.
    geometric : bool, optional
        Use geometric compounding.  Default is ``True``.

    Returns
    -------
    float or pd.Series
        Cumulative return.
    r"""
    def _calc(s: pd.Series, geom: bool) -> float:
        s = s.dropna()
        if len(s) == 0:
            return np.nan
        if geom:
            return (1 + s).prod() - 1
        else:
            return s.sum()

    if isinstance(R, pd.DataFrame):
        return R.apply(_calc, geom=geometric)
    else:
        return _calc(R, geometric)


# ---------------------------------------------------------------------------
# Standard deviation
# ---------------------------------------------------------------------------


def std_dev_annualized(R: pd.Series | pd.DataFrame, scale: int | None = None) -> float | pd.Series:
    r"""
    Calculate annualized standard deviation.

    Uses a sample standard deviation (``ddof=1``) to match R's ``sd()``
    default, then multiplies by :math:`\sqrt{scale}`.

    Parameters
    ----------
    R : pd.Series or pd.DataFrame
        Asset returns.
    scale : int, optional
        Periods per year.

    Returns
    -------
    float or pd.Series
        Annualized standard deviation.
    r"""
    if scale is None:
        scale = _get_scale(R)
    if isinstance(R, pd.DataFrame):
        return R.std(ddof=1) * np.sqrt(scale)
    else:
        return R.dropna().std(ddof=1) * np.sqrt(scale)


# ---------------------------------------------------------------------------
# Downside / Upside statistics
# ---------------------------------------------------------------------------


def downside_deviation(
    R: pd.Series | pd.DataFrame,
    MAR: float = 0.0,
    method: str = "full",
    potential: bool = False,
) -> float | pd.Series:
    r"""
    Calculate downside deviation (or downside potential).

    Downside deviation measures the volatility of returns below a minimum
    acceptable return (*MAR*).  Unlike standard deviation, it only penalises
    losses.

    Formula:

    .. math::

        \delta_{MAR} = \sqrt{\frac{1}{n} \sum_{t=1}^n \min(R_t - MAR,\, 0)^2}

    Parameters
    ----------
    R : pd.Series or pd.DataFrame
        Asset returns.
    MAR : float, optional
        Minimum acceptable return.  Default is 0.0.
    method : str, optional
        ``"full"`` divides by total *n*; ``"subset"`` divides only by the
        number of downside periods.
    potential : bool, optional
        If ``True``, returns the first lower partial moment (downside
        potential) instead of the square-root.

    Returns
    -------
    float or pd.Series
        Downside deviation (or potential).
    r"""
    def _calc(s: pd.Series, mar: float, meth: str, pot: bool) -> float:
        s = s.dropna()
        under = s[s < mar]
        n = len(s) if meth == "full" else len(under)
        if n == 0:
            return 0.0
        diff = mar - under
        return diff.sum() / n if pot else np.sqrt((diff**2).sum() / n)

    if isinstance(R, pd.DataFrame):
        return R.apply(_calc, mar=MAR, meth=method, pot=potential)
    else:
        return _calc(R, MAR, method, potential)


def downside_potential(R: pd.Series | pd.DataFrame, MAR: float = 0.0) -> float | pd.Series:
    r"""
    Calculate downside potential (first lower partial moment).

    Equivalent to calling :func:`downside_deviation` with ``potential=True``.
    r"""
    return downside_deviation(R, MAR=MAR, method="full", potential=True)


def upside_risk(
    R: pd.Series | pd.DataFrame,
    MAR: float = 0.0,
    method: str = "full",
    stat: str = "risk",
) -> float | pd.Series:
    r"""
    Calculate upside risk, variance, or potential.

    Depending on *stat*, measures the variability or sum of returns above a
    specified Minimum Acceptable Return (*MAR*).

    Formula (Variance):

    .. math::

        UV = \frac{1}{n} \sum_{R > MAR} (R - MAR)^2

    Parameters
    ----------
    R : pd.Series or pd.DataFrame
        Asset returns.
    MAR : float, optional
        Minimum Acceptable Return.  Default is 0.0.
    method : str, optional
        ``"full"`` or ``"subset"``.  Default is ``"full"``.
    stat : str, optional
        ``"risk"`` (square-root), ``"variance"``, or ``"potential"`` (sum).
        Default is ``"risk"``.

    Returns
    -------
    float or pd.Series
        Upside risk, variance, or potential.
    r"""
    def _calc(s: pd.Series, mar: float, meth: str, st: str) -> float:
        s = s.dropna()
        above = s[s > mar]
        n = len(s) if meth == "full" else len(above)
        if n == 0:
            return 0.0
        diff = above - mar
        if st == "risk":
            return np.sqrt((diff**2).sum() / n)
        elif st == "variance":
            return (diff**2).sum() / n
        elif st == "potential":
            return diff.sum() / n
        else:
            raise ValueError(f"Unknown stat: {st!r}")

    if isinstance(R, pd.DataFrame):
        return R.apply(_calc, mar=MAR, meth=method, st=stat)
    else:
        return _calc(R, MAR, method, stat)


def upside_potential(R: pd.Series | pd.DataFrame, MAR: float = 0.0) -> float | pd.Series:
    r"""Return the upside potential (first upper partial moment above *MAR*)."""
    return upside_risk(R, MAR=MAR, method="full", stat="potential")


# ---------------------------------------------------------------------------
# Semi-deviation family
# ---------------------------------------------------------------------------


def semi_deviation(R: pd.Series | pd.DataFrame) -> float | pd.Series:
    r"""
    Calculate semi-deviation.

    Semi-deviation is the downside deviation where *MAR* equals the mean
    return.  It corresponds to the square root of the second lower partial
    moment evaluated at the mean.
    r"""
    if isinstance(R, pd.DataFrame):
        return R.apply(lambda x: downside_deviation(x, MAR=x.mean(), method="full"))
    else:
        return downside_deviation(R, MAR=R.mean(), method="full")


def semi_variance(R: pd.Series | pd.DataFrame) -> float | pd.Series:
    r"""
    Calculate semi-variance (MAR = mean, method = subset).
    r"""
    if isinstance(R, pd.DataFrame):
        return R.apply(lambda x: downside_deviation(x, MAR=x.mean(), method="subset") ** 2)
    else:
        return downside_deviation(R, MAR=R.mean(), method="subset") ** 2


# ---------------------------------------------------------------------------
# Gain / loss deviation, MAD, frequency
# ---------------------------------------------------------------------------


def gain_deviation(R: pd.Series | pd.DataFrame) -> float | pd.Series:
    r"""Standard deviation of positive-return periods (sample, ``ddof=1``)."""

    def _calc(s: pd.Series) -> float:
        subset = s.dropna()
        subset = subset[subset > 0]
        if len(subset) < 2:
            return np.nan
        return subset.std(ddof=1)

    return R.apply(_calc) if isinstance(R, pd.DataFrame) else _calc(R)


def loss_deviation(R: pd.Series | pd.DataFrame) -> float | pd.Series:
    r"""Standard deviation of negative-return periods (sample, ``ddof=1``)."""

    def _calc(s: pd.Series) -> float:
        subset = s.dropna()
        subset = subset[subset < 0]
        if len(subset) < 2:
            return np.nan
        return subset.std(ddof=1)

    return R.apply(_calc) if isinstance(R, pd.DataFrame) else _calc(R)


def mean_absolute_deviation(R: pd.Series | pd.DataFrame) -> float | pd.Series:
    r"""
    Calculate Mean Absolute Deviation (MAD).

    .. math::

        MAD = \frac{1}{n} \sum_{t=1}^n |R_t - \bar{R}|

    Matches R's ``MeanAbsoluteDeviation``.
    r"""
    def _calc(s: pd.Series) -> float:
        s = s.dropna()
        if len(s) == 0:
            return np.nan
        return (s - s.mean()).abs().mean()

    return R.apply(_calc) if isinstance(R, pd.DataFrame) else _calc(R)


def downside_frequency(R: pd.Series | pd.DataFrame, MAR: float = 0.0) -> float | pd.Series:
    r"""
    Calculate the frequency of returns falling below *MAR*.

    .. math::

        f_{down} = \frac{\#\{R_t < MAR\}}{n}
    r"""
    def _calc(s: pd.Series, mar: float) -> float:
        s = s.dropna()
        if len(s) == 0:
            return np.nan
        return (s < mar).sum() / len(s)

    if isinstance(R, pd.DataFrame):
        return R.apply(_calc, mar=MAR)
    else:
        return _calc(R, MAR)


# ---------------------------------------------------------------------------
# Volatility skewness
# ---------------------------------------------------------------------------


def volatility_skewness(
    R: pd.Series | pd.DataFrame,
    MAR: float = 0.0,
    stat: str = "volatility",
) -> float | pd.Series:
    r"""
    Calculate Volatility Skewness.

    The ratio of upside risk to downside deviation, optionally using variance
    or potential variants.

    Parameters
    ----------
    R : pd.Series or pd.DataFrame
        Asset returns.
    MAR : float, optional
        Minimum Acceptable Return.  Default is 0.0.
    stat : str, optional
        ``"volatility"`` (risk/risk), ``"variance"`` (var/var), or
        ``"potential"`` (potential/potential).  Default is ``"volatility"``.

    Returns
    -------
    float or pd.Series
        Volatility skewness ratio.
    r"""
    def _calc(s: pd.Series, mar: float, st: str) -> float:
        s = s.dropna()
        if st == "volatility":
            up = upside_risk(s, MAR=mar, method="full", stat="risk")
            dn = downside_deviation(s, MAR=mar, method="full")
        elif st == "variance":
            up = upside_risk(s, MAR=mar, method="full", stat="variance")
            dn = downside_deviation(s, MAR=mar, method="full") ** 2
        elif st == "potential":
            up = upside_risk(s, MAR=mar, method="full", stat="potential")
            dn = downside_deviation(s, MAR=mar, method="full", potential=True)
        else:
            raise ValueError(f"Unknown stat: {st!r}")
        return float(up / dn) if dn != 0 else np.nan

    if isinstance(R, pd.DataFrame):
        return R.apply(_calc, mar=MAR, st=stat)
    else:
        return _calc(R, MAR, stat)
