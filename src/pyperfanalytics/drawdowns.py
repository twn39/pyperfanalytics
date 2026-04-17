import numpy as np
import pandas as pd


def drawdowns(R: pd.Series | pd.DataFrame, geometric: bool = True) -> pd.Series | pd.DataFrame:
    r"""
    Calculate the drawdown levels in a timeseries.

    Drawdown at time :math:`t` is calculated as the percentage drop from the highest cumulative return peak
    observed up to time :math:`t`. Any time the cumulative return drops below the maximum cumulative return,
    it's a drawdown. Drawdowns are measured as a percentage of that maximum cumulative return.

    Formula:

    .. math::

        D_t = \frac{C_t}{\max_{j=1}^t (C_j)} - 1
    where :math:`C_t` is the cumulative return at time :math:`t`.

    Parameters
    ----------
    R : pd.Series or pd.DataFrame
        Asset returns.
    geometric : bool, optional
        If True, calculates geometric cumulative returns (compound).
        If False, calculates arithmetic cumulative returns (simple sum). Default is True.

    Returns
    -------
    pd.Series or pd.DataFrame
        A timeseries of drawdown percentages (<= 0), with the same shape as `R`.
    r"""

    def _calc(s: pd.Series, geom: bool) -> pd.Series:
        s = s.dropna()
        if geom:
            cumulative = (1 + s).cumprod()
        else:
            cumulative = 1 + s.cumsum()

        # In PA, maxCumulativeReturn = cummax(c(1, Return.cumulative))[-1]
        # This handles the case where the series starts with a drawdown
        hwm = cumulative.expanding().max()
        # If the first value is less than 1, the HWM should be 1
        hwm = np.maximum(hwm, 1.0)

        return cumulative / hwm - 1

    if isinstance(R, pd.DataFrame):
        return R.apply(_calc, geom=geometric)
    else:
        return _calc(R, geometric)


def max_drawdown(R: pd.Series | pd.DataFrame, geometric: bool = True) -> float | pd.Series:
    r"""
    Calculate the maximum drawdown of a return series.

    Maximum drawdown is the absolute value of the worst-case continuous loss
    from a peak to a trough. It is the largest drop in a portfolio's equity curve.

    Formula:

    .. math::

        MaxDD = \max(|D_t|)
    where :math:`D_t` is the drawdown at time :math:`t`.

    Parameters
    ----------
    R : pd.Series or pd.DataFrame
        Asset returns.
    geometric : bool, optional
        If True, calculates geometric drawdowns. Default is True.

    Returns
    -------
    float or pd.Series
        The absolute value of the maximum drawdown (positive number).
    r"""
    dd = drawdowns(R, geometric=geometric)
    return np.abs(dd.min())


def find_drawdowns(R: pd.Series | pd.DataFrame, geometric: bool = True) -> dict[str, np.ndarray]:
    r"""
    Find drawdowns in a return series.

    Finds all the drawdowns in a timeseries and gives a structured output of
    their characteristics: the depth, the start period, the trough period,
    the end period, the length (in periods), the peak-to-trough length, and the recovery length.

    Parameters
    ----------
    R : pd.Series or pd.DataFrame
        Asset returns. Only the first column is processed if a DataFrame is provided.
    geometric : bool, optional
        If True, calculates compound drawdowns. Default is True.

    Returns
    -------
    Dict[str, np.ndarray]
        A dictionary containing numpy arrays for:

        - 'return': depth of the drawdown
        - 'from': start index (1-indexed)
        - 'trough': trough index (1-indexed)
        - 'to': end/recovery index (1-indexed)
        - 'length': total length from peak to full recovery
        - 'peaktotrough': length from peak to trough
        - 'recovery': length from trough to full recovery

    Notes
    -----
    - Output indices are 1-indexed to strictly match the behavior of R's PerformanceAnalytics.
    r"""
    if isinstance(R, pd.DataFrame):
        # Implementation for multiple columns should probably return a list of dicts or handle it differently
        # PerformanceAnalytics findDrawdowns seems to handle one column at a time
        dd_series = drawdowns(R.iloc[:, 0], geometric=geometric).dropna()
    else:
        dd_series = drawdowns(R, geometric=geometric).dropna()

    dd_vals = dd_series.values
    n = len(dd_vals)

    draw = []
    begin = []
    end = []
    trough = []

    if n == 0:
        return {
            "return": np.array([]),
            "from": np.array([]),
            "trough": np.array([]),
            "to": np.array([]),
            "length": np.array([]),
            "peaktotrough": np.array([]),
            "recovery": np.array([]),
        }

    # Initialize like R's findDrawdowns.R
    if dd_vals[0] >= 0:
        prior_sign = 1
    else:
        prior_sign = 0

    from_idx = 1
    sofar = float(dd_vals[0])
    to_idx = 1
    dmin = 1

    # R loop: for (i in 1:length(drawdowns))
    for i_1 in range(1, n + 1):
        i_0 = i_1 - 1
        val = float(dd_vals[i_0])
        this_sign = 0 if val < 0 else 1

        if this_sign == prior_sign:
            if val < sofar:
                sofar = val
                dmin = i_1
            to_idx = i_1 + 1
        else:
            # Save previous
            draw.append(sofar)
            begin.append(from_idx)
            trough.append(dmin)
            end.append(to_idx)

            # Start new
            from_idx = i_1
            sofar = val
            to_idx = i_1 + 1
            dmin = i_1
            prior_sign = this_sign

    # Save last
    draw.append(sofar)
    begin.append(from_idx)
    trough.append(dmin)
    end.append(to_idx)

    draw = np.array(draw)
    begin = np.array(begin)
    trough = np.array(trough)
    end = np.array(end)

    # R's indices are 1-based.
    # length = (end - begin + 1)
    # peaktotrough = (trough - begin + 1)
    # recovery = (end - trough)

    lengths = end - begin + 1
    peaktotroughs = trough - begin + 1
    recoveries = end - trough

    return {
        "return": draw,
        "from": begin,
        "trough": trough,
        "to": end,
        "length": lengths,
        "peaktotrough": peaktotroughs,
        "recovery": recoveries,
    }


def average_drawdown(R: pd.Series | pd.DataFrame, geometric: bool = True) -> float | pd.Series:
    r"""
    Calculate the average depth of the observed drawdowns.

    Formula:

    .. math::

        AvgDrawdown = \frac{1}{d} \sum_{j=1}^d |D_j|
    where :math:`D_j` represents the depth of the :math:`j`-th drawdown and :math:`d` is the number of drawdowns.

    Parameters
    ----------
    R : pd.Series or pd.DataFrame
        Asset returns.
    geometric : bool, optional
        If True, calculates geometric drawdowns. Default is True.

    Returns
    -------
    float or pd.Series
        The average drawdown depth as a positive float.
    r"""

    def _calc(s: pd.Series) -> float:
        dd_info = find_drawdowns(s, geometric=geometric)
        returns = dd_info["return"]
        returns = returns[returns < 0]
        if len(returns) == 0:
            return 0.0
        return float(np.abs(np.mean(returns)))

    if isinstance(R, pd.DataFrame):
        return R.apply(_calc)
    else:
        return _calc(R)


def average_length(R: pd.Series | pd.DataFrame, geometric: bool = True) -> float | pd.Series:
    r"""
    Calculate the average length (in periods) of the observed drawdowns.

    Formula:

    .. math::

        AvgLength = \frac{1}{d} \sum_{j=1}^d L_j
    where :math:`L_j` is the total length of the :math:`j`-th drawdown (from peak to recovery),
    and :math:`d` is the number of drawdowns.

    Parameters
    ----------
    R : pd.Series or pd.DataFrame
        Asset returns.
    geometric : bool, optional
        If True, calculate drawdowns geometrically. Default is True.

    Returns
    -------
    float or pd.Series
        Average length of the drawdowns.
    r"""

    def _calc(s: pd.Series) -> float:
        dd_info = find_drawdowns(s, geometric=geometric)
        mask = dd_info["return"] < 0
        lengths = dd_info["length"][mask]
        if len(lengths) == 0:
            return 0.0
        return float(np.mean(lengths))

    if isinstance(R, pd.DataFrame):
        return R.apply(_calc)
    else:
        return _calc(R)


def average_recovery(R: pd.Series | pd.DataFrame, geometric: bool = True) -> float | pd.Series:
    r"""
    Calculate the average length (in periods) of the observed recovery period.

    Formula:

    .. math::

        AvgRecovery = \frac{1}{d} \sum_{j=1}^d R_j
    where :math:`R_j` is the recovery length of the :math:`j`-th drawdown (from trough to recovery).

    Parameters
    ----------
    R : pd.Series or pd.DataFrame
        Asset returns.
    geometric : bool, optional
        If True, calculate drawdowns geometrically. Default is True.

    Returns
    -------
    float or pd.Series
        Average recovery length.
    r"""

    def _calc(s: pd.Series) -> float:
        dd_info = find_drawdowns(s, geometric=geometric)
        mask = dd_info["return"] < 0
        recoveries = dd_info["recovery"][mask]
        if len(recoveries) == 0:
            return 0.0
        return float(np.mean(recoveries))

    if isinstance(R, pd.DataFrame):
        return R.apply(_calc)
    else:
        return _calc(R)


def drawdown_deviation(R: pd.Series | pd.DataFrame, geometric: bool = True) -> float | pd.Series:
    r"""
    Calculate a standard deviation-type statistic using individual drawdowns.

    Drawdown deviation models the dispersion of drawdown depths.

    Formula:

    .. math::

        DD = \sqrt{\frac{1}{n} \sum_{j=1}^d D_j^2}
    where :math:`D_j` is the depth of the :math:`j`-th drawdown, :math:`d` is the number of drawdowns,
    and :math:`n` is the total number of periods in the series.

    Parameters
    ----------
    R : pd.Series or pd.DataFrame
        Asset returns.
    geometric : bool, optional
        If True, calculates geometric drawdowns. Default is True.

    Returns
    -------
    float or pd.Series
        The drawdown deviation.
    r"""

    def _calc(s: pd.Series) -> float:
        s = s.dropna()
        n = len(s)
        if n == 0:
            return 0.0
        dd_info = find_drawdowns(s, geometric=geometric)
        returns = dd_info["return"]
        returns = returns[returns < 0]
        if len(returns) == 0:
            return 0.0
        return float(np.sqrt(np.sum(returns**2) / n))

    if isinstance(R, pd.DataFrame):
        return R.apply(_calc)
    else:
        return _calc(R)


def sort_drawdowns(runs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    r"""
    Sort drawdowns from worst to best.
    r"""
    if len(runs["return"]) == 0:
        return runs
    idx = np.argsort(runs["return"])
    return {k: v[idx] for k, v in runs.items()}


def drawdown_peak(R: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    r"""
    Replicate R's DrawdownPeak logic.
    This function tracks drawdown since the last peak.
    It is used by UlcerIndex and PainIndex in PerformanceAnalytics.

    Implementation follows R's vectorized accumulation to ensure 
    exact matching with continuous compounding drawdowns.
    r"""

    def _calc(s: pd.Series) -> pd.Series:
        s = s.dropna()
        n = len(s)
        if n == 0:
            return pd.Series([], dtype=float)

        # R's DrawdownPeak logic:
        # cumm.x <- cumprod(1 + x)
        # max.cumm.x <- cummax(c(1, cumm.x))[-1]
        # drawdown <- cumm.x / max.cumm.x - 1
        cumm_x = (1.0 + s).cumprod()
        max_cumm_x = np.maximum.accumulate(np.insert(cumm_x.values, 0, 1.0))[1:]
        res = (cumm_x / max_cumm_x) - 1.0

        return pd.Series(res, index=s.index)

    if isinstance(R, pd.DataFrame):
        return R.apply(_calc)
    else:
        return _calc(R)

def cdd(
    R: pd.Series | pd.DataFrame, p: float = 0.95, geometric: bool = True, invert: bool = True, method: str = "discrete"
) -> float | pd.Series | pd.DataFrame:
    r"""
    Calculate Uryasev's proposed Conditional Drawdown at Risk (CDD or CDaR) measure.

    For some confidence level :math:`p`, the conditional drawdown is the mean of the worst :math:`p\%` drawdowns.
    It is a modification of the Expected Shortfall (ES) applied to drawdowns instead of returns.

    Formula:

    .. math::

        CDD = \text{Quantile}(D, 1-p)

    Parameters
    ----------
    R : pd.Series or pd.DataFrame
        Asset returns.
    p : float, optional
        Confidence level for calculation, default is 0.95.
    geometric : bool, optional
        Use geometric compounding. Default is True.
    invert : bool, optional
        If True, inverts the sign to present risk as a positive number (like R).
    method : str, optional
        The calculation method: "discrete" (default) or "average" or "quantile".

    Returns
    -------
    float or pd.Series
        The CDD risk value.
    r"""

    def _calc(s: pd.Series, prob: float, meth: str) -> float:
        s = s.dropna()
        if len(s) == 0:
            return np.nan
        # R's .setalphaprob(p) converts p>0.5 to 1-p. So p=0.95 becomes 0.05.
        if prob > 0.5:
            prob = 1 - prob

        if meth == "quantile":
            dd_info = find_drawdowns(s, geometric=geometric)
            returns = dd_info["return"]
            if len(returns) == 0:
                return 0.0
            val = np.quantile(returns, prob)
        elif meth == "discrete":
            dd_info = find_drawdowns(s, geometric=geometric)
            returns = dd_info["return"]
            if len(returns) == 0:
                return 0.0
            q_val = np.quantile(returns, prob)
            # mean of drawdowns <= q_val
            returns_exceed = returns[returns <= q_val]
            val = returns_exceed.mean() if len(returns_exceed) > 0 else np.nan
        elif meth == "average":
            dd = drawdown_peak(s) if geometric else s.cumsum() - s.cumsum().cummax()
            # To be strictly identical to PA's geometric vs non-geometric
            # R does `Drawdowns(R, geometric=geometric)`. `Drawdowns` uses cumprod vs cumsum.
            # We'll use our drawdowns function directly on the series.
            from pyperfanalytics.drawdowns import drawdowns as compute_dd
            dd = compute_dd(s, geometric=geometric).dropna()
            if len(dd) == 0:
                return 0.0
            # R uses type=8 quantile for average
            # numpy's closest to R type 8 is 'median_unbiased'
            q_val = np.quantile(dd, prob, method="median_unbiased")
            val = dd[dd <= q_val].mean()
        else:
            raise ValueError(f"Unknown CDD method: {meth}")

        if invert:
            return float(-val)
        return float(val)

    if isinstance(R, pd.DataFrame):
        return R.apply(_calc, prob=p, meth=method)
    else:
        return _calc(R, p, method)
