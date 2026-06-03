from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from pyperfanalytics.utils import (
    _get_scale,
    centered_moment,
    co_kurtosis_matrix,
    co_skewness_matrix,
    kurtosis,
    skewness,
)


def _verify_portfolio_args(R, weights, portfolio_method):
    if portfolio_method not in ["single", "component", "marginal"]:
        raise ValueError("portfolio_method must be one of 'single', 'component', 'marginal'")
    if weights is not None:
        w = np.array(weights, dtype=float)
        if isinstance(R, pd.DataFrame):
            if len(w) != R.shape[1]:
                raise ValueError("Length of weights must match the number of columns in R.")
        return w
    return None


def var_historical(
    R: pd.Series | pd.DataFrame,
    p: float = 0.95,
    weights: np.ndarray | list | pd.Series | None = None,
    portfolio_method: str = "single",
) -> Any:
    r"""
    Calculate Historical Value at Risk (VaR), with optional portfolio decomposition.
    """
    w = _verify_portfolio_args(R, weights, portfolio_method)
    alpha = 1 - p if p >= 0.5 else p

    if weights is None or portfolio_method == "single":
        if weights is not None and isinstance(R, pd.DataFrame):
            Rp = R @ w
            return -np.percentile(Rp.dropna(), alpha * 100)
        else:
            def _calc(s: pd.Series) -> float:
                s = s.dropna()
                return -np.percentile(s, alpha * 100) if len(s) > 0 else np.nan
            return R.apply(_calc) if isinstance(R, pd.DataFrame) else _calc(R)

    if not isinstance(R, pd.DataFrame):
        raise ValueError("R must be a DataFrame for portfolio risk decomposition.")

    Rp = R @ w
    hvar = -np.percentile(Rp.dropna(), alpha * 100)

    neg_mask = Rp < 0
    if not neg_mask.any():
        neg_mask = pd.Series(True, index=Rp.index)

    zl_contrib = (R[neg_mask] * w).mean().values
    total_neg_mean = Rp[neg_mask].mean()
    ratio = hvar / total_neg_mean if total_neg_mean != 0 else 0.0
    contrib = zl_contrib * ratio
    pct_contrib = contrib / contrib.sum() if contrib.sum() != 0 else np.zeros_like(contrib)

    if portfolio_method == "marginal":
        return pd.Series(contrib / w, index=R.columns, name="marginal_hVaR")
    else:
        return {
            "hVaR": hvar,
            "contribution": pd.Series(contrib, index=R.columns),
            "pct_contrib": pd.Series(pct_contrib, index=R.columns),
        }


def var_gaussian(
    R: pd.Series | pd.DataFrame,
    p: float = 0.95,
    weights: np.ndarray | list | pd.Series | None = None,
    portfolio_method: str = "single",
) -> Any:
    r"""
    Calculate Gaussian (Parametric) Value at Risk (VaR), with optional portfolio decomposition.
    """
    w = _verify_portfolio_args(R, weights, portfolio_method)
    alpha = 1 - p if p >= 0.5 else p
    z = norm.ppf(alpha)

    if weights is None or portfolio_method == "single":
        if weights is not None and isinstance(R, pd.DataFrame):
            Rp = R @ w
            return -Rp.mean() - z * Rp.std(ddof=1)
        else:
            mu = R.mean()
            m2 = centered_moment(R, 2)
            return -mu - z * np.sqrt(m2)

    if not isinstance(R, pd.DataFrame):
        raise ValueError("R must be a DataFrame for portfolio risk decomposition.")

    mu = R.mean().values
    location = float(w.T @ mu)
    sigma = R.cov(ddof=1).values * (len(R) - 1) / len(R)
    pm2 = float(w.T @ sigma @ w)
    VaR = -location - z * np.sqrt(pm2)

    dpm2 = 2.0 * sigma @ w
    derVaR = -mu - z * (0.5 * dpm2) / np.sqrt(pm2)
    contrib = w * derVaR
    pct_contrib = contrib / VaR

    if portfolio_method == "marginal":
        return pd.Series(derVaR, index=R.columns, name="marginal_gVaR")
    else:
        return {
            "gVaR": VaR,
            "contribution": pd.Series(contrib, index=R.columns),
            "pct_contrib": pd.Series(pct_contrib, index=R.columns),
        }


def var_modified(
    R: pd.Series | pd.DataFrame,
    p: float = 0.95,
    weights: np.ndarray | list | pd.Series | None = None,
    portfolio_method: str = "single",
    M3: np.ndarray | None = None,
    M4: np.ndarray | None = None,
) -> Any:
    r"""
    Calculate Modified (Cornish-Fisher) Value at Risk (VaR), with optional portfolio decomposition.
    """
    w = _verify_portfolio_args(R, weights, portfolio_method)
    alpha = 1 - p if p >= 0.5 else p
    z = norm.ppf(alpha)

    if weights is None or portfolio_method == "single":
        if weights is not None and isinstance(R, pd.DataFrame):
            Rp = R @ w
            return _var_modified_single_series(Rp, p)
        else:
            def _calc(s: pd.Series) -> float:
                s = s.dropna()
                if len(s) == 0:
                    return np.nan
                mu = s.mean()
                m2 = centered_moment(s, 2)
                skew = skewness(s, method="moment")
                exkurt = kurtosis(s, method="excess")
                h = (
                    z
                    + (z**2 - 1.0) * skew / 6.0
                    + (z**3 - 3.0 * z) * exkurt / 24.0
                    - (2.0 * z**3 - 5.0 * z) * skew**2 / 36.0
                )
                return -mu - h * np.sqrt(m2)
            return R.apply(_calc) if isinstance(R, pd.DataFrame) else _calc(R)

    if not isinstance(R, pd.DataFrame):
        raise ValueError("R must be a DataFrame for portfolio risk decomposition.")

    mu = R.mean().values
    location = float(w.T @ mu)
    T = len(R)

    if M3 is None and M4 is None:
        Xc = (R - R.mean()).values
        y = Xc @ w
        pm2 = float(np.mean(y**2))
        pm3 = float(np.mean(y**3))
        pm4 = float(np.mean(y**4))

        dpm2 = 2.0 * (Xc.T @ y) / T
        dpm3 = 3.0 * (Xc.T @ (y**2)) / T
        dpm4 = 4.0 * (Xc.T @ (y**3)) / T
    else:
        sigma = R.cov(ddof=1).values * (T - 1) / T
        pm2 = float(w.T @ sigma @ w)
        _M3 = M3 if M3 is not None else co_skewness_matrix(R, unbiased=False)
        _M4 = M4 if M4 is not None else co_kurtosis_matrix(R)

        pm3 = float(w.T @ _M3 @ np.kron(w, w))
        pm4 = float(w.T @ _M4 @ np.kron(w, np.kron(w, w)))
        dpm2 = 2.0 * sigma @ w
        dpm3 = 3.0 * _M3 @ np.kron(w, w)
        dpm4 = 4.0 * _M4 @ np.kron(w, np.kron(w, w))

    skew = pm3 / (pm2 ** 1.5) if pm2 > 0 else 0.0
    exkurt = pm4 / (pm2 ** 2) - 3.0 if pm2 > 0 else 0.0

    derskew = (2.0 * (pm2 ** 1.5) * dpm3 - 3.0 * pm3 * np.sqrt(pm2) * dpm2) / (2.0 * pm2**3)
    derexkurt = (pm2 * dpm4 - 2.0 * pm4 * dpm2) / pm2**3

    h = (
        z
        + (z**2 - 1.0) * skew / 6.0
        + (z**3 - 3.0 * z) * exkurt / 24.0
        - (2.0 * z**3 - 5.0 * z) * skew**2 / 36.0
    )
    MVaR = -location - h * np.sqrt(pm2)

    derh = (
        (z**2 - 1.0) * derskew / 6.0
        + (z**3 - 3.0 * z) * derexkurt / 24.0
        - (2.0 * z**3 - 5.0 * z) * skew * derskew / 18.0
    )
    derMVaR = -mu - h * dpm2 / (2.0 * np.sqrt(pm2)) - np.sqrt(pm2) * derh

    contrib = w * derMVaR
    pct_contrib = contrib / MVaR

    if portfolio_method == "marginal":
        return pd.Series(derMVaR, index=R.columns, name="marginal_MVaR")
    else:
        return {
            "MVaR": MVaR,
            "contribution": pd.Series(contrib, index=R.columns),
            "pct_contrib": pd.Series(pct_contrib, index=R.columns),
        }


def _var_modified_single_series(s: pd.Series, p: float) -> float:
    s = s.dropna()
    if len(s) == 0:
        return np.nan
    alpha = 1 - p if p >= 0.5 else p
    z = norm.ppf(alpha)
    mu = s.mean()
    m2 = centered_moment(s, 2)
    skew = skewness(s, method="moment")
    exkurt = kurtosis(s, method="excess")
    h = (
        z
        + (z**2 - 1.0) * skew / 6.0
        + (z**3 - 3.0 * z) * exkurt / 24.0
        - (2.0 * z**3 - 5.0 * z) * skew**2 / 36.0
    )
    return -mu - h * np.sqrt(m2)


def es_historical(
    R: pd.Series | pd.DataFrame,
    p: float = 0.95,
    weights: np.ndarray | list | pd.Series | None = None,
    portfolio_method: str = "single",
) -> Any:
    r"""
    Calculate Historical Expected Shortfall (Conditional VaR), with optional portfolio decomposition.
    """
    w = _verify_portfolio_args(R, weights, portfolio_method)
    alpha = 1 - p if p >= 0.5 else p

    if weights is None or portfolio_method == "single":
        if weights is not None and isinstance(R, pd.DataFrame):
            Rp = R @ w
            return _es_historical_single_series(Rp, p)
        else:
            def _calc(s: pd.Series) -> float:
                s = s.dropna()
                if len(s) == 0:
                    return np.nan
                q = np.percentile(s, alpha * 100)
                subset = s[s < q]
                return -subset.mean() if len(subset) > 0 else -q
            return R.apply(_calc) if isinstance(R, pd.DataFrame) else _calc(R)

    if not isinstance(R, pd.DataFrame):
        raise ValueError("R must be a DataFrame for portfolio risk decomposition.")

    Rp = R @ w
    hvar = -np.percentile(Rp.dropna(), alpha * 100)
    exceed_mask = Rp <= -hvar
    if not exceed_mask.any():
        exceed_mask = pd.Series(True, index=Rp.index)

    c_exceed = exceed_mask.sum()
    r_exceed = Rp[exceed_mask].sum()
    realized_contrib = (R[exceed_mask] * w).sum().values

    contrib = -realized_contrib / c_exceed
    total_ES = -r_exceed / c_exceed
    pct_contrib = contrib / total_ES if total_ES != 0 else np.zeros_like(contrib)

    if portfolio_method == "marginal":
        return pd.Series(contrib / w, index=R.columns, name="marginal_hES")
    else:
        return {
            "hES": total_ES,
            "contribution": pd.Series(contrib, index=R.columns),
            "pct_contrib": pd.Series(pct_contrib, index=R.columns),
        }


def _es_historical_single_series(s: pd.Series, p: float) -> float:
    s = s.dropna()
    if len(s) == 0:
        return np.nan
    alpha = 1 - p if p >= 0.5 else p
    q = np.percentile(s, alpha * 100)
    subset = s[s < q]
    return -subset.mean() if len(subset) > 0 else -q


def es_gaussian(
    R: pd.Series | pd.DataFrame,
    p: float = 0.95,
    weights: np.ndarray | list | pd.Series | None = None,
    portfolio_method: str = "single",
) -> Any:
    r"""
    Calculate Gaussian Expected Shortfall (Conditional VaR), with optional portfolio decomposition.
    """
    w = _verify_portfolio_args(R, weights, portfolio_method)
    alpha = 1 - p if p >= 0.5 else p
    z = norm.ppf(alpha)

    if weights is None or portfolio_method == "single":
        if weights is not None and isinstance(R, pd.DataFrame):
            Rp = R @ w
            return -Rp.mean() + norm.pdf(z) * Rp.std(ddof=1) / alpha
        else:
            mu = R.mean()
            m2 = centered_moment(R, 2)
            return -mu + norm.pdf(z) * np.sqrt(m2) / alpha

    if not isinstance(R, pd.DataFrame):
        raise ValueError("R must be a DataFrame for portfolio risk decomposition.")

    mu = R.mean().values
    location = float(w.T @ mu)
    sigma = R.cov(ddof=1).values * (len(R) - 1) / len(R)
    pm2 = float(w.T @ sigma @ w)
    ES = -location + norm.pdf(z) * np.sqrt(pm2) / alpha

    dpm2 = 2.0 * sigma @ w
    derES = -mu + (1.0 / alpha) * norm.pdf(z) * (0.5 * dpm2) / np.sqrt(pm2)
    contrib = w * derES
    pct_contrib = contrib / ES

    if portfolio_method == "marginal":
        return pd.Series(derES, index=R.columns, name="marginal_gES")
    else:
        return {
            "gES": ES,
            "contribution": pd.Series(contrib, index=R.columns),
            "pct_contrib": pd.Series(pct_contrib, index=R.columns),
        }


def es_modified(
    R: pd.Series | pd.DataFrame,
    p: float = 0.95,
    weights: np.ndarray | list | pd.Series | None = None,
    portfolio_method: str = "single",
    M3: np.ndarray | None = None,
    M4: np.ndarray | None = None,
) -> Any:
    r"""
    Calculate Modified (Cornish-Fisher) Expected Shortfall, with optional portfolio decomposition.
    """
    w = _verify_portfolio_args(R, weights, portfolio_method)
    alpha = 1 - p if p >= 0.5 else p
    z = norm.ppf(alpha)

    if weights is None or portfolio_method == "single":
        if weights is not None and isinstance(R, pd.DataFrame):
            Rp = R @ w
            return _es_modified_single_series(Rp, p)
        else:
            def _calc(s: pd.Series) -> float:
                s = s.dropna()
                if len(s) == 0:
                    return np.nan
                mu = s.mean()
                m2 = centered_moment(s, 2)
                skew = skewness(s, method="moment")
                exkurt = kurtosis(s, method="excess")
                h = (
                    z
                    + (z**2 - 1.0) * skew / 6.0
                    + (z**3 - 3.0 * z) * exkurt / 24.0
                    - (2.0 * z**3 - 5.0 * z) * skew**2 / 36.0
                )
                E = (
                    norm.pdf(h)
                    * (
                        1.0
                        + (h**3) * skew / 6.0
                        + (h**6 - 9.0 * h**4 + 9.0 * h**2 + 3.0) * skew**2 / 72.0
                        + (h**4 - 2.0 * h**2 - 1.0) * exkurt / 24.0
                    )
                    / alpha
                )
                return -mu + np.sqrt(m2) * E
            return R.apply(_calc) if isinstance(R, pd.DataFrame) else _calc(R)

    if not isinstance(R, pd.DataFrame):
        raise ValueError("R must be a DataFrame for portfolio risk decomposition.")

    mu = R.mean().values
    location = float(w.T @ mu)
    T = len(R)

    if M3 is None and M4 is None:
        Xc = (R - R.mean()).values
        y = Xc @ w
        pm2 = float(np.mean(y**2))
        pm3 = float(np.mean(y**3))
        pm4 = float(np.mean(y**4))

        dpm2 = 2.0 * (Xc.T @ y) / T
        dpm3 = 3.0 * (Xc.T @ (y**2)) / T
        dpm4 = 4.0 * (Xc.T @ (y**3)) / T
    else:
        sigma = R.cov(ddof=1).values * (T - 1) / T
        pm2 = float(w.T @ sigma @ w)
        _M3 = M3 if M3 is not None else co_skewness_matrix(R, unbiased=False)
        _M4 = M4 if M4 is not None else co_kurtosis_matrix(R)

        pm3 = float(w.T @ _M3 @ np.kron(w, w))
        pm4 = float(w.T @ _M4 @ np.kron(w, np.kron(w, w)))
        dpm2 = 2.0 * sigma @ w
        dpm3 = 3.0 * _M3 @ np.kron(w, w)
        dpm4 = 4.0 * _M4 @ np.kron(w, np.kron(w, w))

    skew = pm3 / (pm2 ** 1.5) if pm2 > 0 else 0.0
    exkurt = pm4 / (pm2 ** 2) - 3.0 if pm2 > 0 else 0.0

    derskew = (2.0 * (pm2 ** 1.5) * dpm3 - 3.0 * pm3 * np.sqrt(pm2) * dpm2) / (2.0 * pm2**3)
    derexkurt = (pm2 * dpm4 - 2.0 * pm4 * dpm2) / pm2**3

    h = (
        z
        + (z**2 - 1.0) * skew / 6.0
        + (z**3 - 3.0 * z) * exkurt / 24.0
        - (2.0 * z**3 - 5.0 * z) * skew**2 / 36.0
    )
    derh = (
        (z**2 - 1.0) * derskew / 6.0
        + (z**3 - 3.0 * z) * derexkurt / 24.0
        - (2.0 * z**3 - 5.0 * z) * skew * derskew / 18.0
    )

    E = (
        norm.pdf(h)
        * (
            1.0
            + (h**3) * skew / 6.0
            + (h**6 - 9.0 * h**4 + 9.0 * h**2 + 3.0) * (skew**2) / 72.0
            + (h**4 - 2.0 * h**2 - 1.0) * exkurt / 24.0
        )
        / alpha
    )
    MES = -location + np.sqrt(pm2) * E

    derMES = (
        -mu
        + E * dpm2 / (2.0 * np.sqrt(pm2))
        - np.sqrt(pm2) * E * h * derh
        + norm.pdf(h)
        * np.sqrt(pm2)
        / alpha
        * (
            derh
            * (
                h**2 * skew / 2.0
                + (6.0 * h**5 - 36.0 * h**3 + 18.0 * h) * (skew**2) / 72.0
                + (4.0 * h**3 - 4.0 * h) * exkurt / 24.0
            )
            + h**3 * derskew / 6.0
            + (h**6 - 9.0 * h**4 + 9.0 * h**2 + 3.0) * skew * derskew / 36.0
            + (h**4 - 2.0 * h**2 - 1.0) * derexkurt / 24.0
        )
    )

    contrib = w * derMES
    pct_contrib = contrib / MES

    if portfolio_method == "marginal":
        return pd.Series(derMES, index=R.columns, name="marginal_MES")
    else:
        return {
            "MES": MES,
            "contribution": pd.Series(contrib, index=R.columns),
            "pct_contrib": pd.Series(pct_contrib, index=R.columns),
        }


def _es_modified_single_series(s: pd.Series, p: float) -> float:
    s = s.dropna()
    if len(s) == 0:
        return np.nan
    alpha = 1 - p if p >= 0.5 else p
    z = norm.ppf(alpha)
    mu = s.mean()
    m2 = centered_moment(s, 2)
    skew = skewness(s, method="moment")
    exkurt = kurtosis(s, method="excess")
    h = (
        z
        + (z**2 - 1.0) * skew / 6.0
        + (z**3 - 3.0 * z) * exkurt / 24.0
        - (2.0 * z**3 - 5.0 * z) * skew**2 / 36.0
    )
    E = (
        norm.pdf(h)
        * (
            1.0
            + (h**3) * skew / 6.0
            + (h**6 - 9.0 * h**4 + 9.0 * h**2 + 3.0) * skew**2 / 72.0
            + (h**4 - 2.0 * h**2 - 1.0) * exkurt / 24.0
        )
        / alpha
    )
    return -mu + np.sqrt(m2) * E


def tracking_error(
    R: pd.Series | pd.DataFrame, Rb: pd.Series | pd.DataFrame, scale: int | None = None
) -> float | pd.Series | pd.DataFrame:
    r"""
    Calculate Tracking Error of returns against a benchmark.

    Tracking error is a measure of how closely a portfolio follows an index.
    It is calculated as the annualized standard deviation of the difference
    between the portfolio's and benchmark's returns.

    Formula:

    .. math::

        TE = \sigma(R_a - R_b) \cdot \sqrt{scale}

    Parameters
    ----------
    R : pd.Series or pd.DataFrame
        Asset returns.
    Rb : pd.Series or pd.DataFrame
        Benchmark returns.
    scale : int, optional
        Number of periods in a year (e.g., 252 for daily, 12 for monthly).
        If None, it is inferred from the index.

    Returns
    -------
    float, pd.Series, or pd.DataFrame
        Annualized Tracking Error.
    r"""
    if scale is None:
        scale = _get_scale(R)

    # Standardize inputs to DataFrame for easier handling of multiple columns
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

            # Difference
            diff = merged.iloc[:, 0] - merged.iloc[:, 1]
            # Annualized SD: sd(diff) * sqrt(scale)
            # R's TrackingError uses sd(diff, na.rm=TRUE) which is N-1 ddof
            te = diff.std() * np.sqrt(scale)
            col_results.append(te)
        results.append(col_results)

    res_df = pd.DataFrame(results, index=rb_cols, columns=ra_cols)

    # Matching R's output format style
    if len(ra_cols) == 1 and len(rb_cols) == 1:
        return res_df.iloc[0, 0]
    elif len(rb_cols) == 1:
        return res_df.iloc[0]
    else:
        return res_df


def capm_beta(
    Ra: pd.Series | pd.DataFrame, Rb: pd.Series | pd.DataFrame, Rf: float | pd.Series | pd.DataFrame = 0
) -> float | pd.Series | pd.DataFrame:
    r"""
    Calculate CAPM Beta of returns against a benchmark.

    Beta is the ratio of the covariance of the asset's excess returns
    with the benchmark's excess returns to the variance of the
    benchmark's excess returns. It measures the systematic risk of the portfolio.

    Formula:

    .. math::

        \beta = \frac{Cov(R_a - R_f, R_b - R_f)}{Var(R_b - R_f)}

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
        CAPM Beta value(s).
    r"""
    from pyperfanalytics.returns import return_excess

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
    for rb_col in rb_cols:
        col_results = []
        for ra_col in ra_cols:
            a = xRa[ra_col]
            b = xRb[rb_col]

            # Align xRa and xRb by pairwise deletion
            merged = pd.concat([a, b], axis=1).dropna()
            if merged.empty or len(merged) < 2:
                col_results.append(np.nan)
                continue

            # beta = cov(ra, rb) / var(rb)
            # R's CAPM.beta uses sample covariance/variance (ddof=1)
            cov_mat = np.cov(merged.iloc[:, 0], merged.iloc[:, 1])
            beta = cov_mat[0, 1] / cov_mat[1, 1]
            col_results.append(beta)
        results.append(col_results)

    res_df = pd.DataFrame(results, index=rb_cols, columns=ra_cols)

    if len(ra_cols) == 1 and len(rb_cols) == 1:
        return res_df.iloc[0, 0]
    elif len(rb_cols) == 1:
        return res_df.iloc[0]
    else:
        return res_df


def ulcer_index(R: pd.Series | pd.DataFrame) -> float | pd.Series:
    r"""
    Calculate the Ulcer Index.

    The Ulcer Index is a measure of downside risk, calculating the quadratic
    mean of the drawdown magnitudes over a period.

    Formula:

    .. math::

        UI = \\sqrt{\frac{1}{n} \\sum_{i=1}^n D_i^2}
    where :math:`D_i` is the drawdown percentage at time :math:`i`.

    Parameters
    ----------
    R : pd.Series or pd.DataFrame
        Asset returns.

    Returns
    -------
    float or pd.Series
        The Ulcer Index.
    r"""
    from pyperfanalytics.drawdowns import drawdown_peak

    dp = drawdown_peak(R)

    def _calc(s: pd.Series) -> float:
        s = s.dropna()
        if len(s) == 0:
            return np.nan
        return np.sqrt((s**2).mean())

    if isinstance(dp, pd.DataFrame):
        return dp.apply(_calc)
    else:
        return _calc(dp)


def pain_index(R: pd.Series | pd.DataFrame) -> float | pd.Series:
    r"""
    Calculate the Pain Index.

    The Pain Index is the mean value of the drawdowns over the entire analysis period.
    It measures both the depth and duration of losses.

    Formula:

    .. math::

        PI = \frac{1}{n} \\sum_{i=1}^n |D_i|

    Parameters
    ----------
    R : pd.Series or pd.DataFrame
        Asset returns.

    Returns
    -------
    float or pd.Series
        The Pain Index.
    r"""
    from pyperfanalytics.drawdowns import drawdown_peak

    dp = drawdown_peak(R)

    def _calc(s: pd.Series) -> float:
        s = s.dropna()
        if len(s) == 0:
            return np.nan
        return np.abs(s).mean()

    if isinstance(dp, pd.DataFrame):
        return dp.apply(_calc)
    else:
        return _calc(dp)


def specific_risk(
    Ra: pd.Series | pd.DataFrame,
    Rb: pd.Series | pd.DataFrame,
    Rf: float | pd.Series | pd.DataFrame = 0,
    scale: int | None = None,
) -> float | pd.Series | pd.DataFrame:
    r"""
    Calculate Specific Risk.

    Specific risk (or idiosyncratic risk) is the annualized standard deviation
    of the error term (alpha) in the CAPM regression. It represents the portion
    of risk that is not explained by the benchmark.

    Formula:

    .. math::

        SpecificRisk = \\sigma(\\epsilon) \\cdot \\sqrt{scale}
    where :math:`\\epsilon_t = R_{a,t} - R_{f,t} - \beta(R_{b,t} - R_{f,t}) - \alpha`.

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
        Annualized Specific Risk.
    r"""
    if scale is None:
        scale = _get_scale(Ra)

    from pyperfanalytics.returns import capm_alpha

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

            beta = capm_beta(a, b, Rf=Rf)
            alpha = capm_alpha(a, b, Rf=Rf)

            # epsilon = Ra - Rb * beta - alpha
            # Note: R's SpecificRisk uses raw returns for epsilon calculation?
            # Let's check R's SpecificRisk.R: epsilon = Ra - Rb * CAPM.beta(Ra,Rb,Rf) - CAPM.alpha(Ra,Rb,Rf)
            # This matches.
            epsilon = a - b * beta - alpha

            # R's SpecificRisk.R (Lestel, Carl Bacon 2008 p.75) formula:
            #   sqrt(sum((epsilon - mean(epsilon))^2) / length(epsilon)) * sqrt(Period)
            # This divides by N (not N-1), i.e. it is the POPULATION standard deviation.
            # NOTE: systematic_risk uses sample SD (ddof=1) via StdDev.annualized in R.
            # This asymmetry exists in the original R package by design (different authors,
            # different conventions).  Both Python functions faithfully replicate their
            # respective R counterparts.
            spec_risk = np.sqrt((epsilon**2).mean() - (epsilon.mean()) ** 2) * np.sqrt(scale)
            col_results.append(spec_risk)
        results.append(col_results)

    res_df = pd.DataFrame(results, index=rb_cols, columns=ra_cols)

    if len(ra_cols) == 1 and len(rb_cols) == 1:
        return res_df.iloc[0, 0]
    elif len(rb_cols) == 1:
        return res_df.iloc[0]
    else:
        return res_df


def total_risk(
    Ra: pd.Series | pd.DataFrame,
    Rb: pd.Series | pd.DataFrame,
    Rf: float | pd.Series | pd.DataFrame = 0,
    scale: int | None = None,
) -> float | pd.Series | pd.DataFrame:
    r"""
    Calculate Total Risk (Systematic + Specific).

    The total risk of an asset can be decomposed into systematic risk (market-related)
    and specific risk (idiosyncratic).

    Formula:

    .. math::

        TotalRisk = \sqrt{SystematicRisk^2 + SpecificRisk^2}

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
        Total Risk.
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
        col_results = []
        for ra_col in ra_cols:
            merged = pd.concat([ra_df[ra_col], rb_df[rb_col]], axis=1).dropna()
            if merged.empty:
                col_results.append(np.nan)
                continue

            a = merged.iloc[:, 0]
            b = merged.iloc[:, 1]

            sys_risk = systematic_risk(a, b, Rf=Rf, scale=scale)
            spec_risk = specific_risk(a, b, Rf=Rf)

            # TotalRisk = sqrt(SystematicRisk^2 + SpecificRisk^2)
            # Both sys_risk and spec_risk are already annualized and use sample SD (ddof=1)
            # R's TotalRisk function has inconsistencies when Rf is a vector,
            # but this implementation is logically consistent with SystematicRisk and SpecificRisk.
            tot_risk = np.sqrt(sys_risk**2 + spec_risk**2)
            col_results.append(tot_risk)
        results.append(col_results)

    res_df = pd.DataFrame(results, index=rb_cols, columns=ra_cols)

    if len(ra_cols) == 1 and len(rb_cols) == 1:
        return res_df.iloc[0, 0]
    elif len(rb_cols) == 1:
        return res_df.iloc[0]
    else:
        return res_df


def herfindahl_index(R: pd.Series | pd.DataFrame) -> float | pd.Series:
    r"""
    Calculate Herfindahl Index based on autocorrelation.

    The Herfindahl Index (or Herfindahl-Hirschman Index) is used here to measure
    the concentration of autocorrelation across different lags.

    Formula:

    .. math::

        HI = \\sum_{i=1}^k \\left( \frac{\\max(0, \rho_i)}{\\sum_{j=1}^k \\max(0, \rho_j)} \right)^2
    where :math:`\rho_i` is the autocorrelation at lag :math:`i`.

    Parameters
    ----------
    R : pd.Series or pd.DataFrame
        Asset returns.

    Returns
    -------
    float or pd.Series
        The Herfindahl Index.
    r"""
    from statsmodels.tsa.stattools import acf

    def _calc(s: pd.Series) -> float:
        s = s.dropna()
        if len(s) < 2:
            return np.nan

        # Calculate ACF
        # PerformanceAnalytics uses R's acf default lag.max = 10 * log10(N/m)
        import math

        nlags = int(10 * math.log10(len(s)))
        nlags = min(len(s) - 1, max(1, nlags))

        r_acf = acf(s, nlags=nlags, fft=True)

        # Get positive ACF values, excluding lag 0
        pos_acf = r_acf[1:][r_acf[1:] >= 0]

        if len(pos_acf) == 0:
            return 0.0

        scaled_acf = pos_acf / pos_acf.sum()
        return float(np.sum(scaled_acf**2))

    if isinstance(R, pd.DataFrame):
        return R.apply(_calc)
    else:
        return _calc(R)


def smoothing_index(R: pd.Series | pd.DataFrame, neg_thetas: bool = False, MAorder: int = 2) -> float | pd.Series:
    r"""
    Calculate Normalized Getmansky Smoothing Index.

    A lower value implies more smoothing (less liquid returns).
    A value of 1 implies no smoothing (highly liquid).

    Formula:

    .. math::

        \xi = \sum_{j=0}^{k} \theta_j^2
    where :math:`\theta_j` are the normalized coefficients of an MA(k) process fitted to the returns.

    Parameters
    ----------
    R : pd.Series or pd.DataFrame
        Asset returns.
    neg_thetas : bool, optional
        If False, constraints the MA coefficients to be non-negative.
    MAorder : int, optional
        The degree of the moving average model. Default is 2.

    Returns
    -------
    float or pd.Series
        Smoothing index value.
    r"""
    from statsmodels.tsa.arima.model import ARIMA

    def _calc(s: pd.Series, neg: bool, order: int) -> float:
        s = s.dropna()
        if len(s) < order + 5:  # Need enough data for ARIMA
            return np.nan

        try:
            import warnings

            from statsmodels.tools.sm_exceptions import ConvergenceWarning, ValueWarning

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ValueWarning)
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                # R arima(..., include.mean=FALSE) applied to demeaned returns
                s_demeaned = s - s.mean()
                model = ARIMA(s_demeaned, order=(0, 0, order), enforce_invertibility=True, trend="n")
                res = model.fit()

            # statsmodels params for MA components are 'ma.L1', 'ma.L2'...
            ma_params = []
            for i in range(1, order + 1):
                name = f"ma.L{i}"
                if name in res.params:
                    ma_params.append(res.params[name])
                else:
                    ma_params.append(0.0)

            if not neg:
                ma_params = [max(0.0, c) for c in ma_params]

            thetas = np.array([1.0] + ma_params)
            thetas /= thetas.sum()

            return float(np.sum(thetas**2))
        except Exception:
            return np.nan

    if isinstance(R, pd.DataFrame):
        return R.apply(_calc, neg=neg_thetas, order=MAorder)
    else:
        return _calc(R, neg_thetas, MAorder)


def fama_beta(
    Ra: pd.Series | pd.DataFrame, Rb: pd.Series | pd.DataFrame, scale: int | None = None
) -> float | pd.Series | pd.DataFrame:
    r"""
    Calculate Fama Beta.

    Fama beta is a measure of systemic risk based on the total risk of the
    portfolio divided by the total risk of the benchmark.

    Formula:

    .. math::

        \beta_F = \frac{\\sigma_a}{\\sigma_b}

    Parameters
    ----------
    Ra : pd.Series or pd.DataFrame
        Asset returns.
    Rb : pd.Series or pd.DataFrame
        Benchmark returns.
    scale : int, optional
        Number of periods in a year.

    Returns
    -------
    float, pd.Series, or pd.DataFrame
        The Fama Beta.
    r"""
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

            # Use population SD (ddof=0)
            std_a = a.std(ddof=0) * np.sqrt(scale)
            std_b = b.std(ddof=0) * np.sqrt(scale)

            if std_b == 0:
                col_results.append(np.nan)
            else:
                col_results.append(std_a / std_b)
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


def cdar_beta(
    Ra: pd.Series | pd.DataFrame,
    Rb: pd.Series | pd.DataFrame,
    p: float = 0.95,
    geometric: bool = True,
    type: str | None = None,
) -> float | pd.Series | pd.DataFrame:
    r"""
    Calculate Conditional Drawdown Beta (CDaR Beta).

    Sensitivity of the portfolio to extreme drawdowns in the benchmark.

    Formula:

    .. math::

        CDaR \beta = \frac{\\sum_{i=1}^k D_{a,i}}{k \\cdot CDaR_b}
    where :math:`D_{a,i}` is the portfolio cumulative return over the benchmark's :math:`k` worst drawdown periods.

    Parameters
    ----------
    Ra : pd.Series or pd.DataFrame
        Asset returns.
    Rb : pd.Series or pd.DataFrame
        Benchmark returns.
    p : float, optional
        Confidence level for calculation, default is 0.95.
    geometric : bool, optional
        Use geometric compounding. Default is True.
    type : str, optional
        Either "average", "max", or None (for standard CDaR).

    Returns
    -------
    float, pd.Series, or pd.DataFrame
        CDaR Beta.
    r"""
    from pyperfanalytics.drawdowns import cdd, find_drawdowns

    if type == "average":
        p = 1.0  # Will be 1-p = 0
    elif type == "max":
        p = 0.0  # Will be 1-p = 1

    p_use = 1.0 - p

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
        b_full = rb_df[rb_col].dropna()

        # Drawdowns of benchmark
        dd_rm = find_drawdowns(b_full, geometric=geometric)
        ret_rm = dd_rm["return"]

        if len(ret_rm) == 0:
            results.append([np.nan] * len(ra_cols))
            continue

        q_quantile = np.quantile(ret_rm, p_use)
        # Indices of worst drawdowns
        indices = np.where(ret_rm <= q_quantile)[0]

        cdd_val = 0.0
        if type == "average":
            cdd_val = np.mean(ret_rm)
        elif type == "max":
            cdd_val = np.min(ret_rm)
        else:
            cdd_val = cdd(b_full, p=p, geometric=geometric, invert=False)

        if cdd_val == 0 or len(indices) == 0:
            results.append([np.nan] * len(ra_cols))
            continue

        for ra_col in ra_cols:
            a_full = ra_df[ra_col].dropna()
            # Align
            merged = pd.concat([a_full, b_full], axis=1).dropna()
            a = merged.iloc[:, 0]

            # Recalculate drawdowns of benchmark on aligned data?
            # PerformanceAnalytics uses full Rm drawdowns, then subsets R.
            # But the dates must match. It just uses row indices: R[from:to]
            # Since R and Rm are checked to have same rows in R code, we use merged indices.

            # To match R perfectly, we must assume aligned indices
            dd_rm_aligned = find_drawdowns(merged.iloc[:, 1], geometric=geometric)
            ret_rm_a = dd_rm_aligned["return"]

            if len(ret_rm_a) == 0:
                col_results.append(np.nan)
                continue

            q_quantile_a = np.quantile(ret_rm_a, p_use)
            indices_a = np.where(ret_rm_a <= q_quantile_a)[0]

            sum_dd_R = 0.0
            # R uses 1-based indexing for from/trough, so we subtract 1 for 0-based
            from_a = dd_rm_aligned["from"] - 1
            trough_a = dd_rm_aligned["trough"] - 1

            for idx in indices_a:
                start = from_a[idx]
                end = trough_a[idx]
                # inclusive range
                temp_r = a.iloc[start : end + 1]
                if geometric:
                    cumul_r = (1 + temp_r).prod() - 1
                else:
                    cumul_r = temp_r.sum()
                sum_dd_R += cumul_r

            beta_dd = sum_dd_R / (len(indices_a) * cdd_val)
            col_results.append(float(beta_dd))

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


def cdar_alpha(
    Ra: pd.Series | pd.DataFrame,
    Rb: pd.Series | pd.DataFrame,
    p: float = 0.95,
    geometric: bool = True,
    type: str | None = None,
    scale: int | None = None,
) -> float | pd.Series | pd.DataFrame:
    r"""
    Calculate Conditional Drawdown Alpha (CDaR Alpha).

    Excess return over the CDaR Beta-adjusted benchmark return.

    Formula:

    .. math::

        CDaR \alpha = R_{a, annualized} - \beta_{CDaR} \cdot R_{b, annualized}

    Parameters
    ----------
    Ra : pd.Series or pd.DataFrame
        Asset returns.
    Rb : pd.Series or pd.DataFrame
        Benchmark returns.
    p : float, optional
        Confidence level. Default is 0.95.
    geometric : bool, optional
        Use geometric compounding. Default is True.
    type : str, optional
        Either "average", "max", or None.
    scale : int, optional
        Number of periods in a year.

    Returns
    -------
    float, pd.Series, or pd.DataFrame
        CDaR Alpha.

    Notes
    -----
    R's ``CDaR.alpha`` hard-codes ``(1 + mean(R))^12 - 1`` for annualisation,
    which (a) assumes monthly data and (b) uses an arithmetic-mean approximation
    rather than the correct geometric compounding formula.  This implementation
    corrects both issues by using :func:`return_annualized` with the inferred or
    supplied ``scale``.
    r"""
    from pyperfanalytics.returns import return_annualized

    if scale is None:
        scale = _get_scale(Ra)

    beta = cdar_beta(Ra, Rb, p=p, geometric=geometric, type=type)

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

            # Use geometric annualized return (CAGR) — the mathematically correct
            # measure of compound growth over the sample period.
            # R uses (1+mean(R))^12-1 which hard-codes monthly frequency AND applies
            # an arithmetic-mean approximation; we correct both issues here.
            rm_exp = return_annualized(b, scale=scale, geometric=True)
            ra_exp = return_annualized(a, scale=scale, geometric=True)

            if isinstance(beta, pd.DataFrame):
                b_val = beta.loc[rb_col, ra_col]
            elif isinstance(beta, pd.Series):
                b_val = beta[ra_col]
            else:
                b_val = beta
            alpha = ra_exp - b_val * rm_exp
            col_results.append(float(alpha))

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


def min_track_record(
    R: pd.Series | pd.DataFrame,
    refSR: float,
    Rf: float | pd.Series | pd.DataFrame = 0.0,
    p: float = 0.95,
    ignore_skewness: bool = False,
    ignore_kurtosis: bool = True,
) -> dict | pd.DataFrame:
    r"""
    Calculate the Minimum Track Record Length.

    Computes the minimum number of observations required to establish that
    the estimated Sharpe Ratio is statistically significantly greater than a reference level.

    Formula:

    .. math::

        T_{min} = 1 + \\left[1 - \\gamma_3 \\widehat{SR} + \frac{\\gamma_4 - 1}{4} \\widehat{SR}^2 \right] \\dots

    .. math::

        \\dots \times \\left(\frac{Z_{\alpha}}{\\widehat{SR} - SR^*}\right)^2

    Parameters
    ----------
    R : pd.Series or pd.DataFrame
        Asset returns.
    refSR : float
        The reference Sharpe Ratio to test against.
    Rf : float, pd.Series, or pd.DataFrame, optional
        Risk-free rate. Default is 0.0.
    p : float, optional
        Confidence level. Default is 0.95.
    ignore_skewness : bool, optional
        If True, assumes zero skewness.
    ignore_kurtosis : bool, optional
        If True, assumes normal kurtosis (3).

    Returns
    -------
    dict or pd.DataFrame
        Minimum track record length, significance boolean, and extra observations needed.
    r"""
    from scipy.stats import norm

    from pyperfanalytics.returns import sharpe_ratio
    from pyperfanalytics.utils import kurtosis, skewness

    if isinstance(R, pd.Series):
        r_df = R.to_frame()
    else:
        r_df = R

    cols = r_df.columns
    min_trl = []
    is_sig = []
    extra_obs = []

    for col in cols:
        s = r_df[col].dropna()
        n = len(s)

        sr = sharpe_ratio(s, Rf=Rf, annualize=False)  # R's MinTrackRecord expects periodic sr
        sk = 0.0 if ignore_skewness else skewness(s)
        kr = 3.0 if ignore_kurtosis else kurtosis(s, method="moment")

        # 1 + (1 - sk*sr + ((kr-1)/4)*sr^2)*(qnorm(p)/(sr-refSR))^2
        if sr <= refSR:
            mtr = np.nan
        else:
            q = norm.ppf(p)
            mtr = 1 + (1 - sk * sr + ((kr - 1) / 4) * sr**2) * (q / (sr - refSR)) ** 2

        min_trl.append(mtr)
        if np.isnan(mtr):
            is_sig.append(False)
            extra_obs.append(np.nan)
        else:
            is_sig.append(n > mtr)
            extra = max(0, np.ceil(mtr - n))
            extra_obs.append(extra)

    res = pd.DataFrame(
        {"min_TRL": min_trl, "IS_SR_SIGNIFICANT": is_sig, "num_of_extra_obs_needed": extra_obs}, index=cols
    ).T

    if len(cols) == 1:
        return res.iloc[:, 0].to_dict()
    return res


def systematic_risk(
    Ra: pd.Series | pd.DataFrame,
    Rb: pd.Series | pd.DataFrame,
    Rf: float | pd.Series | pd.DataFrame = 0,
    scale: int | None = None,
) -> float | pd.Series | pd.DataFrame:
    r"""
    Calculate Systematic Risk.

    The portion of total risk (standard deviation) explained by the benchmark.

    Formula:

    .. math::

        SystematicRisk = \beta \\cdot \\sigma_{b}

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
        Annualized Systematic Risk.
    r"""
    if scale is None:
        scale = _get_scale(Ra)

    from pyperfanalytics.returns import return_excess

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
            xRa = return_excess(ra_df[ra_col], Rf)
            xRb = return_excess(rb_df[rb_col], Rf)

            merged = pd.concat([xRa, xRb], axis=1).dropna()
            if merged.empty:
                col_results.append(np.nan)
                continue

            a = merged.iloc[:, 0]
            b = merged.iloc[:, 1]

            bta = capm_beta(a, b)
            # R uses sample SD (ddof=1) for SystematicRisk
            sigm = b.std(ddof=1) * np.sqrt(scale)
            col_results.append(float(bta * sigm))
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


# ===========================================================================
# High-Order Comoment Shrinkage and Structured Estimation
# ===========================================================================
import scipy.optimize as opt
from itertools import permutations

def _align_returns_and_factor(R, f):
    X = pd.DataFrame(R)
    T_obs, N_assets = X.shape
    if N_assets < 2:
        raise ValueError("R must have at least 2 variables")

    f_values = None
    if f is not None:
        if isinstance(f, (pd.Series, pd.DataFrame)):
            common_idx = X.index.intersection(f.index)
            X = X.loc[common_idx]
            f_aligned = f.loc[common_idx]
            T_obs = len(common_idx)
            if isinstance(f_aligned, pd.DataFrame):
                f_values = f_aligned.values
            else:
                f_values = f_aligned.values[:, np.newaxis]
        else:
            f_arr = np.asarray(f)
            if f_arr.ndim == 1:
                f_values = f_arr[:, np.newaxis]
            else:
                f_values = f_arr
            if len(f_values) != T_obs:
                raise ValueError("Length of factor f must match R")
    return X, f_values, T_obs, N_assets

def _m3_vec_to_mat(vec: np.ndarray, N: int) -> np.ndarray:
    M3 = np.zeros((N, N * N))
    iter_idx = 0
    for ii in range(N):
        for jj in range(ii, N):
            for kk in range(jj, N):
                val = vec[iter_idx]
                M3[ii, jj * N + kk] = val
                M3[ii, kk * N + jj] = val
                M3[jj, ii * N + kk] = val
                M3[jj, kk * N + ii] = val
                M3[kk, ii * N + jj] = val
                M3[kk, jj * N + ii] = val
                iter_idx += 1
    return M3

def _m3_mat_to_vec(M3: np.ndarray, N: int) -> np.ndarray:
    vec = []
    for ii in range(N):
        for jj in range(ii, N):
            for kk in range(jj, N):
                vec.append(M3[ii, jj * N + kk])
    return np.array(vec, dtype=float)

def _m4_vec_to_mat(vec: np.ndarray, N: int) -> np.ndarray:
    M4 = np.zeros((N, N * N * N))
    iter_idx = 0
    for ii in range(N):
        for jj in range(ii, N):
            for kk in range(jj, N):
                for ll in range(kk, N):
                    val = vec[iter_idx]
                    for p_i, p_j, p_k, p_l in set(permutations([ii, jj, kk, ll])):
                        M4[p_l, p_i * N * N + p_j * N + p_k] = val
                    iter_idx += 1
    return M4

def _m4_mat_to_vec(M4: np.ndarray, N: int) -> np.ndarray:
    vec = []
    for ii in range(N):
        for jj in range(ii, N):
            for kk in range(jj, N):
                for ll in range(kk, N):
                    vec.append(M4[ll, ii * N * N + jj * N + kk])
    return np.array(vec, dtype=float)

def _m3_multipliers(N: int) -> np.ndarray:
    mult = []
    for ii in range(N):
        for jj in range(ii, N):
            for kk in range(jj, N):
                if ii == jj == kk:
                    mult.append(1.0)
                elif ii == jj or jj == kk or ii == kk:
                    mult.append(3.0)
                else:
                    mult.append(6.0)
    return np.array(mult)

def _m4_multipliers(N: int) -> np.ndarray:
    mult = []
    for ii in range(N):
        for jj in range(ii, N):
            for kk in range(jj, N):
                for ll in range(kk, N):
                    num_unique = len(set(permutations([ii, jj, kk, ll])))
                    mult.append(float(num_unique))
    return np.array(mult)

def solve_qp(A, b):
    nT = len(b)
    if nT == 1:
        val = b[0] / A[0, 0]
        return np.array([max(0.0, min(1.0, val))])
    
    # Scale A and b to prevent optimization failure on tiny values
    scale_factor = 1.0 / np.max(np.abs(A)) if np.max(np.abs(A)) > 0 else 1.0
    A_scaled = A * scale_factor
    b_scaled = b * scale_factor
    
    def obj(x):
        return 0.5 * x.T @ A_scaled @ x - b_scaled.T @ x

    bounds = [(0.0, 1.0) for _ in range(nT)]
    cons = {'type': 'ineq', 'fun': lambda x: 1.0 - np.sum(x)}
    x0 = np.ones(nT) / (2.0 * nT)
    res = opt.minimize(obj, x0, bounds=bounds, constraints=cons, method='SLSQP', tol=1e-12)
    return res.x

def _calc_VM2(M11, M22, T_obs, N_assets):
    term_diff = M22 - M11**2
    vm2_0 = np.sum(term_diff) / T_obs
    vm2_2 = np.sum(np.diagonal(term_diff)) / T_obs
    v = np.diagonal(M11)
    vm2_1 = np.sum(M22 - np.outer(v, v)) / (T_obs * N_assets)
    return np.array([vm2_0, vm2_1, vm2_2])

def _calc_CM2_1F(Xc, M11, M22, fc, fvar, T_obs, N_assets):
    if fc.ndim == 2:
        fc = fc[:, 0]
    cov_X_f = np.mean(Xc * fc[:, np.newaxis], axis=0)
    S211 = ((Xc**2 * fc[:, np.newaxis]).T @ Xc) / T_obs
    S112 = ((Xc * (fc**2)[:, np.newaxis]).T @ Xc) / T_obs
    
    temp_i0 = S211 - M11 * cov_X_f[:, np.newaxis]
    temp_j0 = S211.T - M11 * cov_X_f[np.newaxis, :]
    temp_var = S112 - M11 * fvar
    
    cov_X_f_col = cov_X_f[:, np.newaxis]
    cov_X_f_row = cov_X_f[np.newaxis, :]
    
    fvar2 = fvar**2
    # Removed the 2.0 multiplier here because sum over symmetric matrix counts non-diagonal entries twice
    Z = (temp_i0 * cov_X_f_row / fvar + temp_j0 * cov_X_f_col / fvar - 
         temp_var * (cov_X_f_col @ cov_X_f_row) / fvar2)
    
    diag_term = np.diagonal(M22 - M11**2)
    np.fill_diagonal(Z, diag_term)
    
    return np.sum(Z) / T_obs

def _calc_CM2_CC(Xc, M11, M22, rcoef, T_obs, N_assets):
    S31 = ((Xc**3).T @ Xc) / T_obs
    v = np.diagonal(M11)
    temp_ii = S31 - M11 * v[:, np.newaxis]
    temp_jj = S31.T - M11 * v[np.newaxis, :]
    
    sqrt_v = np.sqrt(np.where(v > 0, v, 1e-15))
    scale_mat = np.outer(1.0 / sqrt_v, sqrt_v)
    
    Z = rcoef * (scale_mat * temp_ii + scale_mat.T * temp_jj)
    # Multiply by 0.5 because sum over symmetric matrix counts non-diagonal entries twice
    Z = Z * 0.5
    
    diag_term = np.diagonal(M22 - M11**2)
    np.fill_diagonal(Z, diag_term)
    
    return np.sum(Z) / T_obs

def _calc_M3_CCoefficients(Xc, margvars, margkurts, m21, m22, T_obs, N_assets):
    P = N_assets
    n = T_obs
    p = float(P)
    
    r2 = 0.0
    r5 = 0.0
    for ii in range(P):
        for jj in range(ii + 1, P):
            r2 += m21[ii, jj] / np.sqrt(margkurts[ii] * margvars[jj])
            r5 += m22[ii, jj] / np.sqrt(margkurts[ii] * margkurts[jj])
    if P > 1:
        r2 *= 2.0 / (p * (p - 1.0))
        r5 *= 2.0 / (p * (p - 1.0))
    else:
        r2 = 0.0
        r5 = 0.0

    r4 = 0.0
    for ii in range(P):
        for jj in range(ii + 1, P):
            for kk in range(jj + 1, P):
                m111 = np.mean(Xc[:, ii] * Xc[:, jj] * Xc[:, kk])
                nc = (np.sqrt(margvars[ii] * r5 * np.sqrt(margkurts[jj] * margkurts[kk])) +
                      np.sqrt(margvars[jj] * r5 * np.sqrt(margkurts[ii] * margkurts[kk])) +
                      np.sqrt(margvars[kk] * r5 * np.sqrt(margkurts[ii] * margkurts[jj]))) / 3.0
                r4 += m111 / (nc if nc != 0 else 1e-15)
    if P > 2:
        r4 *= 6.0 / (p * (p - 1.0) * (p - 2.0))
    else:
        r4 = 0.0
        
    return r2, r4, r5

def _calc_M3_CC(margvars, margskews, margkurts, r2, r4, r5, N_assets):
    P = N_assets
    vec = []
    for ii in range(P):
        for jj in range(ii, P):
            for kk in range(jj, P):
                if ii == jj:
                    if jj == kk:
                        elem = margskews[ii]
                    else:
                        elem = r2 * np.sqrt(margvars[kk] * margkurts[ii])
                else:
                    if jj == kk:
                        elem = r2 * np.sqrt(margvars[ii] * margkurts[jj])
                    else:
                        elem = r4 * np.sqrt(r5) * (
                            np.sqrt(margvars[kk] * np.sqrt(margkurts[ii] * margkurts[jj])) +
                            np.sqrt(margvars[jj] * np.sqrt(margkurts[ii] * margkurts[kk])) +
                            np.sqrt(margvars[ii] * np.sqrt(margkurts[jj] * margkurts[kk]))
                        ) / 3.0
                vec.append(elem)
    return np.array(vec)

def _calc_VM3(Xc, Xc2, M11, M21, M22, M31, M42, M33, T_obs, N_assets):
    V_d = np.diagonal(M11)
    M21_diag = np.diagonal(M21)
    M22_diag = np.diagonal(M22)
    
    vm3_diag = (np.diagonal(M42) - M21_diag**2 - 6.0 * M22_diag * V_d + 9.0 * V_d**3) / T_obs
    vm3_2 = np.sum(vm3_diag)
    
    sum_phi_iik = 0.0
    for ii in range(N_assets):
        for kk in range(N_assets):
            if ii != kk:
                val = (M42[kk, ii] - M21[kk, ii]**2 - 4.0 * M31[kk, ii] * M11[kk, ii] -
                       2.0 * M22[kk, ii] * M11[ii, ii] + 8.0 * M11[ii, ii] * M11[kk, ii]**2 +
                       M11[kk, kk] * M11[ii, ii]**2) / T_obs
                sum_phi_iik += 3.0 * val
                
    sum_phi_ijk = 0.0
    for ii in range(N_assets):
        for jj in range(ii + 1, N_assets):
            for kk in range(jj + 1, N_assets):
                S211 = np.sum(Xc2[:, ii] * Xc[:, jj] * Xc[:, kk])
                S121 = np.sum(Xc[:, ii] * Xc2[:, jj] * Xc[:, kk])
                S112 = np.sum(Xc[:, ii] * Xc[:, jj] * Xc2[:, kk])
                S111 = np.sum(Xc[:, ii] * Xc[:, jj] * Xc[:, kk])
                S222 = np.sum(Xc2[:, ii] * Xc2[:, jj] * Xc2[:, kk])
                
                val = (S222 - S111**2 / T_obs - 
                       2.0 * S211 * M11[kk, jj] - 2.0 * S121 * M11[kk, ii] - 2.0 * S112 * M11[jj, ii] + 
                       6.0 * M11[kk, ii] * M11[kk, jj] * M11[jj, ii] * T_obs +
                       M11[ii, ii] * M11[kk, jj]**2 * T_obs + 
                       M11[jj, jj] * M11[kk, ii]**2 * T_obs +
                       M11[kk, kk] * M11[jj, ii]**2 * T_obs) / T_obs
                sum_phi_ijk += 6.0 * val / T_obs
                
    vm3_0 = vm3_2 + sum_phi_iik + sum_phi_ijk
    
    term2 = np.outer(M21_diag, M21_diag)
    term3 = 3.0 * M31 * V_d[np.newaxis, :]
    term4 = 3.0 * M31.T * V_d[:, np.newaxis]
    term5 = 9.0 * np.outer(V_d, V_d) * M11
    Mat2 = M33 - term2 - term3 - term4 + term5
    vm3_1 = (vm3_2 + (np.sum(Mat2) - np.sum(np.diagonal(Mat2))) / T_obs) / N_assets
    
    return np.array([vm3_0, vm3_1, vm3_2])

def _calc_VM3kstat(Xc, Xc2, M11, M21, M22, M31, M42, M33, T_obs, N_assets):
    S11 = M11 * T_obs
    S21 = M21 * T_obs
    S22 = M22 * T_obs
    S31 = M31 * T_obs
    S42 = M42 * T_obs
    S33 = M33 * T_obs
    
    N = float(T_obs)
    N2 = N * N
    N3 = N2 * N
    N4 = N3 * N
    N5 = N4 * N
    N122 = (N - 1.0) * (N - 1.0) * (N - 2.0) * (N - 2.0)
    
    alpha = N * N122 * (N - 3.0) * (N - 4.0) * (N - 5.0)
    ciii_S2_3 = (9.0 * N4 - 72.0 * N3 + 213.0 * N2 - 270.0 * N + 120.0) / alpha
    ciii_S4S2 = (-6.0 * N5 + 33.0 * N4 - 42.0 * N3 - 75.0 * N2 + 210.0 * N - 120.0) / alpha
    ciii_S3_2 = (-N5 - 4.0 * N4 + 41.0 * N3 - 40.0 * N2 - 100.0 * N + 80.0) / alpha
    ciii_S6 = (N5 - 5.0 * N4 + 13.0 * N3 - 23.0 * N2 + 22.0 * N - 8.0) / (N122 * (N - 3.0) * (N - 4.0) * (N - 5.0))
    
    ciij_S02S20_2 = (N4 - 8.0 * N3 + 29.0 * N2 - 46.0 * N + 24.0) / alpha
    ciij_S20S11_2 = (8.0 * N4 - 64.0 * N3 + 184.0 * N2 - 224.0 * N + 96.0) / alpha
    ciij_S20S22 = (-2.0 * N5 + 10.0 * N4 - 10.0 * N3 - 34.0 * N2 + 84.0 * N - 48.0) / alpha
    ciij_S31S11 = (-4.0 * N5 + 24.0 * N4 - 36.0 * N3 - 32.0 * N2 + 112.0 * N - 64.0) / alpha
    ciij_S40S02 = (-N4 + 4.0 * N3 - 9.0 * N2 + 14.0 * N - 8.0) / alpha
    ciij_S21_2 = (-N5 + 25.0 * N3 - 36.0 * N2 - 60.0 * N + 48.0) / alpha
    ciij_S12S30 = (-4.0 * N4 + 16.0 * N3 - 4.0 * N2 - 40.0 * N + 32.0) / alpha
    
    cijk_S002S110_2 = (N4 - 8.0 * N3 + 25.0 * N2 - 34.0 * N + 16.0) / alpha
    cijk_S011S101S110 = (6.0 * N4 - 48.0 * N3 + 134.0 * N2 - 156.0 * N + 64.0) / alpha
    cijk_S112S110 = (-2.0 * N5 + 12.0 * N4 - 18.0 * N3 - 16.0 * N2 + 56.0 * N - 32.0) / alpha
    cijk_S002S020S200 = (4.0 * N2 - 12.0 * N + 8.0) / alpha
    cijk_S022S200 = (-N4 + 4.0 * N3 - 9.0 * N2 + 14.0 * N - 8.0) / alpha
    cijk_S111_2 = (-N5 + 2.0 * N4 + 17.0 * N3 - 34.0 * N2 - 40.0 * N + 32.0) / alpha
    cijk_S102S120 = (-2.0 * N4 + 8.0 * N3 - 2.0 * N2 - 20.0 * N + 16.0) / alpha
    
    vm3_diag = []
    for i in range(N_assets):
        val = ciii_S2_3 * S11[i, i]**3 + ciii_S4S2 * S22[i, i] * S11[i, i] + ciii_S3_2 * S21[i, i]**2 + ciii_S6 * S42[i, i]
        vm3_diag.append(val)
    vm3_diag = np.array(vm3_diag)
    vm3_2 = np.sum(vm3_diag)
    
    sum_phi_iik = 0.0
    for i in range(N_assets):
        for k in range(N_assets):
            if i != k:
                val = ciij_S02S20_2 * S11[k, k] * S11[i, i]**2 + \
                      ciij_S20S11_2 * S11[i, i] * S11[k, i]**2 + \
                      ciij_S20S22 * S11[i, i] * S22[k, i] + \
                      ciij_S31S11 * S31[k, i] * S11[k, i] + \
                      ciij_S40S02 * S22[i, i] * S11[k, k] + \
                      ciij_S21_2 * S21[k, i]**2 + \
                      ciij_S12S30 * S21[i, k] * S21[i, i] + \
                      ciii_S6 * S42[k, i]
                sum_phi_iik += 3.0 * val
                
    sum_phi_ijk = 0.0
    for i in range(N_assets):
        for j in range(i + 1, N_assets):
            for k in range(j + 1, N_assets):
                S211i = np.sum(Xc2[:, i] * Xc[:, j] * Xc[:, k])
                S211j = np.sum(Xc[:, i] * Xc2[:, j] * Xc[:, k])
                S211k = np.sum(Xc[:, i] * Xc[:, j] * Xc2[:, k])
                S111 = np.sum(Xc[:, i] * Xc[:, j] * Xc[:, k])
                S222 = np.sum(Xc2[:, i] * Xc2[:, j] * Xc2[:, k])
                
                val = cijk_S002S110_2 * S11[k, k] * S11[j, i]**2 + \
                      (cijk_S011S101S110 * S11[k, j] * S11[k, i] + cijk_S112S110 * S211k) * S11[j, i] + \
                      cijk_S002S110_2 * S11[j, j] * S11[k, i]**2 + \
                      cijk_S112S110 * S211j * S11[k, i] + \
                      cijk_S002S110_2 * S11[i, i] * S11[k, j]**2 + \
                      cijk_S112S110 * S211i * S11[k, j] + \
                      (cijk_S002S020S200 * S11[j, j] * S11[k, k] + cijk_S022S200 * S22[k, j]) * S11[i, i] + \
                      cijk_S022S200 * S22[k, i] * S11[j, j] + \
                      cijk_S022S200 * S22[j, i] * S11[k, k] + \
                      cijk_S111_2 * S111**2 + \
                      cijk_S102S120 * S21[i, k] * S21[i, j] + \
                      cijk_S102S120 * S21[j, k] * S21[j, i] + \
                      cijk_S102S120 * S21[k, j] * S21[k, i] + \
                      ciii_S6 * S222
                sum_phi_ijk += 6.0 * val
                
    vm3_0 = vm3_2 + sum_phi_iik + sum_phi_ijk
    
    c_S11_3 = (24.0 * N2 - 72.0 * N + 48.0) / alpha
    c_S02S20S11 = (9.0 * N4 - 72.0 * N3 + 189.0 * N2 - 198.0 * N + 72.0) / alpha
    c_S22S11 = (-9.0 * N4 + 36.0 * N3 - 81.0 * N2 + 126.0 * N - 72.0) / alpha
    c_S13S20 = (-3.0 * N5 + 21.0 * N4 - 39.0 * N3 + 3.0 * N2 + 42.0 * N - 24.0) / alpha
    c_S21S12 = (-9.0 * N4 + 36.0 * N3 - 9.0 * N2 - 90.0 * N + 72.0) / alpha
    c_S03S30 = (-N5 + 5.0 * N4 + 5.0 * N3 - 31.0 * N2 - 10.0 * N + 8.0) / alpha
    
    sum_cov = 0.0
    for i in range(N_assets):
        for j in range(i + 1, N_assets):
            val = c_S11_3 * S11[j, i]**3 + \
                  (c_S02S20S11 * S11[i, i] * S11[j, j] + c_S22S11 * S22[j, i]) * S11[j, i] + \
                  c_S13S20 * S31[i, j] * S11[i, i] + \
                  c_S13S20 * S31[j, i] * S11[j, j] + \
                  c_S21S12 * S21[j, i] * S21[i, j] + \
                  c_S03S30 * S21[i, i] * S21[j, j] + \
                  ciii_S6 * S33[j, i]
            sum_cov += 2.0 * val
    vm3_1 = (vm3_2 + sum_cov) / N_assets
    
    return np.array([vm3_0, vm3_1, vm3_2])

def _calc_CM3_1F(Xc, Xc2, fc, fvar, fskew, M11, M21, M22, M42, T_obs, N_assets):
    covXf = np.mean(Xc * fc, axis=0)
    X1f2 = np.mean(Xc * (fc**2), axis=0)
    X1f3 = np.mean(Xc * (fc**3), axis=0)
    X11f1 = (Xc.T @ (Xc * fc)) / T_obs
    fvar3 = fvar**3
    
    cm3_sum = 0.0
    for i in range(N_assets):
        for j in range(i, N_assets):
            for k in range(j, N_assets):
                mult = 1.0
                if i == j:
                    if j == k:
                        val = M42[i, i] - M21[i, i]**2 - 6.0 * M22[i, i] * M11[i, i] + 9.0 * M11[i, i]**3
                        cm3_sum += val
                        continue
                    else:
                        mult = 3.0
                        S311 = np.mean(Xc2[:, i] * Xc[:, i] * Xc[:, k] * fc[:, 0])
                        S221 = np.mean(Xc2[:, i] * Xc2[:, k] * fc[:, 0])
                        S213 = np.mean(Xc2[:, i] * Xc[:, k] * fc[:, 0]**3)
                        S211_val = np.mean(Xc2[:, i] * Xc[:, k] * fc[:, 0])
                        S212 = np.mean(Xc2[:, i] * Xc[:, k] * fc[:, 0]**2)
                        
                        temp_ii = S311 - M21[i, k] * covXf[i] - 2.0 * M11[k, i] * X11f1[i, i] - M11[i, i] * X11f1[k, i]
                        temp_kk = S221 - M21[i, k] * covXf[k] - M11[i, i] * X11f1[k, k] - 2.0 * M11[k, i] * X11f1[k, i]
                        temp_fskew = S213 - M21[i, k] * fskew - 3.0 * S211_val * fvar - 2.0 * X1f3[i] * M11[k, i] - X1f3[k] * M11[i, i] + \
                                     3.0 * M11[i, i] * covXf[k] * fvar + 6.0 * M11[k, i] * covXf[i] * fvar
                        temp_fvar = S212 - M21[i, k] * fvar - 2.0 * X1f2[i] * M11[k, i] - X1f2[k] * M11[i, i]
                        
                        val = ((2.0 * covXf[i] * covXf[k] * temp_ii + covXf[i]**2 * temp_kk) * fskew + \
                               covXf[i]**2 * covXf[k] * temp_fskew - 3.0 * covXf[i]**2 * covXf[k] * fskew * temp_fvar / fvar) / fvar3
                else:
                    if j == k:
                        mult = 3.0
                        S221 = np.mean(Xc2[:, i] * Xc2[:, j] * fc[:, 0])
                        S131 = np.mean(Xc[:, i] * Xc2[:, j] * Xc[:, j] * fc[:, 0])
                        S123 = np.mean(Xc[:, i] * Xc2[:, j] * fc[:, 0]**3)
                        S121 = np.mean(Xc[:, i] * Xc2[:, j] * fc[:, 0])
                        S122 = np.mean(Xc[:, i] * Xc2[:, j] * fc[:, 0]**2)
                        
                        temp_ii = S221 - M21[j, i] * covXf[i] - M11[j, j] * X11f1[i, i] - 2.0 * M11[j, i] * X11f1[j, i]
                        temp_jj = S131 - M21[j, i] * covXf[j] - 2.0 * M11[j, i] * X11f1[j, j] - M11[j, j] * X11f1[j, i]
                        temp_fskew = S123 - M21[j, i] * fskew - 3.0 * S121 * fvar - X1f3[i] * M11[j, j] - 2.0 * X1f3[j] * M11[j, i] + \
                                     6.0 * M11[j, i] * covXf[j] * fvar + 3.0 * M11[j, j] * covXf[i] * fvar
                        temp_fvar = S122 - M21[j, i] * fvar - X1f2[i] * M11[j, j] - 2.0 * X1f2[j] * M11[j, i]
                        
                        val = ((covXf[j]**2 * temp_ii + 2.0 * covXf[i] * covXf[j] * temp_jj) * fskew + \
                               covXf[i] * covXf[j]**2 * temp_fskew - 3.0 * covXf[i] * covXf[j]**2 * fskew * temp_fvar / fvar) / fvar3
                    else:
                        mult = 6.0
                        S2111 = np.mean(Xc2[:, i] * Xc[:, j] * Xc[:, k] * fc[:, 0])
                        S1211 = np.mean(Xc[:, i] * Xc2[:, j] * Xc[:, k] * fc[:, 0])
                        S1121 = np.mean(Xc[:, i] * Xc[:, j] * Xc2[:, k] * fc[:, 0])
                        S1112 = np.mean(Xc[:, i] * Xc[:, j] * Xc[:, k] * fc[:, 0]**2)
                        S1113 = np.mean(Xc[:, i] * Xc[:, j] * Xc[:, k] * fc[:, 0]**3)
                        S1111 = np.mean(Xc[:, i] * Xc[:, j] * Xc[:, k] * fc[:, 0])
                        S1110 = np.mean(Xc[:, i] * Xc[:, j] * Xc[:, k])
                        
                        temp_ii = S2111 - S1110 * covXf[i] - M11[k, j] * X11f1[i, i] - M11[k, i] * X11f1[j, i] - M11[j, i] * X11f1[k, i]
                        temp_jj = S1211 - S1110 * covXf[j] - M11[k, i] * X11f1[j, j] - M11[k, j] * X11f1[i, j] - M11[i, j] * X11f1[k, j]
                        temp_kk = S1121 - S1110 * covXf[k] - M11[j, i] * X11f1[k, k] - M11[j, k] * X11f1[i, k] - M11[i, k] * X11f1[j, k]
                        temp_fskew = S1113 - S1110 * fskew - 3.0 * S1111 * fvar - X1f3[i] * M11[k, j] - X1f3[j] * M11[k, i] - X1f3[k] * M11[j, i] + \
                                     3.0 * M11[j, i] * covXf[k] * fvar + 3.0 * M11[k, i] * covXf[j] * fvar + 3.0 * M11[k, j] * covXf[i] * fvar
                        temp_fvar = S1112 - S1110 * fvar - X1f2[i] * M11[k, j] - X1f2[j] * M11[k, i] - X1f2[k] * M11[j, i]
                        
                        val = ((covXf[j] * covXf[k] * temp_ii + covXf[i] * covXf[k] * temp_jj + covXf[i] * covXf[j] * temp_kk) * fskew + \
                               covXf[i] * covXf[j] * covXf[k] * temp_fskew - 3.0 * covXf[i] * covXf[j] * covXf[k] * fskew * temp_fvar / fvar) / fvar3
                cm3_sum += mult * val
    return cm3_sum / T_obs

def _calc_CM3_Simaan(Xc, Xc2, margskewsroot, M11, M21, M22, M31, M42, M51, T_obs, N_assets):
    cm3_sum = 0.0
    for i in range(N_assets):
        for j in range(i, N_assets):
            for k in range(j, N_assets):
                mult = 1.0
                if i == j:
                    if j == k:
                        val = M42[i, i] - M21[i, i]**2 - 6.0 * M22[i, i] * M11[i, i] + 9.0 * M11[i, i]**3
                        cm3_sum += val
                        continue
                    else:
                        mult = 3.0
                        temp_ii = M51[i, k] - M21[i, k] * M21[i, i] - 4.0 * M31[i, k] * M11[i, i] - 2.0 * M22[i, i] * M11[i, k] + 9.0 * M11[i, i]**2 * M11[i, k]
                        temp_kk = M42[k, i] - M21[i, k] * M21[k, k] - 3.0 * M22[i, k] * M11[k, k] - M22[k, k] * M11[i, i] - 2.0 * M31[k, i] * M11[i, k] + \
                                  6.0 * M11[k, k] * M11[i, k]**2 + 3.0 * M11[k, k]**2 * M11[i, i]
                        val = margskewsroot[i]**2 * margskewsroot[k] * (2.0 * M21[i, i] * M21[k, k] * temp_ii + M21[i, i]**2 * temp_kk)
                else:
                    if j == k:
                        mult = 3.0
                        temp_ii = M42[i, j] - M21[j, i] * M21[i, i] - 3.0 * M22[j, i] * M11[i, i] - M22[i, i] * M11[j, j] - 2.0 * M31[i, j] * M11[j, i] + \
                                  6.0 * M11[i, i] * M11[j, i]**2 + 3.0 * M11[i, i]**2 * M11[j, j]
                        temp_jj = M51[j, i] - M21[j, i] * M21[j, j] - 4.0 * M31[j, i] * M11[j, j] - 2.0 * M22[j, j] * M11[j, i] + 9.0 * M11[j, j]**2 * M11[j, i]
                        val = margskewsroot[i] * margskewsroot[j]**2 * (M21[j, j]**2 * temp_ii + 2.0 * M21[i, i] * M21[j, j] * temp_jj)
                    else:
                        mult = 6.0
                        S411 = np.mean(Xc[:, i]**4 * Xc[:, j] * Xc[:, k])
                        S141 = np.mean(Xc[:, i] * Xc[:, j]**4 * Xc[:, k])
                        S114 = np.mean(Xc[:, i] * Xc[:, j] * Xc[:, k]**4)
                        S111 = np.mean(Xc[:, i] * Xc[:, j] * Xc[:, k])
                        S211 = np.mean(Xc[:, i]**2 * Xc[:, j] * Xc[:, k])
                        S121 = np.mean(Xc[:, i] * Xc[:, j]**2 * Xc[:, k])
                        S112 = np.mean(Xc[:, i] * Xc[:, j] * Xc[:, k]**2)
                        
                        temp_ii = S411 - S111 * M21[i, i] - 3.0 * S211 * M11[i, i] - M22[i, i] * M11[k, j] - M31[i, j] * M11[k, i] - M31[i, k] * M11[j, i] + \
                                  6.0 * M11[i, i] * M11[j, i] * M11[k, i] + 3.0 * M11[i, i]**2 * M11[k, j]
                        temp_jj = S141 - S111 * M21[j, j] - 3.0 * S121 * M11[j, j] - M22[j, j] * M11[k, i] - M31[j, i] * M11[k, j] - M31[j, k] * M11[i, j] + \
                                  6.0 * M11[j, j] * M11[i, j] * M11[k, j] + 3.0 * M11[j, j]**2 * M11[k, i]
                        temp_kk = S114 - S111 * M21[k, k] - 3.0 * S112 * M11[k, k] - M22[k, k] * M11[j, i] - M31[k, j] * M11[i, k] - M31[k, i] * M11[j, k] + \
                                  6.0 * M11[k, k] * M11[j, k] * M11[i, k] + 3.0 * M11[k, k]**2 * M11[j, i]
                        val = margskewsroot[i] * margskewsroot[j] * margskewsroot[k] * \
                              (M21[j, j] * M21[k, k] * temp_ii + M21[i, i] * M21[k, k] * temp_jj + M21[i, i] * M21[j, j] * temp_kk)
                cm3_sum += mult * val
    return cm3_sum / T_obs

def _calc_CM3_CC(Xc, Xc2, margvars, margskews, margkurts, marg5s, marg6s, M11, M21, M31, M32, M41, M61, r2, r4, r5, T_obs, N_assets):
    cm3_sum = 0.0
    for i in range(N_assets):
        for j in range(i, N_assets):
            for k in range(j, N_assets):
                mult = 1.0
                if i == j:
                    if j == k:
                        val = marg6s[i] - margskews[i]**2 - 6.0 * margkurts[i] * margvars[i] + 9.0 * margvars[i]**3
                        cm3_sum += val
                        continue
                    else:
                        mult = 2.0
                        temp_ii4 = M61[i, k] - M21[i, k] * margkurts[i] - 4.0 * M31[i, k] * margskews[i] - \
                                   2.0 * M11[i, k] * marg5s[i] - margvars[i] * M41[i, k] + 12.0 * margvars[i] * M11[i, k] * margskews[i]
                        temp_kk2 = M32[k, i] - margvars[k] * M21[i, k] - margskews[k] * margvars[i] - 2.0 * M21[i, k] * M11[i, k]
                        val = 3.0 * r2 * (np.sqrt(margvars[k] / margkurts[i]) * temp_ii4 + np.sqrt(margkurts[i] / margvars[k]) * temp_kk2) / 2.0
                else:
                    if j == k:
                        mult = 2.0
                        temp_ii2 = M32[i, j] - margvars[i] * M21[j, i] - margskews[i] * margvars[j] - 2.0 * M21[i, j] * M11[i, j]
                        temp_jj4 = M61[j, i] - M21[j, i] * margkurts[j] - 4.0 * M31[j, i] * margskews[j] - \
                                   2.0 * M11[i, j] * marg5s[j] - margvars[j] * M41[j, i] + 12.0 * margvars[j] * M11[i, j] * margskews[j]
                        val = 3.0 * r2 * (np.sqrt(margvars[i] / margkurts[j]) * temp_jj4 + np.sqrt(margkurts[j] / margvars[i]) * temp_ii2) / 2.0
                    else:
                        mult = 6.0
                        S511 = np.mean(Xc[:, i]**5 * Xc[:, j] * Xc[:, k])
                        S151 = np.mean(Xc[:, i] * Xc[:, j]**5 * Xc[:, k])
                        S115 = np.mean(Xc[:, i] * Xc[:, j] * Xc[:, k]**5)
                        S211 = np.mean(Xc[:, i]**2 * Xc[:, j] * Xc[:, k])
                        S121 = np.mean(Xc[:, i] * Xc[:, j]**2 * Xc[:, k])
                        S112 = np.mean(Xc[:, i] * Xc[:, j] * Xc[:, k]**2)
                        S311 = np.mean(Xc[:, i]**3 * Xc[:, j] * Xc[:, k])
                        S131 = np.mean(Xc[:, i] * Xc[:, j]**3 * Xc[:, k])
                        S113 = np.mean(Xc[:, i] * Xc[:, j] * Xc[:, k]**3)
                        m111 = np.mean(Xc[:, i] * Xc[:, j] * Xc[:, k])
                        
                        temp_ii4 = S511 - m111 * margkurts[i] - 4.0 * margskews[i] * S211 - \
                                   marg5s[i] * M11[j, k] - M41[i, j] * M11[i, k] - M41[i, k] * M11[i, j] + \
                                   8.0 * margskews[i] * M11[i, j] * M11[i, k] + 4.0 * margskews[i] * margvars[i] * M11[j, k]
                        temp_jj4 = S151 - m111 * margkurts[j] - 4.0 * margskews[j] * S121 - \
                                   marg5s[j] * M11[i, k] - M41[j, k] * M11[j, i] - M41[j, i] * M11[j, k] + \
                                   8.0 * margskews[j] * M11[j, k] * M11[j, i] + 4.0 * margskews[j] * margvars[j] * M11[i, k]
                        temp_kk4 = S115 - m111 * margkurts[k] - 4.0 * margskews[k] * S112 - \
                                   marg5s[k] * M11[i, j] - M41[k, i] * M11[k, j] - M41[k, j] * M11[k, i] + \
                                   8.0 * margskews[k] * M11[k, i] * M11[k, j] + 4.0 * margskews[k] * margvars[k] * M11[i, j]
                                   
                        temp_ii2 = S311 - margvars[i] * m111 - margskews[i] * M11[j, k] - \
                                   M21[i, j] * M11[i, k] - M21[i, k] * M11[i, j]
                        temp_jj2 = S131 - margvars[j] * m111 - margskews[j] * M11[i, k] - \
                                   M21[j, i] * M11[j, k] - M21[j, k] * M11[j, i]
                        temp_kk2 = S113 - margvars[k] * m111 - margskews[k] * M11[i, j] - \
                                   M21[k, i] * M11[k, j] - M21[k, j] * M11[k, i]
                        
                        cov_ii = np.sqrt(margvars[i] * np.sqrt(margkurts[j] / margkurts[k]**3)) * temp_kk4 / 2.0 + \
                                 np.sqrt(margvars[i] * np.sqrt(margkurts[k] / margkurts[j]**3)) * temp_jj4 / 2.0 + \
                                 np.sqrt(np.sqrt(margkurts[j] * margkurts[k]) / margvars[i]) * temp_ii2
                        cov_jj = np.sqrt(margvars[j] * np.sqrt(margkurts[i] / margkurts[k]**3)) * temp_kk4 / 2.0 + \
                                 np.sqrt(margvars[j] * np.sqrt(margkurts[k] / margkurts[i]**3)) * temp_ii4 / 2.0 + \
                                 np.sqrt(np.sqrt(margkurts[i] * margkurts[k]) / margvars[j]) * temp_jj2
                        cov_kk = np.sqrt(margvars[k] * np.sqrt(margkurts[j] / margkurts[i]**3)) * temp_ii4 / 2.0 + \
                                 np.sqrt(margvars[k] * np.sqrt(margkurts[i] / margkurts[j]**3)) * temp_jj4 / 2.0 + \
                                 np.sqrt(np.sqrt(margkurts[i] * margkurts[j]) / margvars[k]) * temp_kk2
                        val = r4 * np.sqrt(r5) * (cov_ii + cov_jj + cov_kk)
                cm3_sum += mult * val
    return cm3_sum / T_obs

def _calc_VM4(Xc, Xc2, M11, M21, M22, M31, M32, M41, M42, T_obs, N_assets):
    vm4_0 = 0.0
    vm4_1 = 0.0
    vm4_2 = 0.0
    
    T_obs = float(T_obs)
    P = N_assets
    
    for ii in range(P):
        for jj in range(ii, P):
            for kk in range(jj, P):
                for ll in range(kk, P):
                    if ii == jj == kk == ll:
                        S8 = np.sum(Xc2[:, ii]**4)
                        temp = (S8 / T_obs - M31[ii, ii]**2 - 
                                8.0 * M32[ii, ii] * M21[ii, ii] + 
                                16.0 * M11[ii, ii] * M21[ii, ii]**2) / T_obs
                        vm4_0 += temp
                        vm4_1 += temp / P
                        vm4_2 += temp
                    elif ii == jj == kk:
                        S62 = np.sum(Xc2[:, ii]**3 * Xc2[:, ll])
                        val = (S62 / T_obs - M31[ii, ll]**2 - 
                               6.0 * M41[ii, ll] * M21[ii, ll] - 
                               2.0 * M32[ii, ll] * M21[ii, ii] + 
                               6.0 * M11[ii, ll] * M21[ii, ll] * M21[ii, ii] + 
                               6.0 * M11[ii, ii] * M21[ii, ll]**2 + 
                               M21[ii, ii]**2 * M11[ll, ll] + 
                               3.0 * M21[ii, ll]**2 * M11[ii, ii]) / T_obs
                        vm4_0 += 4.0 * val
                    elif ii == jj and kk == ll:
                        S44 = np.sum(Xc2[:, ii]**2 * Xc2[:, kk]**2)
                        val = (S44 / T_obs - M22[ii, kk]**2 - 
                               4.0 * M32[ii, kk] * M21[kk, ii] - 
                               4.0 * M32[kk, ii] * M21[ii, kk] + 
                               8.0 * M11[ii, kk] * M21[kk, ii] * M21[ii, kk] + 
                               4.0 * M21[ii, kk]**2 * M11[kk, kk] + 
                               4.0 * M21[kk, ii]**2 * M11[ii, ii]) / T_obs
                        vm4_0 += 6.0 * val
                        
                        # cov with T (equal marginals)
                        temp_cov = 0.0
                        for mm in range(P):
                            S222 = np.sum(Xc2[:, ii] * Xc2[:, kk] * Xc2[:, mm])
                            temp_cov += 2.0 * M11[mm, mm] * (S222 / T_obs - M11[mm, mm] * M22[ii, kk] - 
                                         2.0 * M21[mm, ii] * M21[kk, ii] - 2.0 * M21[mm, kk] * M21[ii, kk]) / T_obs
                        vm4_1 += 6.0 * temp_cov / P
                        
                        # cov with T (unequal marginals)
                        temp_ii = M42[ii, kk] - M11[ii, ii] * M22[ii, kk] - 2.0 * M21[ii, ii] * M21[kk, ii] - 2.0 * M21[ii, kk]**2
                        temp_kk = M42[kk, ii] - M11[kk, kk] * M22[kk, ii] - 2.0 * M21[kk, kk] * M21[ii, kk] - 2.0 * M21[kk, ii]**2
                        vm4_2 += 6.0 * (M11[kk, kk] * temp_ii + M11[ii, ii] * temp_kk) / T_obs
                    elif ii == jj:
                        S422 = np.sum(Xc2[:, ii]**2 * Xc2[:, kk] * Xc2[:, ll])
                        S311 = np.sum(Xc2[:, ii] * Xc[:, ii] * Xc[:, kk] * Xc[:, ll])
                        S221 = np.sum(Xc2[:, ii] * Xc2[:, kk] * Xc[:, ll])
                        S212 = np.sum(Xc2[:, ii] * Xc[:, kk] * Xc2[:, ll])
                        m211 = np.mean(Xc2[:, ii] * Xc[:, kk] * Xc[:, ll])
                        m111 = np.mean(Xc[:, ii] * Xc[:, kk] * Xc[:, ll])
                        
                        val = (S422 / T_obs - m211**2 - 4.0 * S311 / T_obs * m111 - 
                               2.0 * S221 / T_obs * M21[ii, ll] - 2.0 * S212 / T_obs * M21[ii, kk] + 
                               4.0 * M11[ii, ll] * M21[ii, kk] * m111 + 4.0 * M11[ii, kk] * m111 * M21[ii, ll] + 
                               2.0 * M11[kk, ll] * M21[ii, ll] * M21[ii, kk] + 
                               4.0 * M11[ii, ii] * m111**2 + M21[ii, kk]**2 * M11[ll, ll] + 
                               M21[ii, ll]**2 * M11[kk, kk]) / T_obs
                        vm4_0 += 12.0 * val
                    elif jj == kk == ll:
                        S62 = np.sum(Xc2[:, jj]**3 * Xc2[:, ii])
                        val = (S62 / T_obs - M31[jj, ii]**2 - 
                               6.0 * M41[jj, ii] * M21[jj, ii] - 
                               2.0 * M32[jj, ii] * M21[jj, jj] + 
                               6.0 * M11[jj, ii] * M21[jj, ii] * M21[jj, jj] + 
                               6.0 * M11[jj, jj] * M21[jj, ii]**2 + 
                               M21[jj, jj]**2 * M11[ii, ii] + 
                               3.0 * M21[jj, ii]**2 * M11[jj, jj]) / T_obs
                        vm4_0 += 4.0 * val
                    elif jj == kk:
                        S422 = np.sum(Xc2[:, jj]**2 * Xc2[:, ii] * Xc2[:, ll])
                        S311 = np.sum(Xc2[:, jj] * Xc[:, jj] * Xc[:, ii] * Xc[:, ll])
                        S221 = np.sum(Xc2[:, jj] * Xc2[:, ii] * Xc[:, ll])
                        S212 = np.sum(Xc2[:, jj] * Xc[:, ii] * Xc2[:, ll])
                        m211 = np.mean(Xc2[:, jj] * Xc[:, ii] * Xc[:, ll])
                        m111 = np.mean(Xc[:, jj] * Xc[:, ii] * Xc[:, ll])
                        
                        val = (S422 / T_obs - m211**2 - 4.0 * S311 / T_obs * m111 - 
                               2.0 * S221 / T_obs * M21[jj, ll] - 2.0 * S212 / T_obs * M21[jj, ii] + 
                               4.0 * M11[jj, ll] * M21[jj, ii] * m111 + 4.0 * M11[jj, ii] * m111 * M21[jj, ll] + 
                               2.0 * M11[ii, ll] * M21[jj, ll] * M21[jj, ii] + 
                               4.0 * M11[jj, jj] * m111**2 + M21[jj, ii]**2 * M11[ll, ll] + 
                               M21[jj, ll]**2 * M11[ii, ii]) / T_obs
                        vm4_0 += 12.0 * val
                    elif kk == ll:
                        S422 = np.sum(Xc2[:, kk]**2 * Xc2[:, ii] * Xc2[:, jj])
                        S311 = np.sum(Xc2[:, kk] * Xc[:, kk] * Xc[:, ii] * Xc[:, jj])
                        S221 = np.sum(Xc2[:, kk] * Xc2[:, ii] * Xc[:, jj])
                        S212 = np.sum(Xc2[:, kk] * Xc[:, ii] * Xc2[:, jj])
                        m211 = np.mean(Xc2[:, kk] * Xc[:, ii] * Xc[:, jj])
                        m111 = np.mean(Xc[:, kk] * Xc[:, ii] * Xc[:, jj])
                        
                        val = (S422 / T_obs - m211**2 - 4.0 * S311 / T_obs * m111 - 
                               2.0 * S221 / T_obs * M21[kk, jj] - 2.0 * S212 / T_obs * M21[kk, ii] + 
                               4.0 * M11[kk, jj] * M21[kk, ii] * m111 + 4.0 * M11[kk, ii] * m111 * M21[kk, jj] + 
                               2.0 * M11[ii, jj] * M21[kk, jj] * M21[kk, ii] + 
                               4.0 * M11[kk, kk] * m111**2 + M21[kk, ii]**2 * M11[jj, jj] + 
                               M21[kk, jj]**2 * M11[ii, ii]) / T_obs
                        vm4_0 += 12.0 * val
                    else:
                        S2222 = np.sum(Xc2[:, ii] * Xc2[:, jj] * Xc2[:, kk] * Xc2[:, ll])
                        S2111 = np.sum(Xc2[:, ii] * Xc[:, jj] * Xc[:, kk] * Xc[:, ll])
                        S1211 = np.sum(Xc[:, ii] * Xc2[:, jj] * Xc[:, kk] * Xc[:, ll])
                        S1121 = np.sum(Xc[:, ii] * Xc[:, jj] * Xc2[:, kk] * Xc[:, ll])
                        S1112 = np.sum(Xc[:, ii] * Xc[:, jj] * Xc[:, kk] * Xc2[:, ll])
                        m1111 = np.mean(Xc[:, ii] * Xc[:, jj] * Xc[:, kk] * Xc[:, ll])
                        m0111 = np.mean(Xc[:, jj] * Xc[:, kk] * Xc[:, ll])
                        m1011 = np.mean(Xc[:, ii] * Xc[:, kk] * Xc[:, ll])
                        m1101 = np.mean(Xc[:, ii] * Xc[:, jj] * Xc[:, ll])
                        m1110 = np.mean(Xc[:, ii] * Xc[:, jj] * Xc[:, kk])
                        
                        val = (S2222 / T_obs - m1111**2 - 2.0 * S2111 / T_obs * m0111 - 
                               2.0 * S1211 / T_obs * m1011 - 2.0 * S1121 / T_obs * m1101 - 
                               2.0 * S1112 / T_obs * m1110 + 
                               2.0 * M11[ll, ii] * m1110 * m0111 + 2.0 * M11[ll, jj] * m1011 * m1110 + 
                               2.0 * M11[ll, kk] * m1101 * m1110 + 2.0 * M11[kk, ii] * m0111 * m1101 + 
                               2.0 * M11[kk, jj] * m1011 * m1101 + 2.0 * M11[jj, ii] * m0111 * m1011 + 
                               m1110**2 * M11[ll, ll] + m1101**2 * M11[kk, kk] + 
                               m1011**2 * M11[jj, jj] + m0111**2 * M11[ii, ii]) / T_obs
                        vm4_0 += 24.0 * val
 
    for ii in range(P):
        for jj in range(ii + 1, P):
            S44 = np.sum(Xc2[:, ii]**2 * Xc2[:, jj]**2)
            val = (S44 / T_obs - M22[ii, ii] * M22[jj, jj] - 
                   4.0 * M41[ii, jj] * M21[jj, jj] - 4.0 * M41[jj, ii] * M21[ii, ii] + 
                   16.0 * M21[ii, ii] * M21[jj, jj] * M11[ii, jj]) / (T_obs * P)
            vm4_1 += 2.0 * val
            
    return np.array([vm4_0, vm4_1, vm4_2])

def _calc_CM4_1F(Xc, Xc2, fc, fvar, fskew, fkurt, M11, M21, M22, M31, T_obs, N_assets):
    T_obs = float(T_obs)
    P = N_assets
    fvar2 = fvar**2
    fvar3 = fvar2 * fvar
    fvar4 = fvar2 * fvar2
    fvar5 = fvar2 * fvar3
    
    covXf = np.mean(Xc * fc, axis=0)
    X1f2 = np.mean(Xc * (fc**2), axis=0)
    X1f4 = np.mean(Xc * (fc**4), axis=0)
    
    X11f1 = (Xc.T @ (Xc * fc)) / T_obs
    
    cm4_sum = 0.0
    
    for ii in range(P):
        for jj in range(ii, P):
            for kk in range(jj, P):
                for ll in range(kk, P):
                    mult = 1.0
                    if ii == jj == kk == ll:
                        S8 = np.sum(Xc2[:, ii]**4)
                        S5 = np.sum(Xc2[:, ii]**2 * Xc[:, ii])
                        val = (S8 / T_obs - M31[ii, ii]**2 - 
                               8.0 * S5 / T_obs * M21[ii, ii] + 
                               16.0 * M11[ii, ii] * M21[ii, ii]**2) / T_obs
                        cm4_sum += val
                        continue
                    elif ii == jj == kk:
                        mult = 4.0
                        S411 = np.mean(Xc2[:, ii]**2 * Xc[:, ll] * fc[:, 0])
                        S321 = np.mean(Xc[:, ii]**3 * Xc2[:, ll] * fc[:, 0])
                        S51 = np.mean(Xc[:, ii]**5 * Xc[:, ll])
                        S312 = np.mean(Xc[:, ii]**3 * Xc[:, ll] * fc[:, 0]**2)
                        S314 = np.mean(Xc[:, ii]**3 * Xc[:, ll] * fc[:, 0]**4)
                        S311 = np.mean(Xc[:, ii]**3 * Xc[:, ll] * fc[:, 0])
                        
                        temp_i0 = S411 - M31[ll, ii] * covXf[ii] - 3.0 * M21[ll, ii] * X11f1[ii, ii] - M21[ii, ii] * X11f1[ll, ii]
                        temp_l0 = S321 - M31[ll, ii] * covXf[ll] - 3.0 * M21[ll, ii] * X11f1[ll, ii] - M21[ii, ii] * X11f1[ll, ll]
                        temp_ii = S51 - M31[ll, ii] * M11[ii, ii] - 4.0 * M21[ll, ii] * M21[ii, ii]
                        temp_var = S312 - M31[ll, ii] * fvar - 3.0 * M21[ll, ii] * X1f2[ii] - M21[ii, ii] * X1f2[ll]
                        temp_kurt = S314 - M31[ll, ii] * fkurt - 4.0 * S311 * fkurt + 4.0 * fskew * (3.0 * M21[ll, ii] * covXf[ii] + M21[ii, ii] * covXf[ll])
                        
                        val = ((3.0 * covXf[ii]**2 * covXf[ll] * fkurt / fvar4 - 9.0 * covXf[ii]**2 * covXf[ll] / fvar2 + 3.0 * covXf[ll] * M11[ii, ii] / fvar) * temp_i0 +
                               (covXf[ii]**3 * fkurt / fvar4 + 3.0 * covXf[ii] * (M11[ii, ii] - covXf[ii]**2 / fvar) / fvar) * temp_l0 +
                               3.0 * covXf[ii] * covXf[ll] / fvar * temp_ii +
                               (-4.0 * covXf[ii]**3 * covXf[ll] * fkurt / fvar5 - 3.0 * covXf[ii] * covXf[ll] * M11[ii, ii] / fvar2 + 6.0 * covXf[ii]**3 * covXf[ll] / fvar3) * temp_var +
                               covXf[ii]**3 * covXf[ll] * temp_kurt / fvar4) / T_obs
                    elif ii == jj and kk == ll:
                        mult = 6.0
                        S321 = np.mean(Xc[:, ii]**3 * Xc2[:, kk] * fc[:, 0])
                        S231 = np.mean(Xc2[:, ii] * Xc[:, kk]**3 * fc[:, 0])
                        S420 = np.mean(Xc2[:, ii]**2 * Xc2[:, kk])
                        S240 = np.mean(Xc2[:, ii] * Xc2[:, kk]**2)
                        S222 = np.mean(Xc2[:, ii] * Xc2[:, kk] * fc[:, 0]**2)
                        S224 = np.mean(Xc2[:, ii] * Xc2[:, kk] * fc[:, 0]**4)
                        
                        temp_i0 = S321 - M22[kk, ii] * covXf[ii] - 2.0 * M21[ii, kk] * X11f1[ii, ii] - 2.0 * M21[kk, ii] * X11f1[kk, ii]
                        temp_k0 = S231 - M22[ii, kk] * covXf[kk] - 2.0 * M21[kk, ii] * X11f1[kk, kk] - 2.0 * M21[ii, kk] * X11f1[ii, kk]
                        temp_ii = S420 - M22[kk, ii] * M11[ii, ii] - 2.0 * M21[ii, kk] * M21[ii, ii] - 2.0 * M21[kk, ii] * M21[kk, ii]
                        temp_kk = S240 - M22[ii, kk] * M11[kk, kk] - 2.0 * M21[kk, ii] * M21[kk, kk] - 2.0 * M21[ii, kk] * M21[ii, kk]
                        temp_var = S222 - M22[kk, ii] * fvar - 2.0 * M21[ii, kk] * X1f2[ii] - 2.0 * M21[kk, ii] * X1f2[kk]
                        temp_kurt = S224 - M22[kk, ii] * fkurt - 2.0 * M21[ii, kk] * X1f4[ii] - 2.0 * M21[kk, ii] * X1f4[kk] + 8.0 * fskew * (M21[ii, kk] * covXf[ii] + M21[kk, ii] * covXf[kk])
                        
                        val = ((2.0 * covXf[ii] * covXf[kk]**2 * fkurt / fvar4 - 2.0 * covXf[ii] * covXf[kk]**2 / fvar2) * temp_i0 +
                               (2.0 * covXf[ii]**2 * covXf[kk] * fkurt / fvar4 - 2.0 * covXf[ii]**2 * covXf[kk] / fvar2) * temp_k0 +
                               M11[kk, kk] * temp_ii + M11[ii, ii] * temp_kk +
                               (-4.0 * covXf[ii]**2 * covXf[kk]**2 * fkurt / fvar5 + 2.0 * covXf[ii]**2 * covXf[kk]**2 / fvar3) * temp_var +
                               covXf[ii]**2 * covXf[kk]**2 * temp_kurt / fvar4) / T_obs
                    elif ii == jj:
                        mult = 12.0
                        S3111 = np.mean(Xc[:, ii]**3 * Xc[:, kk] * Xc[:, ll] * fc[:, 0])
                        S2211 = np.mean(Xc2[:, ii] * Xc2[:, kk] * Xc[:, ll] * fc[:, 0])
                        S2121 = np.mean(Xc2[:, ii] * Xc[:, kk] * Xc2[:, ll] * fc[:, 0])
                        S4110 = np.mean(Xc2[:, ii]**2 * Xc[:, kk] * Xc[:, ll])
                        S2112 = np.mean(Xc2[:, ii] * Xc[:, kk] * Xc[:, ll] * fc[:, 0]**2)
                        S2114 = np.mean(Xc2[:, ii] * Xc[:, kk] * Xc[:, ll] * fc[:, 0]**4)
                        S2111 = np.mean(Xc2[:, ii] * Xc[:, kk] * Xc[:, ll] * fc[:, 0])
                        m2110 = np.mean(Xc2[:, ii] * Xc[:, kk] * Xc[:, ll])
                        m1110 = np.mean(Xc[:, ii] * Xc[:, kk] * Xc[:, ll])
                        
                        temp_i0 = S3111 - m2110 * covXf[ii] - 2.0 * m1110 * X11f1[ii, ii] - M21[ll, ii] * X11f1[kk, ii] - M21[kk, ii] * X11f1[ll, ii]
                        temp_k0 = S2211 - m2110 * covXf[kk] - 2.0 * m1110 * X11f1[kk, ii] - M21[ll, ii] * X11f1[kk, kk] - M21[kk, ii] * X11f1[ll, kk]
                        temp_l0 = S2121 - m2110 * covXf[ll] - 2.0 * m1110 * X11f1[ll, ii] - M21[kk, ii] * X11f1[ll, ll] - M21[ll, ii] * X11f1[kk, ll]
                        temp_ii = S4110 - m2110 * M11[ii, ii] - 2.0 * m1110 * M21[ii, ii] - 2.0 * M21[ll, ii] * M21[kk, ii]
                        temp_var = S2112 - m2110 * fvar - 2.0 * m1110 * X1f2[ii] - M21[ll, ii] * X1f2[kk] - M21[kk, ii] * X1f2[ll]
                        temp_kurt = S2114 - m2110 * fkurt - 2.0 * m1110 * X1f4[ii] - M21[ll, ii] * X1f4[kk] - M21[kk, ii] * X1f4[ll] - 4.0 * S2111 * fskew + \
                                    4.0 * fskew * (2.0 * m1110 * covXf[ii] + M21[ll, ii] * covXf[kk] + M21[kk, ii] * covXf[ll])
                        
                        val = ((2.0 * covXf[ii] * covXf[kk] * covXf[ll] * fkurt / fvar4 - 2.0 * covXf[ii] * covXf[kk] * covXf[ll] / fvar2) * temp_i0 +
                               (covXf[ii]**2 * covXf[ll] * fkurt / fvar4 + covXf[ll] * (M11[ii, ii] - covXf[ii]**2 / fvar) / fvar) * temp_k0 +
                               (covXf[ii]**2 * covXf[kk] * fkurt / fvar4 + covXf[kk] * (M11[ii, ii] - covXf[ii]**2 / fvar) / fvar) * temp_l0 +
                               covXf[kk] * covXf[ll] / fvar * temp_ii +
                               (-4.0 * covXf[ii]**2 * covXf[kk] * covXf[ll] * fkurt / fvar5 + 2.0 * covXf[kk] * covXf[ll] * covXf[ii]**2 / fvar3 - covXf[kk] * covXf[ll] * M11[ii, ii] / fvar2) * temp_var +
                               covXf[ii]**2 * covXf[kk] * covXf[ll] * temp_kurt / fvar4) / T_obs
                    elif jj == kk == ll:
                        mult = 4.0
                        S411 = np.mean(Xc2[:, jj]**2 * Xc[:, ii] * fc[:, 0])
                        S321 = np.mean(Xc[:, jj]**3 * Xc2[:, ii] * fc[:, 0])
                        S51 = np.mean(Xc[:, jj]**5 * Xc[:, ii])
                        S312 = np.mean(Xc[:, jj]**3 * Xc[:, ii] * fc[:, 0]**2)
                        S314 = np.mean(Xc[:, jj]**3 * Xc[:, ii] * fc[:, 0]**4)
                        S311 = np.mean(Xc[:, jj]**3 * Xc[:, ii] * fc[:, 0])
                        
                        temp_j0 = S411 - M31[ii, jj] * covXf[jj] - 3.0 * M21[ii, jj] * X11f1[jj, jj] - M21[jj, jj] * X11f1[ii, jj]
                        temp_i0 = S321 - M31[ii, jj] * covXf[ii] - 3.0 * M21[ii, jj] * X11f1[jj, jj] - M21[jj, jj] * X11f1[ii, ii]
                        temp_jj = S51 - M31[ii, jj] * M11[jj, jj] - 4.0 * M21[ii, jj] * M21[jj, jj]
                        temp_var = S312 - M31[ii, jj] * fvar - 3.0 * M21[ii, jj] * X1f2[jj] - M21[jj, jj] * X1f2[ii]
                        temp_kurt = S314 - M31[ii, jj] * fkurt - 4.0 * S311 * fkurt + 4.0 * fskew * (3.0 * M21[ii, jj] * covXf[jj] + M21[jj, jj] * covXf[ii])
                        
                        val = ((3.0 * covXf[jj]**2 * covXf[ii] * fkurt / fvar4 - 9.0 * covXf[jj]**2 * covXf[ii] / fvar2 + 3.0 * covXf[ii] * M11[jj, jj] / fvar) * temp_j0 +
                               (covXf[jj]**3 * fkurt / fvar4 + 3.0 * covXf[jj] * (M11[jj, jj] - covXf[jj]**2 / fvar) / fvar) * temp_i0 +
                               3.0 * covXf[jj] * covXf[ii] / fvar * temp_jj +
                               (-4.0 * covXf[jj]**3 * covXf[ii] * fkurt / fvar5 - 3.0 * covXf[jj] * covXf[ii] * M11[jj, jj] / fvar2 + 6.0 * covXf[jj]**3 * covXf[ii] / fvar3) * temp_var +
                               covXf[jj]**3 * covXf[ii] * temp_kurt / fvar4) / T_obs
                    elif jj == kk:
                        mult = 12.0
                        S3111 = np.mean(Xc[:, jj]**3 * Xc[:, ii] * Xc[:, ll] * fc[:, 0])
                        S2211 = np.mean(Xc2[:, jj] * Xc2[:, ii] * Xc[:, ll] * fc[:, 0])
                        S2121 = np.mean(Xc2[:, jj] * Xc[:, ii] * Xc2[:, ll] * fc[:, 0])
                        S4110 = np.mean(Xc2[:, jj]**2 * Xc[:, ii] * Xc[:, ll])
                        S2112 = np.mean(Xc2[:, jj] * Xc[:, ii] * Xc[:, ll] * fc[:, 0]**2)
                        S2114 = np.mean(Xc2[:, jj] * Xc[:, ii] * Xc[:, ll] * fc[:, 0]**4)
                        S2111 = np.mean(Xc2[:, jj] * Xc[:, ii] * Xc[:, ll] * fc[:, 0])
                        m2110 = np.mean(Xc2[:, jj] * Xc[:, ii] * Xc[:, ll])
                        m1110 = np.mean(Xc[:, jj] * Xc[:, ii] * Xc[:, ll])
                        
                        temp_j0 = S3111 - m2110 * covXf[jj] - 2.0 * m1110 * X11f1[jj, jj] - M21[ll, jj] * X11f1[ii, jj] - M21[ii, jj] * X11f1[ll, jj]
                        temp_i0 = S2211 - m2110 * covXf[ii] - 2.0 * m1110 * X11f1[ii, jj] - M21[ll, jj] * X11f1[ii, ii] - M21[ii, jj] * X11f1[ll, ii]
                        temp_l0 = S2121 - m2110 * covXf[ll] - 2.0 * m1110 * X11f1[ll, jj] - M21[ii, jj] * X11f1[ll, ll] - M21[ll, jj] * X11f1[ii, ll]
                        temp_jj = S4110 - m2110 * M11[jj, jj] - 2.0 * m1110 * M21[jj, jj] - 2.0 * M21[ll, jj] * M21[ii, jj]
                        temp_var = S2112 - m2110 * fvar - 2.0 * m1110 * X1f2[jj] - M21[ll, jj] * X1f2[ii] - M21[ii, jj] * X1f2[ll]
                        temp_kurt = S2114 - m2110 * fkurt - 2.0 * m1110 * X1f4[jj] - M21[ll, jj] * X1f4[ii] - M21[ii, jj] * X1f4[ll] - 4.0 * S2111 * fskew + \
                                    4.0 * fskew * (2.0 * m1110 * covXf[jj] + M21[ll, jj] * covXf[ii] + M21[ii, jj] * covXf[ll])
                        
                        val = ((2.0 * covXf[jj] * covXf[ii] * covXf[ll] * fkurt / fvar4 - 2.0 * covXf[jj] * covXf[ii] * covXf[ll] / fvar2) * temp_j0 +
                               (covXf[jj]**2 * covXf[ll] * fkurt / fvar4 + covXf[ll] * (M11[jj, jj] - covXf[jj]**2 / fvar) / fvar) * temp_i0 +
                               (covXf[jj]**2 * covXf[ii] * fkurt / fvar4 + covXf[ii] * (M11[jj, jj] - covXf[jj]**2 / fvar) / fvar) * temp_l0 +
                               covXf[ii] * covXf[ll] / fvar * temp_jj +
                               (-4.0 * covXf[jj]**2 * covXf[ii] * covXf[ll] * fkurt / fvar5 + 2.0 * covXf[ii] * covXf[ll] * covXf[jj]**2 / fvar3 - covXf[ii] * covXf[ll] * M11[jj, jj] / fvar2) * temp_var +
                               covXf[jj]**2 * covXf[ii] * covXf[ll] * temp_kurt / fvar4) / T_obs
                    elif kk == ll:
                        mult = 12.0
                        S3111 = np.mean(Xc[:, kk]**3 * Xc[:, ii] * Xc[:, jj] * fc[:, 0])
                        S2211 = np.mean(Xc2[:, kk] * Xc2[:, ii] * Xc[:, jj] * fc[:, 0])
                        S2121 = np.mean(Xc2[:, kk] * Xc[:, ii] * Xc2[:, jj] * fc[:, 0])
                        S4110 = np.mean(Xc2[:, kk]**2 * Xc[:, ii] * Xc[:, jj])
                        S2112 = np.mean(Xc2[:, kk] * Xc[:, ii] * Xc[:, jj] * fc[:, 0]**2)
                        S2114 = np.mean(Xc2[:, kk] * Xc[:, ii] * Xc[:, jj] * fc[:, 0]**4)
                        S2111 = np.mean(Xc2[:, kk] * Xc[:, ii] * Xc[:, jj] * fc[:, 0])
                        m2110 = np.mean(Xc2[:, kk] * Xc[:, ii] * Xc[:, jj])
                        m1110 = np.mean(Xc[:, kk] * Xc[:, ii] * Xc[:, jj])
                        
                        temp_k0 = S3111 - m2110 * covXf[kk] - 2.0 * m1110 * X11f1[kk, kk] - M21[jj, kk] * X11f1[ii, kk] - M21[ii, kk] * X11f1[jj, kk]
                        temp_i0 = S2211 - m2110 * covXf[ii] - 2.0 * m1110 * X11f1[ii, kk] - M21[jj, kk] * X11f1[ii, ii] - M21[ii, kk] * X11f1[jj, ii]
                        temp_j0 = S2121 - m2110 * covXf[jj] - 2.0 * m1110 * X11f1[jj, kk] - M21[ii, kk] * X11f1[jj, jj] - M21[jj, kk] * X11f1[ii, jj]
                        temp_kk = S4110 - m2110 * M11[kk, kk] - 2.0 * m1110 * M21[kk, kk] - 2.0 * M21[jj, kk] * M21[ii, kk]
                        temp_var = S2112 - m2110 * fvar - 2.0 * m1110 * X1f2[kk] - M21[jj, kk] * X1f2[ii] - M21[ii, kk] * X1f2[jj]
                        temp_kurt = S2114 - m2110 * fkurt - 2.0 * m1110 * X1f4[kk] - M21[jj, kk] * X1f4[ii] - M21[ii, kk] * X1f4[jj] - 4.0 * S2111 * fskew + \
                                    4.0 * fskew * (2.0 * m1110 * covXf[kk] + M21[jj, kk] * covXf[ii] + M21[ii, kk] * covXf[jj])
                        
                        val = ((2.0 * covXf[kk] * covXf[ii] * covXf[jj] * fkurt / fvar4 - 2.0 * covXf[kk] * covXf[ii] * covXf[jj] / fvar2) * temp_k0 +
                               (covXf[kk]**2 * covXf[jj] * fkurt / fvar4 + covXf[jj] * (M11[kk, kk] - covXf[kk]**2 / fvar) / fvar) * temp_i0 +
                               (covXf[kk]**2 * covXf[ii] * fkurt / fvar4 + covXf[ii] * (M11[kk, kk] - covXf[kk]**2 / fvar) / fvar) * temp_j0 +
                               covXf[ii] * covXf[jj] / fvar * temp_kk +
                               (-4.0 * covXf[kk]**2 * covXf[ii] * covXf[jj] * fkurt / fvar5 + 2.0 * covXf[ii] * covXf[jj] * covXf[kk]**2 / fvar3 - covXf[ii] * covXf[jj] * M11[kk, kk] / fvar2) * temp_var +
                               covXf[kk]**2 * covXf[ii] * covXf[jj] * temp_kurt / fvar4) / T_obs
                    else:
                        mult = 24.0
                        S11114 = np.mean(Xc[:, ii] * Xc[:, jj] * Xc[:, kk] * Xc[:, ll] * fc[:, 0]**4)
                        S11112 = np.mean(Xc[:, ii] * Xc[:, jj] * Xc[:, kk] * Xc[:, ll] * fc[:, 0]**2)
                        S11111 = np.mean(Xc[:, ii] * Xc[:, jj] * Xc[:, kk] * Xc[:, ll] * fc[:, 0])
                        S21111 = np.mean(Xc2[:, ii] * Xc[:, jj] * Xc[:, kk] * Xc[:, ll] * fc[:, 0])
                        S12111 = np.mean(Xc[:, ii] * Xc2[:, jj] * Xc[:, kk] * Xc[:, ll] * fc[:, 0])
                        S11211 = np.mean(Xc[:, ii] * Xc[:, jj] * Xc2[:, kk] * Xc[:, ll] * fc[:, 0])
                        S11121 = np.mean(Xc[:, ii] * Xc[:, jj] * Xc[:, kk] * Xc2[:, ll] * fc[:, 0])
                        m1111 = np.mean(Xc[:, ii] * Xc[:, jj] * Xc[:, kk] * Xc[:, ll])
                        m01110 = np.mean(Xc[:, jj] * Xc[:, kk] * Xc[:, ll])
                        m10110 = np.mean(Xc[:, ii] * Xc[:, kk] * Xc[:, ll])
                        m11010 = np.mean(Xc[:, ii] * Xc[:, jj] * Xc[:, ll])
                        m11100 = np.mean(Xc[:, ii] * Xc[:, jj] * Xc[:, kk])
                        
                        temp_ii = S21111 - m1111 * covXf[ii] - m01110 * X11f1[ii, ii] - m10110 * X11f1[jj, ii] - m11010 * X11f1[kk, ii] - m11100 * X11f1[ll, ii]
                        temp_jj = S12111 - m1111 * covXf[jj] - m01110 * X11f1[ii, jj] - m10110 * X11f1[jj, jj] - m11010 * X11f1[kk, jj] - m11100 * X11f1[ll, jj]
                        temp_kk = S11211 - m1111 * covXf[kk] - m01110 * X11f1[ii, kk] - m10110 * X11f1[jj, kk] - m11010 * X11f1[kk, kk] - m11100 * X11f1[ll, kk]
                        temp_ll = S11121 - m1111 * covXf[ll] - m01110 * X11f1[ii, ll] - m10110 * X11f1[jj, ll] - m11010 * X11f1[kk, ll] - m11100 * X11f1[ll, ll]
                        temp_kurt = S11114 - m1111 * fkurt - m01110 * X1f4[ii] - m10110 * X1f4[jj] - m11010 * X1f4[kk] - m11100 * X1f4[ll] - 4.0 * S11111 * fskew + \
                                    4.0 * fskew * (m01110 * covXf[ii] + m10110 * covXf[jj] + m11010 * covXf[kk] + m11100 * covXf[ll])
                        temp_var = S11112 - m1111 * fvar - m01110 * X1f2[ii] - m10110 * X1f2[jj] - m11010 * X1f2[kk] - m11100 * X1f2[ll]
                        
                        val = ((covXf[jj] * covXf[kk] * covXf[ll] * temp_ii +
                                covXf[ii] * covXf[kk] * covXf[ll] * temp_jj +
                                covXf[ii] * covXf[jj] * covXf[ll] * temp_kk +
                                covXf[ii] * covXf[jj] * covXf[kk] * temp_ll) * fkurt +
                                covXf[ii] * covXf[jj] * covXf[kk] * covXf[ll] * temp_kurt -
                                4.0 * covXf[ii] * covXf[jj] * covXf[kk] * covXf[ll] * fkurt * temp_var / fvar) / fvar4
                    cm4_sum += mult * val
                    
    return cm4_sum / T_obs

def _calc_CM4_CC(Xc, Xc2, m11, m21, m22, m31, m32, m33, m41, r3, r5, r6, r7, marg6s, marg7s, T_obs, N_assets):
    T_obs = float(T_obs)
    P = N_assets
    rCM4 = 0.0
    
    for ii in range(P):
        for jj in range(ii, P):
            for kk in range(jj, P):
                for ll in range(kk, P):
                    if ii == jj == kk == ll:
                        S8 = np.sum(Xc2[:, ii]**4)
                        val = (S8 / T_obs - m31[ii, ii]**2 - 
                               8.0 * m41[ii, ii] * m21[ii, ii] + 
                               16.0 * m11[ii, ii] * m21[ii, ii]**2) / T_obs
                        rCM4 += val
                    elif ii == jj == kk:
                        S91 = np.sum(Xc2[:, ii]**4 * Xc[:, ii] * Xc[:, ll])
                        S62 = np.sum(Xc2[:, ii]**3 * Xc2[:, ll])
                        
                        temp_ii = S91 / T_obs - m31[ll, ii] * marg6s[ii] - \
                                  3.0 * m21[ll, ii] * marg7s[ii] - m21[ii, ii] * S62 / T_obs - \
                                  6.0 * m41[ii, ii] * m41[ll, ii] + \
                                  18.0 * m41[ii, ii] * m21[ll, ii] * m11[ii, ii] + \
                                  6.0 * m41[ii, ii] * m21[ll, ii] * m11[ll, ii]
                        temp_ll = m33[ll, ii] - m31[ll, ii] * m11[ll, ll] - \
                                  3.0 * m21[ii, ll] * m21[ll, ii] - m21[ii, ii] * m21[ll, ll]
                        
                        val = 2.0 * r3 * (np.sqrt(m11[ll, ll] / marg6s[ii]) * temp_ii + \
                                         np.sqrt(marg6s[ii] / m11[ll, ll]) * temp_ll) / T_obs
                        rCM4 += val
                    elif ii == jj and kk == ll:
                        S62 = np.sum(Xc2[:, ii]**3 * Xc2[:, kk])
                        S26 = np.sum(Xc2[:, ii] * Xc2[:, kk]**3)
                        
                        temp_ii = S62 / T_obs - m22[kk, ii] * m22[ii, ii] - \
                                  4.0 * m32[kk, ii] * m21[ii, ii] - 2.0 * m21[ii, kk] * m32[ii, ii] - \
                                  2.0 * m21[kk, ii] * m41[kk, ii] + 8.0 * m21[kk, ii] * m21[ii, ii] * m11[kk, ii] + \
                                  8.0 * m21[ii, kk] * m21[ii, ii] * m11[ii, ii]
                        temp_kk = S26 / T_obs - m22[ii, kk] * m22[kk, kk] - \
                                  4.0 * m32[ii, kk] * m21[kk, kk] - 2.0 * m21[kk, ii] * m32[kk, kk] - \
                                  2.0 * m21[ii, kk] * m41[ii, kk] + 8.0 * m21[ii, kk] * m21[kk, kk] * m11[ii, kk] + \
                                  8.0 * m21[kk, ii] * m21[kk, kk] * m11[kk, kk]
                        
                        val = 3.0 * r5 * (np.sqrt(m22[kk, kk] / m22[ii, ii]) * temp_ii + \
                                         np.sqrt(m22[ii, ii] / m22[kk, kk]) * temp_kk) / T_obs
                        rCM4 += val
                    elif ii == jj:
                        S611 = np.sum(Xc2[:, ii]**3 * Xc[:, kk] * Xc[:, ll])
                        S251 = np.sum(Xc2[:, ii] * Xc[:, kk]**5 * Xc[:, ll])
                        S215 = np.sum(Xc2[:, ii] * Xc[:, kk] * Xc[:, ll]**5)
                        S311 = np.sum(Xc2[:, ii] * Xc[:, ii] * Xc[:, kk] * Xc[:, ll])
                        S221 = np.sum(Xc2[:, ii] * Xc2[:, kk] * Xc[:, ll])
                        S212 = np.sum(Xc2[:, ii] * Xc[:, kk] * Xc2[:, ll])
                        m211 = np.mean(Xc2[:, ii] * Xc[:, kk] * Xc[:, ll])
                        m111 = np.mean(Xc[:, ii] * Xc[:, kk] * Xc[:, ll])
                        
                        temp_ii = S611 / T_obs - m211 * m22[ii, ii] - 4.0 * S311 / T_obs * m21[ii, ii] - \
                                  2.0 * m111 * m32[ii, ii] - m21[ll, ii] * m41[kk, ii] - m21[kk, ii] * m41[ll, ii] + \
                                  4.0 * M21_val(m21, kk, ii, P) * m21[ii, ii] * m11[ll, ii] + 4.0 * M21_val(m21, ll, ii, P) * m21[ii, ii] * m11[kk, ii] + \
                                  8.0 * m111 * m21[ii, ii] * m11[ii, ii]
                        temp_kk = S251 / T_obs - m211 * m22[kk, kk] - 4.0 * S221 / T_obs * m21[kk, kk] - \
                                  m21[ll, ii] * m32[kk, kk] - 2.0 * m111 * m41[ii, kk] - M21_val(m21, kk, ii, P) * m41[ll, kk] + \
                                  4.0 * M21_val(m21, kk, ii, P) * m21[kk, kk] * m11[ll, kk] + 8.0 * m111 * m21[kk, kk] * m11[kk, ii] + \
                                  4.0 * M21_val(m21, ll, ii, P) * m11[kk, kk] * m21[kk, kk]
                        temp_ll = S215 / T_obs - m211 * m22[ll, ll] - 4.0 * S212 / T_obs * m21[ll, ll] - \
                                  M21_val(m21, kk, ii, P) * m32[ll, ll] - 2.0 * m111 * m41[ii, ll] - M21_val(m21, ll, ii, P) * m41[kk, ll] + \
                                  4.0 * M21_val(m21, ll, ii, P) * m21[ll, ll] * m11[kk, ll] + 8.0 * m111 * m21[ll, ll] * m11[ll, ii] + \
                                  4.0 * M21_val(m21, kk, ii, P) * m11[ll, ll] * m21[ll, ll]
                        
                        val = 3.0 * r6 * np.sqrt(r5) * \
                              (2.0 * np.sqrt(np.sqrt(m22[kk, kk] * m22[ll, ll]) / m22[ii, ii]) * temp_ii + \
                               np.sqrt(m22[ii, ii] * np.sqrt(m22[ll, ll] / m22[kk, kk]**3)) * temp_kk + \
                               np.sqrt(m22[ii, ii] * np.sqrt(m22[kk, kk] / m22[ll, ll]**3)) * temp_ll) / T_obs
                        rCM4 += val
                    elif jj == kk == ll:
                        S91 = np.sum(Xc2[:, jj]**4 * Xc[:, jj] * Xc[:, ii])
                        S62 = np.sum(Xc2[:, jj]**3 * Xc2[:, ii])
                        
                        temp_jj = S91 / T_obs - m31[ii, jj] * marg6s[jj] - \
                                  3.0 * m21[ii, jj] * marg7s[jj] - m21[jj, jj] * S62 / T_obs - \
                                  6.0 * m41[jj, jj] * m41[ii, jj] + \
                                  18.0 * m41[jj, jj] * m21[ii, jj] * m11[jj, jj] + \
                                  6.0 * m41[jj, jj] * m21[jj, jj] * m11[ii, jj]
                        temp_ii = m33[ii, jj] - m31[ii, jj] * m11[ii, ii] - \
                                  3.0 * m21[jj, ii] * m21[ii, jj] - m21[jj, jj] * m21[ii, ii]
                        
                        val = 2.0 * r3 * (np.sqrt(m11[ii, ii] / marg6s[jj]) * temp_jj + \
                                         np.sqrt(marg6s[jj] / m11[ii, ii]) * temp_ii) / T_obs
                        rCM4 += val
                    elif jj == kk:
                        S161 = np.sum(Xc[:, ii] * Xc2[:, jj]**3 * Xc[:, ll])
                        S521 = np.sum(Xc[:, ii]**5 * Xc2[:, jj] * Xc[:, ll])
                        S125 = np.sum(Xc[:, ii] * Xc2[:, jj] * Xc[:, ll]**5)
                        S131 = np.sum(Xc[:, ii] * Xc2[:, jj] * Xc[:, jj] * Xc[:, ll])
                        S221 = np.sum(Xc2[:, ii] * Xc2[:, jj] * Xc[:, ll])
                        S122 = np.sum(Xc[:, ii] * Xc2[:, jj] * Xc2[:, ll])
                        m121 = np.mean(Xc[:, ii] * Xc2[:, jj] * Xc[:, ll])
                        m111 = np.mean(Xc[:, ii] * Xc[:, jj] * Xc[:, ll])
                        
                        temp_jj = S161 / T_obs - m121 * m22[jj, jj] - 4.0 * S131 / T_obs * m21[jj, jj] - \
                                  2.0 * m111 * m32[jj, jj] - M21_val(m21, ll, jj, P) * m41[ii, jj] - M21_val(m21, ii, jj, P) * m41[ll, jj] + \
                                  4.0 * M21_val(m21, ii, jj, P) * m21[jj, jj] * m11[ll, jj] + 4.0 * M21_val(m21, ll, jj, P) * m21[jj, jj] * m11[ii, jj] + \
                                  8.0 * m111 * m21[jj, jj] * m11[jj, jj]
                        temp_ii = S521 / T_obs - m121 * m22[ii, ii] - 4.0 * S221 / T_obs * m21[ii, ii] - \
                                  M21_val(m21, ll, jj, P) * m32[ii, ii] - 2.0 * m111 * m41[jj, ii] - M21_val(m21, ii, jj, P) * m41[ll, ii] + \
                                  4.0 * M21_val(m21, ii, jj, P) * m21[ii, ii] * m11[ll, ii] + 8.0 * m111 * m21[ii, ii] * m11[jj, ii] + \
                                  4.0 * M21_val(m21, ll, jj, P) * m11[ii, ii] * m21[ii, ii]
                        temp_ll = S125 / T_obs - m121 * m22[ll, ll] - 4.0 * S122 / T_obs * m21[ll, ll] - \
                                  M21_val(m21, ii, jj, P) * m32[ll, ll] - 2.0 * m111 * m41[jj, ll] - M21_val(m21, ll, jj, P) * m41[ii, ll] + \
                                  4.0 * M21_val(m21, ll, jj, P) * m21[ll, ll] * m11[ii, ll] + 8.0 * m111 * m21[ll, ll] * m11[jj, ll] + \
                                  4.0 * M21_val(m21, ii, jj, P) * m11[ll, ll] * m21[ll, ll]
                        
                        val = 3.0 * r6 * np.sqrt(r5) * \
                              (2.0 * np.sqrt(np.sqrt(m22[ii, ii] * m22[ll, ll]) / m22[jj, jj]) * temp_jj + \
                               np.sqrt(m22[jj, jj] * np.sqrt(m22[ll, ll] / m22[ii, ii]**3)) * temp_ii + \
                               np.sqrt(m22[jj, jj] * np.sqrt(m22[ii, ii] / m22[ll, ll]**3)) * temp_ll) / T_obs
                        rCM4 += val
                    elif kk == ll:
                        S116 = np.sum(Xc[:, ii] * Xc[:, jj] * Xc[:, kk]**6)
                        S152 = np.sum(Xc[:, ii] * Xc[:, jj]**5 * Xc2[:, kk])
                        S512 = np.sum(Xc[:, ii]**5 * Xc[:, jj] * Xc2[:, kk])
                        S113 = np.sum(Xc[:, ii] * Xc[:, jj] * Xc2[:, kk] * Xc[:, kk])
                        S122 = np.sum(Xc[:, ii] * Xc2[:, jj] * Xc2[:, kk])
                        S212 = np.sum(Xc2[:, ii] * Xc[:, jj] * Xc2[:, kk])
                        m112 = np.mean(Xc[:, ii] * Xc[:, jj] * Xc2[:, kk])
                        m111 = np.mean(Xc[:, ii] * Xc[:, jj] * Xc[:, kk])
                        
                        temp_kk = S116 / T_obs - m112 * m22[kk, kk] - 4.0 * S113 / T_obs * m21[kk, kk] - \
                                  2.0 * m111 * m32[kk, kk] - M21_val(m21, jj, kk, P) * m41[ii, kk] - M21_val(m21, ii, kk, P) * m41[jj, kk] + \
                                  4.0 * M21_val(m21, ii, kk, P) * m21[kk, kk] * m11[jj, kk] + 4.0 * M21_val(m21, jj, kk, P) * m21[kk, kk] * m11[ii, kk] + \
                                  8.0 * m111 * m21[kk, kk] * m11[kk, kk]
                        temp_ii = S512 / T_obs - m112 * m22[ii, ii] - 4.0 * S212 / T_obs * m21[ii, ii] - \
                                  M21_val(m21, jj, kk, P) * m32[ii, ii] - 2.0 * m111 * m41[kk, ii] - M21_val(m21, ii, kk, P) * m41[jj, ii] + \
                                  4.0 * M21_val(m21, ii, kk, P) * m21[ii, ii] * m11[jj, ii] + 8.0 * m111 * m21[ii, ii] * m11[kk, ii] + \
                                  4.0 * M21_val(m21, jj, kk, P) * m11[ii, ii] * m21[ii, ii]
                        temp_jj = S152 / T_obs - m112 * m22[jj, jj] - 4.0 * S122 / T_obs * m21[jj, jj] - \
                                  M21_val(m21, ii, kk, P) * m32[jj, jj] - 2.0 * m111 * m41[kk, jj] - M21_val(m21, jj, kk, P) * m41[ii, jj] + \
                                  4.0 * M21_val(m21, jj, kk, P) * m21[jj, jj] * m11[ii, jj] + 8.0 * m111 * m21[jj, jj] * m11[kk, jj] + \
                                  4.0 * M21_val(m21, ii, kk, P) * m11[jj, jj] * m21[jj, jj]
                        
                        val = 3.0 * r6 * np.sqrt(r5) * \
                              (2.0 * np.sqrt(np.sqrt(m22[ii, ii] * m22[jj, jj]) / m22[kk, kk]) * temp_kk + \
                               np.sqrt(m22[kk, kk] * np.sqrt(m22[jj, jj] / m22[ii, ii]**3)) * temp_ii + \
                               np.sqrt(m22[kk, kk] * np.sqrt(m22[ii, ii] / m22[jj, jj]**3)) * temp_jj) / T_obs
                        rCM4 += val
                    else:
                        S5111 = np.sum(Xc2[:, ii]**2 * Xc[:, ii] * Xc[:, jj] * Xc[:, kk] * Xc[:, ll])
                        S1511 = np.sum(Xc[:, ii] * Xc2[:, jj]**2 * Xc[:, jj] * Xc[:, kk] * Xc[:, ll])
                        S1151 = np.sum(Xc[:, ii] * Xc[:, jj] * Xc2[:, kk]**2 * Xc[:, kk] * Xc[:, ll])
                        S1115 = np.sum(Xc[:, ii] * Xc[:, jj] * Xc[:, kk] * Xc2[:, ll]**2 * Xc[:, ll])
                        S2111 = np.sum(Xc2[:, ii] * Xc[:, jj] * Xc[:, kk] * Xc[:, ll])
                        S1211 = np.sum(Xc[:, ii] * Xc2[:, jj] * Xc[:, kk] * Xc[:, ll])
                        S1121 = np.sum(Xc[:, ii] * Xc[:, jj] * Xc2[:, kk] * Xc[:, ll])
                        S1112 = np.sum(Xc[:, ii] * Xc[:, jj] * Xc[:, kk] * Xc2[:, ll])
                        m1111 = np.mean(Xc[:, ii] * Xc[:, jj] * Xc[:, kk] * Xc[:, ll])
                        m0111 = np.mean(Xc[:, jj] * Xc[:, kk] * Xc[:, ll])
                        m1011 = np.mean(Xc[:, ii] * Xc[:, kk] * Xc[:, ll])
                        m1101 = np.mean(Xc[:, ii] * Xc[:, jj] * Xc[:, ll])
                        m1110 = np.mean(Xc[:, ii] * Xc[:, jj] * Xc[:, kk])
                        
                        temp_ii = S5111 / T_obs - m1111 * m22[ii, ii] - 4.0 * S2111 / T_obs * m21[ii, ii] - m0111 * m41[ii, ii] - m1011 * m41[jj, ii] - m1101 * m41[kk, ii] - m1110 * m41[ll, ii] + 4.0 * m1110 * m21[ii, ii] * m11[ll, ii] + 4.0 * m1101 * m21[ii, ii] * m11[kk, ii] + 4.0 * m1011 * m21[ii, ii] * m11[jj, ii] + 4.0 * m0111 * m11[ii, ii] * m21[ii, ii]
                        temp_jj = S1511 / T_obs - m1111 * m22[jj, jj] - 4.0 * S1211 / T_obs * m21[jj, jj] - m0111 * m41[ii, jj] - m1011 * m41[jj, jj] - m1101 * m41[kk, jj] - m1110 * m41[ll, jj] + 4.0 * m1110 * m21[jj, jj] * m11[ll, jj] + 4.0 * m1101 * m21[jj, jj] * m11[kk, jj] + 4.0 * m1011 * m21[jj, jj] * m11[jj, jj] + 4.0 * m0111 * m11[ii, jj] * m21[jj, jj]
                        temp_kk = S1151 / T_obs - m1111 * m22[kk, kk] - 4.0 * S1121 / T_obs * m21[kk, kk] - m0111 * m41[ii, kk] - m1011 * m41[jj, kk] - m1101 * m41[kk, kk] - m1110 * m41[ll, kk] + 4.0 * m1110 * m21[kk, kk] * m11[ll, kk] + 4.0 * m1101 * m21[kk, kk] * m11[kk, kk] + 4.0 * m1011 * m21[kk, kk] * m11[jj, kk] + 4.0 * m0111 * m11[ii, kk] * M21_val(m21, kk, kk, P)
                        temp_ll = S1115 / T_obs - m1111 * m22[ll, ll] - 4.0 * S1112 / T_obs * m21[ll, ll] - m0111 * m41[ii, ll] - m1011 * m41[jj, ll] - m1101 * m41[kk, ll] - m1110 * m41[ll, ll] + 4.0 * m1110 * m21[ll, ll] * m11[ll, ll] + 4.0 * m1101 * m21[ll, ll] * m11[kk, ll] + 4.0 * m1011 * m21[ll, ll] * m11[jj, ll] + 4.0 * m0111 * m11[ii, ll] * M21_val(m21, ll, ll, P)
                        
                        val = 6.0 * r7 * r5 * (np.sqrt(np.sqrt((m22[jj, jj] * m22[kk, kk] * m22[ll, ll]) / m22[ii, ii]**3)) * temp_ii + \
                                             np.sqrt(np.sqrt((m22[ii, ii] * m22[kk, kk] * m22[ll, ll]) / m22[jj, jj]**3)) * temp_jj + \
                                             np.sqrt(np.sqrt((m22[ii, ii] * m22[jj, jj] * m22[ll, ll]) / m22[kk, kk]**3)) * temp_kk + \
                                             np.sqrt(np.sqrt((m22[ii, ii] * m22[jj, jj] * m22[kk, kk]) / m22[ll, ll]**3)) * temp_ll) / T_obs
                        rCM4 += val
                        
    return rCM4

def M21_val(m21, a, b, P):
    return m21[a, b]

def _calc_M4_T12(margkurts, margvars, N_assets):
    P = N_assets
    vec = []
    for ii in range(P):
        for jj in range(ii, P):
            for kk in range(jj, P):
                for ll in range(kk, P):
                    elem = 0.0
                    if ii == jj == kk == ll:
                        elem = margkurts[ii]
                    elif ii == jj and kk == ll: 
                        elem = margvars[ii] * margvars[kk]
                    vec.append(elem)
    return np.array(vec)

def _calc_M4_1f(margkurts, fvar, fkurt, epsvars, beta, N_assets):
    P = N_assets
    vec = []
    for ii in range(P):
        for jj in range(ii, P):
            for kk in range(jj, P):
                for ll in range(kk, P):
                    if ii == jj == kk == ll:
                        elem = margkurts[ii]
                    elif ii == jj == kk:
                        elem = beta[ii]**3 * beta[ll] * fkurt + 3.0 * beta[ii] * beta[ll] * fvar * epsvars[ii]
                    elif ii == jj and kk == ll:
                        elem = beta[ii]**2 * beta[kk]**2 * fkurt + fvar * (beta[ii]**2 * epsvars[kk] + beta[kk]**2 * epsvars[ii]) + epsvars[ii] * epsvars[kk]
                    elif ii == jj:
                        elem = beta[ii]**2 * beta[kk] * beta[ll] * fkurt + beta[kk] * beta[ll] * fvar * epsvars[ii]
                    elif jj == kk == ll:
                        elem = beta[ii] * beta[jj]**3 * fkurt + 3.0 * beta[ii] * beta[jj] * fvar * epsvars[jj]
                    elif jj == kk:
                        elem = beta[ii] * beta[jj]**2 * beta[ll] * fkurt + beta[ii] * beta[ll] * fvar * epsvars[jj]
                    elif kk == ll:
                        elem = beta[ii] * beta[jj] * beta[kk]**2 * fkurt + beta[ii] * beta[jj] * fvar * epsvars[kk]
                    else:
                        elem = beta[ii] * beta[jj] * beta[kk] * beta[ll] * fkurt
                    vec.append(elem)
    return np.array(vec)

def _calc_M4_MFresid(Stransf, epsvars, N_assets):
    P = N_assets
    vec = []
    for ii in range(P):
        for jj in range(ii, P):
            for kk in range(jj, P):
                for ll in range(kk, P):
                    elem = 0.0
                    if ii == jj == kk == ll:
                        elem = 6.0 * Stransf[ii, ii] * epsvars[ii]
                    elif ii == jj == kk:
                        elem = 3.0 * Stransf[ii, ll] * epsvars[ii]
                    elif ii == jj and kk == ll:
                        elem = Stransf[ii, ii] * epsvars[kk] + Stransf[kk, kk] * epsvars[ii]
                    elif ii == jj:
                        elem = Stransf[kk, ll] * epsvars[ii]
                    elif jj == kk == ll:
                        elem = 3.0 * Stransf[ii, jj] * epsvars[jj]
                    elif jj == kk:
                        elem = Stransf[ii, ll] * epsvars[jj]
                    elif kk == ll:
                        elem = Stransf[ii, jj] * epsvars[kk]
                    vec.append(elem)
    return np.array(vec)

def _calc_M4_CCoefficients(Xc, margvars, margkurts, marg6s, m22, m31, T_obs, N_assets):
    P = N_assets
    r3 = 0.0
    r5 = 0.0
    for ii in range(P):
        for jj in range(ii + 1, P):
            r3 += m31[jj, ii] / np.sqrt(marg6s[ii] * margvars[jj])
            r5 += m22[jj, ii] / np.sqrt(margkurts[ii] * margkurts[jj])
    r3 *= 2.0 / (P * (P - 1.0))
    r5 *= 2.0 / (P * (P - 1.0))
    
    r6 = 0.0
    r7 = 0.0
    for ii in range(P):
        for jj in range(ii + 1, P):
            for kk in range(jj + 1, P):
                m211 = np.mean(Xc[:, ii]**2 * Xc[:, jj] * Xc[:, kk])
                r6 += m211 / np.sqrt(margkurts[ii] * r5 * np.sqrt(margkurts[jj] * margkurts[kk]))
                
                for ll in range(kk + 1, P):
                    m1111 = np.mean(Xc[:, ii] * Xc[:, jj] * Xc[:, kk] * Xc[:, ll])
                    r7 += m1111 / (r5 * np.sqrt(np.sqrt(margkurts[ii] * margkurts[jj] * margkurts[kk] * margkurts[ll])))
    if P > 2:
        r6 *= 6.0 / (P * (P - 1.0) * (P - 2.0))
    else:
        r6 = 0.0
    if P > 3:
        r7 *= 24.0 / (P * (P - 1.0) * (P - 2.0) * (P - 3.0))
    else:
        r7 = 0.0
    return r3, r5, r6, r7

def _calc_M4_CC(margvars, margkurts, marg6s, r3, r5, r6, r7, N_assets):
    P = N_assets
    vec = []
    for ii in range(P):
        for jj in range(ii, P):
            for kk in range(jj, P):
                for ll in range(kk, P):
                    if ii == jj:
                        if jj == kk:
                            if kk == ll:
                                elem = margkurts[ii]
                            else:
                                elem = r3 * np.sqrt(marg6s[ii] * margvars[ll])
                        else:
                            if kk == ll:
                                elem = r5 * np.sqrt(margkurts[ii] * margkurts[kk])
                            else:
                                elem = r6 * np.sqrt(margvars[ii] * r5 * np.sqrt(margkurts[kk] * margkurts[ll]))
                    else:
                        if jj == kk:
                            if kk == ll:
                                elem = r3 * np.sqrt(margvars[ii] * marg6s[jj])
                            else:
                                elem = r6 * np.sqrt(margvars[jj] * r5 * np.sqrt(margkurts[ii] * margkurts[ll]))
                        else:
                            if kk == ll:
                                elem = r6 * np.sqrt(margvars[kk] * r5 * np.sqrt(margkurts[ii] * margkurts[jj]))
                            else:
                                elem = r7 * r5 * np.sqrt(np.sqrt(margkurts[ii] * margkurts[jj] * margkurts[kk] * margkurts[ll]))
                    vec.append(elem)
    return np.array(vec)

def m2_struct(R, struct="Indep", f=None):
    """
    Calculate covariance matrix as structured estimator.
    """
    X, f_val, T_obs, N_assets = _align_returns_and_factor(R, f)
    Xc = X.values - np.mean(X.values, axis=0, keepdims=True)
    margvars = np.mean(Xc**2, axis=0)
    
    if struct == "Indep":
        return np.diag(margvars)
    elif struct == "IndepId":
        return np.mean(margvars) * np.eye(N_assets)
    elif struct == "observedfactor":
        if f_val is None:
            raise ValueError("Provide factor observations f")
        if f_val.shape[1] == 1:
            f_centered = f_val[:, 0] - np.mean(f_val[:, 0])
            fvar = np.mean(f_centered**2)
            fvar_ddof1 = np.var(f_centered, ddof=1)
            beta = np.array([np.cov(Xc[:, i], f_centered)[0, 1] / fvar_ddof1 for i in range(N_assets)])
            T2 = fvar * np.outer(beta, beta)
            np.fill_diagonal(T2, margvars)
            return T2
        else:
            f_design = np.column_stack([np.ones(T_obs), f_val])
            Beta = np.linalg.lstsq(f_design, Xc, rcond=None)[0]
            beta = Beta[1:, :]
            fcov = np.cov(f_val, rowvar=False, ddof=0)
            if f_val.shape[1] == 1:
                fcov = np.atleast_2d(fcov)
            T2 = beta.T @ fcov @ beta
            fc = f_val - np.mean(f_val, axis=0, keepdims=True)
            residuals = Xc - fc @ beta
            epsvars = np.mean(residuals**2, axis=0)
            T2 = T2 + np.diag(epsvars)
            return T2
    elif struct == "CC":
        sd_vec = np.sqrt(margvars)
        M2 = (Xc.T @ Xc) / T_obs
        sd_vec_nonzero = np.where(sd_vec > 0, sd_vec, 1e-15)
        R2 = M2 / np.outer(sd_vec_nonzero, sd_vec_nonzero)
        triu_indices = np.triu_indices(N_assets, k=1)
        rcoef = np.mean(R2[triu_indices])
        R2_new = np.full((N_assets, N_assets), rcoef)
        np.fill_diagonal(R2_new, 1.0)
        T2 = np.diag(sd_vec) @ R2_new @ np.diag(sd_vec)
        return T2
    else:
        raise ValueError(f"Unknown structure method '{struct}'")

def _m3_struct_Indep(margskews, N_assets):
    vec = []
    for ii in range(N_assets):
        for jj in range(ii, N_assets):
            for kk in range(jj, N_assets):
                elem = margskews[ii] if ii == jj == kk else 0.0
                vec.append(elem)
    return np.array(vec)

def _m3_struct_1f(margskews, beta, fskew, N_assets):
    vec = []
    for ii in range(N_assets):
        for jj in range(ii, N_assets):
            for kk in range(jj, N_assets):
                if ii == jj == kk:
                    elem = margskews[ii]
                else:
                    elem = beta[ii] * beta[jj] * beta[kk] * fskew
                vec.append(elem)
    return np.array(vec)

def _m3_struct_Simaan(margskewsroot, N_assets):
    vec = []
    for ii in range(N_assets):
        for jj in range(ii, N_assets):
            for kk in range(jj, N_assets):
                vec.append(margskewsroot[ii] * margskewsroot[jj] * margskewsroot[kk])
    return np.array(vec)

def m3_struct(R, struct="Indep", f=None, unbiasedMarg=False, as_mat=True):
    """
    Calculate coskewness matrix as structured estimator.
    """
    X, f_val, T_obs, N_assets = _align_returns_and_factor(R, f)
    Xc = X.values - np.mean(X.values, axis=0, keepdims=True)
    margvars = np.mean(Xc**2, axis=0)
    margskews = np.mean(Xc**3, axis=0)
    
    if unbiasedMarg:
        if T_obs < 6:
            raise ValueError("R should have at least 6 observations for unbiasedMarg")
        if struct not in ["Indep", "IndepId", "CS"]:
            raise ValueError("unbiasedMarg can only be combined with Indep, IndepId, and CS")
        margskews = margskews * (T_obs**2) / ((T_obs - 1) * (T_obs - 2))
        
    ncosk = N_assets * (N_assets + 1) * (N_assets + 2) // 6
    
    if struct == "latent1factor":
        margskewsroot = np.sign(margskews) * np.abs(margskews)**(1.0 / 3.0)
        vec = []
        for ii in range(N_assets):
            for jj in range(ii, N_assets):
                for kk in range(jj, N_assets):
                    vec.append(margskewsroot[ii] * margskewsroot[jj] * margskewsroot[kk])
        T3 = np.array(vec)
    elif struct == "Indep":
        T3 = _m3_struct_Indep(margskews, N_assets)
    elif struct == "IndepId":
        skews = np.repeat(np.mean(margskews), N_assets)
        T3 = _m3_struct_Indep(skews, N_assets)
    elif struct == "observedfactor":
        if f_val is None:
            raise ValueError("Provide factor observations f")
        if f_val.shape[1] == 1:
            f_centered = f_val[:, 0] - np.mean(f_val[:, 0])
            fvar = np.mean(f_centered**2)
            fvar_ddof1 = np.var(f_centered, ddof=1)
            fskew = np.mean(f_centered**3)
            beta = np.array([np.cov(Xc[:, i], f_centered)[0, 1] / fvar_ddof1 for i in range(N_assets)])
            T3 = _m3_struct_1f(margskews, beta, fskew, N_assets)
        else:
            f_design = np.column_stack([np.ones(T_obs), f_val])
            Beta = np.linalg.lstsq(f_design, Xc, rcond=None)[0]
            beta = Beta[1:, :]
            fc = f_val - np.mean(f_val, axis=0, keepdims=True)
            fc_kron = np.einsum('ti,tj->tij', fc, fc).reshape(T_obs, -1)
            M3_factor_mat = (fc.T @ fc_kron) / T_obs
            M3_mapped = beta.T @ M3_factor_mat @ np.kron(beta, beta)
            residuals = Xc - fc @ beta
            epsskews = np.mean(residuals**3, axis=0)
            M3_eps = np.zeros((N_assets, N_assets * N_assets))
            for i in range(N_assets):
                M3_eps[i, i * N_assets + i] = epsskews[i]
            T3_mat = M3_mapped + M3_eps
            T3 = _m3_mat_to_vec(T3_mat, N_assets)
    elif struct == "CC":
        margkurts = np.mean(Xc**4, axis=0)
        Xc2 = Xc**2
        m21 = (Xc2.T @ Xc) / T_obs
        m22 = (Xc2.T @ Xc2) / T_obs
        r2, r4, r5 = _calc_M3_CCoefficients(Xc, margvars, margkurts, m21, m22, T_obs, N_assets)
        T3 = _calc_M3_CC(margvars, margskews, margkurts, r2, r4, r5, N_assets)
    elif struct == "CS":
        T3 = np.zeros(ncosk)
    else:
        raise ValueError(f"Unknown structure method '{struct}'")
        
    if as_mat:
        return _m3_vec_to_mat(T3, N_assets)
    return T3

def m4_struct(R, struct="Indep", f=None, as_mat=True):
    """
    Calculate cokurtosis matrix as structured estimator.
    """
    X, f_val, T_obs, N_assets = _align_returns_and_factor(R, f)
    Xc = X.values - np.mean(X.values, axis=0, keepdims=True)
    margvars = np.mean(Xc**2, axis=0)
    margkurts = np.mean(Xc**4, axis=0)
    
    if struct == "Indep":
        T4 = _calc_M4_T12(margkurts, margvars, N_assets)
    elif struct == "IndepId":
        meanmargkurts = np.mean(margkurts)
        meank_iikk = np.sqrt(np.mean(margvars**2))
        T4 = _calc_M4_T12(np.repeat(meanmargkurts, N_assets), np.repeat(meank_iikk, N_assets), N_assets)
    elif struct == "observedfactor":
        if f_val is None:
            raise ValueError("Provide factor observations f")
        if f_val.shape[1] == 1:
            f_centered = f_val[:, 0] - np.mean(f_val[:, 0])
            fvar = np.mean(f_centered**2)
            fvar_ddof1 = np.var(f_centered, ddof=1)
            fkurt = np.mean(f_centered**4)
            beta = np.array([np.cov(Xc[:, i], f_centered)[0, 1] / fvar_ddof1 for i in range(N_assets)])
            epsvars = margvars - beta**2 * fvar
            T4 = _calc_M4_1f(margkurts, fvar, fkurt, epsvars, beta, N_assets)
        else:
            f_design = np.column_stack([np.ones(T_obs), f_val])
            Beta = np.linalg.lstsq(f_design, Xc, rcond=None)[0]
            beta = Beta[1:, :]
            fc = f_val - np.mean(f_val, axis=0, keepdims=True)
            fc_kron = np.einsum('ti,tj,tk->tijk', fc, fc, fc).reshape(T_obs, -1)
            M4_factor_mat = (fc.T @ fc_kron) / T_obs
            M4_mapped = beta.T @ M4_factor_mat @ np.kron(beta, np.kron(beta, beta))
            residuals = Xc - fc @ beta
            epskurts = np.mean(residuals**4, axis=0)
            epsvars = np.mean(residuals**2, axis=0)
            fcov = np.cov(f_val, rowvar=False, ddof=0)
            if f_val.shape[1] == 1:
                fcov = np.atleast_2d(fcov)
            Stransf = beta.T @ fcov @ beta
            T4_MF = _calc_M4_MFresid(Stransf, epsvars, N_assets)
            T4_ind = _calc_M4_T12(epskurts, epsvars, N_assets)
            T4 = _m4_mat_to_vec(M4_mapped, N_assets) + T4_MF + T4_ind
    elif struct == "CC":
        marg6s = np.mean(Xc**6, axis=0)
        Xc2 = Xc**2
        m22 = (Xc2.T @ Xc2) / T_obs
        m31 = ((Xc**3).T @ Xc) / T_obs
        r3, r5, r6, r7 = _calc_M4_CCoefficients(Xc, margvars, margkurts, marg6s, m22, m31, T_obs, N_assets)
        T4 = _calc_M4_CC(margvars, margkurts, marg6s, r3, r5, r6, r7, N_assets)
    else:
        raise ValueError(f"Unknown structure method '{struct}'")
        
    if as_mat:
        return _m4_vec_to_mat(T4, N_assets)
    return T4

def m2_shrink(R, targets=1, f=None):
    """
    Calculate covariance matrix shrinkage estimator.
    """
    X, f_val, T_obs, N_assets = _align_returns_and_factor(R, f)
    
    if isinstance(targets, int):
        targets = [targets]
    targets = list(targets)
    if not targets:
        raise ValueError("No targets selected")
    for t in targets:
        if t not in [1, 2, 3, 4]:
            raise ValueError("Select valid targets (out of 1, 2, 3, 4)")
            
    has_t3 = 3 in targets
    if has_t3 and f is None:
        raise ValueError("Provide factor observations f")
        
    n_factors = 1
    f_other = None
    extra_factors = False
    if has_t3 and f_val is not None:
        n_factors = f_val.shape[1]
        if n_factors > 1:
            f_other = f_val[:, 1:]
            f_val = f_val[:, 0:1]
            extra_factors = True
            
    nT = len(targets)
    if extra_factors:
        nT += n_factors - 1
        
    Xc = X.values - np.mean(X.values, axis=0, keepdims=True)
    margvars = np.mean(Xc**2, axis=0)
    M2 = np.cov(X.values, rowvar=False, ddof=0)
    
    T2_list = []
    for t in targets:
        if t == 1:
            T2_list.append(np.diag(margvars))
        elif t == 2:
            T2_list.append(np.mean(margvars) * np.eye(N_assets))
        elif t == 3:
            f_centered = f_val[:, 0] - np.mean(f_val[:, 0])
            fvar = np.mean(f_centered**2)
            fvar_ddof1 = np.var(f_centered, ddof=1)
            beta = np.array([np.cov(Xc[:, i], f_centered)[0, 1] / fvar_ddof1 for i in range(N_assets)])
            T2_1f = fvar * np.outer(beta, beta)
            np.fill_diagonal(T2_1f, margvars)
            T2_list.append(T2_1f)
        elif t == 4:
            sd_vec = np.sqrt(margvars)
            sd_vec_nonzero = np.where(sd_vec > 0, sd_vec, 1e-15)
            R2 = M2 / np.outer(sd_vec_nonzero, sd_vec_nonzero)
            triu_indices = np.triu_indices(N_assets, k=1)
            rcoef = np.mean(R2[triu_indices])
            R2_new = np.full((N_assets, N_assets), rcoef)
            np.fill_diagonal(R2_new, 1.0)
            T2_list.append(np.diag(sd_vec) @ R2_new @ np.diag(sd_vec))
            
    if extra_factors:
        assert f_other is not None
        for ii in range(n_factors - 1):
            f_centered = f_other[:, ii] - np.mean(f_other[:, ii])
            fvar = np.mean(f_centered**2)
            fvar_ddof1 = np.var(f_centered, ddof=1)
            beta = np.array([np.cov(Xc[:, i], f_centered)[0, 1] / fvar_ddof1 for i in range(N_assets)])
            T2_1f = fvar * np.outer(beta, beta)
            np.fill_diagonal(T2_1f, margvars)
            T2_list.append(T2_1f)
            
    T2_mat = np.column_stack([t2.flatten() for t2 in T2_list])
    M2_flat = M2.flatten()
    
    A = np.zeros((nT, nT))
    for ii in range(nT):
        for jj in range(ii, nT):
            val = np.sum((T2_mat[:, ii] - M2_flat) * (T2_mat[:, jj] - M2_flat))
            A[ii, jj] = val
            A[jj, ii] = val
            
    m11 = (Xc.T @ Xc) / T_obs
    Xc2 = Xc**2
    m22 = (Xc2.T @ Xc2) / T_obs
    VM2vec = _calc_VM2(m11, m22, T_obs, N_assets)
    
    b = np.full(nT, VM2vec[0])
    
    iter_idx = 0
    for t in targets:
        if t == 1:
            b[iter_idx] -= VM2vec[2]
            iter_idx += 1
        elif t == 2:
            b[iter_idx] -= VM2vec[1]
            iter_idx += 1
        elif t == 3:
            f_centered = f_val[:, 0] - np.mean(f_val[:, 0])
            fvar = np.mean(f_centered**2)
            b[iter_idx] -= _calc_CM2_1F(Xc, m11, m22, f_centered, fvar, T_obs, N_assets)
            iter_idx += 1
        elif t == 4:
            sd_vec = np.sqrt(margvars)
            sd_vec_nonzero = np.where(sd_vec > 0, sd_vec, 1e-15)
            R2 = M2 / np.outer(sd_vec_nonzero, sd_vec_nonzero)
            triu_indices = np.triu_indices(N_assets, k=1)
            rcoef = np.mean(R2[triu_indices])
            b[iter_idx] -= _calc_CM2_CC(Xc, m11, m22, rcoef, T_obs, N_assets)
            iter_idx += 1
            
    if extra_factors:
        assert f_other is not None
        for ii in range(n_factors - 1):
            f_centered = f_other[:, ii] - np.mean(f_other[:, ii])
            fvar = np.mean(f_centered**2)
            b[iter_idx] -= _calc_CM2_1F(Xc, m11, m22, f_centered, fvar, T_obs, N_assets)
            iter_idx += 1
            
    lambda_weights = solve_qp(A, b)
    M2sh = (1.0 - np.sum(lambda_weights)) * M2
    for tt in range(nT):
        M2sh += lambda_weights[tt] * T2_list[tt]
        
    return {"M2sh": M2sh, "lambda": lambda_weights, "A": A, "b": b}

def m3_shrink(R, targets=1, f=None, unbiasedMSE=False, as_mat=True):
    """
    Calculate coskewness matrix shrinkage estimator.
    """
    X, f_val, T_obs, N_assets = _align_returns_and_factor(R, f)
    
    if isinstance(targets, int):
        targets = [targets]
    targets = list(targets)
    if not targets:
        raise ValueError("No targets selected")
    for t in targets:
        if t not in [1, 2, 3, 4, 5, 6]:
            raise ValueError("Select valid targets (out of 1, 2, 3, 4, 5, 6)")
            
    if 3 in targets and f is None:
        raise ValueError("Provide factor observations f")
    if unbiasedMSE and any(t in targets for t in [3, 4, 5]):
        raise ValueError("unbiasedMSE can only be combined with T1, T2, and T6")
        
    n_factors = 1
    f_other = None
    extra_factors = False
    if 3 in targets and f_val is not None:
        n_factors = f_val.shape[1]
        if n_factors > 1:
            f_other = f_val[:, 1:]
            f_val = f_val[:, 0:1]
            extra_factors = True
            
    nT = len(targets)
    if extra_factors:
        nT += n_factors - 1
        
    if unbiasedMSE and T_obs < 6:
        raise ValueError("R should have at least 6 observations")
        
    Xc = X.values - np.mean(X.values, axis=0, keepdims=True)
    Xc2 = Xc**2
    margvars = np.mean(Xc2, axis=0)
    margskews = np.mean(Xc**3, axis=0)
    
    m11 = (Xc.T @ Xc) / T_obs
    m21 = (Xc2.T @ Xc) / T_obs
    m22 = (Xc2.T @ Xc2) / T_obs
    m31 = ((Xc**3).T @ Xc) / T_obs
    m42 = ((Xc**4).T @ Xc2) / T_obs
    m33 = ((Xc**3).T @ Xc**3) / T_obs
    
    fc_kron = np.einsum('ti,tj->tij', Xc, Xc).reshape(T_obs, -1)
    if unbiasedMSE:
        CC = T_obs / ((T_obs - 1) * (T_obs - 2))
    else:
        CC = 1.0 / T_obs
    M3_mat = CC * (Xc.T @ fc_kron)
    M3 = _m3_mat_to_vec(M3_mat, N_assets)
    
    T3_list = []
    for t in targets:
        if t == 1:
            skews = margskews.copy()
            if unbiasedMSE:
                skews = skews * (T_obs**2) / ((T_obs - 1) * (T_obs - 2))
            T3_list.append(_m3_struct_Indep(skews, N_assets))
        elif t == 2:
            skews = margskews.copy()
            if unbiasedMSE:
                skews = skews * (T_obs**2) / ((T_obs - 1) * (T_obs - 2))
            skews = np.repeat(np.mean(skews), N_assets)
            T3_list.append(_m3_struct_Indep(skews, N_assets))
        elif t == 3:
            f_centered = f_val[:, 0] - np.mean(f_val[:, 0])
            fvar = np.mean(f_centered**2)
            fvar_ddof1 = np.var(f_centered, ddof=1)
            fskew = np.mean(f_centered**3)
            beta = np.array([np.cov(Xc[:, i], f_centered)[0, 1] / fvar_ddof1 for i in range(N_assets)])
            T3_list.append(_m3_struct_1f(margskews, beta, fskew, N_assets))
        elif t == 4:
            margkurts = np.mean(Xc**4, axis=0)
            r2, r4, r5 = _calc_M3_CCoefficients(Xc, margvars, margkurts, m21, m22, T_obs, N_assets)
            T3_list.append(_calc_M3_CC(margvars, margskews, margkurts, r2, r4, r5, N_assets))
        elif t == 5:
            margskewsroot = np.sign(margskews) * np.abs(margskews)**(1.0 / 3.0)
            T3_list.append(_m3_struct_Simaan(margskewsroot, N_assets))
        elif t == 6:
            T3_list.append(np.zeros(len(M3)))
            
    if extra_factors:
        assert f_other is not None
        for ii in range(n_factors - 1):
            f_centered = f_other[:, ii] - np.mean(f_other[:, ii])
            fvar = np.mean(f_centered**2)
            fvar_ddof1 = np.var(f_centered, ddof=1)
            fskew = np.mean(f_centered**3)
            beta = np.array([np.cov(Xc[:, i], f_centered)[0, 1] / fvar_ddof1 for i in range(N_assets)])
            T3_list.append(_m3_struct_1f(margskews, beta, fskew, N_assets))
            
    T3_mat = np.column_stack(T3_list)
    mult = _m3_multipliers(N_assets)
    
    A = np.zeros((nT, nT))
    for ii in range(nT):
        for jj in range(ii, nT):
            val = np.sum((T3_mat[:, ii] - M3) * (T3_mat[:, jj] - M3) * mult)
            A[ii, jj] = val
            A[jj, ii] = val
            
    if unbiasedMSE:
        VM3vec = _calc_VM3kstat(Xc, Xc2, m11, m21, m22, m31, m42, m33, T_obs, N_assets)
    else:
        VM3vec = _calc_VM3(Xc, Xc2, m11, m21, m22, m31, m42, m33, T_obs, N_assets)
        
    b = np.full(nT, VM3vec[0])
    
    iter_idx = 0
    for t in targets:
        if t == 1:
            b[iter_idx] -= VM3vec[2]
            iter_idx += 1
        elif t == 2:
            b[iter_idx] -= VM3vec[1]
            iter_idx += 1
        elif t == 3:
            f_centered = f_val[:, 0] - np.mean(f_val[:, 0])
            fvar = np.mean(f_centered**2)
            fskew = np.mean(f_centered**3)
            b[iter_idx] -= _calc_CM3_1F(Xc, Xc2, f_centered[:, np.newaxis], fvar, fskew, m11, m21, m22, m42, T_obs, N_assets)
            iter_idx += 1
        elif t == 4:
            marg5s = np.mean(Xc**5, axis=0)
            marg6s = np.mean(Xc**6, axis=0)
            margkurts = np.mean(Xc**4, axis=0)
            m41 = ((Xc**4).T @ Xc) / T_obs
            m61 = ((Xc**6).T @ Xc) / T_obs
            m32 = ((Xc**3).T @ Xc2) / T_obs
            r2, r4, r5 = _calc_M3_CCoefficients(Xc, margvars, margkurts, m21, m22, T_obs, N_assets)
            b[iter_idx] -= _calc_CM3_CC(Xc, Xc2, margvars, margskews, margkurts, marg5s, marg6s, m11, m21, m31, m32, m41, m61, r2, r4, r5, T_obs, N_assets)
            iter_idx += 1
        elif t == 5:
            margskewsroot = np.sign(margskews) * np.abs(margskews)**(1.0 / 3.0)
            m51 = ((Xc**5).T @ Xc) / T_obs
            b[iter_idx] -= _calc_CM3_Simaan(Xc, Xc2, margskewsroot, m11, m21, m22, m31, m42, m51, T_obs, N_assets)
            iter_idx += 1
        elif t == 6:
            iter_idx += 1
            
    if extra_factors:
        assert f_other is not None
        for ii in range(n_factors - 1):
            f_centered = f_other[:, ii] - np.mean(f_other[:, ii])
            fvar = np.mean(f_centered**2)
            fskew = np.mean(f_centered**3)
            b[iter_idx] -= _calc_CM3_1F(Xc, Xc2, f_centered[:, np.newaxis], fvar, fskew, m11, m21, m22, m42, T_obs, N_assets)
            iter_idx += 1
            
    lambda_weights = solve_qp(A, b)
    M3sh = (1.0 - np.sum(lambda_weights)) * M3
    for tt in range(nT):
        M3sh += lambda_weights[tt] * T3_list[tt]
        
    if as_mat:
        return {"M3sh": _m3_vec_to_mat(M3sh, N_assets), "lambda": lambda_weights, "A": A, "b": b}
    return {"M3sh": M3sh, "lambda": lambda_weights, "A": A, "b": b}

def m4_shrink(R, targets=1, f=None, as_mat=True):
    """
    Calculate cokurtosis matrix shrinkage estimator.
    """
    X, f_val, T_obs, N_assets = _align_returns_and_factor(R, f)
    
    if isinstance(targets, int):
        targets = [targets]
    targets = list(targets)
    if not targets:
        raise ValueError("No targets selected")
    for t in targets:
        if t not in [1, 2, 3, 4]:
            raise ValueError("Select valid targets (out of 1, 2, 3, 4)")
            
    if 3 in targets and f is None:
        raise ValueError("Provide factor observations f")
        
    n_factors = 1
    f_other = None
    extra_factors = False
    if 3 in targets and f_val is not None:
        n_factors = f_val.shape[1]
        if n_factors > 1:
            f_other = f_val[:, 1:]
            f_val = f_val[:, 0:1]
            extra_factors = True
            
    nT = len(targets)
    if extra_factors:
        nT += n_factors - 1
        
    Xc = X.values - np.mean(X.values, axis=0, keepdims=True)
    Xc2 = Xc**2
    margvars = np.mean(Xc2, axis=0)
    margkurts = np.mean(Xc**4, axis=0)
    
    m11 = (Xc.T @ Xc) / T_obs
    m21 = (Xc2.T @ Xc) / T_obs
    m22 = (Xc2.T @ Xc2) / T_obs
    m31 = ((Xc**3).T @ Xc) / T_obs
    m32 = ((Xc**3).T @ Xc2) / T_obs
    m41 = ((Xc**4).T @ Xc) / T_obs
    m42 = ((Xc**4).T @ Xc2) / T_obs
    
    fc_kron = np.einsum('ti,tj,tk->tijk', Xc, Xc, Xc).reshape(T_obs, -1)
    M4_mat = (Xc.T @ fc_kron) / T_obs
    M4 = _m4_mat_to_vec(M4_mat, N_assets)
    
    T4_list = []
    for t in targets:
        if t == 1:
            T4_list.append(_calc_M4_T12(margkurts, margvars, N_assets))
        elif t == 2:
            meanmargkurts = np.mean(margkurts)
            meank_iikk = np.sqrt(np.mean(margvars**2))
            T4_list.append(_calc_M4_T12(np.repeat(meanmargkurts, N_assets), np.repeat(meank_iikk, N_assets), N_assets))
        elif t == 3:
            f_centered = f_val[:, 0] - np.mean(f_val[:, 0])
            fvar = np.mean(f_centered**2)
            fvar_ddof1 = np.var(f_centered, ddof=1)
            fkurt = np.mean(f_centered**4)
            beta = np.array([np.cov(Xc[:, i], f_centered)[0, 1] / fvar_ddof1 for i in range(N_assets)])
            epsvars = margvars - beta**2 * fvar
            T4_list.append(_calc_M4_1f(margkurts, fvar, fkurt, epsvars, beta, N_assets))
        elif t == 4:
            marg6s = np.mean(Xc**6, axis=0)
            r3, r5, r6, r7 = _calc_M4_CCoefficients(Xc, margvars, margkurts, marg6s, m22, m31, T_obs, N_assets)
            T4_list.append(_calc_M4_CC(margvars, margkurts, marg6s, r3, r5, r6, r7, N_assets))
            
    if extra_factors:
        assert f_other is not None
        for ii in range(n_factors - 1):
            f_centered = f_other[:, ii] - np.mean(f_other[:, ii])
            fvar = np.mean(f_centered**2)
            fvar_ddof1 = np.var(f_centered, ddof=1)
            fkurt = np.mean(f_centered**4)
            beta = np.array([np.cov(Xc[:, i], f_centered)[0, 1] / fvar_ddof1 for i in range(N_assets)])
            epsvars = margvars - beta**2 * fvar
            T4_list.append(_calc_M4_1f(margkurts, fvar, fkurt, epsvars, beta, N_assets))
            
    T4_mat = np.column_stack(T4_list)
    mult = _m4_multipliers(N_assets)
    
    A = np.zeros((nT, nT))
    for ii in range(nT):
        for jj in range(ii, nT):
            val = np.sum((T4_mat[:, ii] - M4) * (T4_mat[:, jj] - M4) * mult)
            A[ii, jj] = val
            A[jj, ii] = val
            
    VM4vec = _calc_VM4(Xc, Xc2, m11, m21, m22, m31, m32, m41, m42, T_obs, N_assets)
    b = np.full(nT, VM4vec[0])
    
    iter_idx = 0
    for t in targets:
        if t == 1:
            b[iter_idx] -= VM4vec[2]
            iter_idx += 1
        elif t == 2:
            b[iter_idx] -= VM4vec[1]
            iter_idx += 1
        elif t == 3:
            f_centered = f_val[:, 0] - np.mean(f_val[:, 0])
            fvar = np.mean(f_centered**2)
            fskew = np.mean(f_centered**3)
            fkurt = np.mean(f_centered**4)
            b[iter_idx] -= _calc_CM4_1F(Xc, Xc2, f_centered[:, np.newaxis], fvar, fskew, fkurt, m11, m21, m22, m31, T_obs, N_assets)
            iter_idx += 1
        elif t == 4:
            marg6s = np.mean(Xc**6, axis=0)
            marg7s = np.mean(Xc**7, axis=0)
            m33 = ((Xc**3).T @ Xc**3) / T_obs
            r3, r5, r6, r7 = _calc_M4_CCoefficients(Xc, margvars, margkurts, marg6s, m22, m31, T_obs, N_assets)
            b[iter_idx] -= _calc_CM4_CC(Xc, Xc2, m11, m21, m22, m31, m32, m33, m41, r3, r5, r6, r7, marg6s, marg7s, T_obs, N_assets)
            iter_idx += 1
            
    if extra_factors:
        assert f_other is not None
        for ii in range(n_factors - 1):
            f_centered = f_other[:, ii] - np.mean(f_other[:, ii])
            fvar = np.mean(f_centered**2)
            fskew = np.mean(f_centered**3)
            fkurt = np.mean(f_centered**4)
            b[iter_idx] -= _calc_CM4_1F(Xc, Xc2, f_centered[:, np.newaxis], fvar, fskew, fkurt, m11, m21, m22, m31, T_obs, N_assets)
            iter_idx += 1
            
    lambda_weights = solve_qp(A, b)
    M4sh = (1.0 - np.sum(lambda_weights)) * M4
    for tt in range(nT):
        M4sh += lambda_weights[tt] * T4_list[tt]
        
    if as_mat:
        return {"M4sh": _m4_vec_to_mat(M4sh, N_assets), "lambda": lambda_weights, "A": A, "b": b}
    return {"M4sh": M4sh, "lambda": lambda_weights, "A": A, "b": b}

