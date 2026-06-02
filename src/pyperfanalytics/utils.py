import numpy as np
import pandas as pd


def centered_moment(R: pd.Series | pd.DataFrame, moment: int) -> float | pd.Series:
    r"""
    Calculate the nth centered moment (population version, matching R's PerformanceAnalytics).
    r"""

    def _calc(s: pd.Series, m: int) -> float:
        s = s.dropna()
        if len(s) == 0:
            return np.nan
        return np.mean((s - s.mean()) ** m)

    if isinstance(R, pd.DataFrame):
        return R.apply(_calc, m=moment)
    else:
        return _calc(R, moment)


def centered_comoment(Ra: pd.Series, Rb: pd.Series, p1: int, p2: int, normalize: bool = False) -> float:
    r"""
    Calculate the joint centered comoment of two series.
    E[ (Ra - E[Ra])^p1 * (Rb - E[Rb])^p2 ]
    r"""
    merged = pd.concat([Ra, Rb], axis=1).dropna()
    if merged.empty:
        return np.nan

    a = merged.iloc[:, 0]
    b = merged.iloc[:, 1]

    centered_a = a - a.mean()
    centered_b = b - b.mean()

    out = (centered_a**p1 * centered_b**p2).mean()

    if normalize:
        # R code: out = out / centeredmoment(Rb, power=(p1+p2))
        m_b = (centered_b ** (p1 + p2)).mean()
        if m_b == 0:
            return np.nan
        out = out / m_b

    return out


def co_variance(Ra: pd.Series, Rb: pd.Series) -> float:
    return centered_comoment(Ra, Rb, 1, 1)


def co_skewness(Ra: pd.Series, Rb: pd.Series) -> float:
    return centered_comoment(Ra, Rb, 1, 2)


def co_kurtosis(Ra: pd.Series, Rb: pd.Series) -> float:
    return centered_comoment(Ra, Rb, 1, 3)


def beta_co_variance(Ra: pd.Series, Rb: pd.Series) -> float:
    return centered_comoment(Ra, Rb, 1, 1, normalize=True)


def beta_co_skewness(Ra: pd.Series, Rb: pd.Series) -> float:
    return centered_comoment(Ra, Rb, 1, 2, normalize=True)


def beta_co_kurtosis(Ra: pd.Series, Rb: pd.Series) -> float:
    return centered_comoment(Ra, Rb, 1, 3, normalize=True)


def skewness(R: pd.Series | pd.DataFrame, method: str = "moment") -> float | pd.Series:
    r"""
    Calculate skewness of the return distribution.
    Methods: 'moment', 'fisher', 'sample'.
    r"""

    def _calc(s: pd.Series, meth: str) -> float:
        s = s.dropna()
        n = len(s)
        if n < 3:
            return np.nan

        # PerformanceAnalytics specific:
        # method="moment" is population skewness (centered)
        # method="fisher" is raw fisher (NO centering)
        # method="sample" is adjusted population skewness (centered)

        if meth == "moment":
            m2 = np.mean((s - s.mean()) ** 2)
            m3 = np.mean((s - s.mean()) ** 3)
            return m3 / (m2 ** (1.5))
        elif meth == "fisher":
            if n < 3:
                return np.nan
            return ((np.sqrt(n * (n - 1)) / (n - 2)) * (np.mean(s**3))) / (np.mean(s**2) ** (1.5))
        elif meth == "sample":
            if n < 3:
                return np.nan
            m2 = np.mean((s - s.mean()) ** 2)
            m3 = np.mean((s - s.mean()) ** 3)
            # R's sample formula: n/((n-1)(n-2)) * sum( (x-mu)^3 / sd_pop^3 )
            return (m3 / (m2 ** (1.5))) * n / ((n - 1) * (n - 2)) * n
            # In R's skewness.R: sum((x-mean(x))^3/sqrt(var(x)*(n-1)/n)^3)*n/((n-1)*(n-2))
            # sd_pop = sqrt(var(x)*(n-1)/n) = sqrt(m2)
            # sum( (x-mu)^3 / m2^1.5 ) = n * m3 / m2^1.5
            # So: (n * m3 / m2^1.5) * n / ((n-1)*(n-2))
        else:
            raise ValueError(f"Unknown skewness method: {meth}")

    if isinstance(R, pd.DataFrame):
        return R.apply(_calc, meth=method)
    else:
        return _calc(R, method)


def kurtosis(R: pd.Series | pd.DataFrame, method: str = "excess") -> float | pd.Series:
    r"""
    Calculate kurtosis of the return distribution.
    Methods: 'excess', 'moment', 'fisher', 'sample', 'sample_excess'.
    r"""

    def _calc(s: pd.Series, meth: str) -> float:
        s = s.dropna()
        n = len(s)
        # Kurtosis ideally needs 4 points. If less than 4, sample formulas break.
        # We enforce a minimum of 4 points to be safe across methods.
        if n < 4:
            return np.nan

        m2 = np.mean((s - s.mean()) ** 2)
        m4 = np.mean((s - s.mean()) ** 4)

        if meth == "moment":
            return m4 / (m2**2)
        elif meth == "excess":
            return (m4 / (m2**2)) - 3
        elif meth == "fisher":
            if n < 4:
                return np.nan
            # R's fisher is raw (NO centering)
            r2 = np.mean(s**2)
            r4 = np.mean(s**4)
            return ((n + 1) * (n - 1) * ((r4 / r2**2) - (3 * (n - 1)) / (n + 1))) / ((n - 2) * (n - 3))
        elif meth == "sample":
            if n < 4:
                return np.nan
            # In R: sum((x-mean(x))^4/var(x)^2)*n*(n+1)/((n-1)*(n-2)*(n-3))
            # var(x) = m2 * n / (n-1)
            # sum((x-mu)^4) = n * m4
            var_x = s.var(ddof=1)
            return (n * m4 / var_x**2) * n * (n + 1) / ((n - 1) * (n - 2) * (n - 3))
        elif meth == "sample_excess":
            if n < 4:
                return np.nan
            k = _calc(s, "sample")
            return k - 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3))
        else:
            raise ValueError(f"Unknown kurtosis method: {meth}")

    if isinstance(R, pd.DataFrame):
        return R.apply(_calc, meth=method)
    else:
        return _calc(R, method)


def frequency(R: pd.Series | pd.DataFrame) -> int | pd.Series:
    r"""
    Identify the frequency label (scale) of the return series.
    Returns 252 for Daily, 52 for Weekly, 12 for Monthly, 4 for Quarterly, 1 for Yearly.
    """
    if not isinstance(R.index, pd.DatetimeIndex):
        raise ValueError("Data index must be a DatetimeIndex to determine frequency.")

    freq = R.index.inferred_freq
    scale = 1
    if freq:
        freq_base = freq.split("-")[0]
        if freq_base in ["B", "D"]:
            scale = 252
        elif freq_base in ["W"]:
            scale = 52
        elif freq_base in ["M", "ME", "MS", "BME", "BMS"]:
            scale = 12
        elif freq_base in ["Q", "QE", "QS", "BQE", "BQS"]:
            scale = 4
        elif freq_base in ["Y", "YE", "YS", "BYE", "BYS", "A"]:
            scale = 1
        else:
            scale = _empirical_scale(R.index)
    else:
        scale = _empirical_scale(R.index)

    if isinstance(R, pd.DataFrame):
        return pd.Series([scale] * len(R.columns), index=R.columns, name="Frequency")
    return scale


def _empirical_scale(index: pd.DatetimeIndex) -> int:
    if len(index) < 2:
        return 1
    days_diff = pd.Series(index).diff().dt.days.median()
    if days_diff <= 1.5:
        return 252
    elif days_diff <= 7.5:
        return 52
    elif days_diff <= 31.5:
        return 12
    elif days_diff <= 92.5:
        return 4
    return 1


def _get_scale(data: pd.Series | pd.DataFrame) -> int:
    res = frequency(data)
    if isinstance(res, pd.Series):
        return int(res.iloc[0])
    return int(res)


def co_skewness_matrix(R: pd.DataFrame, unbiased: bool = False) -> np.ndarray:
    r"""
    Calculate the N x N^2 multivariate sample coskewness matrix.
    """
    Xc = (R - R.mean()).values
    T, N = Xc.shape
    if unbiased:
        if T < 3:
            raise ValueError("R must have at least 3 rows for unbiased coskewness.")
        CC = T / ((T - 1) * (T - 2))
    else:
        CC = 1.0 / T

    R_kron = np.einsum('ti,tj->tij', Xc, Xc).reshape(T, N * N)
    M3 = CC * (Xc.T @ R_kron)
    return M3


def co_kurtosis_matrix(R: pd.DataFrame) -> np.ndarray:
    r"""
    Calculate the N x N^3 multivariate sample cokurtosis matrix.
    """
    Xc = (R - R.mean()).values
    T, N = Xc.shape
    R_kron3 = np.einsum('ti,tj,tk->tijk', Xc, Xc, Xc).reshape(T, N * N * N)
    M4 = (Xc.T @ R_kron3) / T
    return M4


def portm3(w: np.ndarray, M3: np.ndarray) -> float:
    r"""
    Calculate the portfolio third central moment w^T * M3 * (w x w).
    """
    w_flat = w.ravel()
    return float(w_flat.T @ M3 @ np.kron(w_flat, w_flat))


def derportm3(w: np.ndarray, M3: np.ndarray) -> np.ndarray:
    r"""
    Calculate the gradient of the portfolio third central moment with respect to weights.
    """
    w_flat = w.ravel()
    return 3.0 * M3 @ np.kron(w_flat, w_flat)


def portm4(w: np.ndarray, M4: np.ndarray) -> float:
    r"""
    Calculate the portfolio fourth central moment w^T * M4 * (w x w x w).
    """
    w_flat = w.ravel()
    return float(w_flat.T @ M4 @ np.kron(w_flat, np.kron(w_flat, w_flat)))


def derportm4(w: np.ndarray, M4: np.ndarray) -> np.ndarray:
    r"""
    Calculate the gradient of the portfolio fourth central moment with respect to weights.
    """
    w_flat = w.ravel()
    return 4.0 * M4 @ np.kron(w_flat, np.kron(w_flat, w_flat))
