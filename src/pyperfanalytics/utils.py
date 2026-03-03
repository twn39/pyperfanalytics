import numpy as np
import pandas as pd
from typing import Union

def centered_moment(R: Union[pd.Series, pd.DataFrame], moment: int) -> Union[float, pd.Series]:
    """
    Calculate the nth centered moment (population version, matching R's PerformanceAnalytics).
    """
    def _calc(s: pd.Series, m: int) -> float:
        s = s.dropna()
        if len(s) == 0: return np.nan
        return np.mean((s - s.mean())**m)

    if isinstance(R, pd.DataFrame):
        return R.apply(_calc, m=moment)
    else:
        return _calc(R, moment)

def centered_comoment(Ra: pd.Series, Rb: pd.Series, p1: int, p2: int, normalize: bool = False) -> float:
    """
    Calculate the joint centered comoment of two series.
    E[ (Ra - E[Ra])^p1 * (Rb - E[Rb])^p2 ]
    """
    merged = pd.concat([Ra, Rb], axis=1).dropna()
    if merged.empty: return np.nan
    
    a = merged.iloc[:, 0]
    b = merged.iloc[:, 1]
    
    centered_a = a - a.mean()
    centered_b = b - b.mean()
    
    out = (centered_a**p1 * centered_b**p2).mean()
    
    if normalize:
        # R code: out = out / centeredmoment(Rb, power=(p1+p2))
        m_b = (centered_b**(p1+p2)).mean()
        if m_b == 0: return np.nan
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

def skewness(R: Union[pd.Series, pd.DataFrame], method: str = "moment") -> Union[float, pd.Series]:
    """
    Calculate skewness of the return distribution.
    Methods: 'moment', 'fisher', 'sample'.
    """
    def _calc(s: pd.Series, meth: str) -> float:
        s = s.dropna()
        n = len(s)
        if n == 0: return np.nan
        
        # PerformanceAnalytics specific:
        # method="moment" is population skewness (centered)
        # method="fisher" is raw fisher (NO centering)
        # method="sample" is adjusted population skewness (centered)
        
        if meth == "moment":
            m2 = np.mean((s - s.mean())**2)
            m3 = np.mean((s - s.mean())**3)
            return m3 / (m2**(1.5))
        elif meth == "fisher":
            if n < 3: return np.nan
            return ((np.sqrt(n*(n-1))/(n-2)) * (np.mean(s**3))) / (np.mean(s**2)**(1.5))
        elif meth == "sample":
            if n < 3: return np.nan
            m2 = np.mean((s - s.mean())**2)
            m3 = np.mean((s - s.mean())**3)
            return (m3 / (m2**(1.5))) * n / ((n-1)*(n-2)) * n # R's sample formula: n/((n-1)(n-2)) * sum( (x-mu)^3 / sd_pop^3 )
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

def kurtosis(R: Union[pd.Series, pd.DataFrame], method: str = "excess") -> Union[float, pd.Series]:
    """
    Calculate kurtosis of the return distribution.
    Methods: 'excess', 'moment', 'fisher', 'sample', 'sample_excess'.
    """
    def _calc(s: pd.Series, meth: str) -> float:
        s = s.dropna()
        n = len(s)
        if n == 0: return np.nan
        
        m2 = np.mean((s - s.mean())**2)
        m4 = np.mean((s - s.mean())**4)
        
        if meth == "moment":
            return m4 / (m2**2)
        elif meth == "excess":
            return (m4 / (m2**2)) - 3
        elif meth == "fisher":
            if n < 4: return np.nan
            # R's fisher is raw (NO centering)
            r2 = np.mean(s**2)
            r4 = np.mean(s**4)
            return ((n+1)*(n-1)*((r4 / r2**2) - (3*(n-1))/(n+1))) / ((n-2)*(n-3))
        elif meth == "sample":
            if n < 4: return np.nan
            # In R: sum((x-mean(x))^4/var(x)^2)*n*(n+1)/((n-1)*(n-2)*(n-3))
            # var(x) = m2 * n / (n-1)
            # sum((x-mu)^4) = n * m4
            var_x = s.var(ddof=1)
            return (n * m4 / var_x**2) * n * (n+1) / ((n-1)*(n-2)*(n-3))
        elif meth == "sample_excess":
            if n < 4: return np.nan
            k = _calc(s, "sample")
            return k - 3.0 * (n-1)**2 / ((n-2)*(n-3))
        else:
            raise ValueError(f"Unknown kurtosis method: {meth}")

    if isinstance(R, pd.DataFrame):
        return R.apply(_calc, meth=method)
    else:
        return _calc(R, method)
def _get_scale(data: Union[pd.Series, pd.DataFrame]) -> int:
    """
    Determine the scale (periods per year) based on index frequency.
    """
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Data index must be a DatetimeIndex to determine scale.")

    # Simple heuristic based on pandas frequency or average spacing
    freq = data.index.inferred_freq
    if freq:
        if 'B' in freq or 'D' in freq:
            return 252
        if 'W' in freq:
            return 52
        if 'ME' in freq or 'M' in freq:
            return 12
        if 'QE' in freq or 'Q' in freq:
            return 4
        if 'YE' in freq or 'Y' in freq:
            return 1

    # Fallback to empirical spacing if freq is not set
    days_diff = pd.Series(data.index).diff().dt.days.median()
    if days_diff <= 1.5:
        return 252
    if days_diff <= 7.5:
        return 52
    if days_diff <= 31.5:
        return 12
    if days_diff <= 92.5:
        return 4
    return 1
