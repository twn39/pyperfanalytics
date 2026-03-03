import pandas as pd
import numpy as np
from typing import Union, List, Dict, Any, Optional

def drawdowns(R: Union[pd.Series, pd.DataFrame], geometric: bool = True) -> Union[pd.Series, pd.DataFrame]:
    """
    Calculate the drawdown levels in a timeseries.
    """
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

def max_drawdown(R: Union[pd.Series, pd.DataFrame], geometric: bool = True) -> Union[float, pd.Series]:
    """
    Calculate the maximum drawdown of a return series.
    Returns the absolute value of the worst-case loss.
    """
    dd = drawdowns(R, geometric=geometric)
    return np.abs(dd.min())

def find_drawdowns(R: Union[pd.Series, pd.DataFrame], geometric: bool = True) -> Dict[str, np.ndarray]:
    """
    Find drawdowns in a return series.
    Returns a dictionary with return, from, trough, to, length, peaktotrough, recovery.
    Indices are 1-indexed to match R.
    """
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
            "return": np.array([]), "from": np.array([]), "trough": np.array([]), 
            "to": np.array([]), "length": np.array([]), "peaktotrough": np.array([]), 
            "recovery": np.array([])
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
        "recovery": recoveries
    }

def average_drawdown(R: Union[pd.Series, pd.DataFrame], geometric: bool = True) -> Union[float, pd.Series]:
    """
    Calculate the average depth of the observed drawdowns.
    """
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

def average_length(R: Union[pd.Series, pd.DataFrame], geometric: bool = True) -> Union[float, pd.Series]:
    """
    Calculate the average length (in periods) of the observed drawdowns.
    """
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

def average_recovery(R: Union[pd.Series, pd.DataFrame], geometric: bool = True) -> Union[float, pd.Series]:
    """
    Calculate the average length (in periods) of the observed recovery period.
    """
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

def drawdown_deviation(R: Union[pd.Series, pd.DataFrame], geometric: bool = True) -> Union[float, pd.Series]:
    """
    Calculate a standard deviation-type statistic using individual drawdowns.
    DD = sqrt(sum[j=1,2,...,d](D_j^2/n))
    """
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

def sort_drawdowns(runs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Sort drawdowns from worst to best.
    """
    if len(runs["return"]) == 0:
        return runs
    idx = np.argsort(runs["return"])
    return {k: v[idx] for k, v in runs.items()}

def drawdown_peak(R: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
    """
    Replicate R's DrawdownPeak logic.
    This function assumes returns are in percent (multiplies/divides by 100).
    It is used by UlcerIndex and PainIndex in PerformanceAnalytics.
    """
    def _calc(s: pd.Series) -> pd.Series:
        s = s.dropna()
        n = len(s)
        res = np.zeros(n)
        peak_idx = 0
        for i in range(n):
            val = 1.0
            for j in range(peak_idx, i + 1):
                val *= (1.0 + s.iloc[j] / 100.0)
            if val > 1.0:
                peak_idx = i + 1
                res[i] = 0.0
            else:
                res[i] = (val - 1.0) * 100.0
        return pd.Series(res, index=s.index)

    if isinstance(R, pd.DataFrame):
        return R.apply(_calc)
    else:
        return _calc(R)

def cdd(R: Union[pd.Series, pd.DataFrame], p: float = 0.95, geometric: bool = True, invert: bool = True) -> Union[float, pd.Series]:
    """
    Calculate Uryasev's proposed Conditional Drawdown at Risk (CDD or CDaR) measure.
    For some confidence level p, the conditional drawdown is the mean of the worst p% drawdowns.
    Wait, R's implementation uses quantile: result = quantile(drawdowns$return, p).
    Actually, R's CDD uses quantile(drawdowns$return, p) but p is usually 0.95.
    Wait! R's implementation says `p=.setalphaprob(p)`. If p=0.95, .setalphaprob gives 0.05.
    Then `quantile(drawdowns$return, 0.05)` is the 5% quantile (which is deep negative).
    """
    def _calc(s: pd.Series, prob: float) -> float:
        s = s.dropna()
        if len(s) == 0:
            return np.nan
        # R's .setalphaprob(p) converts p>0.5 to 1-p. So p=0.95 becomes 0.05.
        if prob > 0.5:
            prob = 1 - prob
        
        dd_info = find_drawdowns(s, geometric=geometric)
        returns = dd_info["return"]
        if len(returns) == 0:
            return 0.0
            
        # PerformanceAnalytics just uses quantile
        val = np.quantile(returns, prob)
        if invert:
            return float(-val)
        return float(val)

    if isinstance(R, pd.DataFrame):
        return R.apply(_calc, prob=p)
    else:
        return _calc(R, p)
