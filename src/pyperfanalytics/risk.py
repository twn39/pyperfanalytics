import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Union, Optional
from pyperfanalytics.utils import centered_moment, skewness, kurtosis, _get_scale

def var_historical(R: Union[pd.Series, pd.DataFrame], p: float = 0.95) -> Union[float, pd.Series]:
    """
    Calculate Historical VaR (negative of the alpha-quantile).
    """
    alpha = 1 - p if p >= 0.5 else p
    def _calc(s: pd.Series, a: float) -> float:
        s = s.dropna()
        if len(s) == 0: return np.nan
        return -np.percentile(s, a * 100)

    if isinstance(R, pd.DataFrame):
        return R.apply(_calc, a=alpha)
    else:
        return _calc(R, alpha)

def var_gaussian(R: Union[pd.Series, pd.DataFrame], p: float = 0.95) -> Union[float, pd.Series]:
    """
    Calculate Gaussian VaR.
    """
    alpha = 1 - p if p >= 0.5 else p
    mu = R.mean()
    m2 = centered_moment(R, 2)
    z = norm.ppf(alpha)
    return -mu - z * np.sqrt(m2)

def var_modified(R: Union[pd.Series, pd.DataFrame], p: float = 0.95) -> Union[float, pd.Series]:
    """
    Calculate Modified (Cornish-Fisher) VaR.
    """
    alpha = 1 - p if p >= 0.5 else p
    z = norm.ppf(alpha)
    
    mu = R.mean()
    m2 = centered_moment(R, 2)
    skew = skewness(R, method="moment")
    exkurt = kurtosis(R, method="excess")
    
    # Cornish-Fisher expansion for the z-score
    h = (z + 
         (1/6.0) * (z**2 - 1.0) * skew + 
         (1/24.0) * (z**3 - 3.0*z) * exkurt - 
         (1/36.0) * (2.0*z**3 - 5.0*z) * skew**2)
    
    return -mu - h * np.sqrt(m2)

def es_historical(R: Union[pd.Series, pd.DataFrame], p: float = 0.95) -> Union[float, pd.Series]:
    """
    Calculate Historical Expected Shortfall.
    """
    alpha = 1 - p if p >= 0.5 else p
    def _calc(s: pd.Series, a: float) -> float:
        s = s.dropna()
        if len(s) == 0: return np.nan
        q = np.percentile(s, a * 100)
        subset = s[s < q]
        if len(subset) == 0:
            return -q # PA's fallback: if no values < VaR, ES = VaR
        return -subset.mean()

    if isinstance(R, pd.DataFrame):
        return R.apply(_calc, a=alpha)
    else:
        return _calc(R, alpha)

def es_gaussian(R: Union[pd.Series, pd.DataFrame], p: float = 0.95) -> Union[float, pd.Series]:
    """
    Calculate Gaussian Expected Shortfall.
    """
    alpha = 1 - p if p >= 0.5 else p
    mu = R.mean()
    m2 = centered_moment(R, 2)
    z = norm.ppf(alpha)
    return -mu + norm.pdf(z) * np.sqrt(m2) / alpha

def es_modified(R: Union[pd.Series, pd.DataFrame], p: float = 0.95) -> Union[float, pd.Series]:
    """
    Calculate Modified (Cornish-Fisher) Expected Shortfall.
    """
    alpha = 1 - p if p >= 0.5 else p
    z = norm.ppf(alpha)
    
    mu = R.mean()
    m2 = centered_moment(R, 2)
    skew = skewness(R, method="moment")
    exkurt = kurtosis(R, method="excess")
    
    # h as used in MES calculation in R code
    h = (z + (1/6.0) * (z**2 - 1.0) * skew + 
         (1/24.0) * (z**3 - 3.0*z) * exkurt - 
         (1/36.0) * (2.0*z**3 - 5.0*z) * skew**2)
    
    # E as defined in ES.CornishFisher in R code
    E = (norm.pdf(h) * (
        1.0 + 
        (h**3) * skew / 6.0 + 
        (h**6 - 9.0*h**4 + 9.0*h**2 + 3.0) * (skew**2) / 72.0 + 
        (h**4 - 2.0*h**2 - 1.0) * exkurt / 24.0
    )) / alpha
    
    return -mu + np.sqrt(m2) * E

def tracking_error(
    R: Union[pd.Series, pd.DataFrame],
    Rb: Union[pd.Series, pd.DataFrame],
    scale: Optional[int] = None
) -> Union[float, pd.Series, pd.DataFrame]:
    """
    Calculate Tracking Error of returns against a benchmark.
    
    Tracking error is calculated by taking the standard deviation of the 
    difference between the investment's returns and the benchmark's returns, 
    then multiplying the result by the square root of the scale of the returns.
    """
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
    Ra: Union[pd.Series, pd.DataFrame],
    Rb: Union[pd.Series, pd.DataFrame],
    Rf: Union[float, pd.Series, pd.DataFrame] = 0
) -> Union[float, pd.Series, pd.DataFrame]:
    """
    Calculate CAPM beta of returns against a benchmark.
    
    Beta is the ratio of the covariance of the asset's excess returns 
    with the benchmark's excess returns to the variance of the 
    benchmark's excess returns.
    """
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
            
            # Match R's strict NA handling
            if a.isna().any() or b.isna().any():
                col_results.append(np.nan)
                continue
                
            # Align xRa and xRb
            merged = pd.concat([a, b], axis=1).dropna()
            if merged.empty:
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
def ulcer_index(R: Union[pd.Series, pd.DataFrame]) -> Union[float, pd.Series]:
    """
    Calculate the Ulcer Index.
    Matches R's PerformanceAnalytics implementation (using DrawdownPeak).
    """
    from pyperfanalytics.drawdowns import drawdown_peak
    dp = drawdown_peak(R)
    
    def _calc(s: pd.Series) -> float:
        s = s.dropna()
        if len(s) == 0: return np.nan
        return np.sqrt((s**2).mean())
        
    if isinstance(dp, pd.DataFrame):
        return dp.apply(_calc)
    else:
        return _calc(dp)

def pain_index(R: Union[pd.Series, pd.DataFrame]) -> Union[float, pd.Series]:
    """
    Calculate the Pain Index.
    Matches R's PerformanceAnalytics implementation (using DrawdownPeak).
    """
    from pyperfanalytics.drawdowns import drawdown_peak
    dp = drawdown_peak(R)
    
    def _calc(s: pd.Series) -> float:
        s = s.dropna()
        if len(s) == 0: return np.nan
        return np.abs(s).mean()
        
    if isinstance(dp, pd.DataFrame):
        return dp.apply(_calc)
    else:
        return _calc(dp)



def specific_risk(
    Ra: Union[pd.Series, pd.DataFrame],
    Rb: Union[pd.Series, pd.DataFrame],
    Rf: Union[float, pd.Series, pd.DataFrame] = 0,
    scale: Optional[int] = None
) -> Union[float, pd.Series, pd.DataFrame]:
    """
    Calculate Specific Risk.
    
    Specific risk is the annualized standard deviation of the error term 
    in the CAPM regression.
    """
    if scale is None:
        scale = _get_scale(Ra)
        
    from pyperfanalytics.returns import return_excess, capm_alpha
    
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
            
            # result = sqrt(sum((epsilon - mean(epsilon))^2)/length(epsilon))*sqrt(Period)
            # This is essentially population SD of epsilon * sqrt(scale)
            spec_risk = np.sqrt((epsilon**2).mean() - (epsilon.mean())**2) * np.sqrt(scale)
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
    Ra: Union[pd.Series, pd.DataFrame],
    Rb: Union[pd.Series, pd.DataFrame],
    Rf: Union[float, pd.Series, pd.DataFrame] = 0,
    scale: Optional[int] = None
) -> Union[float, pd.Series, pd.DataFrame]:
    """
    Calculate Total Risk (Systematic + Specific).
    Note: R's TotalRisk uses sample variance for both Systematic and Specific 
    components, whereas SpecificRisk uses population variance. We match R here.
    """
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

def herfindahl_index(R: Union[pd.Series, pd.DataFrame]) -> Union[float, pd.Series]:
    """
    Calculate Herfindahl Index based on autocorrelation.
    Note: This follows R's PerformanceAnalytics implementation which uses squared 
    scaled positive autocorrelation coefficients.
    """
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


def smoothing_index(
    R: Union[pd.Series, pd.DataFrame],
    neg_thetas: bool = False,
    MAorder: int = 2
) -> Union[float, pd.Series]:
    """
    Calculate Normalized Getmansky Smoothing Index.
    A lower value implies more smoothing (less liquid).
    A value of 1 implies no smoothing (more liquid).
    """
    from statsmodels.tsa.arima.model import ARIMA
    
    def _calc(s: pd.Series, neg: bool, order: int) -> float:
        s = s.dropna()
        if len(s) < order + 5: # Need enough data for ARIMA
            return np.nan
        
        try:
            import warnings
            from statsmodels.tools.sm_exceptions import ValueWarning
            
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ValueWarning)
                # R arima(..., include.mean=FALSE)
                model = ARIMA(s, order=(0, 0, order), enforce_invertibility=True, trend='n')
                res = model.fit()
            
            # statsmodels params for MA components are 'ma.L1', 'ma.L2'...
            ma_params = []
            for i in range(1, order + 1):
                name = f'ma.L{i}'
                if name in res.params:
                    ma_params.append(res.params[name])
                else:
                    ma_params.append(0.0)
            
            if not neg:
                ma_params = [max(0.0, c) for c in ma_params]
                
            thetas = np.array([1.0] + ma_params)
            thetas /= thetas.sum()
            
            return float(np.sum(thetas**2))
        except:
            return np.nan

    if isinstance(R, pd.DataFrame):
        return R.apply(_calc, neg=neg_thetas, order=MAorder)
    else:
        return _calc(R, neg_thetas, MAorder)

def fama_beta(
    Ra: Union[pd.Series, pd.DataFrame],
    Rb: Union[pd.Series, pd.DataFrame],
    scale: Optional[int] = None
) -> Union[float, pd.Series, pd.DataFrame]:
    """
    Calculate Fama Beta.
    Fama beta = portfolio standard deviation / benchmark standard deviation.
    Annualized and using population standard deviation.
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
    Ra: Union[pd.Series, pd.DataFrame],
    Rb: Union[pd.Series, pd.DataFrame],
    p: float = 0.95,
    geometric: bool = True,
    type: Optional[str] = None
) -> Union[float, pd.Series, pd.DataFrame]:
    """
    Conditional Drawdown Beta.
    """
    from pyperfanalytics.drawdowns import find_drawdowns, cdd
    
    if type == "average":
        p = 1.0 # Will be 1-p = 0
    elif type == "max":
        p = 0.0 # Will be 1-p = 1
        
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
                temp_r = a.iloc[start:end+1]
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
    Ra: Union[pd.Series, pd.DataFrame],
    Rb: Union[pd.Series, pd.DataFrame],
    p: float = 0.95,
    geometric: bool = True,
    type: Optional[str] = None,
    scale: Optional[int] = None
) -> Union[float, pd.Series, pd.DataFrame]:
    """
    Conditional Drawdown Alpha.
    """
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
            
            rm_exp = (1 + b.mean())**scale - 1
            ra_exp = (1 + a.mean())**scale - 1
            
            b_val = beta.loc[rb_col, ra_col] if isinstance(beta, pd.DataFrame) else (beta[ra_col] if isinstance(beta, pd.Series) else beta)
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
    R: Union[pd.Series, pd.DataFrame],
    refSR: float,
    Rf: Union[float, pd.Series, pd.DataFrame] = 0.0,
    p: float = 0.95,
    ignore_skewness: bool = False,
    ignore_kurtosis: bool = True
) -> Union[dict, pd.DataFrame]:
    """
    Minimum Track Record Length.
    """
    from pyperfanalytics.returns import sharpe_ratio
    from pyperfanalytics.utils import skewness, kurtosis
    from scipy.stats import norm
    
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
        
        sr = sharpe_ratio(s, Rf=Rf, annualize=False) # R's MinTrackRecord expects periodic sr
        sk = 0.0 if ignore_skewness else skewness(s)
        kr = 3.0 if ignore_kurtosis else kurtosis(s, method="moment")
        
        # 1 + (1 - sk*sr + ((kr-1)/4)*sr^2)*(qnorm(p)/(sr-refSR))^2
        if sr <= refSR:
            mtr = np.nan
        else:
            q = norm.ppf(p)
            mtr = 1 + (1 - sk*sr + ((kr-1)/4)*sr**2) * (q / (sr - refSR))**2
            
        min_trl.append(mtr)
        if np.isnan(mtr):
            is_sig.append(False)
            extra_obs.append(np.nan)
        else:
            is_sig.append(n > mtr)
            extra = max(0, np.ceil(mtr - n))
            extra_obs.append(extra)
            
    res = pd.DataFrame({
        "min_TRL": min_trl,
        "IS_SR_SIGNIFICANT": is_sig,
        "num_of_extra_obs_needed": extra_obs
    }, index=cols).T
    
    if len(cols) == 1:
        return res.iloc[:, 0].to_dict()
    return res

def systematic_risk(
    Ra: Union[pd.Series, pd.DataFrame],
    Rb: Union[pd.Series, pd.DataFrame],
    Rf: Union[float, pd.Series, pd.DataFrame] = 0,
    scale: Optional[int] = None
) -> Union[float, pd.Series, pd.DataFrame]:
    """
    Calculate Systematic Risk.
    systematic risk = beta * market risk
    """
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

def specific_risk(
    Ra: Union[pd.Series, pd.DataFrame],
    Rb: Union[pd.Series, pd.DataFrame],
    Rf: Union[float, pd.Series, pd.DataFrame] = 0,
    scale: Optional[int] = None
) -> Union[float, pd.Series, pd.DataFrame]:
    """
    Calculate Specific Risk.
    Specific risk is the standard deviation of the error term in the regression.
    """
    if scale is None:
        scale = _get_scale(Ra)

    from pyperfanalytics.returns import capm_alpha
    
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
            a_raw = ra_df[ra_col]
            b_raw = rb_df[rb_col]
            
            merged = pd.concat([a_raw, b_raw], axis=1).dropna()
            if merged.empty:
                col_results.append(np.nan)
                continue
            
            a = merged.iloc[:, 0]
            b = merged.iloc[:, 1]
            
            bta = capm_beta(a, b, Rf=Rf)
            alpha = capm_alpha(a, b, Rf=Rf) # periodic
            
            epsilon = a - b * bta - alpha
            # R uses population SD (ddof=0)
            res_std = epsilon.std(ddof=0) * np.sqrt(scale)
            col_results.append(float(res_std))
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

