import numpy as np
import pandas as pd
import pytest

import pyperfanalytics as pa


def test_pairwise_deletion():
    """Test that functions requiring two series drop missing pairs correctly."""
    idx = pd.date_range("2020-01-01", periods=5, freq="ME")
    ra = pd.Series([0.01, 0.02, np.nan, 0.04, 0.05], index=idx)  # Missing 3rd
    rb = pd.Series([0.02, np.nan, 0.03, 0.02, 0.01], index=idx)  # Missing 2nd

    # Only periods 1, 4, 5 are completely overlapping
    # ra_valid = [0.01, 0.04, 0.05]
    # rb_valid = [0.02, 0.02, 0.01]

    # Test 1: Tracking error (which computes diffs, needs aligned data)
    te = pa.tracking_error(ra, rb, scale=12)
    # Validate against manual calc on valid overlapping data
    ra_valid = pd.Series([0.01, 0.04, 0.05])
    rb_valid = pd.Series([0.02, 0.02, 0.01])
    diff = ra_valid - rb_valid
    expected_te = np.sqrt(12) * diff.std(ddof=1)
    assert te == pytest.approx(expected_te, abs=1e-6)

    # Test 2: CAPM Beta
    beta = pa.capm_beta(ra, rb, Rf=0.0)
    expected_beta = diff.cov(rb_valid) / rb_valid.var(ddof=1) + 1  # Cov(Ra,Rb)/Var(Rb)
    assert beta == pytest.approx(expected_beta, rel=1e-4)


def test_all_nan_series():
    """Test functions against a series of pure NaNs to ensure graceful nan returns."""
    idx = pd.date_range("2020-01-01", periods=10, freq="ME")
    s_nan = pd.Series([np.nan] * 10, index=idx)

    # Simple returns
    assert np.isnan(pa.return_annualized(s_nan))
    assert np.isnan(pa.return_cumulative(s_nan))
    
    # Risk
    assert np.isnan(pa.std_dev_annualized(s_nan))
    assert np.isnan(pa.max_drawdown(s_nan))
    assert np.isnan(pa.sharpe_ratio(s_nan, Rf=0.0))

    # Bivariate
    rb = pd.Series(np.random.randn(10), index=idx)
    assert np.isnan(pa.capm_beta(s_nan, rb))
    assert np.isnan(pa.jensen_alpha(s_nan, rb))
    assert np.isnan(pa.information_ratio(s_nan, rb))


def test_insufficient_data():
    """Test functions that require a minimum number of observations."""
    idx = pd.date_range("2020-01-01", periods=3, freq="ME")
    
    # Needs at least 4 for kurtosis
    s = pd.Series([0.01, 0.02, 0.03], index=idx)
    assert np.isnan(pa.utils.kurtosis(s))
    
    # Regression needs at least 2 overlapping points. We'll give it 1.
    s1 = pd.Series([0.01, np.nan, np.nan], index=idx)
    s2 = pd.Series([0.02, 0.03, 0.04], index=idx)
    
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        beta = pa.capm_beta(s1, s2)
        assert np.isnan(beta)

        te = pa.tracking_error(s1, s2)
        assert np.isnan(te)

        m2 = pa.m_squared(s1, s2)
        assert np.isnan(m2)

