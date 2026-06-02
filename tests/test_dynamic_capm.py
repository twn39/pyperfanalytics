import pytest
import numpy as np
import pandas as pd
from pyperfanalytics import capm_dynamic

def test_ols_r_compatibility(managers_data):
    """
    Verify that our OLS Ferson-Schadt conditional CAPM matches R's CAPM.dynamic
    to floating-point precision when using position alignment and global de-meaning.
    """
    Ra = managers_data["HAM1"]
    Rb = managers_data["SP500 TR"]
    Rf = 0.035 / 12
    Z = managers_data[["US 10Y TR", "US 3m TR"]]

    # Run Python CAPM.dynamic with R compatibility mode
    res = capm_dynamic(
        Ra, Rb, Rf=Rf, Z=Z, lags=1,
        method="ols", demean="global", align="position"
    )

    assert isinstance(res, pd.DataFrame)
    assert res.shape == (1, 6)

    # R ground truth coefficients computed via CAPM.dynamic
    expected_coefs = np.array([
        7.09649966e-03,   # Average alpha
        -1.96351030e-01,  # US 10Y TR alpha at t - 1
        1.66538074e-01,   # US 3m TR alpha at t - 1
        3.24801510e-01,   # Average beta
        3.49333616e+00,   # US 10Y TR beta at t - 1
        -6.37481371e+01   # US 3m TR beta at t - 1
    ])

    actual_coefs = res.iloc[0].values
    np.testing.assert_allclose(actual_coefs, expected_coefs, atol=1e-6, rtol=1e-6)
    # Actually, they match up to 1e-12 in local testing
    np.testing.assert_allclose(actual_coefs, expected_coefs, atol=1e-12)


def test_ols_strict_alignment(managers_data):
    """Verify OLS with strict alignment (correct date-time index alignment)."""
    Ra = managers_data["HAM1"]
    Rb = managers_data["SP500 TR"]
    Rf = 0.035 / 12
    Z = managers_data[["US 10Y TR", "US 3m TR"]]

    res_strict = capm_dynamic(
        Ra, Rb, Rf=Rf, Z=Z, lags=1,
        method="ols", demean="column", align="strict"
    )
    assert isinstance(res_strict, pd.DataFrame)
    assert res_strict.shape == (1, 6)
    # Check that coefficients are reasonable and distinct from the buggy R alignment
    assert res_strict.iloc[0]["Average alpha"] != pytest.approx(7.09649966e-03, abs=1e-5)


def test_kalman_dynamic_single_asset(managers_data):
    """Verify Kalman filter TVP-CAPM for a single asset (Series input)."""
    Ra = managers_data["HAM1"].dropna()
    # SP500 has no NaN during HAM1 period
    Rb = managers_data["SP500 TR"].loc[Ra.index]
    
    res = capm_dynamic(Ra, Rb, Rf=0.0, method="kalman", kalman_method="smooth")
    
    assert isinstance(res, pd.DataFrame)
    assert list(res.columns) == ["alpha", "beta", "alpha_se", "beta_se"]
    assert len(res) == len(Ra)
    assert res.index.equals(Ra.index)
    
    # Beta should be time-varying
    assert res["beta"].std() > 0
    # The mean beta should be positive and reasonable for an equity fund
    assert 0.1 < res["beta"].mean() < 0.6
    
    # Verify standard errors are positive
    assert (res["alpha_se"] > 0).all()
    assert (res["beta_se"] > 0).all()


def test_kalman_dynamic_multi_asset(managers_data):
    """Verify Kalman filter TVP-CAPM for multiple assets (DataFrame input)."""
    Ra = managers_data[["HAM1", "HAM2"]].dropna()
    Rb = managers_data["SP500 TR"].loc[Ra.index]
    
    # Returns a dict of DataFrames mapped by asset column name
    res = capm_dynamic(Ra, Rb, Rf=0.0, method="kalman", kalman_method="smooth")
    
    assert isinstance(res, dict)
    assert set(res.keys()) == {"HAM1", "HAM2"}
    
    for key, df in res.items():
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["alpha", "beta", "alpha_se", "beta_se"]
        assert len(df) == len(Ra)
        assert df.index.equals(Ra.index)


def test_kalman_parallel_execution(managers_data):
    """Verify parallel ProcessPoolExecutor execution for multi-asset Kalman fitting."""
    Ra = managers_data[["HAM1", "HAM2", "HAM3", "HAM4"]].dropna()
    Rb = managers_data["SP500 TR"].loc[Ra.index]
    
    # Run with parallel workers
    res_parallel = capm_dynamic(Ra, Rb, Rf=0.0, method="kalman", n_jobs=2)
    # Run in serial
    res_serial = capm_dynamic(Ra, Rb, Rf=0.0, method="kalman", n_jobs=1)
    
    assert isinstance(res_parallel, dict)
    assert set(res_parallel.keys()) == {"HAM1", "HAM2", "HAM3", "HAM4"}
    
    # Check that outputs are identical
    for asset in Ra.columns:
        pd.testing.assert_frame_equal(res_parallel[asset], res_serial[asset])


def test_input_validation(managers_data):
    """Verify parameter checks and validation errors."""
    Ra = managers_data["HAM1"]
    Rb = managers_data["SP500 TR"]
    
    # 1. Invalid method name
    with pytest.raises(ValueError, match="method must be"):
        capm_dynamic(Ra, Rb, method="invalid")
        
    # 2. Missing Z in OLS mode
    with pytest.raises(ValueError, match="Z must be provided"):
        capm_dynamic(Ra, Rb, method="ols", Z=None)
