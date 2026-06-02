import pytest
import numpy as np
import pandas as pd
from pyperfanalytics import (
    co_skewness_matrix,
    co_kurtosis_matrix,
    portm3,
    derportm3,
    portm4,
    derportm4,
    var_historical,
    var_gaussian,
    var_modified,
    es_historical,
    es_gaussian,
    es_modified,
)

@pytest.fixture
def sample_portfolio_data():
    """Generates simple mock returns for testing."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    df = pd.DataFrame(
        np.random.normal(0.0005, 0.01, size=(100, 3)),
        index=dates,
        columns=["AssetA", "AssetB", "AssetC"]
    )
    weights = np.array([0.4, 0.3, 0.3])
    return df, weights


def test_comoments_equivalence(sample_portfolio_data):
    """Verify that using the comoment matrices yields exact mathematical equivalence with deviations."""
    df, w = sample_portfolio_data
    Xc = (df - df.mean()).values
    y = Xc @ w
    T, N = Xc.shape

    # Directly calculate portfolio moments
    pm3_direct = float(np.mean(y**3))
    pm4_direct = float(np.mean(y**4))

    dpm3_direct = 3.0 * (Xc.T @ (y**2)) / T
    dpm4_direct = 4.0 * (Xc.T @ (y**3)) / T

    # Calculate via comoment matrices
    M3 = co_skewness_matrix(df, unbiased=False)
    M4 = co_kurtosis_matrix(df)

    pm3_matrix = portm3(w, M3)
    pm4_matrix = portm4(w, M4)

    dpm3_matrix = derportm3(w, M3)
    dpm4_matrix = derportm4(w, M4)

    # Assert close values
    assert pm3_direct == pytest.approx(pm3_matrix, abs=1e-12)
    assert pm4_direct == pytest.approx(pm4_matrix, abs=1e-12)
    np.testing.assert_allclose(dpm3_direct, dpm3_matrix, atol=1e-12)
    np.testing.assert_allclose(dpm4_direct, dpm4_matrix, atol=1e-12)


def test_euler_allocation(sample_portfolio_data):
    """
    Verify Euler's allocation theorem: sum(contribution) == total_risk
    for all 6 risk estimators.
    """
    df, w = sample_portfolio_data

    estimators = [
        ("var_historical", var_historical, "hVaR"),
        ("var_gaussian", var_gaussian, "gVaR"),
        ("var_modified", var_modified, "MVaR"),
        ("es_historical", es_historical, "hES"),
        ("es_gaussian", es_gaussian, "gES"),
        ("es_modified", es_modified, "MES"),
    ]

    for name, func, key in estimators:
        # Run component decomposition
        res = func(df, p=0.95, weights=w, portfolio_method="component")
        
        assert isinstance(res, dict)
        assert key in res
        assert "contribution" in res
        assert "pct_contrib" in res

        total_risk = res[key]
        contrib = res["contribution"]
        pct_contrib = res["pct_contrib"]

        # 1. sum(contribution) == total_risk
        assert contrib.sum() == pytest.approx(total_risk, abs=1e-12)

        # 2. pct_contrib == contrib / total_risk
        expected_pct = contrib / total_risk if total_risk != 0 else np.zeros_like(contrib)
        np.testing.assert_allclose(pct_contrib, expected_pct, atol=1e-12)

        # 3. Check compatibility with marginal return
        marginal = func(df, p=0.95, weights=w, portfolio_method="marginal")
        assert isinstance(marginal, pd.Series)
        
        # contribution = weights * marginal
        np.testing.assert_allclose(contrib, w * marginal, atol=1e-12)


def test_path_a_vs_path_b_equivalence(sample_portfolio_data):
    """
    For modified VaR and ES, verify that Path A (M3/M4 is None)
    and Path B (M3/M4 is explicitly passed or computed internally)
    produce identical outputs.
    """
    df, w = sample_portfolio_data

    # Test modified VaR
    res_a_var = var_modified(df, p=0.95, weights=w, portfolio_method="component")
    M3 = co_skewness_matrix(df, unbiased=False)
    M4 = co_kurtosis_matrix(df)
    res_b_var = var_modified(df, p=0.95, weights=w, portfolio_method="component", M3=M3, M4=M4)

    assert res_a_var["MVaR"] == pytest.approx(res_b_var["MVaR"], abs=1e-12)
    np.testing.assert_allclose(res_a_var["contribution"], res_b_var["contribution"], atol=1e-12)
    np.testing.assert_allclose(res_a_var["pct_contrib"], res_b_var["pct_contrib"], atol=1e-12)

    # Test modified ES
    res_a_es = es_modified(df, p=0.95, weights=w, portfolio_method="component")
    res_b_es = es_modified(df, p=0.95, weights=w, portfolio_method="component", M3=M3, M4=M4)

    assert res_a_es["MES"] == pytest.approx(res_b_es["MES"], abs=1e-12)
    np.testing.assert_allclose(res_a_es["contribution"], res_b_es["contribution"], atol=1e-12)
    np.testing.assert_allclose(res_a_es["pct_contrib"], res_b_es["pct_contrib"], atol=1e-12)


def test_input_validation(sample_portfolio_data):
    """Verify input validation and error handling for weights and portfolio methods."""
    df, w = sample_portfolio_data

    # 1. R is not DataFrame but component/marginal method requested
    series = df["AssetA"]
    with pytest.raises(ValueError, match="R must be a DataFrame"):
        var_gaussian(series, weights=[1.0], portfolio_method="component")

    # 2. Weights length does not match columns
    with pytest.raises(ValueError, match="Length of weights must match the number of columns"):
        var_gaussian(df, weights=[0.5, 0.5], portfolio_method="single")

    # 3. Invalid portfolio_method
    with pytest.raises(ValueError, match="portfolio_method must be one of"):
        var_gaussian(df, weights=w, portfolio_method="invalid_method")


def test_high_dimensional_smoke():
    """Smoke test to verify that the O(TN) Direct Data path does not crash or hang for high dimensions."""
    np.random.seed(123)
    T, N = 1000, 50
    df = pd.DataFrame(
        np.random.normal(0.0001, 0.02, size=(T, N)),
        columns=[f"Asset_{i}" for i in range(N)]
    )
    w = np.ones(N) / N

    # Running with None M3/M4 (uses the fast direct deviations path)
    res = var_modified(df, p=0.95, weights=w, portfolio_method="component")
    assert "MVaR" in res
    assert len(res["contribution"]) == N
    assert res["contribution"].sum() == pytest.approx(res["MVaR"], abs=1e-12)
