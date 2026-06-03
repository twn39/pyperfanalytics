import numpy as np
import pandas as pd
import pytest
import pyperfanalytics as pa


def test_standard_deviation_aliases_and_sample_methods():
    # Create test data
    idx = pd.date_range("2020-01-01", periods=12, freq="ME")
    R = pd.Series([0.01, -0.02, 0.03, 0.01, -0.01, 0.02, 0.04, -0.03, 0.01, 0.02, -0.01, 0.03], index=idx)

    # 1. Test unbiased (ddof=1)
    std_unbiased = pa.std_dev_annualized(R, scale=12, sample_method="unbiased")
    expected_unbiased = R.std(ddof=1) * np.sqrt(12)
    assert std_unbiased == pytest.approx(expected_unbiased, abs=1e-12)

    # 2. Test ML (ddof=0)
    std_ml = pa.std_dev_annualized(R, scale=12, sample_method="ML")
    expected_ml = R.std(ddof=0) * np.sqrt(12)
    assert std_ml == pytest.approx(expected_ml, abs=1e-12)

    # 3. Test aliases
    assert pa.sd_annualized(R, scale=12) == pytest.approx(expected_unbiased, abs=1e-12)
    assert pa.sd_multiperiod(R, scale=3) == pytest.approx(R.std(ddof=1) * np.sqrt(3), abs=1e-12)


def test_capm_auxiliary_functions():
    idx = pd.date_range("2020-01-01", periods=12, freq="ME")
    Ra = pd.Series([0.02, 0.03, -0.01, 0.04, 0.01, 0.02, 0.03, -0.02, 0.01, 0.04, -0.01, 0.03], index=idx)
    Rb = pd.Series([0.01, 0.02, 0.00, 0.03, 0.01, 0.01, 0.02, -0.01, 0.00, 0.03, -0.02, 0.02], index=idx)
    Rf = 0.001

    # 1. CML slope (Sharpe Ratio of Rb)
    slope = pa.capm_cml_slope(Rb, Rf=Rf)
    expected_slope = (Rb - Rf).mean() / (Rb - Rf).std(ddof=1)
    assert slope == pytest.approx(expected_slope, abs=1e-12)

    # 2. CML expected return: Rf_mean + CML_Slope * std_dev(Ra)
    cml = pa.capm_cml(Ra, Rb, Rf=Rf)
    expected_cml = Rf + expected_slope * Ra.std(ddof=1)
    assert cml == pytest.approx(expected_cml, abs=1e-12)

    # 3. Risk Premium
    rp = pa.capm_risk_premium(Ra, Rf=Rf)
    assert rp == pytest.approx((Ra - Rf).mean(), abs=1e-12)

    # 4. SML slope
    sml_slope = pa.capm_sml_slope(Rb, Rf=Rf)
    assert sml_slope == pytest.approx(1.0 / (Rb - Rf).mean(), abs=1e-12)


def test_capm_epsilon_ann():
    idx = pd.date_range("2020-01-01", periods=12, freq="ME")
    Ra = pd.Series([0.02, 0.03, -0.01, 0.04, 0.01, 0.02, 0.03, -0.02, 0.01, 0.04, -0.01, 0.03], index=idx)
    Rb = pd.Series([0.01, 0.02, 0.00, 0.03, 0.01, 0.01, 0.02, -0.01, 0.00, 0.03, -0.02, 0.02], index=idx)
    Rf = 0.001
    scale = 12

    # Expected values
    rp_ann = (1 + Ra).prod() ** (scale / len(Ra)) - 1
    rpb_ann = (1 + Rb).prod() ** (scale / len(Rb)) - 1
    rf_ann = (1 + Rf) ** scale - 1

    # periodic regression intercept & slope
    beta = pa.capm_beta(Ra, Rb, Rf=Rf)
    alpha = pa.capm_alpha(Ra, Rb, Rf=Rf)
    alpha_ann = (1 + alpha) ** scale - 1

    # Annualized Epsilon: Rp_ann - ( Rf_ann + alpha_ann + beta * (Rpb_ann - Rf_ann) )
    expected_epsilon = rp_ann - (rf_ann + alpha_ann + beta * (rpb_ann - rf_ann))

    epsilon = pa.capm_epsilon(Ra, Rb, Rf=Rf, scale=scale)
    assert epsilon == pytest.approx(expected_epsilon, abs=1e-12)


def test_period_contribution_wrappers():
    idx = pd.date_range("2020-01-01", periods=12, freq="ME")
    C = pd.DataFrame({
        "Asset1": [0.01, 0.02, -0.01, 0.01, 0.00, 0.02, 0.03, -0.01, 0.01, 0.02, -0.01, 0.02],
        "Asset2": [0.005, -0.01, 0.02, 0.01, -0.005, 0.01, 0.01, 0.00, -0.01, 0.01, 0.02, -0.01]
    }, index=idx)

    # test weekly, monthly, quarterly, yearly wrappers
    # monthly on monthly data should aggregate but check that it runs
    y_contrib = pa.to_yearly_contributions(C)
    expected_y = pa.to_period_contributions(C, period="years")
    pd.testing.assert_frame_equal(y_contrib, expected_y)

    q_contrib = pa.to_quarterly_contributions(C)
    expected_q = pa.to_period_contributions(C, period="quarters")
    pd.testing.assert_frame_equal(q_contrib, expected_q)


def test_table_aliases():
    assert pa.table_sfm == pa.table_capm
    assert pa.table_trailing_periods == pa.table_rolling_periods


def test_data_loaders():
    # Load managers
    managers = pa.load_managers()
    assert isinstance(managers, pd.DataFrame)
    assert isinstance(managers.index, pd.DatetimeIndex)
    assert managers.shape == (132, 10)  # R managers has 12 columns, wait, CSV has 132 rows & 10 columns

    # Load edhec
    edhec = pa.load_edhec()
    assert isinstance(edhec, pd.DataFrame)
    assert isinstance(edhec.index, pd.DatetimeIndex)
    assert edhec.shape == (152, 13)

    # Load portfolio_bacon
    portfolio_bacon = pa.load_portfolio_bacon()
    assert isinstance(portfolio_bacon, pd.DataFrame)
    assert isinstance(portfolio_bacon.index, pd.DatetimeIndex)
    assert portfolio_bacon.shape == (24, 2)
