import numpy as np
import pandas as pd
import pytest
import pyperfanalytics as pa


def test_upside_frequency():
    dates = pd.date_range("2026-01-01", periods=10, freq="D")
    R_data = [-0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04, -0.03, np.nan, np.nan]
    s = pd.Series(R_data, index=dates)

    # 1. Scalar MAR = 0.0
    # Positive: 0.01, 0.02, 0.03, 0.04 (4 items). Non-NaN count is 8.
    assert pa.upside_frequency(s, MAR=0.0) == pytest.approx(0.5)

    # 2. Scalar MAR = 0.01
    # Strictly above 0.01: 0.02, 0.03, 0.04 (3 items)
    assert pa.upside_frequency(s, MAR=0.01) == pytest.approx(0.375)

    # 3. Dynamic MAR (Series)
    mar_data = [0.0] * 10
    mar_s = pd.Series(mar_data, index=dates)
    assert pa.upside_frequency(s, MAR=mar_s) == pytest.approx(0.5)

    # 4. DataFrame
    df = pd.DataFrame({"A": s, "B": s})
    res_df = pa.upside_frequency(df, MAR=0.0)
    assert isinstance(res_df, pd.Series)
    assert res_df["A"] == pytest.approx(0.5)
    assert res_df["B"] == pytest.approx(0.5)

    # 5. All NaN
    all_nan = pd.Series([np.nan] * 5, index=dates[:5])
    assert pd.isna(pa.upside_frequency(all_nan))


def test_level_calculate_positive_and_reverse():
    dates = pd.date_range("2026-01-31", periods=6, freq="ME")
    ret = pd.Series([0.01, -0.02, 0.03, 0.0, 0.02, 0.01], index=dates)

    # ==================== 1. DISCRETE METHOD ====================
    # A. Positive (initial=True)
    res_discrete = pa.level_calculate(ret, seed_value=100.0, initial=True, method="discrete")
    assert res_discrete.index[0] == pd.Timestamp("2025-12-31")
    assert res_discrete.iloc[0] == pytest.approx(100.0)
    
    expected_lvl = 100.0 * (1.0 + 0.01)
    assert res_discrete.loc["2026-01-31"] == pytest.approx(expected_lvl)

    # B. Reverse (initial=False)
    last_val = res_discrete.iloc[-1]
    res_rev_discrete = pa.level_calculate(ret, seed_value=last_val, initial=False, method="discrete")
    pd.testing.assert_series_equal(res_discrete, res_rev_discrete, atol=1e-14)

    # ==================== 2. LOG METHOD ====================
    # A. Positive
    res_log = pa.level_calculate(ret, seed_value=100.0, initial=True, method="log")
    assert res_log.iloc[0] == pytest.approx(100.0)
    assert res_log.loc["2026-01-31"] == pytest.approx(100.0 * np.exp(0.01))

    # B. Reverse
    last_val_log = res_log.iloc[-1]
    res_rev_log = pa.level_calculate(ret, seed_value=last_val_log, initial=False, method="log")
    pd.testing.assert_series_equal(res_log, res_rev_log, atol=1e-14)

    # ==================== 3. DIFFERENCE METHOD ====================
    # A. Positive
    res_diff = pa.level_calculate(ret, seed_value=100.0, initial=True, method="difference")
    assert res_diff.iloc[0] == pytest.approx(100.0)
    assert res_diff.loc["2026-01-31"] == pytest.approx(100.0 + 0.01)

    # B. Reverse
    last_val_diff = res_diff.iloc[-1]
    res_rev_diff = pa.level_calculate(ret, seed_value=last_val_diff, initial=False, method="difference")
    pd.testing.assert_series_equal(res_diff, res_rev_diff, atol=1e-14)
    assert res_rev_diff.iloc[-1] == pytest.approx(last_val_diff)


def test_level_calculate_dataframe():
    dates = pd.date_range("2026-01-31", periods=3, freq="ME")
    df_ret = pd.DataFrame({
        "X": [0.01, -0.02, 0.03],
        "Y": [0.02, 0.01, -0.01]
    }, index=dates)

    res_df = pa.level_calculate(df_ret, seed_value=10.0, initial=True, method="discrete")
    assert isinstance(res_df, pd.DataFrame)
    assert res_df.index[0] == pd.Timestamp("2025-12-31")
    assert res_df.loc["2025-12-31", "X"] == pytest.approx(10.0)
    assert res_df.loc["2025-12-31", "Y"] == pytest.approx(10.0)

    res_rev_df = pa.level_calculate(df_ret, seed_value=10.0, initial=False, method="discrete")
    assert res_rev_df.iloc[-1, 0] == pytest.approx(10.0)


def test_return_relative():
    dates = pd.date_range("2026-01-01", periods=5, freq="D")
    ra = pd.Series([0.01, 0.02, -0.01, 0.03, 0.0], index=dates)
    rb = pd.Series([0.02, 0.01, 0.0, -0.02, 0.01], index=dates)

    # 1. Series + Series (Fast Path)
    res = pa.return_relative(ra, rb)
    assert isinstance(res, pd.Series)
    
    cum_a = (1.0 + ra).cumprod()
    cum_b = (1.0 + rb).cumprod()
    expected = cum_a / cum_b
    pd.testing.assert_series_equal(res, expected)

    # 2. DataFrame + Series
    df_a = pd.DataFrame({"A1": ra, "A2": ra * 2})
    res_mixed = pa.return_relative(df_a, rb)
    assert isinstance(res_mixed, pd.DataFrame)
    assert list(res_mixed.columns) == ["A1/Rb", "A2/Rb"]

    # 3. Slow Path (with NaNs)
    ra_nan = ra.copy()
    ra_nan.iloc[2] = np.nan
    res_nan = pa.return_relative(ra_nan, rb)
    assert len(res_nan) == 4
    assert pd.Timestamp("2026-01-03") not in res_nan.index


def test_frequency():
    # 1. Daily
    dates_d = pd.date_range("2026-01-01", periods=10, freq="D")
    s_d = pd.Series(range(10), index=dates_d)
    assert pa.frequency(s_d) == 252

    # 2. Weekly
    dates_w = pd.date_range("2026-01-01", periods=10, freq="W")
    s_w = pd.Series(range(10), index=dates_w)
    assert pa.frequency(s_w) == 52

    # 3. Monthly
    dates_m = pd.date_range("2026-01-01", periods=10, freq="ME")
    s_m = pd.Series(range(10), index=dates_m)
    assert pa.frequency(s_m) == 12

    # 4. DataFrame
    df = pd.DataFrame({"A": s_d, "B": s_d})
    res_df = pa.frequency(df)
    assert isinstance(res_df, pd.Series)
    assert res_df["A"] == 252
    assert res_df["B"] == 252


def test_real_data_cross_verification():
    # 1. Verify upside_frequency on real portfolio_bacon.csv (expected 0.542 in R)
    bacon_path = "data/portfolio_bacon.csv"
    bacon_df = pd.read_csv(bacon_path, index_col=0)
    bacon_df.index = pd.to_datetime(bacon_df.index)
    
    res_bacon = pa.upside_frequency(bacon_df.iloc[:, 0], MAR=0.005)
    assert res_bacon == pytest.approx(13 / 24)
    assert round(res_bacon, 3) == 0.542

    # 2. Verify level_calculate on real managers.csv data (multivariate, with NaNs)
    managers_path = "data/managers.csv"
    mgr_df = pd.read_csv(managers_path, index_col=0)
    mgr_df.index = pd.to_datetime(mgr_df.index)
    mgr_eq = mgr_df.iloc[:, :6]  # Select 6 Equities

    # Discrete positive & reverse calculations
    lvl_discrete = pa.level_calculate(mgr_eq, seed_value=100.0, initial=True, method="discrete")
    lvl_rev_discrete = pa.level_calculate(mgr_eq, seed_value=lvl_discrete.iloc[-1], initial=False, method="discrete")
    pd.testing.assert_frame_equal(lvl_discrete, lvl_rev_discrete, atol=1e-14)

    # Log positive & reverse calculations
    lvl_log = pa.level_calculate(mgr_eq, seed_value=1.0, initial=True, method="log")
    lvl_rev_log = pa.level_calculate(mgr_eq, seed_value=lvl_log.iloc[-1], initial=False, method="log")
    pd.testing.assert_frame_equal(lvl_log, lvl_rev_log, atol=1e-14)

    # 3. Verify frequency on real monthly managers.csv dataset (expected scale = 12)
    mgr_freq = pa.frequency(mgr_df)
    assert isinstance(mgr_freq, pd.Series)
    assert (mgr_freq == 12).all()
