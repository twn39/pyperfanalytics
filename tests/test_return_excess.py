"""
Tests for return_excess edge cases: multi-column Rf, Rf extending beyond R,
NaN gaps with ffill propagation, and Series vs DataFrame input consistency.

These tests cover the fix for Bug 4 identified in the 2026-04-28 audit:
  return_excess crashed with ValueError when Rf was a multi-column DataFrame
  whose date range extended beyond R's date range.
"""

import numpy as np
import pandas as pd
import pytest

import pyperfanalytics as pa


def _make_idx(n=5, freq="ME"):
    return pd.date_range("2020-01", periods=n, freq=freq)


# ---------------------------------------------------------------------------
# 1. Basic correctness (scalar Rf)
# ---------------------------------------------------------------------------


def test_return_excess_scalar_rf_series():
    idx = _make_idx(4)
    r = pd.Series([0.01, 0.02, 0.03, 0.04], index=idx)
    result = pa.return_excess(r, Rf=0.005)
    expected = r - 0.005
    pd.testing.assert_series_equal(result, expected)


def test_return_excess_scalar_rf_dataframe():
    idx = _make_idx(4)
    df = pd.DataFrame({"A": [0.01, 0.02], "B": [0.03, 0.04]},
                      index=idx[:2])
    result = pa.return_excess(df, Rf=0.001)
    pd.testing.assert_frame_equal(result, df - 0.001)


# ---------------------------------------------------------------------------
# 2. Single-column vector Rf
# ---------------------------------------------------------------------------


def test_return_excess_single_series_rf_same_index():
    idx = _make_idx(4)
    r = pd.Series([0.01, 0.02, 0.03, 0.04], index=idx, name="R")
    rf = pd.Series([0.001, 0.002, 0.003, 0.004], index=idx, name="RF")
    result = pa.return_excess(r, rf)
    expected = r - rf
    pd.testing.assert_series_equal(result, expected)


def test_return_excess_single_series_rf_longer_than_r():
    """Rf extends 2 periods beyond R — result must still have R's shape."""
    idx_r = _make_idx(3)
    idx_rf = _make_idx(5)
    r = pd.Series([0.01, 0.02, 0.03], index=idx_r, name="R")
    rf = pd.Series([0.001, 0.002, 0.003, 0.004, 0.005], index=idx_rf, name="RF")
    result = pa.return_excess(r, rf)
    assert result.shape == r.shape
    assert result.index.equals(r.index)
    np.testing.assert_allclose(result.values, [0.009, 0.018, 0.027], rtol=1e-10)


def test_return_excess_dataframe_rf_longer_than_r():
    """DataFrame R with single-col Rf that is longer — no crash, correct shape."""
    idx_r = _make_idx(3)
    idx_rf = _make_idx(5)
    r = pd.DataFrame({"A": [0.01, 0.02, 0.03]}, index=idx_r)
    rf = pd.Series([0.001, 0.002, 0.003, 0.004, 0.005], index=idx_rf, name="RF")
    result = pa.return_excess(r, rf)
    assert result.shape == r.shape
    assert result.index.equals(r.index)
    np.testing.assert_allclose(result["A"].values, [0.009, 0.018, 0.027], rtol=1e-10)


# ---------------------------------------------------------------------------
# 3. Multi-column Rf (previously crashing)
# ---------------------------------------------------------------------------


def test_return_excess_multi_col_rf_same_index():
    """Two-column Rf with same index as R: each R column minus matching Rf column."""
    idx = _make_idx(3)
    r = pd.DataFrame({"A": [0.01, 0.02, 0.03],
                      "B": [0.04, 0.05, 0.06]}, index=idx)
    rf = pd.DataFrame({"RF1": [0.001, 0.002, 0.003],
                       "RF2": [0.002, 0.003, 0.004]}, index=idx)
    # return_excess subtracts RF columns positionally (1-to-1) from R columns
    result = pa.return_excess(r, rf)
    assert result.shape == r.shape
    np.testing.assert_allclose(result["A"].values, [0.009, 0.018, 0.027], rtol=1e-10)
    np.testing.assert_allclose(result["B"].values, [0.038, 0.047, 0.056], rtol=1e-10)


def test_return_excess_multi_col_rf_longer_than_r():
    """Critical regression: multi-col Rf with matching R columns, Rf longer than R.

    When Rf is a multi-col DataFrame, each R column is subtracted by the
    corresponding Rf column (positional 1-to-1 mapping).  The fix ensures that
    having extra *rows* in Rf (date range beyond R) no longer raises ValueError.
    """
    idx_r = _make_idx(3)
    idx_rf = _make_idx(5)  # 2 extra rows
    # R has 2 columns, Rf has 2 columns — same column count, different row count
    r = pd.DataFrame({"A": [0.01, 0.02, 0.03],
                      "B": [0.04, 0.05, 0.06]}, index=idx_r)
    rf = pd.DataFrame({"RF1": [0.001, 0.002, 0.003, 0.004, 0.005],
                       "RF2": [0.002, 0.003, 0.004, 0.005, 0.006]}, index=idx_rf)
    # Should NOT raise ValueError (this was the crash before the fix)
    result = pa.return_excess(r, rf)
    assert result.shape == r.shape
    assert result.index.equals(r.index)
    # A - RF1 for first 3 rows
    np.testing.assert_allclose(result["A"].values, [0.009, 0.018, 0.027], rtol=1e-10)
    # B - RF2 for first 3 rows
    np.testing.assert_allclose(result["B"].values, [0.038, 0.047, 0.056], rtol=1e-10)


# ---------------------------------------------------------------------------
# 4. NaN propagation: ffill behaviour
# ---------------------------------------------------------------------------


def test_return_excess_rf_with_nan_gap_ffilled():
    """NaN in Rf should be forward-filled (matching R's na.locf behaviour)."""
    idx = _make_idx(5)
    r = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05], index=idx)
    rf = pd.Series([0.001, np.nan, np.nan, 0.004, 0.005], index=idx, name="RF")
    result = pa.return_excess(r, rf)
    # After ffill: rf = [0.001, 0.001, 0.001, 0.004, 0.005]
    expected = r - pd.Series([0.001, 0.001, 0.001, 0.004, 0.005], index=idx)
    pd.testing.assert_series_equal(result, expected)


def test_return_excess_r_with_nan_preserved():
    """NaN values in R must be preserved (not dropped, not filled)."""
    idx = _make_idx(4)
    r = pd.Series([0.01, np.nan, 0.03, 0.04], index=idx)
    rf = pd.Series([0.001, 0.002, 0.003, 0.004], index=idx, name="RF")
    result = pa.return_excess(r, rf)
    assert np.isnan(result.iloc[1])
    assert not np.isnan(result.iloc[0])
    assert not np.isnan(result.iloc[2])


# ---------------------------------------------------------------------------
# 5. Return type consistency
# ---------------------------------------------------------------------------


def test_return_excess_series_in_series_out():
    idx = _make_idx(3)
    r = pd.Series([0.01, 0.02, 0.03], index=idx, name="R")
    rf = pd.Series([0.001, 0.002, 0.003], index=idx, name="RF")
    result = pa.return_excess(r, rf)
    assert isinstance(result, pd.Series)


def test_return_excess_dataframe_in_dataframe_out():
    idx = _make_idx(3)
    r = pd.DataFrame({"A": [0.01, 0.02, 0.03]}, index=idx)
    rf = pd.Series([0.001, 0.002, 0.003], index=idx, name="RF")
    result = pa.return_excess(r, rf)
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["A"]
