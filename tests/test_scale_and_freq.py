import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

import pyperfanalytics as pa
from pyperfanalytics.utils import _get_scale


def test_scale_inferred_freq():
    """Test standard pandas frequency strings."""
    # Daily (Business days)
    idx_b = pd.date_range("2020-01-01", periods=10, freq="B")
    s_b = pd.Series(np.random.randn(10), index=idx_b)
    assert _get_scale(s_b) == 252

    # Daily (All days)
    idx_d = pd.date_range("2020-01-01", periods=10, freq="D")
    s_d = pd.Series(np.random.randn(10), index=idx_d)
    assert _get_scale(s_d) == 252

    # Weekly
    idx_w = pd.date_range("2020-01-01", periods=10, freq="W")
    s_w = pd.Series(np.random.randn(10), index=idx_w)
    assert _get_scale(s_w) == 52

    # Monthly (ME or MS depending on pandas version)
    idx_m = pd.date_range("2020-01-01", periods=10, freq="ME")
    s_m = pd.Series(np.random.randn(10), index=idx_m)
    assert _get_scale(s_m) == 12

    # Quarterly
    idx_q = pd.date_range("2020-01-01", periods=10, freq="QE")
    s_q = pd.Series(np.random.randn(10), index=idx_q)
    assert _get_scale(s_q) == 4

    # Yearly
    idx_y = pd.date_range("2020-01-01", periods=10, freq="YE")
    s_y = pd.Series(np.random.randn(10), index=idx_y)
    assert _get_scale(s_y) == 1


def test_scale_empirical_spacing():
    """Test fallback heuristic when index has no formal frequency."""
    # Irregular daily (e.g., missing some days but mostly 1 day apart)
    idx = pd.DatetimeIndex(["2020-01-01", "2020-01-02", "2020-01-04", "2020-01-05", "2020-01-06"])
    s = pd.Series(np.random.randn(5), index=idx)
    assert _get_scale(s) == 252

    # Irregular weekly
    idx = pd.DatetimeIndex(["2020-01-01", "2020-01-08", "2020-01-16", "2020-01-22", "2020-01-29"])
    s = pd.Series(np.random.randn(5), index=idx)
    assert _get_scale(s) == 52

    # Irregular monthly
    idx = pd.DatetimeIndex(["2020-01-31", "2020-02-28", "2020-03-25", "2020-04-30", "2020-05-31"])
    s = pd.Series(np.random.randn(5), index=idx)
    assert _get_scale(s) == 12


def test_scale_errors():
    """Test exceptions for invalid indexes without scale parameter."""
    s = pd.Series([0.01, 0.02, 0.03])
    with pytest.raises(ValueError, match="Data index must be a DatetimeIndex"):
        _get_scale(s)

    with pytest.raises(ValueError, match="Data index must be a DatetimeIndex"):
        pa.return_annualized(s)

    # Providing scale manually bypasses the error
    expected = ( (1+0.01) * (1+0.02) * (1+0.03) ) ** (12 / 3) - 1
    assert pa.return_annualized(s, scale=12) == pytest.approx(expected, abs=1e-6)

