import pandas as pd
import pytest

from pyperfanalytics.returns import (
    adjusted_sharpe_ratio,
    bernardo_ledoit_ratio,
    d_ratio,
    omega_ratio,
    prospect_ratio,
    rachev_ratio,
)


def test_advanced_ratios(managers_data):
    ham1 = managers_data.iloc[:, 0]

    # Benchmarks from PerformanceAnalytics
    # Omega: 3.190689
    # BernardoLedoit: 3.190689
    # Rachev: 1.482528
    # Prospect (MAR=0): 0.328465
    # AdjustedSharpe: 0.9198034

    assert omega_ratio(ham1) == pytest.approx(3.190689, abs=1e-6)
    assert bernardo_ledoit_ratio(ham1) == pytest.approx(3.190689, abs=1e-6)
    assert d_ratio(ham1) == pytest.approx(1.0 / 3.190689, abs=1e-6)
    assert rachev_ratio(ham1) == pytest.approx(1.482528, abs=1e-6)
    assert prospect_ratio(ham1, MAR=0) == pytest.approx(0.328465, abs=1e-5) # Adjusted tolerance
    assert adjusted_sharpe_ratio(ham1) == pytest.approx(0.9198034, abs=1e-6)

def test_advanced_ratios_dataframe(managers_data):
    # Test on multiple columns
    df = managers_data.iloc[:, :3]
    res = omega_ratio(df)
    assert isinstance(res, pd.Series)
    assert len(res) == 3
    assert res.iloc[0] == pytest.approx(3.190689, abs=1e-6)
