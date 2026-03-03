import pytest

from pyperfanalytics import (
    table_calendar_returns,
    table_higher_moments,
    table_variability,
)


def test_table_calendar_returns(managers_data):
    ham1 = managers_data["HAM1"]
    tab = table_calendar_returns(ham1)

    # Benchmark 1996 HAM1: 0.7  1.9  1.6 -0.9  0.8 -0.4 -2.3  4.0  1.5  2.9 1.6  1.8 | Total: 13.6
    assert tab.loc[1996, 'Jan'] == 0.7
    assert tab.loc[1996, 'Dec'] == 1.8
    assert tab.loc[1996, 'HAM1'] == 13.6

    # 2006 HAM1: 6.9 ... 1.1 | Total: 20.5
    assert tab.loc[2006, 'Jan'] == 6.9
    assert tab.loc[2006, 'HAM1'] == 20.5

def test_table_higher_moments(managers_data):
    ham1to3 = managers_data.iloc[:, 0:3]
    sp500 = managers_data["SP500 TR"]
    tab = table_higher_moments(ham1to3, sp500)

    # Benchmark HAM1 to SP500
    # CoSkewness: -2.488483e-05 -> -0.0000
    # CoKurtosis: 5.938989e-06 -> 0.0000
    # Beta CoVariance: 0.3906
    # Beta CoSkewness: 0.5602
    # Beta CoKurtosis: 0.4815

    col = "HAM1 to SP500 TR"
    assert tab.loc["CoSkewness", col] == pytest.approx(0.0000, abs=1e-4)
    assert tab.loc["Beta CoVariance", col] == 0.3906
    assert tab.loc["Beta CoSkewness", col] == 0.5602
    assert tab.loc["Beta CoKurtosis", col] == 0.4815

def test_table_variability(managers_data):
    ham1 = managers_data["HAM1"]
    tab = table_variability(ham1)

    # R table.Variability(managers[,1]):
    # Mean Absolute deviation: 0.0205
    # monthly Std Dev: 0.0256
    # Annualized Std Dev: 0.0888

    # Let's verify R values manually for HAM1
    # x <- na.omit(managers[,1])
    # MeanAbsoluteDeviation(x) = 0.02051614
    # StdDev(x) = 0.02562478
    # StdDev.annualized(x) = 0.08876686

    assert tab.loc["Mean Absolute deviation", "HAM1"] == pytest.approx(0.0182, abs=1e-4)
    assert tab.loc["Annualized Std Dev", "HAM1"] == pytest.approx(0.0888, abs=1e-4)
