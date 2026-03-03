import pandas as pd
import pytest

from pyperfanalytics import (
    mean_absolute_deviation,
    sterling_ratio,
    table_autocorrelation,
    table_correlation,
    table_distributions,
    table_downside_risk_ratio,
    table_drawdowns_ratio,
    table_stats,
)


def load_bench(name):
    df = pd.read_csv(f"tests/benchmarks/{name}.csv", index_col=0)
    return df

def test_mean_absolute_deviation(managers_data):
    # R: MeanAbsoluteDeviation(managers[,1:8])
    ham1to8 = managers_data.iloc[:, 0:8]
    py_res = mean_absolute_deviation(ham1to8)
    r_res = pd.read_csv("tests/benchmarks/mean_absolute_deviation.csv", index_col=0)

    # Compare each asset
    for col in ham1to8.columns:
        assert py_res[col] == pytest.approx(r_res.loc[col, "MAD"], abs=1e-6)

def test_sterling_ratio(managers_data):
    # R: SterlingRatio(managers[,1:8])
    ham1to8 = managers_data.iloc[:, 0:8]
    py_res = sterling_ratio(ham1to8)
    r_res = pd.read_csv("tests/benchmarks/sterling_ratio.csv", index_col=0)

    for col in ham1to8.columns:
        assert py_res[col] == pytest.approx(r_res.loc[col, "SterlingRatio"], abs=1e-6)

def test_table_autocorrelation(managers_data):
    ham1to8 = managers_data.iloc[:, 0:8]
    py_tab = table_autocorrelation(ham1to8)
    r_tab = load_bench("table_autocorrelation")

    # R table.Autocorrelation returns rho1, rho2... as columns if t() is not used.
    # Actually my generate script did: tab_auto <- table.Autocorrelation(managers[,1:8])
    # Which returns assets as columns.

    for col in py_tab.columns:
        for idx in py_tab.index:
            py_val = py_tab.loc[idx, col]
            r_val = r_tab.loc[idx, col]
            assert py_val == pytest.approx(r_val, abs=1e-4)

def test_table_correlation(managers_data):
    ham1to6 = managers_data.iloc[:, 0:6]
    sp500 = managers_data["SP500 TR"]
    py_tab = table_correlation(ham1to6, sp500)
    r_tab = load_bench("table_correlation")

    # r_tab index: HAM1 to SP500 TR, etc.
    for col in py_tab.columns:
        for row in py_tab.index:
            assert py_tab.loc[row, col] == pytest.approx(r_tab.loc[row, col], abs=1e-4)

def test_table_distributions(managers_data):
    ham1to8 = managers_data.iloc[:, 0:8]
    py_tab = table_distributions(ham1to8)
    r_tab = load_bench("table_distributions")

    for col in py_tab.columns:
        for idx in py_tab.index:
            # Special case: label might different (e.g. "Monthly Std Dev" vs "Period Std Dev")
            # My Python code uses freq_map
            # R code uses periodicity(y)$scale
            pass # We'll just check values by positional match if labels differ slightly

    # Check values Positional check because index names might vary slightly
    for i in range(len(py_tab.columns)):
        for j in range(len(py_tab.index)):
            assert py_tab.iloc[j, i] == pytest.approx(r_tab.iloc[j, i], abs=1e-4)

def test_table_downside_risk_ratio(managers_data):
    ham1to8 = managers_data.iloc[:, 0:8]
    py_tab = table_downside_risk_ratio(ham1to8)
    r_tab = load_bench("table_downside_risk_ratio")

    for i in range(len(py_tab.columns)):
        for j in range(len(py_tab.index)):
            assert py_tab.iloc[j, i] == pytest.approx(r_tab.iloc[j, i], abs=1e-4)

def test_table_drawdowns_ratio(managers_data):
    ham1to8 = managers_data.iloc[:, 0:8]
    py_tab = table_drawdowns_ratio(ham1to8)
    r_tab = load_bench("table_drawdowns_ratio")

    for i in range(len(py_tab.columns)):
        for j in range(len(py_tab.index)):
            assert py_tab.iloc[j, i] == pytest.approx(r_tab.iloc[j, i], abs=1e-4)

def test_table_stats(managers_data):
    ham1to8 = managers_data.iloc[:, 0:8]
    py_tab = table_stats(ham1to8)
    r_tab = load_bench("table_stats")

    for i in range(len(py_tab.columns)):
        for j in range(len(py_tab.index)):
            # Geometric mean might be NA in R if there are negative returns <= -1,
            # but managers data is usually safe.
            if pd.isna(r_tab.iloc[j, i]):
                assert pd.isna(py_tab.iloc[j, i])
            else:
                assert py_tab.iloc[j, i] == pytest.approx(r_tab.iloc[j, i], abs=1e-4)
