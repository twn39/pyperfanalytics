import pandas as pd

import pyperfanalytics as pa


def test_hurst_index():
    # Load managers data
    m = pd.read_csv("data/managers.csv", index_col=0)
    m.index = pd.to_datetime(m.index)
    selected = ["HAM1", "HAM2", "HAM3", "HAM4", "HAM5", "HAM6"]

    actual = pa.hurst_index(m[selected])
    expected = pd.read_csv("tests/benchmarks/hurst_index.csv", index_col=0).iloc[0]
    pd.testing.assert_series_equal(actual, expected, check_names=False, atol=1e-4)


def test_smoothing_index():
    m = pd.read_csv("data/managers.csv", index_col=0)
    m.index = pd.to_datetime(m.index)
    selected = ["HAM1", "HAM2", "HAM3", "HAM4", "HAM5", "HAM6"]

    actual = pa.smoothing_index(m[selected])
    expected = pd.read_csv("tests/benchmarks/smoothing_index.csv", index_col=0).iloc[0]
    pd.testing.assert_series_equal(actual, expected, check_names=False, atol=1e-2)


def test_table_prob_outperformance():
    e = pd.read_csv("data/edhec.csv", index_col=0)
    e.index = pd.to_datetime(e.index)

    actual = pa.table_prob_outperformance(e.iloc[:, 0], e.iloc[:, 1])
    expected = pd.read_csv("tests/benchmarks/table_prob_outperformance.csv")

    expected["period_lengths"] = expected["period_lengths"].astype(int)
    actual.columns = expected.columns

    pd.testing.assert_frame_equal(actual, expected, atol=1e-4)


def test_table_rolling_periods():
    e = pd.read_csv("data/edhec.csv", index_col=0)
    e.index = pd.to_datetime(e.index)
    selected = e.columns[:6]

    actual = pa.table_rolling_periods(e[selected], periods=[12, 24, 36])
    expected = pd.read_csv("tests/benchmarks/table_rolling_periods.csv", index_col=0)

    pd.testing.assert_frame_equal(actual, expected, atol=1e-3)


def test_to_period_contributions():
    contrib = pd.read_csv("tests/benchmarks/input_contribution.csv", index_col=0)
    contrib.index = pd.to_datetime(contrib.index)

    actual = pa.to_period_contributions(contrib, period="years")
    expected = pd.read_csv("tests/benchmarks/to_period_contributions.csv", index_col=0)

    # R CSV index might be strings of years or similar
    actual_reset = actual.reset_index(drop=True)
    expected_reset = expected.reset_index(drop=True)

    pd.testing.assert_frame_equal(actual_reset, expected_reset, atol=1e-4)
