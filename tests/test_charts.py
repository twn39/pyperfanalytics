import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from pyperfanalytics import charts as pa_charts


@pytest.fixture
def sample_data():
    """Create sample returns data for testing charts."""
    dates = pd.date_range(start="2020-01-01", periods=100, freq="B")
    np.random.seed(42)
    df = pd.DataFrame(
        {
            "Asset_A": np.random.normal(0.001, 0.02, 100),
            "Asset_B": np.random.normal(0.0005, 0.015, 100),
            "Benchmark": np.random.normal(0.0008, 0.018, 100),
        },
        index=dates,
    )
    return df


def test_chart_cum_returns(sample_data):
    fig = pa_charts.chart_cum_returns(sample_data[["Asset_A", "Asset_B"]])
    assert isinstance(fig, go.Figure)


def test_chart_bar_returns(sample_data):
    fig = pa_charts.chart_bar_returns(sample_data[["Asset_A"]])
    assert isinstance(fig, go.Figure)


def test_chart_drawdown(sample_data):
    fig = pa_charts.chart_drawdown(sample_data[["Asset_A", "Asset_B"]])
    assert isinstance(fig, go.Figure)


def test_chart_rolling_performance(sample_data):
    fig = pa_charts.chart_rolling_performance(sample_data[["Asset_A"]], width=20)
    assert isinstance(fig, go.Figure)


def test_chart_histogram(sample_data):
    fig = pa_charts.chart_histogram(sample_data["Asset_A"])
    assert isinstance(fig, go.Figure)


def test_chart_boxplot(sample_data):
    fig = pa_charts.chart_boxplot(sample_data[["Asset_A", "Asset_B"]])
    assert isinstance(fig, go.Figure)


def test_chart_correlation(sample_data):
    fig = pa_charts.chart_correlation(sample_data)
    assert isinstance(fig, go.Figure)


def test_chart_rolling_correlation(sample_data):
    fig = pa_charts.chart_rolling_correlation(sample_data[["Asset_A"]], sample_data[["Benchmark"]], width=20)
    assert isinstance(fig, go.Figure)


def test_chart_risk_return_scatter(sample_data):
    fig = pa_charts.chart_risk_return_scatter(sample_data)
    assert isinstance(fig, go.Figure)


def test_chart_relative_performance(sample_data):
    fig = pa_charts.chart_relative_performance(sample_data["Asset_A"], sample_data["Benchmark"])
    assert isinstance(fig, go.Figure)


def test_chart_capture_ratios(sample_data):
    fig = pa_charts.chart_capture_ratios(sample_data[["Asset_A", "Asset_B"]], sample_data[["Benchmark"]])
    assert isinstance(fig, go.Figure)


def test_chart_acf(sample_data):
    fig = pa_charts.chart_acf(sample_data["Asset_A"])
    assert isinstance(fig, go.Figure)


def test_chart_stacked_bar(sample_data):
    fig = pa_charts.chart_stacked_bar(sample_data)
    assert isinstance(fig, go.Figure)


def test_chart_ecdf(sample_data):
    fig = pa_charts.chart_ecdf(sample_data[["Asset_A", "Asset_B"]])
    assert isinstance(fig, go.Figure)
