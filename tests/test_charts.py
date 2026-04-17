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


@pytest.mark.parametrize("geometric, wealth_index", [(True, False), (False, True), (False, False)])
def test_chart_cum_returns(sample_data, geometric, wealth_index):
    fig = pa_charts.chart_cum_returns(sample_data[["Asset_A", "Asset_B"]], geometric=geometric, wealth_index=wealth_index)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2
    assert len(fig.data[0].x) == 100
    if wealth_index:
        assert fig.layout.yaxis.title.text == "Value of $1"
    else:
        assert fig.layout.yaxis.title.text == "Cumulative Return"


def test_chart_bar_returns(sample_data):
    fig = pa_charts.chart_bar_returns(sample_data[["Asset_A"]], colorset=["blue"])
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert fig.data[0].type == "bar"


@pytest.mark.parametrize("geometric", [True, False])
def test_chart_drawdown(sample_data, geometric):
    fig = pa_charts.chart_drawdown(sample_data[["Asset_A", "Asset_B"]], geometric=geometric)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2
    assert all(val <= 0 for val in fig.data[0].y if pd.notna(val))


def test_charts_performance_summary(sample_data):
    fig = pa_charts.charts_performance_summary(sample_data[["Asset_A", "Asset_B"]], wealth_index=True)
    assert isinstance(fig, go.Figure)
    # 3 subplots * 2 assets
    assert len(fig.data) == 6


@pytest.mark.parametrize("methods", [
    "HistoricalVaR", 
    ["ModifiedVaR", "GaussianES"]
])
def test_chart_bar_var(sample_data, methods):
    fig = pa_charts.chart_bar_var(sample_data["Asset_A"], methods=methods, width=20)
    assert isinstance(fig, go.Figure)
    # Data traces: 1 bar + N line traces for VaR/ES + optional mean
    assert len(fig.data) > 1


def test_chart_var_sensitivity(sample_data):
    fig = pa_charts.chart_var_sensitivity(sample_data["Asset_A"])
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 6  # 6 default methods


def test_chart_rolling_performance(sample_data):
    fig = pa_charts.chart_rolling_performance(sample_data[["Asset_A"]], width=20, FUN="sharpe_ratio")
    assert isinstance(fig, go.Figure)


@pytest.mark.parametrize("methods", [
    ["add.normal"],
    ["add.density", "add.risk"]
])
def test_chart_histogram(sample_data, methods):
    fig = pa_charts.chart_histogram(sample_data["Asset_A"], methods=methods)
    assert isinstance(fig, go.Figure)


def test_chart_boxplot(sample_data):
    fig = pa_charts.chart_boxplot(sample_data[["Asset_A", "Asset_B"]])
    assert isinstance(fig, go.Figure)
    assert fig.data[0].type == "box"


def test_chart_qqplot(sample_data):
    fig = pa_charts.chart_qqplot(sample_data[["Asset_A"]])
    assert isinstance(fig, go.Figure)


def test_chart_correlation(sample_data):
    fig = pa_charts.chart_correlation(sample_data)
    assert isinstance(fig, go.Figure)
    # The correlation chart is a subplot matrix of scatter/histograms, not a single heatmap
    assert fig.data[0].type in ["histogram", "scatter"]


def test_chart_rolling_correlation(sample_data):
    fig = pa_charts.chart_rolling_correlation(sample_data[["Asset_A", "Asset_B"]], sample_data[["Benchmark"]], width=20)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2


def test_chart_risk_return_scatter(sample_data):
    fig = pa_charts.chart_risk_return_scatter(sample_data)
    assert isinstance(fig, go.Figure)
    assert fig.data[0].type == "scatter"


def test_chart_relative_performance(sample_data):
    fig = pa_charts.chart_relative_performance(sample_data["Asset_A"], sample_data["Benchmark"])
    assert isinstance(fig, go.Figure)


def test_chart_capture_ratios(sample_data):
    fig = pa_charts.chart_capture_ratios(sample_data[["Asset_A", "Asset_B"]], sample_data[["Benchmark"]])
    assert isinstance(fig, go.Figure)


def test_chart_rolling_regression(sample_data):
    fig = pa_charts.chart_rolling_regression(sample_data["Asset_A"], sample_data["Benchmark"], width=20)
    assert isinstance(fig, go.Figure)
    # Default is Beta. One asset, one benchmark -> 1 trace
    assert len(fig.data) == 1


def test_charts_rolling_regression(sample_data):
    fig = pa_charts.charts_rolling_regression(sample_data[["Asset_A", "Asset_B"]], sample_data[["Benchmark"]], width=20)
    assert isinstance(fig, go.Figure)


def test_chart_acf(sample_data):
    fig = pa_charts.chart_acf(sample_data["Asset_A"])
    assert isinstance(fig, go.Figure)
    assert fig.data[0].type == "bar"  # ACF bars


def test_chart_acf_plus(sample_data):
    fig = pa_charts.chart_acf_plus(sample_data["Asset_A"])
    assert isinstance(fig, go.Figure)
    # ACF and PACF subplots
    assert len(fig.data) >= 2


def test_chart_events(sample_data):
    # Create pseudo event dates
    events = [sample_data.index[20], sample_data.index[50]]
    fig = pa_charts.chart_events(sample_data["Asset_A"], dates=events, pre=10, post=10)
    assert isinstance(fig, go.Figure)


def test_chart_snail_trail(sample_data):
    fig = pa_charts.chart_snail_trail(sample_data["Asset_A"], sample_data["Benchmark"], width=20)
    assert isinstance(fig, go.Figure)


def test_chart_stacked_bar(sample_data):
    fig = pa_charts.chart_stacked_bar(sample_data)
    assert isinstance(fig, go.Figure)


def test_chart_component_returns(sample_data):
    weights = [0.4, 0.4, 0.2]
    fig = pa_charts.chart_component_returns(sample_data, weights=weights)
    assert isinstance(fig, go.Figure)


def test_chart_ecdf(sample_data):
    fig = pa_charts.chart_ecdf(sample_data[["Asset_A", "Asset_B"]])
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2


def test_chart_scatter(sample_data):
    fig = pa_charts.chart_scatter(sample_data["Asset_A"], sample_data["Benchmark"])
    assert isinstance(fig, go.Figure)
