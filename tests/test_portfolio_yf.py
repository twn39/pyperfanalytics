import numpy as np
import pandas as pd
import pytest

from pyperfanalytics.returns import return_portfolio


@pytest.fixture
def yf_data():
    return pd.read_csv("data/yfinance_etfs.csv", index_col=0, parse_dates=True)

@pytest.fixture
def weights():
    return [0.4, 0.4, 0.2]

def test_yf_portfolio_geom_none(yf_data, weights):
    bench = pd.read_csv("tests/benchmarks/yf_port_geom_none.csv", index_col=0)
    res = return_portfolio(yf_data, weights=weights, geometric=True, rebalance_on="none")
    np.testing.assert_allclose(res.values, bench.iloc[:, 0].values, atol=1e-4)

def test_yf_portfolio_arith_none(yf_data, weights):
    bench = pd.read_csv("tests/benchmarks/yf_port_arith_none.csv", index_col=0)
    res = return_portfolio(yf_data, weights=weights, geometric=False, rebalance_on="none")
    np.testing.assert_allclose(res.values, bench.iloc[:, 0].values, atol=1e-4)

def test_yf_portfolio_geom_years(yf_data, weights):
    bench = pd.read_csv("tests/benchmarks/yf_port_geom_years.csv", index_col=0)
    res = return_portfolio(yf_data, weights=weights, geometric=True, rebalance_on="years")
    np.testing.assert_allclose(res.values, bench.iloc[:, 0].values, atol=1e-4)
