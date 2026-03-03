import pytest
import numpy as np
import pandas as pd
from pyperfanalytics.returns import return_portfolio, return_clean, return_geltner

@pytest.fixture
def test_data():
    df = pd.read_csv("data/managers.csv", index_col=0, parse_dates=True)
    m = df["1996":"2006"].iloc[:, 0:6]
    return m

def test_return_geltner(test_data):
    bench = pd.read_csv("tests/benchmarks/return_geltner.csv", index_col=0)
    res = return_geltner(test_data)
    np.testing.assert_allclose(res.values, bench.values, atol=1e-4)

def test_return_clean_boudt(test_data):
    bench = pd.read_csv("tests/benchmarks/return_clean_boudt.csv", index_col=0)
    m_sub = test_data.iloc[:, 0:4].dropna()
    res = return_clean(m_sub, method="boudt")
    np.testing.assert_allclose(res.values, bench.values, atol=0.05)

def test_return_portfolio_geom_none(test_data):
    bench = pd.read_csv("tests/benchmarks/return_portfolio_geom_none.csv", index_col=0)
    m_sub = test_data.iloc[:, 0:4].dropna()
    res = return_portfolio(m_sub, geometric=True, rebalance_on="none")
    np.testing.assert_allclose(res.values, bench.iloc[:, 0].values, atol=1e-4)

def test_return_portfolio_arith_none(test_data):
    bench = pd.read_csv("tests/benchmarks/return_portfolio_arith_none.csv", index_col=0)
    m_sub = test_data.iloc[:, 0:4].dropna()
    res = return_portfolio(m_sub, geometric=False, rebalance_on="none")
    np.testing.assert_allclose(res.values, bench.iloc[:, 0].values, atol=1e-4)

def test_return_portfolio_geom_months(test_data):
    bench = pd.read_csv("tests/benchmarks/return_portfolio_geom_months.csv", index_col=0)
    m_sub = test_data.iloc[:, 0:4].dropna()
    res = return_portfolio(m_sub, geometric=True, rebalance_on="months")
    np.testing.assert_allclose(res.values, bench.iloc[:, 0].values, atol=1e-4)

def test_return_portfolio_geom_years(test_data):
    bench = pd.read_csv("tests/benchmarks/return_portfolio_geom_years.csv", index_col=0)
    m_sub = test_data.iloc[:, 0:4].dropna()
    res = return_portfolio(m_sub, geometric=True, rebalance_on="years")
    np.testing.assert_allclose(res.values, bench.iloc[:, 0].values, atol=1e-4)
