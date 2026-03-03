import pytest
import pandas as pd
import numpy as np
from pyperfanalytics import (
    m_squared, m_squared_excess, appraisal_ratio, net_selectivity, 
    kappa, prospect_ratio, table_capture_ratios, table_up_down_ratios,
    table_annualized_returns
)
from pandas.testing import assert_frame_equal

@pytest.fixture
def managers_data():
    df = pd.read_csv("data/managers.csv", index_col=0, parse_dates=True)
    return df["1996":"2006"]

def test_msquared(managers_data):
    m = managers_data
    bench = pd.read_csv("tests/benchmarks/msquared.csv", index_col=0)
    
    # Calculate Python results
    # Ra: columns 1-5, Rb: column 8 (SP500 TR)
    res = m_squared(m.iloc[:, 0:5], m.iloc[:, 7], Rf=0)
    
    # Compare (bench has SP500 as index, asset names as columns)
    # Our m_squared returns a series/df with Rb as index and Ra as columns if multi
    if isinstance(res, pd.Series):
        res_df = res.to_frame().T
    else:
        res_df = res
        
    np.testing.assert_allclose(res_df.values, bench.values, atol=1e-4)

def test_msquared_excess(managers_data):
    m = managers_data
    bench_geo = pd.read_csv("tests/benchmarks/msquared_excess_geo.csv", index_col=0)
    bench_ari = pd.read_csv("tests/benchmarks/msquared_excess_ari.csv", index_col=0)
    
    res_geo = m_squared_excess(m.iloc[:, 0:5], m.iloc[:, 7], Rf=0, method="geometric")
    res_ari = m_squared_excess(m.iloc[:, 0:5], m.iloc[:, 7], Rf=0, method="arithmetic")
    
    if isinstance(res_geo, pd.Series): res_geo = res_geo.to_frame().T
    if isinstance(res_ari, pd.Series): res_ari = res_ari.to_frame().T
    
    np.testing.assert_allclose(res_geo.values, bench_geo.values, atol=1e-4)
    np.testing.assert_allclose(res_ari.values, bench_ari.values, atol=1e-4)

def test_appraisal_ratio(managers_data):
    m = managers_data
    # We'll test appraisal method
    bench = pd.read_csv("tests/benchmarks/appraisal_ratio_app.csv", index_col=0)
    res = appraisal_ratio(m.iloc[:, 0:5], m.iloc[:, 7], Rf=0)
    
    if isinstance(res, pd.Series): res = res.to_frame().T
    np.testing.assert_allclose(res.values, bench.values, atol=1e-4)

def test_net_selectivity(managers_data):
    m = managers_data
    bench = pd.read_csv("tests/benchmarks/net_selectivity.csv", index_col=0)
    res = net_selectivity(m.iloc[:, 0:5], m.iloc[:, 7], Rf=0)
    
    if isinstance(res, pd.Series): res = res.to_frame().T
    np.testing.assert_allclose(res.values, bench.values, atol=1e-4)

def test_kappa(managers_data):
    m = managers_data
    bench_l1 = pd.read_csv("tests/benchmarks/kappa_l1.csv", index_col=0)
    bench_l2 = pd.read_csv("tests/benchmarks/kappa_l2.csv", index_col=0)
    
    res_l1 = kappa(m.iloc[:, 0:5], MAR=0, l=1)
    res_l2 = kappa(m.iloc[:, 0:5], MAR=0, l=2)
    
    # Python returns a Series, Bench is typically 1xN or Nx1 in CSV
    # np.testing.assert_allclose handles shape mismatch if broadcasting is possible
    # but (5,) and (1, 5) need explicit reshapping or .values comparison
    np.testing.assert_allclose(res_l1.values.reshape(1, -1), bench_l1.values, atol=1e-4)
    np.testing.assert_allclose(res_l2.values.reshape(1, -1), bench_l2.values, atol=1e-4)

def test_prospect_ratio(managers_data):
    m = managers_data
    bench = pd.read_csv("tests/benchmarks/prospect_ratio.csv", index_col=0)
    res = prospect_ratio(m.iloc[:, 0:5], MAR=0)
    
    np.testing.assert_allclose(res.values.reshape(1, -1), bench.values, atol=1e-4)

def test_table_capture_ratios(managers_data):
    m = managers_data
    bench = pd.read_csv("tests/benchmarks/table_capture_ratios.csv", index_col=0)
    res = table_capture_ratios(m.iloc[:, 0:6], m.iloc[:, [7]])
    
    # R column names: "Up Capture", "Down Capture"
    # Python might have different column names, let's align
    res.columns = bench.columns
    assert_frame_equal(res, bench, atol=1e-4)

def test_table_up_down_ratios(managers_data):
    m = managers_data
    bench = pd.read_csv("tests/benchmarks/table_up_down_ratios.csv", index_col=0)
    res = table_up_down_ratios(m.iloc[:, 0:6], m.iloc[:, [7]])
    
    res.columns = bench.columns
    assert_frame_equal(res, bench, atol=1e-4)

def test_table_annualized_returns(managers_data):
    m = managers_data
    bench = pd.read_csv("tests/benchmarks/table_annualized_returns.csv", index_col=0)
    # Rf is column 10 (EDHEC LS EQ) - wait, column 10 is Rf in managers?
    # R script used Rf=m[,10,drop=FALSE]
    res = table_annualized_returns(m.iloc[:, 0:6], Rf=m.iloc[:, 9])
    
    res.columns = bench.columns
    # Sortino ratio might have slightly different names/rounding
    assert_frame_equal(res, bench, atol=1e-4)
