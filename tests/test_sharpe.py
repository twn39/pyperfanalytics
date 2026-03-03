import pytest
import pandas as pd
import numpy as np
from pyperfanalytics import sharpe_ratio

def test_sharpe_ratio_stddev(edhec_data):
    # Benchmark results from R PerformanceAnalytics::SharpeRatio(edhec, Rf=0, FUN="StdDev")
    # Generated using read.csv("edhec.csv") in R to match exactly.
    r_results = {
        "Convertible Arbitrage": 0.3196702,
        "CTA Global": 0.2582269,
        "Distressed Securities": 0.4334711,
        "Emerging Markets": 0.2137865,
        "Equity Market Neutral": 0.6665282,
        "Event Driven": 0.4153772,
        "Fixed Income Arbitrage": 0.2985557,
        "Global Macro": 0.4507954,
        "Long/Short Equity": 0.3499564,
        "Merger Arbitrage": 0.6075128,
        "Relative Value": 0.5078801,
        "Short Selling": 0.07552172,
        "Funds of Funds": 0.3249744
    }
    
    py_results = sharpe_ratio(edhec_data, Rf=0, FUN="StdDev")
    
    for asset, r_val in r_results.items():
        assert py_results[asset] == pytest.approx(r_val, abs=1e-6)

def test_sharpe_ratio_semisd(edhec_data):
    # Verified with direct R script on repo CSV: SharpeRatio(edhec, Rf=0, FUN="SemiSD")
    r_results = {
        "Convertible Arbitrage": 0.2712727,
        "CTA Global": 0.2657195,
        "Distressed Securities": 0.3807032,
        "Emerging Markets": 0.1915231,
        "Equity Market Neutral": 0.5734602,
        "Event Driven": 0.3610422,
        "Fixed Income Arbitrage": 0.2363237,
        "Global Macro": 0.5024986,
        "Long/Short Equity": 0.3336567,
        "Merger Arbitrage": 0.5299353,
        "Relative Value": 0.4325989,
        "Short Selling": 0.0806424,
        "Funds of Funds": 0.3125013
    }
    
    py_results = sharpe_ratio(edhec_data, Rf=0, FUN="SemiSD")
    
    for asset, r_val in r_results.items():
        assert py_results[asset] == pytest.approx(r_val, abs=1e-6)

@pytest.mark.parametrize("scenario, rf_val, annualize, expected", [
    ("SR1", 0, False, {"HAM1": 0.4339932, "HAM3": 0.3408953, "HAM4": 0.2070881, "HAM6": 0.4642393}),
    ("SR2", 0.035/12, False, {"HAM1": 0.3201889, "HAM3": 0.2610141, "HAM4": 0.1522615, "HAM6": 0.3417545}),
    ("SR4", 0, True, {"HAM1": 1.5491189, "HAM3": 1.1955305, "HAM4": 0.6592017, "HAM6": 1.664170}),
])
def test_sharpe_ratio_managers_scenarios(managers_data, scenario, rf_val, annualize, expected):
    R = managers_data[['HAM1', 'HAM2', 'HAM3', 'HAM4', 'HAM5', 'HAM6']]
    py_results = sharpe_ratio(R, Rf=rf_val, annualize=annualize)
    for asset, r_val in expected.items():
        assert py_results[asset] == pytest.approx(r_val, abs=1e-6)

def test_sharpe_ratio_managers_rf_seq(managers_data):
    # SR3: Rf as sequence
    R = managers_data[['HAM1', 'HAM2', 'HAM3', 'HAM4', 'HAM5', 'HAM6']]
    Rf_seq = managers_data['US 3m TR']
    expected = {"HAM1": 0.3081020, "HAM3": 0.2525301, "HAM4": 0.1464385, "HAM6": 0.3785371}
    py_results = sharpe_ratio(R, Rf=Rf_seq)
    for asset, r_val in expected.items():
        assert py_results[asset] == pytest.approx(r_val, abs=1e-6)

def test_sharpe_ratio_semisd_managers(managers_data):
    # From test_sharpe_semisd.py
    R = managers_data[['HAM1', 'HAM2', 'HAM3', 'HAM4', 'HAM5', 'HAM6']]
    expected = {"HAM1": 0.4122201, "HAM3": 0.3714733, "HAM4": 0.1972034}
    py_results = sharpe_ratio(R, Rf=0, FUN="SemiSD")
    for asset, r_val in expected.items():
        assert py_results[asset] == pytest.approx(r_val, abs=1e-6)
