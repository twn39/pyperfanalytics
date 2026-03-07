import pytest

from pyperfanalytics import burke_ratio, downside_deviation, sortino_ratio


def test_downside_metrics_edhec(edhec_data):
    # Benchmark results from R PerformanceAnalytics
    r_dd_0 = {
        "Convertible Arbitrage": 0.01470482,
        "CTA Global": 0.01371832,
        "Distressed Securities": 0.0118771,
        "Emerging Markets": 0.02693192,
        "Equity Market Neutral": 0.005745902,
        "Event Driven": 0.0121099,
        "Fixed Income Arbitrage": 0.01156365,
        "Global Macro": 0.0068386,
        "Long/Short Equity": 0.01278646,
        "Merger Arbitrage": 0.00667517,
        "Relative Value": 0.008723539,
        "Short Selling": 0.03421968,
        "Funds of Funds": 0.01088799,
    }

    r_sortino_0 = {
        "Convertible Arbitrage": 0.4358131,
        "CTA Global": 0.4730515,
        "Distressed Securities": 0.6696325,
        "Emerging Markets": 0.3061814,
        "Equity Market Neutral": 1.04468,
        "Event Driven": 0.6294327,
        "Fixed Income Arbitrage": 0.3658812,
        "Global Macro": 1.121921,
        "Long/Short Equity": 0.6068817,
        "Merger Arbitrage": 1.016434,
        "Relative Value": 0.7681878,
        "Short Selling": 0.1216021,
        "Funds of Funds": 0.5435736,
    }

    py_dd = downside_deviation(edhec_data, MAR=0)
    py_sortino = sortino_ratio(edhec_data, MAR=0)

    for asset in r_dd_0.keys():
        assert py_dd[asset] == pytest.approx(r_dd_0[asset], abs=1e-6)
        assert py_sortino[asset] == pytest.approx(r_sortino_0[asset], abs=1e-6)


def test_burke_ratio_corrected():
    import pandas as pd

    # portfolio_bacon column 1, d=7 drawdown events, n=24 observations
    # Fixed two deviations from paper/R:
    #   1. Paper fix: denominator is sqrt(sum(D²)/d) not sqrt(sum(D²))  → ratio * sqrt(d) larger
    #   2. R fix:     R uses *0.01 scaling (expects %-inputs), Python works with decimals
    # New paper-correct values (Burke 1994 definition with RMS denominator):
    #   Burke Ratio = 2.000773, Modified Burke Ratio = 9.801745
    # Supersedes old values: Burke=0.756221, ModBurke=3.704711 (were based on sqrt(sum) form)
    pb = pd.read_csv("data/portfolio_bacon.csv", index_col=0)
    col1 = pb.iloc[:, 0]

    b_ratio = burke_ratio(col1, scale=12)
    mod_b_ratio = burke_ratio(col1, modified=True, scale=12)

    assert b_ratio == pytest.approx(2.000773, abs=1e-5)
    assert mod_b_ratio == pytest.approx(9.801745, abs=1e-5)
