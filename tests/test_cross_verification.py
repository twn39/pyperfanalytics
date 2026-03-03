import pytest
import pandas as pd
import numpy as np
import pyperfanalytics as pa

@pytest.mark.parametrize("dataset_f, bench_f, asset, rb_key", [
    ("test_data", "r_benchmarks", "AGG", "SPY"),
    ("test_data", "r_benchmarks", "GLD", "SPY"),
    ("test_data", "r_benchmarks", "SPY", "SPY"),
    ("test_data_v2", "r_benchmarks_v2", "QQQ", "QQQ"),
    ("test_data_v2", "r_benchmarks_v2", "IWM", "QQQ"),
    ("test_data_v2", "r_benchmarks_v2", "EEM", "QQQ"),
    ("test_data_v3", "r_benchmarks_v3", "TSLA", "QQQ"),
    ("test_data_v3", "r_benchmarks_v3", "NVDA", "QQQ"),
    ("test_data_v3", "r_benchmarks_v3", "AMD", "QQQ"),
])
def test_cross_verification(request, dataset_f, bench_f, asset, rb_key):
    data = request.getfixturevalue(dataset_f)
    benchmarks = request.getfixturevalue(bench_f)
    
    ra = data[asset]
    rb = data[rb_key]
    rf = data["BIL"]
    bench = benchmarks[asset]
    
    # Tolerances
    tol = 1e-4 # Standard for diverse data comparison
    
    # Returns & Standard Deviation
    assert pa.return_annualized(ra) == pytest.approx(bench["Return.annualized"], abs=tol)
    assert pa.std_dev_annualized(ra) == pytest.approx(bench["StdDev.annualized"], abs=tol)
    
    # Ratios
    assert pa.sharpe_ratio(ra, Rf=rf, annualize=True) == pytest.approx(bench["SharpeRatio"], abs=tol)
    assert pa.sortino_ratio(ra, MAR=rf.mean()) == pytest.approx(bench["SortinoRatio"], abs=tol)
    assert pa.omega_ratio(ra, L=rf) == pytest.approx(bench["OmegaRatio"], abs=tol)
    assert pa.rachev_ratio(ra) == pytest.approx(bench["RachevRatio"], abs=tol)
    assert pa.prospect_ratio(ra, MAR=rf.mean()) == pytest.approx(bench["ProspectRatio"], abs=tol)
    assert pa.adjusted_sharpe_ratio(ra) == pytest.approx(bench["AdjustedSharpeRatio"], abs=tol)
    
    # Risk (Note: signs aligned with R)
    assert -pa.var_historical(ra, p=0.95) == pytest.approx(bench["VaR.Hist"], abs=tol)
    assert -pa.var_gaussian(ra, p=0.95) == pytest.approx(bench["VaR.Gaus"], abs=tol)
    assert -pa.var_modified(ra, p=0.95) == pytest.approx(bench["VaR.Mod"], abs=tol)
    
    assert -pa.es_historical(ra, p=0.95) == pytest.approx(bench["ES.Hist"], abs=tol)
    assert -pa.es_gaussian(ra, p=0.95) == pytest.approx(bench["ES.Gaus"], abs=tol)
    assert -pa.es_modified(ra, p=0.95) == pytest.approx(bench["ES.Mod"], abs=tol)
    
    # Drawdowns
    assert pa.max_drawdown(ra) == pytest.approx(bench["MaxDrawdown"], abs=tol)
    assert pa.ulcer_index(ra) == pytest.approx(bench["UlcerIndex"], abs=tol)
    
    # Attribution
    assert pa.capm_beta(ra, rb, Rf=rf) == pytest.approx(bench["CAPM.beta"], abs=tol)
    assert pa.capm_alpha(ra, rb, Rf=rf) == pytest.approx(bench["CAPM.alpha"], abs=tol)
    assert pa.specific_risk(ra, rb, Rf=rf) == pytest.approx(bench["SpecificRisk"], abs=tol)
    assert pa.systematic_risk(ra, rb, Rf=rf) == pytest.approx(bench["SystematicRisk"], abs=tol)

    # New Metrics
    assert pa.burke_ratio(ra, Rf=rf) == pytest.approx(bench["BurkeRatio"], abs=tol)
    assert pa.burke_ratio(ra, Rf=rf, modified=True) == pytest.approx(bench["ModifiedBurkeRatio"], abs=tol)
    assert pa.modigliani(ra, rb, Rf=rf) == pytest.approx(bench["Modigliani"], abs=tol)
    assert pa.fama_beta(ra, rb) == pytest.approx(bench["FamaBeta"], abs=tol)

    assert pa.mean_absolute_deviation(ra) == pytest.approx(bench["MeanAbsoluteDeviation"], abs=tol)
    assert pa.downside_frequency(ra, MAR=rf.mean()) == pytest.approx(bench["DownsideFrequency"], abs=tol)
    assert pa.m2_sortino(ra, rb, MAR=rf.mean()) == pytest.approx(bench["M2Sortino"], abs=tol)
    assert pa.m_squared(ra, rb, Rf=rf) == pytest.approx(bench["MSquared"], abs=tol)
    assert pa.m_squared_excess(ra, rb, Rf=rf, method="geometric") == pytest.approx(bench["MSquaredExcess_geom"], abs=tol)
    assert pa.m_squared_excess(ra, rb, Rf=rf, method="arithmetic") == pytest.approx(bench["MSquaredExcess_arith"], abs=tol)
    assert pa.net_selectivity(ra, rb, Rf=rf) == pytest.approx(bench["NetSelectivity"], abs=tol)
    assert pa.omega_excess_return(ra, rb, MAR=rf.mean()) == pytest.approx(bench["OmegaExcessReturn"], abs=tol)
    assert pa.omega_sharpe_ratio(ra, MAR=rf.mean()) == pytest.approx(bench["OmegaSharpeRatio"], abs=tol)
    assert pa.downside_sharpe_ratio(ra, Rf=rf) == pytest.approx(bench["DownsideSharpeRatio"], abs=tol)
    assert pa.return_cumulative(ra) == pytest.approx(bench["Return.cumulative"], abs=tol)
    assert pa.kappa(ra, MAR=rf.mean(), l=2) == pytest.approx(bench["Kappa"], abs=tol)
    assert pa.annualized_excess_return(ra, rb) == pytest.approx(bench["AnnualizedExcessReturn"], abs=tol)

    assert pa.average_drawdown(ra) == pytest.approx(bench["AverageDrawdown"], abs=tol)
    assert pa.average_length(ra) == pytest.approx(bench["AverageLength"], abs=tol)
    assert pa.average_recovery(ra) == pytest.approx(bench["AverageRecovery"], abs=tol)
    assert pa.drawdown_deviation(ra) == pytest.approx(bench["DrawdownDeviation"], abs=tol)
    assert pa.cdd(ra, p=0.95) == pytest.approx(bench["CDD"], abs=tol)
    assert pa.cdar_beta(ra, rb, p=0.95) == pytest.approx(bench["CDaR.beta"], abs=tol)
    assert pa.cdar_alpha(ra, rb, p=0.95) == pytest.approx(bench["CDaR.alpha"], abs=tol)

    # Phase 10 Metrics
    refSR_val = 0.0 if not (asset == "AGG" and "test_data_v2" not in dataset_f) else -0.01

    assert pa.prob_sharpe_ratio(ra, Rf=rf, refSR=refSR_val) == pytest.approx(bench["ProbSharpeRatio"], abs=tol)

    mtr_res = pa.min_track_record(ra, refSR=refSR_val, Rf=rf.mean())
    assert mtr_res["min_TRL"] == pytest.approx(bench["MinTrackRecord"], abs=tol)
    # Looser tolerance for ACF differences across Python and R for Herfindahl Index
    assert pa.herfindahl_index(ra) == pytest.approx(bench["HerfindahlIndex"], abs=tol * 10)

    py_ir_table = pa.table_information_ratio(ra, rb, scale=252)
    bench_ir = bench["table.InformationRatio"]
    assert py_ir_table.iloc[0, 0] == pytest.approx(bench_ir["Tracking Error"], abs=tol)
    assert py_ir_table.iloc[1, 0] == pytest.approx(bench_ir["Annualised Tracking Error"], abs=tol)
    
    bench_val = bench_ir["Information Ratio"]
    if pd.isna(bench_val) or bench_val == "NaN" or (isinstance(bench_val, float) and np.isnan(bench_val)):
        assert pd.isna(py_ir_table.iloc[2, 0])
    else:
        assert py_ir_table.iloc[2, 0] == pytest.approx(bench_val, abs=tol)

    py_sr_table = pa.table_specific_risk(ra, rb, Rf=rf)
    bench_sr = bench["table.SpecificRisk"]
    assert py_sr_table.iloc[0, 0] == pytest.approx(bench_sr["Specific Risk"], abs=tol)
    assert py_sr_table.iloc[1, 0] == pytest.approx(bench_sr["Systematic Risk"], abs=tol)
    assert py_sr_table.iloc[2, 0] == pytest.approx(bench_sr["Total Risk"], abs=tol)
