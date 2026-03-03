import pytest
import pandas as pd
import numpy as np
from pyperfanalytics.risk import systematic_risk, specific_risk, total_risk
from pyperfanalytics.returns import jensen_alpha, appraisal_ratio, market_timing

def test_risk_decomposition(managers_data):
    ham1 = managers_data['HAM1']
    sp500 = managers_data['SP500 TR']
    
    # R Benchmarks
    # SpecificRisk: 0.06643962 
    # SystematicRisk: 0.05860128 
    
    spec = specific_risk(ham1, sp500)
    sys = systematic_risk(ham1, sp500)
    tot = total_risk(ham1, sp500)
    
    assert spec == pytest.approx(0.06643962, abs=1e-6)
    assert sys == pytest.approx(0.05860128, abs=1e-6)
    # total_risk should be sqrt(spec^2 + sys^2)
    assert tot == pytest.approx(np.sqrt(spec**2 + sys**2), abs=1e-6)

def test_attribution_ratios(managers_data):
    ham1 = managers_data['HAM1']
    sp500 = managers_data['SP500 TR']
    
    # R Benchmarks
    # CAPM.jensenAlpha: 0.09974296
    # AppraisalRatio: 1.501257 
    
    ja = jensen_alpha(ham1, sp500)
    ar = appraisal_ratio(ham1, sp500)
    
    assert ja == pytest.approx(0.09974296, abs=1e-6)
    assert ar == pytest.approx(1.501257, abs=1e-5)

def test_market_timing(managers_data):
    ham1 = managers_data['HAM1']
    sp500 = managers_data['SP500 TR']
    
    # R Benchmarks TM
    #                      Alpha      Beta      Gamma
    # HAM1 to SP500 TR 0.00966066 0.3843016 -0.9646121
    
    res_tm = market_timing(ham1, sp500, method="TM")
    assert res_tm.loc["HAM1 to SP500 TR", "Alpha"] == pytest.approx(0.00966066, abs=1e-6)
    assert res_tm.loc["HAM1 to SP500 TR", "Beta"] == pytest.approx(0.3843016, abs=1e-6)
    assert res_tm.loc["HAM1 to SP500 TR", "Gamma"] == pytest.approx(-0.9646121, abs=1e-6)
    
    # R Benchmarks HM
    #                      Alpha     Beta    Gamma
    # HAM1 to SP500 TR 0.01006957 0.324894 0.133821
    
    res_hm = market_timing(ham1, sp500, method="HM")
    assert res_hm.loc["HAM1 to SP500 TR", "Alpha"] == pytest.approx(0.01006957, abs=1e-6)
    assert res_hm.loc["HAM1 to SP500 TR", "Beta"] == pytest.approx(0.324894, abs=1e-6)
    assert res_hm.loc["HAM1 to SP500 TR", "Gamma"] == pytest.approx(0.133821, abs=1e-6)
