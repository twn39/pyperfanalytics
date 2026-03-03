import pytest
from pyperfanalytics.returns import up_capture, down_capture, up_down_ratios, gain_deviation, loss_deviation

def test_capture_ratios(managers_data):
    # Benchmark from R:
    # Up Capture: 0.6346612
    # Down Capture: 0.2076304
    ra = managers_data['HAM1']
    rb = managers_data['SP500 TR']
    
    uc = up_capture(ra, rb)
    dc = down_capture(ra, rb)
    
    assert uc == pytest.approx(0.6346612, abs=1e-6)
    assert dc == pytest.approx(0.2076304, abs=1e-6)

def test_up_down_ratios_methods(managers_data):
    ra = managers_data['HAM1']
    rb = managers_data['SP500 TR']
    
    # Capture matches
    res = up_down_ratios(ra, rb, method="Capture", side="Up")
    assert res == pytest.approx(0.6346612, abs=1e-6)

def test_deviation_metrics(managers_data):
    # Benchmark from R:
    # Gain Deviation: 0.01693097
    # Loss Deviation: 0.02113806
    ra = managers_data['HAM1']
    
    gd = gain_deviation(ra)
    ld = loss_deviation(ra)
    
    assert gd == pytest.approx(0.01693097, abs=1e-7)
    assert ld == pytest.approx(0.02113806, abs=1e-7)
