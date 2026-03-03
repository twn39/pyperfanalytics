import pytest
from pyperfanalytics.tables import table_capture_ratios, table_up_down_ratios

def test_table_capture_ratios(managers_data):
    # Benchmark from R (HAM1-6 vs SP500):
    # HAM1: Up 0.6347, Down 0.2076
    # HAM2: Up 0.6616, Down 0.0439
    
    ra = managers_data[['HAM1', 'HAM2', 'HAM3', 'HAM4', 'HAM5', 'HAM6']]
    rb = managers_data[['SP500 TR']]
    
    table = table_capture_ratios(ra, rb)
    
    assert table.loc['HAM1', 'Up Capture'] == pytest.approx(0.6347, abs=1e-4)
    assert table.loc['HAM1', 'Down Capture'] == pytest.approx(0.2076, abs=1e-4)
    assert table.loc['HAM2', 'Up Capture'] == pytest.approx(0.6616, abs=1e-4)
    assert table.loc['HAM2', 'Down Capture'] == pytest.approx(0.0439, abs=1e-4)

def test_table_up_down_ratios(managers_data):
    # Benchmark from R (HAM1 vs SP500):
    # Up Capture: 0.6347, Down Capture: 0.2076
    # Up Number: 0.8941, Down Number: 0.5106
    # Up Percent: 0.2941, Down Percent: 0.8085
    
    ra = managers_data[['HAM1']]
    rb = managers_data[['SP500 TR']]
    
    table = table_up_down_ratios(ra, rb)
    
    idx = 'HAM1 to SP500 TR'
    assert table.loc[idx, 'Up Capture'] == pytest.approx(0.6347, abs=1e-4)
    assert table.loc[idx, 'Down Capture'] == pytest.approx(0.2076, abs=1e-4)
    assert table.loc[idx, 'Up Number'] == pytest.approx(0.8941, abs=1e-4)
    assert table.loc[idx, 'Down Number'] == pytest.approx(0.5106, abs=1e-4)
    assert table.loc[idx, 'Up Percent'] == pytest.approx(0.2941, abs=1e-4)
    assert table.loc[idx, 'Down Percent'] == pytest.approx(0.8085, abs=1e-4)
