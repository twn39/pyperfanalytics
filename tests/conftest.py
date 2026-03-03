import pytest
import pandas as pd
import os

@pytest.fixture
def edhec_data():
    """Fixture to load EDHEC risk managers data."""
    path = "third_party/PerformanceAnalytics/data/edhec.csv"
    data = pd.read_csv(path, index_col=0)
    data.index = pd.to_datetime(data.index)
    return data

@pytest.fixture
def managers_data():
    """Fixture to load Portfolio Managers data."""
    path = "third_party/PerformanceAnalytics/data/managers.csv"
    data = pd.read_csv(path, index_col=0)
    data.index = pd.to_datetime(data.index)
    return data

@pytest.fixture
def test_data():
    """Fixture to load yfinance ETF dataset."""
    import os
    path = "data/test_data.csv"
    if not os.path.exists(path):
        pytest.skip("Test data not found. Run scripts/generate_test_data.py first.")
    data = pd.read_csv(path, index_col=0)
    data.index = pd.to_datetime(data.index)
    return data

@pytest.fixture
def r_benchmarks():
    """Fixture to load R benchmarks for the ETF dataset."""
    import json
    import os
    path = "data/r_benchmarks.json"
    if not os.path.exists(path):
        pytest.skip("R benchmarks not found. Run scripts/generate_r_benchmarks.R first.")
    with open(path, "r") as f:
        return json.load(f)

@pytest.fixture
def r_benchmarks_phase10():
    """Fixture to load R benchmarks for Phase 10 advanced metrics."""
    import json
    import os
    path = "data/r_benchmarks_phase10.json"
    if not os.path.exists(path):
        pytest.skip("R benchmarks phase10 not found. Run scripts/generate_r_benchmarks_phase10.R first.")
    with open(path, "r") as f:
        return json.load(f)
@pytest.fixture
def test_data_v2():
    """Fixture to load yfinance ETF dataset v2 (Growth/Small/Emerging)."""
    import os
    path = "data/test_data_v2.csv"
    if not os.path.exists(path):
        pytest.skip("Test data v2 not found. Run scripts/generate_test_data_v2.py first.")
    data = pd.read_csv(path, index_col=0)
    data.index = pd.to_datetime(data.index)
    return data

@pytest.fixture
def r_benchmarks_v2():
    """Fixture to load R benchmarks for the ETF dataset v2."""
    import json
    import os
    path = "data/r_benchmarks_v2.json"
    if not os.path.exists(path):
        pytest.skip("R benchmarks v2 not found. Run scripts/generate_r_benchmarks_v2.R first.")
    with open(path, "r") as f:
        return json.load(f)

@pytest.fixture
def test_data_v3():
    """Fixture to load yfinance ETF dataset v3 (Tech/High Vol)."""
    import os
    path = "data/test_data_v3.csv"
    if not os.path.exists(path):
        pytest.skip("Test data v3 not found.")
    data = pd.read_csv(path, index_col=0)
    data.index = pd.to_datetime(data.index)
    return data

@pytest.fixture
def r_benchmarks_v3():
    """Fixture to load R benchmarks for the ETF dataset v3."""
    import json
    import os
    path = "data/r_benchmarks_v3.json"
    if not os.path.exists(path):
        pytest.skip("R benchmarks v3 not found.")
    with open(path, "r") as f:
        return json.load(f)
