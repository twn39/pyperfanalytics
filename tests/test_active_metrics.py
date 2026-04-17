import pandas as pd
import pytest

from pyperfanalytics import active_premium, information_ratio, tracking_error


def test_active_metrics_managers(managers_data):
    # Benchmark from R PerformanceAnalytics with managers data
    # Ra = managers[,1:2], Rb = managers[,8]
    r_results = {
        "TE": {"HAM1": 0.1131667, "HAM2": 0.1533647},
        "AP": {"HAM1": 0.04078668, "HAM2": 0.07759873},
        "IR": {"HAM1": 0.3604125, "HAM2": 0.5059751},
    }

    ra = managers_data[["HAM1", "HAM2"]]
    rb = managers_data["SP500 TR"]

    te_py = tracking_error(ra, rb)
    ap_py = active_premium(ra, rb)
    ir_py = information_ratio(ra, rb)

    assert isinstance(te_py, pd.Series)
    assert isinstance(ap_py, pd.Series)
    assert isinstance(ir_py, pd.Series)

    for asset in ["HAM1", "HAM2"]:
        assert te_py[asset] == pytest.approx(r_results["TE"][asset], abs=1e-6)
        assert ap_py[asset] == pytest.approx(r_results["AP"][asset], abs=1e-6)
        assert ir_py[asset] == pytest.approx(r_results["IR"][asset], abs=1e-6)
