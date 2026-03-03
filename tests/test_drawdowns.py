import pandas as pd
import pytest

from pyperfanalytics.tables import table_drawdowns


def test_table_drawdowns_ham1(managers_data):
    # Benchmark from R table.Drawdowns(managers[,1])
    r_results = [
        {"From": "2002-02-28", "Trough": "2003-02-28", "To": "2003-07-31", "Depth": -0.1518, "Length": 18},
        {"From": "1998-05-31", "Trough": "1998-08-31", "To": "1999-03-31", "Depth": -0.1239, "Length": 11},
        {"From": "2005-03-31", "Trough": "2005-04-30", "To": "2005-09-30", "Depth": -0.0412, "Length": 7},
    ]

    ham1 = managers_data['HAM1']
    py_table = table_drawdowns(ham1)

    for i, r_row in enumerate(r_results):
        py_row = py_table.iloc[i]
        assert str(pd.to_datetime(py_row["From"]).date()) == r_row["From"]
        assert str(pd.to_datetime(py_row["To"]).date()) == r_row["To"]
        assert py_row["Depth"] == pytest.approx(r_row["Depth"], abs=1e-4)
        assert py_row["Length"] == r_row["Length"]
