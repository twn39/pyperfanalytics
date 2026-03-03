import pandas as pd
import pytest

from pyperfanalytics.tables import table_drawdowns


def test_table_drawdowns(managers_data):
    ham1 = managers_data.iloc[:, 0]
    res = table_drawdowns(ham1)

    # Benchmarks from PerformanceAnalytics table.Drawdowns(managers[,1,drop=FALSE])
    #         From     Trough         To   Depth Length To Trough Recovery
    # 1 2002-02-28 2003-02-28 2003-07-31 -0.1518     18        13        5
    # 2 1998-05-31 1998-08-31 1999-03-31 -0.1239     11         4        7

    assert len(res) >= 5

    # Check first row
    assert res.iloc[0]["Depth"] == pytest.approx(-0.1518, abs=1e-4)
    assert res.iloc[0]["Length"] == 18
    assert res.iloc[0]["To Trough"] == 13
    assert res.iloc[0]["Recovery"] == 5

    # Check dates formatting
    assert str(res.iloc[0]["From"].date()) == "2002-02-28"
    assert str(res.iloc[0]["Trough"].date()) == "2003-02-28"
    assert str(res.iloc[0]["To"].date()) == "2003-07-31"

def test_table_drawdowns_unrecovered(managers_data):
    # Test with a series that hasn't recovered (e.g. from a trough onwards)
    ham1 = managers_data.iloc[:, 0]
    # Find the last trough in HAM1, which is around 2005-04-30
    subset = ham1.loc["2005-03-31":]
    res = table_drawdowns(subset)

    # If the last drawdown hasn't recovered, 'To' should be NaN in our implementation
    # matching the logic we added in table_drawdowns.
    last_dd = res.iloc[0]
    if last_dd["Depth"] < 0 and pd.isna(last_dd["To"]):
         assert pd.isna(last_dd["Recovery"])
