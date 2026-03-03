import pytest

from pyperfanalytics.tables import table_annualized_returns


def test_table_annualized_returns(managers_data):
    # Verified with direct R script: Return.annualized(managers$HAM1) -> 0.137532
    # Verified with direct R script: StdDev.annualized(managers$HAM1) -> 0.0887808
    ra = managers_data.iloc[:, 0:6]
    py_table = table_annualized_returns(ra)

    # Check HAM1
    assert py_table.loc["Annualized Return", "HAM1"] == pytest.approx(0.137532, abs=1e-6)
    assert py_table.loc["Annualized Std Dev", "HAM1"] == pytest.approx(0.0887808, abs=1e-6)
