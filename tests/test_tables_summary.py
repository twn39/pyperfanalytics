import pytest

from pyperfanalytics.tables import table_capm, table_downside_risk


def test_table_capm(managers_data):
    # Benchmark from R (HAM1 to SP500):
    # Alpha: 0.0058
    # Beta: 0.3901
    # Beta+: 0.3005
    # Beta-: 0.4264
    # R-squared: 0.4339
    # Annualized Alpha: 0.0715
    # Correlation: 0.6587
    # Tracking Error: 0.1132
    # Active Premium: 0.0408
    # Information Ratio: 0.3604
    # Treynor Ratio: 0.2428

    ra = managers_data["HAM1"]
    rb = managers_data["SP500 TR"]
    rf = managers_data["US 3m TR"]

    table = table_capm(ra, rb, Rf=rf)

    col = "HAM1 to SP500 TR"
    assert table.loc["Alpha", col] == pytest.approx(0.0058, abs=1e-4)
    assert table.loc["Beta", col] == pytest.approx(0.3901, abs=1e-4)
    assert table.loc["Beta+", col] == pytest.approx(0.3005, abs=1e-4)
    assert table.loc["Beta-", col] == pytest.approx(0.4264, abs=1e-4)
    assert table.loc["R-squared", col] == pytest.approx(0.4339, abs=1e-4)
    assert table.loc["Annualized Alpha", col] == pytest.approx(0.0715, abs=1e-4)
    assert table.loc["Correlation", col] == pytest.approx(0.6587, abs=1e-4)
    assert table.loc["Tracking Error", col] == pytest.approx(0.1132, abs=1e-4)
    assert table.loc["Active Premium", col] == pytest.approx(0.0408, abs=1e-4)
    assert table.loc["Information Ratio", col] == pytest.approx(0.3604, abs=1e-4)
    assert table.loc["Treynor Ratio", col] == pytest.approx(0.2428, abs=1e-4)


def test_table_downside_risk(managers_data):
    # Benchmark from R (HAM1):
    # Semi Deviation: 0.0191
    # Gain Deviation: 0.0169
    # Loss Deviation: 0.0211
    # Maximum Drawdown: 0.1518
    # Historical VaR (95%): -0.0258
    # Historical ES (95%): -0.0513
    # Modified VaR (95%): -0.0342
    # Modified ES (95%): -0.0610

    ra = managers_data["HAM1"]
    rf = managers_data["US 3m TR"]

    # Note: R benchmark uses MAR=0.1/12
    table = table_downside_risk(ra, Rf=rf, MAR=0.1 / 12)

    col = "HAM1"
    assert table.loc["Semi Deviation", col] == pytest.approx(0.0191, abs=1e-4)
    assert table.loc["Gain Deviation", col] == pytest.approx(0.0169, abs=1e-4)
    assert table.loc["Loss Deviation", col] == pytest.approx(0.0211, abs=1e-4)
    assert table.loc["Maximum Drawdown", col] == pytest.approx(0.1518, abs=1e-4)
    assert table.loc["Historical VaR (95%)", col] == pytest.approx(-0.0258, abs=1e-4)
    assert table.loc["Historical ES (95%)", col] == pytest.approx(-0.0513, abs=1e-4)
    assert table.loc["Modified VaR (95%)", col] == pytest.approx(-0.0342, abs=1e-4)
    assert table.loc["Modified ES (95%)", col] == pytest.approx(-0.0610, abs=1e-4)
