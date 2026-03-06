import pytest

from pyperfanalytics.returns import calmar_ratio, omega_ratio, treynor_ratio


def test_treynor_ratio(managers_data):
    # Benchmark from R:
    # TreynorRatio(ra, rb, Rf=rf) -> 0.2428042
    ra = managers_data["HAM1"]
    rb = managers_data["SP500 TR"]
    rf = managers_data["US 3m TR"]

    tr = treynor_ratio(ra, rb, Rf=rf)
    assert tr == pytest.approx(0.2428042, abs=1e-6)


def test_calmar_ratio(managers_data):
    # Benchmark from R:
    # CalmarRatio(ra) -> 0.9061697
    ra = managers_data["HAM1"]

    cr = calmar_ratio(ra)
    assert cr == pytest.approx(0.9061697, abs=1e-6)


def test_omega_ratio(edhec_data):
    # Benchmark from R (with repo CSV):
    # Omega(r_edhec, L=0) -> 2.602138
    ra = edhec_data["Convertible Arbitrage"]

    omega = omega_ratio(ra, L=0)
    assert omega == pytest.approx(2.602138, abs=1e-6)
