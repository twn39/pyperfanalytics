import pytest

from pyperfanalytics import (
    beta_co_kurtosis,
    beta_co_skewness,
    beta_co_variance,
    co_kurtosis,
    co_skewness,
    kurtosis,
    skewness,
)


def test_skewness_kurtosis(managers_data):
    ham1 = managers_data["HAM1"].dropna()

    # R Benchmarks for HAM1
    # moment: -0.6588445
    # fisher: 0.5695854 (raw)
    # sample: -0.6740873

    assert skewness(ham1, method="moment") == pytest.approx(-0.6588445, abs=1e-6)
    assert skewness(ham1, method="fisher") == pytest.approx(0.5695854, abs=1e-6)
    assert skewness(ham1, method="sample") == pytest.approx(-0.6740873, abs=1e-6)

    # Kurtosis Benchmarks for HAM1
    # moment: 5.361589
    # excess: 2.361589
    # fisher: 0.8846114 (raw)
    # sample: 5.570361
    # sample_excess: 2.500415

    assert kurtosis(ham1, method="moment") == pytest.approx(5.361589, abs=1e-6)
    assert kurtosis(ham1, method="excess") == pytest.approx(2.361589, abs=1e-6)
    assert kurtosis(ham1, method="fisher") == pytest.approx(0.8846114, abs=1e-6)
    assert kurtosis(ham1, method="sample") == pytest.approx(5.570361, abs=1e-6)
    assert kurtosis(ham1, method="sample_excess") == pytest.approx(2.500415, abs=1e-6)

def test_co_moments(managers_data):
    ham1 = managers_data["HAM1"]
    sp500 = managers_data["SP500 TR"]

    # R Benchmarks for HAM1 vs SP500
    # CoSkewness: -2.488483e-05
    # CoKurtosis: 5.938989e-06
    # BetaCoVariance: 0.3906033
    # BetaCoSkewness: 0.560197
    # BetaCoKurtosis: 0.4814681

    assert co_skewness(ham1, sp500) == pytest.approx(-2.488483e-05, abs=1e-10)
    assert co_kurtosis(ham1, sp500) == pytest.approx(5.938989e-06, abs=1e-10)
    assert beta_co_variance(ham1, sp500) == pytest.approx(0.3906033, abs=1e-6)
    assert beta_co_skewness(ham1, sp500) == pytest.approx(0.560197, abs=1e-6)
    assert beta_co_kurtosis(ham1, sp500) == pytest.approx(0.4814681, abs=1e-6)
