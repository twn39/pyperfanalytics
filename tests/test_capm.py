import pytest

from pyperfanalytics.returns import capm_alpha
from pyperfanalytics.risk import capm_beta


def test_capm_metrics(managers_data):
    # Benchmark from R:
    # ra <- managers[,1,drop=FALSE]; rb <- managers[,8,drop=FALSE]; rf <- managers[,10,drop=FALSE]
    # CAPM.beta(ra, rb, Rf=rf) -> 0.3900712
    # CAPM.alpha(ra, rb, Rf=rf) -> 0.005774729

    ra = managers_data['HAM1']
    rb = managers_data['SP500 TR']
    rf = managers_data['US 3m TR']

    beta = capm_beta(ra, rb, Rf=rf)
    alpha = capm_alpha(ra, rb, Rf=rf)

    assert beta == pytest.approx(0.3900712, abs=1e-6)
    assert alpha == pytest.approx(0.005774729, abs=1e-7)

def test_capm_beta_multiple(managers_data):
    # Test multiple columns
    ra = managers_data[['HAM1', 'HAM2']]
    rb = managers_data['SP500 TR']
    rf = managers_data['US 3m TR']

    beta = capm_beta(ra, rb, Rf=rf)
    assert len(beta) == 2
    assert beta['HAM1'] == pytest.approx(0.3900712, abs=1e-6)
