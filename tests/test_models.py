import numpy as np
import pytest

from statarb.data import simulate_ou
from statarb.models import engle_granger, fit_static_hedge, half_life, ols_hedge_ratio


def test_ols_recovers_exact_linear_relation():
    x = np.linspace(10, 50, 100)
    alpha, beta = ols_hedge_ratio(3.0 + 2.5 * x, x)
    assert alpha == pytest.approx(3.0)
    assert beta == pytest.approx(2.5)


@pytest.mark.parametrize("true_hl", [5.0, 20.0])
def test_half_life_recovers_simulated_ou(true_hl):
    s = simulate_ou(50_000, true_hl, 1.0, rng=0)
    assert half_life(s) == pytest.approx(true_hl, rel=0.1)


def test_half_life_of_random_walk_is_infinite_or_huge():
    walk = np.cumsum(np.random.default_rng(0).standard_normal(5000))
    assert half_life(walk) > 200


def test_cointegrated_pair_is_detected(coint_pair):
    fit = fit_static_hedge(coint_pair.prices["y"], coint_pair.prices["x"])
    assert fit.pvalue < 0.01
    assert fit.beta == pytest.approx(coint_pair.beta, rel=0.02)
    assert fit.half_life == pytest.approx(coint_pair.half_life, rel=0.3)


def test_independent_walks_are_not_cointegrated(independent_walks):
    _, pvalue = engle_granger(independent_walks["y"], independent_walks["x"])
    assert pvalue > 0.1


def test_misaligned_series_rejected(coint_pair):
    y = coint_pair.prices["y"]
    with pytest.raises(ValueError, match="same index"):
        fit_static_hedge(y, coint_pair.prices["x"].shift(1, freq="B"))
