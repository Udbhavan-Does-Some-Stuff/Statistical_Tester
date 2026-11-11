import math

import numpy as np
import pandas as pd
import pytest

from statarb.analytics import (
    cagr,
    max_drawdown,
    performance_summary,
    position_stats,
    sharpe_ratio,
    sortino_ratio,
    total_return,
)


def test_total_return_and_cagr():
    r = pd.Series([0.01, -0.02, 0.03])
    assert total_return(r) == pytest.approx(1.01 * 0.98 * 1.03 - 1)
    one_year = pd.Series([0.0] * 251 + [0.1])
    assert cagr(one_year, 252) == pytest.approx(0.1)


def test_sharpe_matches_formula():
    r = pd.Series([0.01, -0.02, 0.03, 0.005])
    expected = r.mean() / r.std(ddof=1) * math.sqrt(252)
    assert sharpe_ratio(r) == pytest.approx(expected)
    assert sharpe_ratio(r, risk_free_rate=0.0252) == pytest.approx((r - 0.0001).mean() / r.std(ddof=1) * math.sqrt(252))


def test_degenerate_series_give_nan():
    flat = pd.Series([0.001] * 10)
    assert math.isnan(sharpe_ratio(flat))
    assert math.isnan(sortino_ratio(flat))


def test_max_drawdown_from_interior_peak():
    # wealth 1.1 -> 0.55 -> 0.66: drawdown from 1.1 to 0.55
    assert max_drawdown(pd.Series([0.1, -0.5, 0.2])) == pytest.approx(-0.5)


def test_max_drawdown_counts_initial_capital_as_peak():
    # loses immediately: the peak is the starting wealth of 1.0
    assert max_drawdown(pd.Series([-0.1, 0.05])) == pytest.approx(-0.1)


def test_position_stats():
    pos = pd.Series([0, 1, 1, 0, -1, -1, -1, 0])
    assert position_stats(pos) == {"n_trades": 2, "time_in_market": 5 / 8, "avg_holding_period": 2.5}


def test_summary_has_all_keys():
    r = pd.Series(np.random.default_rng(0).normal(0.0005, 0.01, 500))
    summary = performance_summary(r, pd.Series(np.zeros(500)))
    assert {"sharpe", "max_drawdown", "cagr", "n_trades"} <= summary.keys()
