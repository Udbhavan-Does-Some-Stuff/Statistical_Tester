import pandas as pd
import pytest

from statarb.backtest import PairsConfig, run_pairs_backtest
from statarb.execution import ZERO_COSTS


def _run(prices, **overrides):
    """Mechanics tests bypass the cointegration filter so they can never pass vacuously."""
    overrides.setdefault("max_pvalue", None)
    return run_pairs_backtest(prices["y"], prices["x"], PairsConfig(**overrides))


def test_profitable_on_true_mean_reverting_pair(coint_pair):
    res = run_pairs_backtest(
        coint_pair.prices["y"], coint_pair.prices["x"], PairsConfig(costs=ZERO_COSTS)
    )
    assert res.traded  # passes the default 5% Engle-Granger filter
    assert res.metrics["n_trades"] > 20
    assert res.metrics["sharpe"] > 1.0
    assert len(res.frame) == 1500 - 252  # trading window only


def test_costs_strictly_reduce_pnl(coint_pair):
    free = _run(coint_pair.prices, costs=ZERO_COSTS)
    costly = _run(coint_pair.prices)
    pd.testing.assert_series_equal(free.frame["position"], costly.frame["position"])
    assert costly.equity.iloc[-1] < free.equity.iloc[-1]
    paid = costly.frame[["transaction_costs", "borrow_costs"]].to_numpy().sum()
    assert free.equity.iloc[-1] - costly.equity.iloc[-1] == pytest.approx(paid)


def test_equity_identity(coint_pair):
    res = _run(coint_pair.prices)
    assert res.equity.iloc[-1] == pytest.approx(res.config.capital + res.frame["net_pnl"].sum())


def test_execution_lag_shifts_fills(coint_pair):
    res = _run(coint_pair.prices, execution_lag=2)
    pd.testing.assert_series_equal(
        res.frame["position"], res.frame["signal"].shift(2, fill_value=0), check_names=False, check_dtype=False
    )


def test_non_cointegrated_pair_is_not_traded(independent_walks):
    res = run_pairs_backtest(independent_walks["y"], independent_walks["x"], PairsConfig())
    assert not res.traded
    assert (res.frame["position"] == 0).all()
    assert res.equity.iloc[-1] == res.config.capital


def test_no_lookahead_truncation_invariance(coint_pair):
    """Results up to bar T must be identical whether or not data after T exists.

    This one test covers every stage (hedge fit, z-score, signal, fills,
    accounting). Any use of future data, anywhere, breaks it.
    """
    full = _run(coint_pair.prices)
    cut = coint_pair.prices.index[1000]
    truncated = _run(coint_pair.prices.loc[:cut])
    assert truncated.metrics["n_trades"] > 10  # guard: the comparison must cover real trades
    pd.testing.assert_frame_equal(full.frame.loc[:cut], truncated.frame)


def test_formation_window_is_out_of_sample(coint_pair):
    """Hedge must depend only on formation data: changing trading-window data leaves it unchanged."""
    prices = coint_pair.prices.copy()
    prices.iloc[300:, 0] *= 1.5
    assert _run(prices).hedge == _run(coint_pair.prices).hedge


def test_config_validation():
    with pytest.raises(ValueError):
        PairsConfig(formation_periods=50, zscore_window=60)
    with pytest.raises(ValueError):
        PairsConfig(execution_lag=-1)
