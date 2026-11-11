from statarb.analytics.plots import _holding_periods, plot_pair_backtest
from statarb.backtest import PairsConfig, run_pairs_backtest

import pandas as pd


def test_holding_periods():
    idx = pd.RangeIndex(8)
    pos = pd.Series([0, 1, 1, 0, -1, -1, 0, 1], idx)
    assert _holding_periods(pos) == [(1, 2, 1), (4, 5, -1), (7, 7, 1)]


def test_plot_builds(coint_pair):
    res = run_pairs_backtest(coint_pair.prices["y"], coint_pair.prices["x"], PairsConfig(max_pvalue=None))
    fig = plot_pair_backtest(res)
    names = {t.name for t in fig.data}
    assert {"z-score", "Gross equity", "Net equity", "Drawdown"} <= names
