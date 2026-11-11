"""Accounting tests against hand-computed numbers.

Setup: y = [10, 11, 12, 11], x = [5, 5, 6, 6], beta = 2, capital = 100,
position = [0, 1, 1, 0] (long spread filled at bar 1, closed at bar 3).

Entry at bar 1: units = 100 / (11 + 2*5) = 100/21
    shares_y = 100/21, shares_x = -200/21
Bar 2 P&L: 100/21 * (12-11) - 200/21 * (6-5) = -100/21
Bar 3 P&L: 100/21 * (11-12) - 200/21 * (6-6) = -100/21
"""

import pandas as pd
import pytest

from statarb.execution import ZERO_COSTS, CostModel
from statarb.portfolio import mark_to_market, size_pair_positions

IDX = pd.RangeIndex(4)
Y = pd.Series([10.0, 11.0, 12.0, 11.0], IDX)
X = pd.Series([5.0, 5.0, 6.0, 6.0], IDX)
BETA = pd.Series(2.0, IDX)
POS = pd.Series([0, 1, 1, 0], IDX)


def _shares():
    return size_pair_positions(POS, Y, X, BETA, capital=100.0)


def test_sizing_sets_gross_exposure_to_capital_at_entry():
    shares = _shares()
    assert shares["shares_y"].tolist() == pytest.approx([0, 100 / 21, 100 / 21, 0])
    assert shares["shares_x"].tolist() == pytest.approx([0, -200 / 21, -200 / 21, 0])
    gross_at_entry = abs(shares["shares_y"][1]) * Y[1] + abs(shares["shares_x"][1]) * X[1]
    assert gross_at_entry == pytest.approx(100.0)


def test_shares_are_held_not_rebalanced_and_beta_frozen_at_entry():
    beta = pd.Series([2.0, 2.0, 3.0, 3.0], IDX)  # beta moves mid-trade
    shares = size_pair_positions(POS, Y, X, beta, capital=100.0)
    assert shares["shares_x"][2] == shares["shares_x"][1]


def test_gross_pnl_without_costs():
    ledger = mark_to_market(_shares(), Y, X, ZERO_COSTS, capital=100.0, periods_per_year=252)
    assert ledger["gross_pnl"].tolist() == pytest.approx([0, 0, -100 / 21, -100 / 21])
    assert ledger["equity"].iloc[-1] == pytest.approx(100 - 200 / 21)
    # per-bar returns compound back to the equity curve exactly
    assert (1 + ledger["returns"]).prod() == pytest.approx(ledger["equity"].iloc[-1] / 100)


def test_transaction_and_borrow_costs():
    costs = CostModel(commission_bps=4.0, slippage_bps=6.0, borrow_bps_annual=252.0)  # 10bp/trade, 1bp/bar borrow
    ledger = mark_to_market(_shares(), Y, X, costs, capital=100.0, periods_per_year=252)
    # entry trades 100 notional; exit trades 100/21*11 + 200/21*6 = 2300/21
    assert ledger["transaction_costs"].tolist() == pytest.approx([0, 0.1, 0, 2300 / 21 * 1e-3])
    # short x leg held into bars 2 and 3, valued at previous close (5 then 6)
    assert ledger["borrow_costs"].tolist() == pytest.approx([0, 0, 1000 / 21 * 1e-4, 1200 / 21 * 1e-4])
    expected_net = ledger["gross_pnl"] - ledger["transaction_costs"] - ledger["borrow_costs"]
    pd.testing.assert_series_equal(ledger["net_pnl"], expected_net, check_names=False)


def test_flat_position_has_no_pnl():
    flat = size_pair_positions(POS * 0, Y, X, BETA, capital=100.0)
    ledger = mark_to_market(flat, Y, X, CostModel(), capital=100.0, periods_per_year=252)
    assert (ledger["net_pnl"] == 0).all()
