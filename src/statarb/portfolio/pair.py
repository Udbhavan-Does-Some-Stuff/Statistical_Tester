"""Sizing and mark-to-market accounting for a single pair.

Timing convention (one bar = one close):

    position_t   spread state *held after* the close of bar t (already lagged
                 by the execution layer, so it's a fill, not a wish)
    shares_t     share holdings after trading at close t
    pnl_t        shares_{t-1} . (P_t - P_{t-1}) - costs incurred at t

Holdings set at close t earn the move from t to t+1, never the move into t.
That shift is what keeps the accounting causal.
"""

from __future__ import annotations

import pandas as pd

from ..execution.costs import CostModel


def _check_aligned(*series: pd.Series) -> None:
    first = series[0].index
    if any(not s.index.equals(first) for s in series[1:]):
        raise ValueError("all inputs must share the same index")


def size_pair_positions(
    position: pd.Series,
    y: pd.Series,
    x: pd.Series,
    hedge_ratio: pd.Series,
    capital: float,
) -> pd.DataFrame:
    """Convert spread states {-1, 0, +1} into share holdings.

    One unit of spread = +1 share of y and -beta shares of x. At each entry we
    buy n = capital / (P_y + |beta| P_x) units, so gross exposure equals
    ``capital`` at the fill, then hold the *shares* fixed until the state
    changes. We don't rebalance daily: as prices drift, gross exposure drifts,
    which is what a trader holding the position actually experiences.

    ``hedge_ratio`` is a series so that a time-varying beta (e.g. a Kalman
    filter) slots in unchanged. The beta at entry is frozen for the life of
    the trade.
    """
    _check_aligned(position, y, x, hedge_ratio)
    if capital <= 0:
        raise ValueError("capital must be positive")

    pos = position.astype(float)
    entry = pos.ne(pos.shift(fill_value=0.0)) & pos.ne(0.0)  # also catches direct flips
    units = (capital / (y + hedge_ratio.abs() * x)).where(entry).ffill()
    beta_at_entry = hedge_ratio.where(entry).ffill()

    shares_y = (pos * units).fillna(0.0)
    shares_x = (-pos * beta_at_entry * units).fillna(0.0)
    return pd.DataFrame({"shares_y": shares_y, "shares_x": shares_x}, index=position.index)


def mark_to_market(
    shares: pd.DataFrame,
    y: pd.Series,
    x: pd.Series,
    costs: CostModel,
    capital: float,
    periods_per_year: int,
) -> pd.DataFrame:
    """Bar-by-bar P&L, costs, equity and returns for a share-holding path."""
    _check_aligned(shares["shares_y"], y, x)
    held = shares.shift(1, fill_value=0.0)  # holdings carried into bar t

    gross_pnl = held["shares_y"] * y.diff().fillna(0.0) + held["shares_x"] * x.diff().fillna(0.0)

    traded = shares.diff().fillna(shares)  # first bar trades from zero
    traded_notional = traded["shares_y"].abs() * y + traded["shares_x"].abs() * x
    transaction_costs = costs.transaction_costs(traded_notional)

    short_notional = (
        (-held["shares_y"]).clip(lower=0.0) * y.shift(1)
        + (-held["shares_x"]).clip(lower=0.0) * x.shift(1)
    ).fillna(0.0)
    borrow_costs = costs.borrow_costs(short_notional, periods_per_year)

    net_pnl = gross_pnl - transaction_costs - borrow_costs
    equity = capital + net_pnl.cumsum()
    # Return on equity at the start of the bar. Sizing uses fixed `capital`
    # (no compounding of position size), but returns are still per-bar, so
    # prod(1 + r) reproduces equity / capital exactly.
    returns = net_pnl / equity.shift(1, fill_value=capital)

    return pd.DataFrame(
        {
            "gross_pnl": gross_pnl,
            "transaction_costs": transaction_costs,
            "borrow_costs": borrow_costs,
            "net_pnl": net_pnl,
            "equity": equity,
            "returns": returns,
            "gross_exposure": shares["shares_y"].abs() * y + shares["shares_x"].abs() * x,
        },
        index=shares.index,
    )
