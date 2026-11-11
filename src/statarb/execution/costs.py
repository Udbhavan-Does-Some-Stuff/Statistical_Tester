"""Transaction and financing cost model.

Stage 1 uses a linear model: cost is proportional to traded notional, per leg.
That's a reasonable approximation for liquid large caps at modest size. It
understates costs when size is large relative to volume; a square-root
market-impact term (cost ~ sigma * sqrt(Q / ADV)) is the natural next step
once volume data is in the pipeline.

Pairs trading always holds a short leg, so borrow fees are charged every bar
on short notional. Ignoring them flatters slow-reverting pairs most, because
those hold positions longest.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

BPS = 1e-4


@dataclass(frozen=True)
class CostModel:
    commission_bps: float = 1.0  # broker/exchange fees, per unit of traded notional
    slippage_bps: float = 2.0  # half-spread + adverse fill vs. the reference close
    borrow_bps_annual: float = 25.0  # stock-loan fee on short notional (general collateral)

    def __post_init__(self) -> None:
        if min(self.commission_bps, self.slippage_bps, self.borrow_bps_annual) < 0:
            raise ValueError("costs must be non-negative")

    @property
    def transaction_bps(self) -> float:
        return self.commission_bps + self.slippage_bps

    def transaction_costs(self, traded_notional: pd.Series) -> pd.Series:
        """Cost of trading ``traded_notional`` (absolute dollars, summed over legs)."""
        return traded_notional * (self.transaction_bps * BPS)

    def borrow_costs(self, short_notional: pd.Series, periods_per_year: int) -> pd.Series:
        """Per-bar financing cost of holding ``short_notional`` dollars short."""
        return short_notional * (self.borrow_bps_annual * BPS / periods_per_year)


ZERO_COSTS = CostModel(commission_bps=0.0, slippage_bps=0.0, borrow_bps_annual=0.0)
