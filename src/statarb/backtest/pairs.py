"""Single-pair backtest with a formation / trading split.

    |<---- formation (in-sample) ---->|<------ trading (out-of-sample) ------>|
     fit alpha, beta; Engle-Granger     spread uses the *frozen* alpha, beta;
     test; half-life                    z-score -> signal -> lagged fill -> P&L

Look-ahead controls, in order of how often they bite in practice:

1. The hedge ratio is estimated on formation data only. Fitting beta on the
   full sample and then trading the same sample is the classic pairs-trading
   leak: the residual is mean-zero *by construction* over the whole period.
2. The z-score is a trailing window ending at t.
3. A decision made at the close of t fills ``execution_lag`` bars later
   (default 1). With lag 0 you trade at the price used to compute the signal,
   which isn't achievable and noticeably inflates results.
4. Holdings set at t earn returns from t+1 onwards (see portfolio.pair).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from ..analytics.metrics import performance_summary
from ..execution.costs import CostModel
from ..models.cointegration import HedgeFit, fit_static_hedge
from ..portfolio.pair import mark_to_market, size_pair_positions
from ..signals.zscore import rolling_zscore, threshold_positions


@dataclass(frozen=True)
class PairsConfig:
    formation_periods: int = 252
    zscore_window: int = 20
    entry_z: float = 2.0
    exit_z: float = 0.5
    stop_z: float | None = 4.0
    execution_lag: int = 1
    capital: float = 1_000_000.0
    max_pvalue: float | None = 0.05  # skip trading if not cointegrated in formation; None = always trade
    costs: CostModel = field(default_factory=CostModel)
    periods_per_year: int = 252

    def __post_init__(self) -> None:
        if self.zscore_window > self.formation_periods:
            raise ValueError("zscore_window must fit inside the formation window")
        if self.execution_lag < 0:
            raise ValueError("execution_lag must be >= 0")
        if self.max_pvalue is not None and not 0 < self.max_pvalue <= 1:
            raise ValueError("max_pvalue must be in (0, 1]")


@dataclass(frozen=True)
class PairsBacktestResult:
    config: PairsConfig
    hedge: HedgeFit
    traded: bool  # False if the formation-period cointegration filter rejected the pair
    frame: pd.DataFrame  # bar-level detail over the trading window
    metrics: dict[str, float]

    @property
    def returns(self) -> pd.Series:
        return self.frame["returns"]

    @property
    def equity(self) -> pd.Series:
        return self.frame["equity"]


def run_pairs_backtest(
    y: pd.Series, x: pd.Series, config: PairsConfig | None = None
) -> PairsBacktestResult:
    """Backtest a z-score pairs strategy on ``y`` (dependent leg) vs ``x``."""
    cfg = config or PairsConfig()
    pair = pd.concat([y.rename("y"), x.rename("x")], axis=1, join="inner").dropna()
    y, x = pair["y"], pair["x"]
    n_form = cfg.formation_periods
    if len(pair) <= n_form + 1:
        raise ValueError(f"need more than {n_form + 1} aligned bars, got {len(pair)}")

    # --- model: formation window only -------------------------------------
    hedge = fit_static_hedge(y.iloc[:n_form], x.iloc[:n_form])
    traded = cfg.max_pvalue is None or hedge.pvalue <= cfg.max_pvalue

    # --- spread & z-score over the full history (both strictly causal) ----
    # Computing them over formation+trading lets the trailing z-score window
    # warm up on formation data, so trading starts on bar one of the window.
    hedge_ratio = pd.Series(hedge.beta, index=pair.index, name="hedge_ratio")
    spread = (y - hedge.alpha - hedge_ratio * x).rename("spread")
    zscore = rolling_zscore(spread, cfg.zscore_window)

    # --- trading window ---------------------------------------------------
    t = slice(n_form, None)
    y_t, x_t, beta_t = y.iloc[t], x.iloc[t], hedge_ratio.iloc[t]
    signal = threshold_positions(zscore.iloc[t], cfg.entry_z, cfg.exit_z, cfg.stop_z)
    if not traded:
        signal = signal * 0
    position = signal.shift(cfg.execution_lag, fill_value=0).astype("int8").rename("position")

    shares = size_pair_positions(position, y_t, x_t, beta_t, cfg.capital)
    ledger = mark_to_market(shares, y_t, x_t, cfg.costs, cfg.capital, cfg.periods_per_year)

    frame = pd.concat(
        [y_t, x_t, beta_t, spread.iloc[t], zscore.iloc[t], signal, position, shares, ledger],
        axis=1,
    )
    metrics = performance_summary(ledger["returns"], position, cfg.periods_per_year)
    return PairsBacktestResult(cfg, hedge, traded, frame, metrics)
