"""Performance metrics computed from a per-bar return series.

Everything is derived from returns r_t, with wealth W_t = prod(1 + r_i),
W_0 = 1. Annualisation uses ``periods_per_year`` (252 for daily bars).

Caveat on Sharpe: scaling by sqrt(periods_per_year) assumes i.i.d. returns.
Pairs-trading returns are serially correlated (positions persist for many
bars), which can bias the annualised Sharpe in either direction. Lo (2002)
gives the correction; it belongs with the overfitting diagnostics (deflated
Sharpe etc.) in a later stage.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def total_return(returns: pd.Series) -> float:
    return float((1.0 + returns).prod() - 1.0)


def cagr(returns: pd.Series, periods_per_year: int = 252) -> float:
    if len(returns) == 0:
        return math.nan
    growth = float((1.0 + returns).prod())
    if growth <= 0:
        return -1.0
    return growth ** (periods_per_year / len(returns)) - 1.0


def annualized_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    return float(returns.std(ddof=1) * math.sqrt(periods_per_year))


def sharpe_ratio(
    returns: pd.Series, periods_per_year: int = 252, risk_free_rate: float = 0.0
) -> float:
    """Annualised Sharpe ratio; NaN when returns have zero variance."""
    excess = returns - risk_free_rate / periods_per_year
    sd = excess.std(ddof=1)
    if not sd > 1e-15:  # also catches NaN and float noise on constant series
        return math.nan
    return float(excess.mean() / sd * math.sqrt(periods_per_year))


def sortino_ratio(
    returns: pd.Series, periods_per_year: int = 252, risk_free_rate: float = 0.0
) -> float:
    """Annualised Sortino ratio using downside deviation below the risk-free rate."""
    excess = returns - risk_free_rate / periods_per_year
    downside = math.sqrt(float(np.mean(np.minimum(excess.to_numpy(), 0.0) ** 2)))
    if downside == 0:
        return math.nan
    return float(excess.mean() / downside * math.sqrt(periods_per_year))


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Fractional drawdown from the running peak, counting initial wealth as a peak."""
    wealth = (1.0 + returns).cumprod()
    peak = wealth.cummax().clip(lower=1.0)
    return (wealth / peak - 1.0).rename("drawdown")


def max_drawdown(returns: pd.Series) -> float:
    """Worst peak-to-trough loss as a non-positive fraction (e.g. -0.12)."""
    if len(returns) == 0:
        return math.nan
    return float(drawdown_series(returns).min())


def calmar_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    mdd = max_drawdown(returns)
    if not mdd < 0:
        return math.nan
    return cagr(returns, periods_per_year) / abs(mdd)


def position_stats(position: pd.Series) -> dict[str, float]:
    """Trade count, fraction of bars in the market, and mean holding period in bars."""
    active = position.ne(0)
    entries = active & position.ne(position.shift(fill_value=0))
    n_trades = int(entries.sum())
    return {
        "n_trades": n_trades,
        "time_in_market": float(active.mean()) if len(position) else math.nan,
        "avg_holding_period": float(active.sum() / n_trades) if n_trades else math.nan,
    }


def performance_summary(
    returns: pd.Series,
    position: pd.Series | None = None,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
) -> dict[str, float]:
    summary = {
        "total_return": total_return(returns),
        "cagr": cagr(returns, periods_per_year),
        "annual_volatility": annualized_volatility(returns, periods_per_year),
        "sharpe": sharpe_ratio(returns, periods_per_year, risk_free_rate),
        "sortino": sortino_ratio(returns, periods_per_year, risk_free_rate),
        "max_drawdown": max_drawdown(returns),
        "calmar": calmar_ratio(returns, periods_per_year),
    }
    if position is not None:
        summary.update(position_stats(position))
    return summary
