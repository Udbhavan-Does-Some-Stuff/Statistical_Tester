"""Backtest orchestration: wires data, models, signals, execution and accounting."""

from .pairs import PairsBacktestResult, PairsConfig, run_pairs_backtest

__all__ = ["PairsBacktestResult", "PairsConfig", "run_pairs_backtest"]
