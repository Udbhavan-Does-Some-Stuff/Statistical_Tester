# statarb

A research-grade statistical arbitrage backtester: cointegration-based pairs
trading with explicit look-ahead controls, realistic costs and out-of-sample
evaluation.

## Layout

```
src/statarb/
  data/        loaders.py (CSV, validation, alignment), synthetic.py (OU / cointegrated pairs)
  models/      cointegration.py (OLS hedge, Engle-Granger), ou.py (AR(1), half-life)
  signals/     zscore.py (rolling z-score, entry/exit/stop state machine)
  execution/   costs.py (commission, slippage, borrow)
  portfolio/   pair.py (sizing, holdings, mark-to-market)
  analytics/   metrics.py (Sharpe, Sortino, drawdown, ...), plots.py (Plotly tear sheet)
  backtest/    pairs.py (formation/trading split, orchestration)
tests/         one file per layer, plus end-to-end and look-ahead tests
examples/      synthetic_pair_demo.py
docs/          ARCHITECTURE.md (design decisions, findings, roadmap)
```

## Usage

```bash
python -m pytest                                    # run the tests
PYTHONPATH=src python examples/synthetic_pair_demo.py
# or: pip install -e .[dev]   (then no PYTHONPATH needed)
```

```python
from statarb.backtest import PairsConfig, run_pairs_backtest
from statarb.data import load_prices_csv, align_pair

prices = load_prices_csv("prices.csv")              # wide: date, TICKER1, TICKER2, ...
y, x = align_pair(prices, "KO", "PEP")
result = run_pairs_backtest(y, x, PairsConfig(formation_periods=252, zscore_window=20))
result.metrics, result.hedge, result.frame          # metrics, formation fit, bar-level detail
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the design.
