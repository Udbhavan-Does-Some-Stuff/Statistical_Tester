# Architecture

## Layers

Each layer depends only on layers above it in this list. Data flows downward
as plain pandas objects on a shared `DatetimeIndex`. There are no framework
classes and no hidden state.

| Layer | Package | Responsibility | Input → output |
|---|---|---|---|
| Data | `statarb.data` | load, validate, align, simulate | files / params → price frame |
| Models | `statarb.models` | hedge ratio, cointegration, mean-reversion dynamics | prices → `HedgeFit`, spread |
| Signals | `statarb.signals` | spread → target state | spread → `{-1, 0, +1}` series |
| Execution | `statarb.execution` | frictions (fill lag lives in the backtest) | traded notional → costs |
| Portfolio | `statarb.portfolio` | sizing, holdings, mark-to-market | states + prices → shares, P&L, equity |
| Analytics | `statarb.analytics` | metrics, tear sheets | returns → metrics, figures |
| Backtest | `statarb.backtest` | orchestration, train/test split | prices + config → result |

## Key decisions

**Vectorised where possible, explicit loop where path-dependent.** Spread,
z-score, sizing and P&L are vectorised pandas operations. The entry/exit
state machine depends on its own history, so it is a plain O(n) loop: correct
first, and fast enough (milliseconds for 10^4 bars).

**Hedge ratio is a time series, even when it is constant.** The static OLS
model broadcasts beta to a series. Portfolio code freezes beta at entry. A
Kalman-filter beta drops in later without touching signals or accounting.

**Fixed-share holdings, no daily rebalancing.** A spread position is
`+n` shares of y and `-n * beta` shares of x, sized at entry so that gross
exposure equals capital, then held. That is what a real position does.

**Price-level regression, not log prices.** A level beta is a share ratio, so
the held portfolio's value is exactly the modelled spread. A log beta is a
dollar ratio, and tracking the spread would need continuous rebalancing.

**Causality is tested, not assumed.** `test_no_lookahead_truncation_invariance`
runs the full pipeline on the full history and on a truncated history, then
requires identical bar-level output up to the cut. Any use of future data,
anywhere in the pipeline, breaks it.

## Timing convention

```
close t:    observe P_t  ->  spread_t, z_t  ->  signal_t
close t+L:  fill signal_t at P_{t+L}            (L = execution_lag, default 1)
bar t+L+1:  holdings earn P_{t+L+1} - P_{t+L}
```

## Findings so far

* **Engle–Granger has low power on short formation windows.** On 252 daily
  bars, the share of truly cointegrated pairs rejected at 5%:
  half-life 5 → 94%, half-life 10 → 52%, half-life 20 → 18%. A strict
  p-value filter on one year of data discards most genuine slow-reverting
  pairs. Scanning many candidate pairs then adds the opposite problem:
  multiple-testing false positives.
* **Same-bar execution inflates results.** On the demo pair, lag 0 vs lag 1
  lifts Sharpe from 1.38 to 1.73 and shrinks max drawdown by a third.
* **OLS alpha is poorly identified** when x spans a narrow range (fitted 6.2
  vs true 10.0, while beta is within 3%). The intercept is an extrapolation to
  x = 0. The rolling z-score absorbs a level error, so it barely matters here,
  but anything using the fitted mean directly (e.g. OU bands) must account
  for it.

## Roadmap

1. ~~Minimal pairs backtester (static OLS hedge, z-score bands, linear costs)~~
2. Walk-forward: rolling formation/trading windows, trade ledger, parameter grids
3. OU model: MLE fit, half-life-driven windows, OU-optimal entry/exit bands
4. Kalman-filter dynamic hedge ratio
5. Multi-pair portfolio: pair selection over a universe, capital allocation, risk limits
6. Baskets: Johansen cointegration, eigenportfolios
7. Overfitting controls: deflated Sharpe, probability of backtest overfitting (CSCV), Lo-adjusted Sharpe
8. Richer execution: square-root impact with volume data, borrow availability
