"""End-to-end demo on a synthetic cointegrated pair.

    python examples/synthetic_pair_demo.py

Prints the formation-window fit and out-of-sample metrics for three
configurations, then writes an interactive tear sheet to output/.
The execution-lag comparison shows how much a same-bar fill (look-ahead)
inflates results.
"""

from pathlib import Path

import pandas as pd

from statarb.analytics.plots import plot_pair_backtest
from statarb.backtest import PairsConfig, run_pairs_backtest
from statarb.data import simulate_cointegrated_pair
from statarb.execution import ZERO_COSTS


def main() -> None:
    pair = simulate_cointegrated_pair(n=1500, half_life=5.0, seed=7)
    y, x = pair.prices["y"], pair.prices["x"]

    configs = {
        "realistic (lag 1, costs)": PairsConfig(),
        "no costs (lag 1)": PairsConfig(costs=ZERO_COSTS),
        "look-ahead (lag 0, costs)": PairsConfig(execution_lag=0),
    }
    results = {name: run_pairs_backtest(y, x, cfg) for name, cfg in configs.items()}

    hedge = results["realistic (lag 1, costs)"].hedge
    print(f"True:   alpha={pair.alpha:.3f}  beta={pair.beta:.4f}  half-life={pair.half_life:.1f}")
    print(f"Fitted: alpha={hedge.alpha:.3f}  beta={hedge.beta:.4f}  half-life={hedge.half_life:.1f}"
          f"  EG t={hedge.eg_stat:.2f}  p={hedge.pvalue:.4f}\n")

    table = pd.DataFrame({name: r.metrics for name, r in results.items()})
    with pd.option_context("display.float_format", "{:,.4f}".format, "display.width", 200, "display.max_columns", None):
        print(table)

    out = Path(__file__).resolve().parents[1] / "output"
    out.mkdir(exist_ok=True)
    path = out / "synthetic_pair_tearsheet.html"
    plot_pair_backtest(results["realistic (lag 1, costs)"], "Synthetic pair, out-of-sample").write_html(
        path, include_plotlyjs="cdn"
    )
    print(f"\nTear sheet: {path}")


if __name__ == "__main__":
    main()
