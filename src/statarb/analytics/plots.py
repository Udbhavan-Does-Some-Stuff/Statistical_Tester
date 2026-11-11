"""Plotly tear sheet for a single-pair backtest.

Three stacked panels on a shared time axis, each with its own y-scale (never
two scales on one panel):

1. z-score with entry/exit bands, holding periods shaded (long = blue, short = red);
2. equity, gross vs. net of costs. The gap between them is the cost drag;
3. drawdown of net equity.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..backtest.pairs import PairsBacktestResult

# Reference palette (light mode). Roles, not raw hex, in the code below.
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e8e7e3"
SERIES_1 = "#2a78d6"  # net equity / drawdown / long holdings
SERIES_2 = "#eb6834"  # gross equity
LONG_FILL = "rgba(42,120,214,0.12)"  # diverging pair: blue pole
SHORT_FILL = "rgba(227,73,72,0.12)"  # diverging pair: red pole


def _holding_periods(position: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp, int]]:
    """(start, end, direction) for each contiguous non-zero run of ``position``."""
    run_id = position.ne(position.shift()).cumsum()
    periods = []
    for _, run in position.groupby(run_id):
        direction = int(run.iloc[0])
        if direction != 0:
            periods.append((run.index[0], run.index[-1], direction))
    return periods


def plot_pair_backtest(result: PairsBacktestResult, title: str | None = None) -> go.Figure:
    f = result.frame
    cfg = result.config
    gross_equity = cfg.capital + f["gross_pnl"].cumsum()
    drawdown = f["equity"] / f["equity"].cummax().clip(lower=cfg.capital) - 1.0

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.4, 0.35, 0.25],
        subplot_titles=("Spread z-score", "Equity", "Drawdown"),
    )

    # --- panel 1: z-score, bands, holding periods -------------------------
    for start, end, direction in _holding_periods(f["position"]):
        fig.add_vrect(
            x0=start, x1=end, row=1, col=1, line_width=0, layer="below",
            fillcolor=LONG_FILL if direction > 0 else SHORT_FILL,
        )
    for level, dash in ((cfg.entry_z, "dash"), (-cfg.entry_z, "dash"), (cfg.exit_z, "dot"), (-cfg.exit_z, "dot")):
        fig.add_hline(y=level, row=1, col=1, line=dict(color=TEXT_SECONDARY, width=1, dash=dash))
    fig.add_trace(
        go.Scatter(x=f.index, y=f["zscore"], name="z-score", line=dict(color=TEXT_PRIMARY, width=1.5),
                   hovertemplate="%{y:.2f}"),
        row=1, col=1,
    )
    # legend entries for the shading
    for label, color in (("Long spread", LONG_FILL), ("Short spread", SHORT_FILL)):
        fig.add_trace(
            go.Scatter(x=[None], y=[None], mode="markers", name=label,
                       marker=dict(size=12, symbol="square", color=color.replace("0.12", "0.5"))),
            row=1, col=1,
        )

    # --- panel 2: equity --------------------------------------------------
    fig.add_trace(
        go.Scatter(x=f.index, y=gross_equity, name="Gross equity", line=dict(color=SERIES_2, width=2),
                   hovertemplate="$%{y:,.0f}"),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(x=f.index, y=f["equity"], name="Net equity", line=dict(color=SERIES_1, width=2),
                   hovertemplate="$%{y:,.0f}"),
        row=2, col=1,
    )
    for series, label in ((gross_equity, "Gross"), (f["equity"], "Net")):
        fig.add_annotation(x=series.index[-1], y=series.iloc[-1], text=label, row=2, col=1,
                           showarrow=False, xanchor="left", xshift=6, font=dict(color=TEXT_SECONDARY))

    # --- panel 3: drawdown ------------------------------------------------
    fig.add_trace(
        go.Scatter(x=f.index, y=drawdown, name="Drawdown", fill="tozeroy", showlegend=False,
                   line=dict(color=SERIES_1, width=1.5), fillcolor="rgba(42,120,214,0.2)",
                   hovertemplate="%{y:.2%}"),
        row=3, col=1,
    )

    m = result.metrics
    subtitle = (
        f"Sharpe {m['sharpe']:.2f} · CAGR {m['cagr']:.2%} · max DD {m['max_drawdown']:.2%} · "
        f"{m['n_trades']} trades · β={result.hedge.beta:.3f} · EG p={result.hedge.pvalue:.3f}"
    )
    fig.update_layout(
        title=dict(text=f"{title or 'Pairs backtest'}<br><sup>{subtitle}</sup>", x=0.01),
        height=850,
        hovermode="x unified",
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(color=TEXT_PRIMARY, family="Inter, Segoe UI, sans-serif", size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=70, r=60, t=110, b=40),
    )
    fig.update_xaxes(showgrid=False, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(tickprefix="$", tickformat=",.0f", row=2, col=1)
    fig.update_yaxes(tickformat=".0%", row=3, col=1)
    return fig
