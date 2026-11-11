"""Price loading and validation.

Conventions used everywhere downstream:

* prices are a wide DataFrame: DatetimeIndex (sorted, unique) x one column per asset;
* prices are *adjusted* closes (splits/dividends), strictly positive;
* missing values are never forward-filled here. A stale price creates fake
  mean reversion (the spread "snaps back" when the stale leg updates), which
  is one of the most common sources of phantom stat-arb profits.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd


def validate_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Raise if ``prices`` violates the price-frame contract; return it unchanged."""
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError("prices must be indexed by a DatetimeIndex")
    if prices.index.has_duplicates:
        dupes = prices.index[prices.index.duplicated()].unique()[:5]
        raise ValueError(f"duplicate timestamps in prices, e.g. {list(dupes)}")
    if not prices.index.is_monotonic_increasing:
        raise ValueError("price index must be sorted ascending")
    non_numeric = [c for c in prices.columns if not pd.api.types.is_numeric_dtype(prices[c])]
    if non_numeric:
        raise TypeError(f"non-numeric price columns: {non_numeric}")
    if (prices <= 0).to_numpy().any():  # NaN compares False, so gaps are allowed
        bad = prices.columns[(prices <= 0).any()].tolist()
        raise ValueError(f"non-positive prices in columns {bad}")
    return prices


def load_prices_csv(
    path: str | Path,
    *,
    date_column: str = "date",
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Load a wide CSV (one date column, one column per asset) into a validated frame."""
    frame = pd.read_csv(path, parse_dates=[date_column], index_col=date_column)
    if columns is not None:
        frame = frame[list(columns)]
    frame = frame.sort_index().astype(float)
    return validate_prices(frame)


def align_pair(prices: pd.DataFrame, y: str, x: str) -> tuple[pd.Series, pd.Series]:
    """Return the two legs on their common timestamps (rows with any gap dropped)."""
    pair = validate_prices(prices[[y, x]]).dropna(how="any")
    return pair[y], pair[x]
