"""Mean-reversion dynamics of a spread.

Stage 1 only needs the half-life. A full OU maximum-likelihood fit (and
OU-optimal entry/exit bands) comes in a later stage.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def ar1_fit(series: ArrayLike) -> tuple[float, float]:
    """OLS fit of s_t = c + phi s_{t-1} + e_t. Returns (c, phi).

    This is the exact discrete-time form of an OU process sampled at a fixed
    interval, with phi = exp(-theta dt) and mu = c / (1 - phi).
    """
    s = np.asarray(series, dtype=float)
    s = s[~np.isnan(s)]
    if s.size < 3:
        raise ValueError("need at least 3 observations to fit an AR(1)")
    design = np.column_stack([np.ones(s.size - 1), s[:-1]])
    (c, phi), *_ = np.linalg.lstsq(design, s[1:], rcond=None)
    return float(c), float(phi)


def half_life(series: ArrayLike) -> float:
    """Mean-reversion half-life in bars: -ln 2 / ln(phi).

    Returns ``inf`` when phi is outside (0, 1): either no mean reversion
    (phi >= 1, a unit root or explosive) or oscillation (phi <= 0), where a
    half-life isn't meaningful.

    Many references use the first-order approximation -ln 2 / (phi - 1); it is
    biased for fast-reverting spreads, so we use the exact expression.
    """
    _, phi = ar1_fit(series)
    if not 0.0 < phi < 1.0:
        return float("inf")
    return float(-np.log(2.0) / np.log(phi))
