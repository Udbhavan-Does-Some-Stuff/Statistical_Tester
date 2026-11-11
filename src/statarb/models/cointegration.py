"""Engle-Granger cointegration and static hedge ratios.

Two I(1) price series y, x are cointegrated if some linear combination
y - alpha - beta x is I(0) (stationary). Engle-Granger is the two-step test:

1. estimate (alpha, beta) by OLS of y on x;
2. ADF-test the residuals for a unit root.

Step 2 can't use the standard ADF critical values: the residuals come from a
regression that was chosen to minimise their variance, so they look more
stationary than they are. statsmodels' ``coint`` uses MacKinnon's
residual-based critical values, which correct for this.

Two caveats worth keeping in mind:

* The test is not symmetric. Regressing y on x and x on y give different
  betas and p-values. We fix the convention: ``y`` is the dependent leg.
* We regress price levels, not log prices. A level hedge ratio is a share
  ratio, so "long 1 share y, short beta shares x" is a fixed-share portfolio
  whose value *is* the spread. A log-price beta is a dollar ratio and would
  need continuous rebalancing to track the modelled spread.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from statsmodels.tsa.stattools import coint

from .ou import half_life

MIN_OBSERVATIONS = 30


@dataclass(frozen=True)
class HedgeFit:
    """Result of fitting a static cointegrating relationship y = alpha + beta x + s."""

    alpha: float
    beta: float
    resid_std: float
    eg_stat: float  # Engle-Granger t-statistic (more negative = stronger evidence)
    pvalue: float  # MacKinnon p-value, H0: no cointegration
    half_life: float  # of the in-sample residual, in bars
    n_obs: int

    def spread(self, y: pd.Series, x: pd.Series) -> pd.Series:
        return (y - self.alpha - self.beta * x).rename("spread")


def _clean_arrays(y: ArrayLike, x: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(y, pd.Series) and isinstance(x, pd.Series) and not y.index.equals(x.index):
        raise ValueError("y and x must share the same index; align them first")
    yv = np.asarray(y, dtype=float)
    xv = np.asarray(x, dtype=float)
    if yv.shape != xv.shape or yv.ndim != 1:
        raise ValueError("y and x must be 1-D and of equal length")
    if np.isnan(yv).any() or np.isnan(xv).any():
        raise ValueError("y and x must not contain NaN")
    if yv.size < MIN_OBSERVATIONS:
        raise ValueError(f"need at least {MIN_OBSERVATIONS} observations, got {yv.size}")
    return yv, xv


def ols_hedge_ratio(y: ArrayLike, x: ArrayLike) -> tuple[float, float]:
    """OLS of y on [1, x]. Returns (alpha, beta)."""
    yv, xv = _clean_arrays(y, x)
    design = np.column_stack([np.ones_like(xv), xv])
    (alpha, beta), *_ = np.linalg.lstsq(design, yv, rcond=None)
    return float(alpha), float(beta)


def engle_granger(y: ArrayLike, x: ArrayLike) -> tuple[float, float]:
    """Engle-Granger test with a constant. Returns (t-statistic, p-value)."""
    yv, xv = _clean_arrays(y, x)
    stat, pvalue, _ = coint(yv, xv, trend="c", autolag="aic")
    return float(stat), float(pvalue)


def fit_static_hedge(y: ArrayLike, x: ArrayLike) -> HedgeFit:
    """Fit alpha, beta by OLS and test the residual for stationarity."""
    yv, xv = _clean_arrays(y, x)
    alpha, beta = ols_hedge_ratio(yv, xv)
    resid = yv - alpha - beta * xv
    stat, pvalue = engle_granger(yv, xv)
    return HedgeFit(
        alpha=alpha,
        beta=beta,
        resid_std=float(resid.std(ddof=2)),  # two estimated parameters
        eg_stat=stat,
        pvalue=pvalue,
        half_life=half_life(resid),
        n_obs=int(yv.size),
    )
