"""Synthetic data with known ground truth.

Every estimator in this project is first validated on data whose true
parameters we know. Without that, you can't tell a bug from a weak signal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import lfilter


def simulate_ou(
    n: int,
    half_life: float,
    sigma_stationary: float,
    *,
    mu: float = 0.0,
    s0: float | None = None,
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    """Simulate an Ornstein-Uhlenbeck process on a unit time grid, exactly.

    The SDE dS = theta (mu - S) dt + sigma dW has the exact discretisation

        S_{t+1} = mu + phi (S_t - mu) + eps_t,   phi = exp(-theta),
        eps_t ~ N(0, sigma_inf^2 (1 - phi^2)),

    where sigma_inf = sigma / sqrt(2 theta) is the stationary std and
    theta = ln 2 / half_life. Unlike an Euler step this has no discretisation
    error, so the simulated half-life is exactly the requested one.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    if half_life <= 0 or sigma_stationary <= 0:
        raise ValueError("half_life and sigma_stationary must be positive")
    rng = np.random.default_rng(rng)
    phi = np.exp(-np.log(2.0) / half_life)
    innovations = np.empty(n)
    # Start in the stationary distribution unless told otherwise.
    innovations[0] = sigma_stationary * rng.standard_normal() if s0 is None else s0 - mu
    innovations[1:] = sigma_stationary * np.sqrt(1.0 - phi**2) * rng.standard_normal(n - 1)
    # AR(1) recursion d_t = phi d_{t-1} + e_t as a linear filter: vectorised C loop.
    return mu + lfilter([1.0], [1.0, -phi], innovations)


@dataclass(frozen=True)
class SyntheticPair:
    """A simulated cointegrated pair together with its true parameters."""

    prices: pd.DataFrame  # columns "y", "x"
    spread: pd.Series  # true stationary component: y - alpha - beta * x
    alpha: float
    beta: float
    half_life: float


def simulate_cointegrated_pair(
    n: int = 1500,
    *,
    alpha: float = 10.0,
    beta: float = 1.5,
    half_life: float = 10.0,
    spread_sd: float = 1.0,
    x0: float = 100.0,
    x_daily_vol: float = 0.015,
    start: str = "2015-01-02",
    seed: int | None = None,
) -> SyntheticPair:
    """Simulate y_t = alpha + beta x_t + s_t, with x a GBM and s an OU process.

    x is I(1) (non-stationary), s is I(0), so (y, x) is cointegrated with
    cointegrating vector (1, -beta) by construction.
    """
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(0.0, x_daily_vol, n)
    log_returns[0] = 0.0
    x = x0 * np.exp(np.cumsum(log_returns))
    spread = simulate_ou(n, half_life, spread_sd, rng=rng)
    y = alpha + beta * x + spread

    index = pd.bdate_range(start, periods=n, name="date")
    prices = pd.DataFrame({"y": y, "x": x}, index=index)
    return SyntheticPair(
        prices=prices,
        spread=pd.Series(spread, index=index, name="spread"),
        alpha=alpha,
        beta=beta,
        half_life=half_life,
    )
