"""Z-score mean-reversion signals.

z_t = (s_t - mean(s_{t-w+1..t})) / std(s_{t-w+1..t})

The window ends at t inclusive: everything in z_t is known at the close of
bar t. The *decision* made from z_t can only be filled later; that lag is the
execution layer's job, not this module's.

Signals are target states, not orders:
    +1 = long the spread  (long y, short beta x), entered when z <= -entry
    -1 = short the spread (short y, long beta x), entered when z >= +entry
     0 = flat
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def rolling_zscore(spread: pd.Series, window: int) -> pd.Series:
    """Causal rolling z-score. NaN until ``window`` observations are available."""
    if window < 2:
        raise ValueError("window must be >= 2")
    rolling = spread.rolling(window, min_periods=window)
    mean = rolling.mean()
    std = rolling.std(ddof=1)
    z = (spread - mean) / std
    return z.where(std > 0).rename("zscore")  # a flat window has no defined z


def threshold_positions(
    z: pd.Series,
    entry: float = 2.0,
    exit: float = 0.5,
    stop: float | None = None,
) -> pd.Series:
    """Turn a z-score path into target states with entry/exit/stop bands.

    The state machine is path-dependent (whether we are in a trade depends on
    the past), so it is an explicit O(n) loop rather than a vectorised trick.
    A plain loop over ~10^4 bars takes milliseconds; Numba is an easy upgrade
    if it ever matters.

    Rules:
    * flat -> enter when |z| >= entry (and |z| < stop, if a stop is set);
    * in a trade -> exit when z reverts inside +/- exit (or overshoots
      through it), or stop out when |z| >= stop;
    * after a stop-out, no new entries until |z| has returned inside the exit
      band. Without this lockout a broken spread gets re-entered every bar.
    * NaN z (warm-up) keeps the current state.
    """
    if not 0.0 <= exit < entry:
        raise ValueError("need 0 <= exit < entry")
    if stop is not None and stop <= entry:
        raise ValueError("stop must be greater than entry")
    stop_level = math.inf if stop is None else stop

    out = np.zeros(len(z), dtype=np.int8)
    pos = 0
    locked = False
    for i, zi in enumerate(z.to_numpy(dtype=float).tolist()):
        if math.isnan(zi):
            out[i] = pos
            continue
        if locked and abs(zi) <= exit:
            locked = False
        if pos == 0:
            if not locked and abs(zi) < stop_level:
                if zi >= entry:
                    pos = -1
                elif zi <= -entry:
                    pos = 1
        elif pos == 1:
            if zi <= -stop_level:
                pos, locked = 0, True
            elif zi >= -exit:
                pos = 0
        else:  # pos == -1
            if zi >= stop_level:
                pos, locked = 0, True
            elif zi <= exit:
                pos = 0
        out[i] = pos
    return pd.Series(out, index=z.index, name="signal")
