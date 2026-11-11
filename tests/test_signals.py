import numpy as np
import pandas as pd
import pytest

from statarb.signals import rolling_zscore, threshold_positions


def _z(values):
    return pd.Series(values, index=pd.RangeIndex(len(values)), dtype=float)


def test_entry_hold_exit_and_stop_band_entry_block():
    z = _z([0, 2.1, 1.5, 0.4, -2.5, -1.0, -0.3, 4.5, 3.0, 0.0])
    got = threshold_positions(z, entry=2, exit=0.5, stop=4).tolist()
    #       flat short hold exit long hold exit  >stop: no entry, then short, exit
    assert got == [0, -1, -1, 0, 1, 1, 0, 0, -1, 0]


def test_stop_out_locks_until_spread_resets():
    z = _z([2.5, 3.0, 4.2, 3.5, 2.5, 0.3, 2.2])
    got = threshold_positions(z, entry=2, exit=0.5, stop=4).tolist()
    # stopped at 4.2; 3.5 and 2.5 must NOT re-enter; 0.3 unlocks; 2.2 re-enters
    assert got == [-1, -1, 0, 0, 0, 0, -1]


def test_nan_keeps_state():
    z = _z([np.nan, -2.5, np.nan, -0.1])
    assert threshold_positions(z, entry=2, exit=0.5).tolist() == [0, 1, 1, 0]


def test_invalid_bands_rejected():
    with pytest.raises(ValueError):
        threshold_positions(_z([0.0]), entry=1.0, exit=1.0)
    with pytest.raises(ValueError):
        threshold_positions(_z([0.0]), entry=2.0, exit=0.5, stop=2.0)


def test_rolling_zscore_matches_definition():
    s = pd.Series(np.random.default_rng(0).standard_normal(50))
    z = rolling_zscore(s, 10)
    window = s.iloc[30:40]
    assert z.iloc[39] == pytest.approx((s.iloc[39] - window.mean()) / window.std(ddof=1))
    assert z.iloc[:9].isna().all()


def test_rolling_zscore_is_causal():
    """Changing the future must not change the past."""
    s = pd.Series(np.random.default_rng(1).standard_normal(200))
    perturbed = s.copy()
    perturbed.iloc[150:] += 100.0
    pd.testing.assert_series_equal(rolling_zscore(s, 20).iloc[:150], rolling_zscore(perturbed, 20).iloc[:150])
