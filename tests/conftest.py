import numpy as np
import pandas as pd
import pytest

from statarb.data import SyntheticPair, simulate_cointegrated_pair


@pytest.fixture(scope="session")
def coint_pair() -> SyntheticPair:
    # Half-life 5: Engle-Granger detects this ~94% of the time on a 252-bar
    # formation window. At half-life 10 it is only ~52%; see docs/ARCHITECTURE.md.
    return simulate_cointegrated_pair(n=1500, alpha=10.0, beta=1.5, half_life=5.0, seed=7)


@pytest.fixture(scope="session")
def independent_walks() -> pd.DataFrame:
    """Two independent geometric random walks: not cointegrated."""
    rng = np.random.default_rng(11)
    n = 1500
    index = pd.bdate_range("2015-01-02", periods=n, name="date")
    log_paths = np.cumsum(rng.normal(0.0, 0.015, size=(n, 2)), axis=0)
    return pd.DataFrame(100.0 * np.exp(log_paths), index=index, columns=["y", "x"])
