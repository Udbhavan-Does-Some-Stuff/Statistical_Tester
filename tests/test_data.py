import numpy as np
import pandas as pd
import pytest

from statarb.data import align_pair, load_prices_csv, simulate_cointegrated_pair, validate_prices


def _frame(values, dates=("2020-01-01", "2020-01-02", "2020-01-03")):
    return pd.DataFrame(values, index=pd.DatetimeIndex(dates))


def test_simulation_is_reproducible_and_consistent():
    a = simulate_cointegrated_pair(n=300, seed=1)
    b = simulate_cointegrated_pair(n=300, seed=1)
    pd.testing.assert_frame_equal(a.prices, b.prices)
    # y - alpha - beta x recovers the true spread exactly
    implied = a.prices["y"] - a.alpha - a.beta * a.prices["x"]
    np.testing.assert_allclose(implied, a.spread, atol=1e-10)
    assert (a.prices > 0).all().all()


def test_validate_rejects_bad_frames():
    with pytest.raises(ValueError, match="non-positive"):
        validate_prices(_frame({"a": [1.0, -1.0, 2.0]}))
    with pytest.raises(ValueError, match="sorted"):
        validate_prices(_frame({"a": [1.0, 2.0, 3.0]}, ("2020-01-03", "2020-01-01", "2020-01-02")))
    with pytest.raises(ValueError, match="duplicate"):
        validate_prices(_frame({"a": [1.0, 2.0, 3.0]}, ("2020-01-01", "2020-01-01", "2020-01-02")))
    with pytest.raises(TypeError):
        validate_prices(pd.DataFrame({"a": [1.0]}))


def test_align_pair_drops_gaps_instead_of_filling():
    prices = _frame({"a": [1.0, np.nan, 3.0], "b": [4.0, 5.0, 6.0]})
    y, x = align_pair(prices, "a", "b")
    assert list(y.index) == [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-03")]
    assert list(x) == [4.0, 6.0]


def test_load_prices_csv_roundtrip(tmp_path):
    path = tmp_path / "prices.csv"
    path.write_text("date,AAA,BBB\n2020-01-02,10,20\n2020-01-01,11,21\n")
    prices = load_prices_csv(path)
    assert prices.index.is_monotonic_increasing
    assert prices.loc["2020-01-01", "AAA"] == 11.0
