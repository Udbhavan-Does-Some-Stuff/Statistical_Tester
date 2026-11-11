"""Data layer: loading, validating and simulating price data."""

from .loaders import align_pair, load_prices_csv, validate_prices
from .synthetic import SyntheticPair, simulate_cointegrated_pair, simulate_ou

__all__ = [
    "SyntheticPair",
    "align_pair",
    "load_prices_csv",
    "simulate_cointegrated_pair",
    "simulate_ou",
    "validate_prices",
]
