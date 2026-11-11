"""Signal generation: from a spread to a target position in {-1, 0, +1}."""

from .zscore import rolling_zscore, threshold_positions

__all__ = ["rolling_zscore", "threshold_positions"]
