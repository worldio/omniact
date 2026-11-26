"""Validation metric aggregators.

``RunningStats`` accumulates per-batch values and exposes ``mean``, ``std``, and ``se``
(standard error of the mean = ``std / sqrt(n)``).  When each ``update`` call
passes a per-batch loss averaged over equal-size batches, the resulting ``se``
equals ``sigma_sample / sqrt(n_total_samples)`` by the central-limit theorem,
so it is the standard error of the overall validation mean regardless of
whether the caller tracks samples or batches.
"""

from __future__ import annotations

import math


class RunningStats:
    """Running mean and standard error in O(1) space."""

    __slots__ = ("n", "_sum", "_sum_sq")

    def __init__(self) -> None:
        self.n = 0
        self._sum = 0.0
        self._sum_sq = 0.0

    def update(self, value: float) -> None:
        self.n += 1
        self._sum += value
        self._sum_sq += value * value

    @property
    def mean(self) -> float:
        return self._sum / self.n if self.n > 0 else 0.0

    @property
    def std(self) -> float:
        """Unbiased sample standard deviation, or ``NaN`` for fewer than two values."""
        if self.n < 2:
            return float("nan")
        mean = self._sum / self.n
        var = (self._sum_sq - self.n * mean * mean) / (self.n - 1)
        return math.sqrt(max(var, 0.0))

    @property
    def se(self) -> float:
        """Standard error of the mean, or ``NaN`` for fewer than two values."""
        if self.n < 2:
            return float("nan")
        return self.std / math.sqrt(self.n)
