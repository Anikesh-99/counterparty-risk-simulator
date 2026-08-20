"""Simulation time grid and Margin-Period-of-Risk (MPoR) lag mapping.

Every array in the engine is indexed against a :class:`TimeGrid`. Times are year
fractions measured from today (t=0 is always the first point). The grid also
precomputes, for each step ``k``, the index of the step one MPoR *earlier* --
this is what the collateral layer uses to lag posted collateral behind exposure.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TimeGrid:
    """An ordered set of exposure dates (year fractions), starting at 0.

    Attributes
    ----------
    times:
        1-D array of year fractions, strictly increasing, with ``times[0] == 0``.
    """

    times: np.ndarray

    def __post_init__(self) -> None:
        t = np.asarray(self.times, dtype=float)
        if t.ndim != 1 or t.size < 2:
            raise ValueError("TimeGrid needs at least two points (0 and a horizon).")
        if t[0] != 0.0:
            raise ValueError("TimeGrid must start at t=0.")
        if np.any(np.diff(t) <= 0):
            raise ValueError("TimeGrid times must be strictly increasing.")
        object.__setattr__(self, "times", t)

    @classmethod
    def regular(cls, horizon: float, n_steps: int) -> "TimeGrid":
        """Uniformly spaced grid from 0 to ``horizon`` with ``n_steps`` intervals."""
        return cls(np.linspace(0.0, horizon, n_steps + 1))

    @classmethod
    def from_tenors(cls, tenors: list[float]) -> "TimeGrid":
        """Grid from an explicit list of positive tenors (0 is prepended)."""
        pts = np.concatenate([[0.0], np.asarray(sorted(tenors), dtype=float)])
        return cls(pts)

    @property
    def n_points(self) -> int:
        """Number of grid points (including t=0)."""
        return self.times.size

    @property
    def n_steps(self) -> int:
        """Number of intervals between points."""
        return self.times.size - 1

    @property
    def dt(self) -> np.ndarray:
        """Interval widths ``times[k+1] - times[k]``; shape (n_steps,)."""
        return np.diff(self.times)

    def mpor_lagged_index(self, mpor_years: float) -> np.ndarray:
        """For each grid point k, the index of the latest point <= times[k]-MPoR.

        Used by the collateral layer: collateral held at step ``k`` reflects the
        margin requirement as of ``times[k] - MPoR``. Points whose lagged time is
        before t=0 map to index 0 (no collateral could have been called yet).
        """
        if mpor_years < 0:
            raise ValueError("MPoR must be non-negative.")
        lagged_time = self.times - mpor_years
        # Nudge by a tolerance so a target landing exactly on a grid node (up to
        # float error) selects that node rather than the one before it.
        idx = np.searchsorted(self.times, lagged_time + 1e-9, side="right") - 1
        return np.clip(idx, 0, self.n_points - 1)
