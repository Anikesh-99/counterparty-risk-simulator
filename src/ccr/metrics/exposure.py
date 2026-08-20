"""Exposure metrics: EE, ENE, EPE, and PFE -- all reductions over the cube.

Every function takes an exposure cube of shape ``(n_paths, n_points)`` and reduces
across the path axis. Discounting, when supplied, is applied *pathwise* (each
path scaled by its own stochastic discount factor) before averaging.
"""

from __future__ import annotations

import numpy as np


def expected_exposure(exposure: np.ndarray, discount: np.ndarray | None = None) -> np.ndarray:
    """EE profile: mean positive exposure at each node; shape (n_points,).

    If ``discount`` (same shape as ``exposure``) is given, returns discounted EE.
    """
    e = exposure if discount is None else exposure * discount
    return e.mean(axis=0)


def expected_negative_exposure(
    neg_exposure: np.ndarray, discount: np.ndarray | None = None
) -> np.ndarray:
    """ENE profile: mean of the (non-positive) negative exposure at each node."""
    e = neg_exposure if discount is None else neg_exposure * discount
    return e.mean(axis=0)


def pfe(exposure: np.ndarray, level: float = 0.975) -> np.ndarray:
    """Potential Future Exposure: the ``level`` quantile across paths per node."""
    if not 0.0 < level < 1.0:
        raise ValueError("PFE level must be in (0, 1).")
    return np.quantile(exposure, level, axis=0)


def expected_positive_exposure(ee: np.ndarray, times: np.ndarray) -> float:
    """EPE: time-weighted average of the EE profile over the grid horizon."""
    times = np.asarray(times, dtype=float)
    horizon = times[-1] - times[0]
    if horizon <= 0:
        return float(ee[0])
    return float(np.trapezoid(ee, times) / horizon)
