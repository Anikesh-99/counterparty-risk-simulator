"""Geometric Brownian Motion equity process with stochastic short-rate drift.

The equity spot follows ``dS = (r_t - q) S dt + sigma_S S dW``. Crucially the
drift uses the *simulated* Hull-White short rate ``r_t`` (not a constant), which
couples equity to rates through the drift in addition to the correlated shocks.
Log-Euler is exact for GBM given the rate path held constant across a step.
"""

from __future__ import annotations

import numpy as np

from .hull_white import RatePaths
from .time_grid import TimeGrid


class GBMEquity:
    """Equity spot process driven by GBM under the simulated rate.

    Parameters
    ----------
    s0:
        Initial spot.
    sigma:
        Lognormal volatility (> 0).
    div_yield:
        Continuous dividend yield ``q``.
    """

    def __init__(self, s0: float, sigma: float, div_yield: float = 0.0) -> None:
        if s0 <= 0:
            raise ValueError("Initial spot must be positive.")
        if sigma < 0:
            raise ValueError("Volatility must be non-negative.")
        self.s0 = float(s0)
        self.sigma = float(sigma)
        self.div_yield = float(div_yield)

    def simulate(
        self, grid: TimeGrid, shocks: np.ndarray, rates: RatePaths
    ) -> np.ndarray:
        """Return spot paths of shape ``(n_paths, n_points)``.

        ``shocks`` are correlated N(0,1) of shape ``(n_paths, n_steps)`` (equity
        column). ``rates`` supplies the short rate used in the drift each step.
        """
        n_paths = shocks.shape[0]
        n_steps = grid.n_steps
        if shocks.shape[1] != n_steps:
            raise ValueError("shocks second dim must equal grid.n_steps.")

        dt = grid.dt
        sig = self.sigma
        s = np.empty((n_paths, grid.n_points))
        s[:, 0] = self.s0
        logs = np.log(self.s0) * np.ones(n_paths)
        for k in range(n_steps):
            r_k = rates.short_rate(k)  # (n_paths,) rate at step start
            drift = (r_k - self.div_yield - 0.5 * sig**2) * dt[k]
            diffusion = sig * np.sqrt(dt[k]) * shocks[:, k]
            logs = logs + drift + diffusion
            s[:, k + 1] = np.exp(logs)
        return s
