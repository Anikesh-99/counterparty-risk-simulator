"""Bundle of simulated risk factors handed to the instruments layer.

Instruments never touch a stochastic process directly -- they ask ``FactorPaths``
for a discount bond, a forward, a spot, or the pathwise discount factor. This is
the seam between the models and instruments layers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .hull_white import RatePaths
from .time_grid import TimeGrid


@dataclass
class FactorPaths:
    """All simulated state on a shared :class:`TimeGrid`.

    Attributes
    ----------
    grid:
        The shared time grid.
    rates:
        Hull-White short-rate paths (with analytic bond reprice).
    equity:
        Equity spot paths, shape ``(n_paths, n_points)``.
    basis_spread:
        Deterministic tenor-basis spread (absolute rate) added to the *simulated*
        OIS curve to form the forecast curve for the swap floating leg. 0 -> single
        curve. Forecast bonds are ``P_fcst(t_k,T) = P_OIS(t_k,T) exp(-s (T-t_k))``,
        so the floating leg stays stochastic (evolves with the simulated rates).
    """

    grid: TimeGrid
    rates: RatePaths
    equity: np.ndarray
    basis_spread: float = 0.0

    @property
    def n_paths(self) -> int:
        return self.equity.shape[0]

    def spot(self, k: int) -> np.ndarray:
        """Equity spot at grid index ``k``; shape (n_paths,)."""
        return self.equity[:, k]

    def discount_bond(self, k: int, T: np.ndarray | float) -> np.ndarray:
        """Stochastic OIS bond ``P(t_k, T)`` per path (see :meth:`RatePaths.discount_bond`)."""
        return self.rates.discount_bond(k, T)

    def forecast_bond(self, k: int, T: np.ndarray | float) -> np.ndarray:
        """Forecast-curve bond ``P_OIS(t_k,T) exp(-basis (T - t_k))`` per path."""
        t = self.grid.times[k]
        p_ois = self.rates.discount_bond(k, T)
        adj = np.exp(-self.basis_spread * (np.asarray(T, float) - t))
        return p_ois * adj

    def numeraire_discount(self, k: int) -> np.ndarray:
        """Pathwise discount ``exp(-int_0^{t_k} r du)`` from 0 to node ``k``."""
        return self.rates.discount_0t[:, k]

    def forecast_forward(self, k: int, T0: float, T1: float) -> np.ndarray:
        """Simple forward for ``[T0, T1]`` observed at node ``k``, per path.

        Derived from the *simulated* forecast bonds, so the floating leg evolves
        with the rate paths (proper dual-curve behaviour under HW).
        """
        p0 = self.forecast_bond(k, T0)
        p1 = self.forecast_bond(k, T1)
        tau = T1 - T0
        return (p0 / p1 - 1.0) / tau
