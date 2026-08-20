"""Hull-White one-factor short-rate model (exact simulation, analytic bonds).

The short rate follows ``dr = (theta(t) - a r) dt + sigma dW``. We use the
standard decomposition ``r(t) = x(t) + alpha(t)`` where ``x`` is a zero-mean
Ornstein-Uhlenbeck process and ``alpha(t)`` is the deterministic shift that makes
the model reprice today's discount curve exactly.

Zero-coupon bonds are affine: ``P(t,T) = A(t,T) exp(-B(t,T) r_t)``, giving an
analytic reprice of the whole curve at any simulated node -- the reason swaps can
be revalued along paths without nested simulation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .curves import DiscountCurve
from .time_grid import TimeGrid


@dataclass
class RatePaths:
    """Simulated short-rate paths plus analytic bond reprice.

    Attributes
    ----------
    grid, model:
        The time grid and originating model (for bond reconstruction).
    r:
        Short rate at each grid node, shape ``(n_paths, n_points)``.
    discount_0t:
        Pathwise discount factor ``exp(-int_0^{t_k} r du)`` from t=0 to each node,
        shape ``(n_paths, n_points)``. Used to discount exposure along each path.
    """

    grid: TimeGrid
    model: "HullWhite1F"
    r: np.ndarray
    discount_0t: np.ndarray

    def short_rate(self, k: int) -> np.ndarray:
        """Short rate at grid index ``k``; shape (n_paths,)."""
        return self.r[:, k]

    def discount_bond(self, k: int, T: np.ndarray | float) -> np.ndarray:
        """``P(t_k, T)`` per path for maturities ``T >= times[k]``.

        Returns shape ``(n_paths,)`` if ``T`` is scalar, else ``(n_paths, len(T))``.
        """
        t = self.grid.times[k]
        r_t = self.r[:, k]
        return self.model.bond_price(t, T, r_t)


class HullWhite1F:
    """Hull-White 1F calibrated to an initial :class:`DiscountCurve`.

    Parameters
    ----------
    a:
        Mean-reversion speed (> 0).
    sigma:
        Short-rate volatility (> 0).
    discount:
        Today's OIS discount curve the model reprices exactly.
    """

    def __init__(self, a: float, sigma: float, discount: DiscountCurve) -> None:
        if a <= 0:
            raise ValueError("Mean reversion 'a' must be positive.")
        if sigma < 0:
            raise ValueError("Volatility 'sigma' must be non-negative.")
        self.a = float(a)
        self.sigma = float(sigma)
        self.discount = discount

    # ---- deterministic curve-fitting pieces -------------------------------
    def _B(self, t: np.ndarray | float, T: np.ndarray | float) -> np.ndarray:
        return (1.0 - np.exp(-self.a * (np.asarray(T, float) - t))) / self.a

    def alpha(self, t: np.ndarray | float) -> np.ndarray:
        """Deterministic shift ``alpha(t)`` fitting the initial forward curve."""
        t = np.asarray(t, dtype=float)
        fwd = self.discount.instantaneous_forward(t)
        return fwd + (self.sigma**2) / (2.0 * self.a**2) * (1.0 - np.exp(-self.a * t)) ** 2

    def bond_price(
        self, t: float, T: np.ndarray | float, r_t: np.ndarray
    ) -> np.ndarray:
        """Affine bond ``P(t,T) = A(t,T) exp(-B(t,T) r_t)``, vectorized over paths.

        ``r_t`` has shape (n_paths,); ``T`` scalar -> (n_paths,), ``T`` vector of
        length m -> (n_paths, m).
        """
        T_arr = np.atleast_1d(np.asarray(T, dtype=float))
        B = (1.0 - np.exp(-self.a * (T_arr - t))) / self.a  # (m,)
        pm_T = self.discount.discount(T_arr)  # (m,)
        pm_t = self.discount.discount(t)  # scalar
        f_t = self.discount.instantaneous_forward(t)  # scalar
        var_term = (self.sigma**2) / (4.0 * self.a) * (1.0 - np.exp(-2.0 * self.a * t))
        lnA = np.log(pm_T / pm_t) + B * f_t - var_term * B**2  # (m,)
        # (n_paths, 1) x (m,) broadcast -> (n_paths, m)
        logP = lnA[None, :] - B[None, :] * r_t[:, None]
        out = np.exp(logP)
        return out[:, 0] if np.isscalar(T) or np.ndim(T) == 0 else out

    # ---- simulation --------------------------------------------------------
    def simulate(self, grid: TimeGrid, shocks: np.ndarray) -> RatePaths:
        """Evolve the short rate on ``grid`` using rate-factor ``shocks``.

        ``shocks`` are correlated N(0,1) of shape ``(n_paths, n_steps)`` (the rate
        column of the correlated driver output).
        """
        n_paths = shocks.shape[0]
        n_steps = grid.n_steps
        if shocks.shape[1] != n_steps:
            raise ValueError("shocks second dim must equal grid.n_steps.")

        a, sigma = self.a, self.sigma
        dt = grid.dt  # (n_steps,)
        # Exact OU transition coefficients per step.
        e = np.exp(-a * dt)  # (n_steps,)
        std = sigma * np.sqrt((1.0 - np.exp(-2.0 * a * dt)) / (2.0 * a))  # (n_steps,)

        x = np.zeros((n_paths, grid.n_points))
        for k in range(n_steps):
            x[:, k + 1] = x[:, k] * e[k] + std[k] * shocks[:, k]

        alpha = self.alpha(grid.times)  # (n_points,)
        r = x + alpha[None, :]

        # Pathwise discount factor via trapezoidal integration of r.
        integ = np.zeros((n_paths, grid.n_points))
        incr = 0.5 * (r[:, 1:] + r[:, :-1]) * dt[None, :]
        integ[:, 1:] = np.cumsum(incr, axis=1)
        discount_0t = np.exp(-integ)

        return RatePaths(grid=grid, model=self, r=r, discount_0t=discount_0t)
