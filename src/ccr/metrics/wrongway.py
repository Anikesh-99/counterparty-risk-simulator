"""Wrong-way risk via a factor-tilted default intensity.

The counterparty's hazard rate is tilted by a standardized simulated driver so
that default becomes correlated with the market state (and hence with exposure):

    lambda_i(t) = lambda_base(t) * exp(beta * z_i(t)) / mean_p[ exp(beta * z(t)) ]

The per-node normalization keeps the cross-path mean intensity equal to the base
hazard, so marginal default probabilities stay calibrated -- ``beta`` injects only
the correlation. ``beta > 0`` raises hazard when the driver is high; pick the
driver and sign so that hazard rises with exposure for genuine wrong-way risk.
"""

from __future__ import annotations

import numpy as np

from ..models.factors import FactorPaths
from .cva import HazardCurve


class WrongWayModel:
    """Factor-tilted stochastic hazard producing per-path survival probabilities.

    Parameters
    ----------
    base_hazard:
        The calibrated (mean) hazard curve.
    beta:
        Tilt strength. 0 -> independence (recovers the base hazard on every path).
    driver:
        Which simulated factor drives the tilt: ``"equity"`` uses the equity
        log-return, ``"rate"`` uses the short rate. The standardized driver is
        used, so scale is irrelevant; only ``beta`` sets strength/sign.
    """

    def __init__(
        self, base_hazard: HazardCurve, beta: float, driver: str = "equity"
    ) -> None:
        if driver not in ("equity", "rate"):
            raise ValueError("driver must be 'equity' or 'rate'.")
        self.base_hazard = base_hazard
        self.beta = float(beta)
        self.driver = driver

    @property
    def lgd(self) -> float:
        return self.base_hazard.lgd

    def _driver_series(self, factors: FactorPaths) -> np.ndarray:
        """Standardized driver, shape (n_paths, n_points), zero-mean per node."""
        if self.driver == "equity":
            raw = np.log(factors.equity)
        else:
            raw = factors.rates.r
        mu = raw.mean(axis=0, keepdims=True)
        sd = raw.std(axis=0, keepdims=True)
        sd = np.where(sd < 1e-12, 1.0, sd)  # t=0 node has zero spread
        return (raw - mu) / sd

    def pathwise_survival(self, factors: FactorPaths) -> np.ndarray:
        """Per-path survival ``S_i(t_k)``, shape (n_paths, n_points)."""
        grid = factors.grid
        times = grid.times
        z = self._driver_series(factors)  # (n_paths, n_points)

        base_h = self.base_hazard.hazard(times)  # (n_points,)
        tilt = np.exp(self.beta * z)  # (n_paths, n_points)
        norm = tilt.mean(axis=0, keepdims=True)  # (1, n_points), keeps mean intensity
        lam = base_h[None, :] * tilt / norm  # (n_paths, n_points)

        # Survival via trapezoidal integration of lambda along each path.
        dt = grid.dt
        incr = 0.5 * (lam[:, 1:] + lam[:, :-1]) * dt[None, :]
        integ = np.zeros_like(lam)
        integ[:, 1:] = np.cumsum(incr, axis=1)
        return np.exp(-integ)
