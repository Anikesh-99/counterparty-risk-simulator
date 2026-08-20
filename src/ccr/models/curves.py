"""Deterministic initial term structures.

The :class:`DiscountCurve` is *today's* OIS discount curve that Hull-White is
calibrated to reproduce. It supports a flat rate plus optional (tenor, zero-rate)
points, linearly interpolated in continuously-compounded zero space.

The forecast curve (for the swap's floating leg) is modelled as the discount
curve plus a deterministic tenor-basis spread -- the analytic dual-curve
approximation (no second stochastic factor).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class DiscountCurve:
    """Continuously-compounded zero curve: ``P(0,T) = exp(-z(T) * T)``.

    Parameters
    ----------
    flat_rate:
        Zero rate used everywhere when ``points`` is empty, and as the value at
        T=0 otherwise.
    points:
        Optional list of ``(tenor, zero_rate)`` used to build a piecewise-linear
        zero curve. Tenors need not be sorted.
    """

    flat_rate: float
    points: list[tuple[float, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.points:
            object.__setattr__(self, "_t", np.array([0.0, 100.0]))
            object.__setattr__(self, "_z", np.array([self.flat_rate, self.flat_rate]))
            return
        pts = sorted(self.points)
        t = np.array([0.0] + [p[0] for p in pts])
        z = np.array([self.flat_rate] + [p[1] for p in pts])
        if np.any(np.diff(t) <= 0):
            raise ValueError("DiscountCurve tenors must be positive and distinct.")
        object.__setattr__(self, "_t", t)
        object.__setattr__(self, "_z", z)

    def zero_rate(self, T: np.ndarray | float) -> np.ndarray:
        """Continuously-compounded zero rate at maturity ``T`` (flat-extrapolated)."""
        T = np.asarray(T, dtype=float)
        return np.interp(T, self._t, self._z)

    def discount(self, T: np.ndarray | float) -> np.ndarray:
        """Today's discount factor ``P(0, T)``."""
        T = np.asarray(T, dtype=float)
        return np.exp(-self.zero_rate(T) * T)

    def instantaneous_forward(self, T: np.ndarray | float, eps: float = 1e-5) -> np.ndarray:
        """Instantaneous forward rate ``f(0,T) = -d ln P(0,T) / dT`` (numerical).

        Hull-White uses this to fit ``theta(t)`` to the initial curve.
        """
        T = np.asarray(T, dtype=float)
        Tp = T + eps
        Tm = np.maximum(T - eps, 0.0)
        lnP_p = -self.zero_rate(Tp) * Tp
        lnP_m = -self.zero_rate(Tm) * Tm
        return -(lnP_p - lnP_m) / (Tp - Tm)


@dataclass(frozen=True)
class ForecastCurve:
    """Forecast curve = discount curve shifted by a constant basis spread.

    A deterministic ``basis_spread`` (in absolute rate, e.g. 0.001 = 10bp) added
    to the discount zero curve. Reduces to single-curve when the spread is 0.
    """

    discount: DiscountCurve
    basis_spread: float = 0.0

    def zero_rate(self, T: np.ndarray | float) -> np.ndarray:
        return self.discount.zero_rate(T) + self.basis_spread

    def discount_factor(self, T: np.ndarray | float) -> np.ndarray:
        T = np.asarray(T, dtype=float)
        return np.exp(-self.zero_rate(T) * T)

    def forward_rate(self, T0: np.ndarray | float, T1: np.ndarray | float) -> np.ndarray:
        """Simple forward rate for the accrual period ``[T0, T1]`` off the curve today."""
        T0 = np.asarray(T0, dtype=float)
        T1 = np.asarray(T1, dtype=float)
        p0 = self.discount_factor(T0)
        p1 = self.discount_factor(T1)
        tau = T1 - T0
        return (p0 / p1 - 1.0) / tau
