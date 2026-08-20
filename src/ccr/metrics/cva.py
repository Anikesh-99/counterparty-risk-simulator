"""Unilateral CVA and the counterparty hazard-rate curve.

CVA is the market price of counterparty default risk: the discounted expected
exposure integrated against the counterparty's marginal default probabilities.

    CVA = LGD * sum_k 0.5*(EE*_{k-1} + EE*_k) * [S(t_{k-1}) - S(t_k)]

where EE* is *discounted* expected exposure, S is the survival probability, and
LGD = 1 - recovery. This assumes independence between exposure and default (no
wrong-way risk) -- stated explicitly because it is the natural next extension.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class HazardCurve:
    """Piecewise-constant hazard-rate (default-intensity) curve.

    Parameters
    ----------
    recovery:
        Recovery rate ``R`` in [0, 1); LGD = 1 - R.
    flat_hazard:
        Constant hazard used when ``points`` is empty.
    points:
        Optional ``(tenor, hazard_rate)`` pairs for a term structure.
    """

    recovery: float = 0.4
    flat_hazard: float = 0.02
    points: list[tuple[float, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not 0.0 <= self.recovery < 1.0:
            raise ValueError("recovery must be in [0, 1).")
        if not self.points:
            object.__setattr__(self, "_t", np.array([0.0, 100.0]))
            object.__setattr__(self, "_h", np.array([self.flat_hazard, self.flat_hazard]))
            return
        pts = sorted(self.points)
        object.__setattr__(self, "_t", np.array([0.0] + [p[0] for p in pts]))
        object.__setattr__(self, "_h", np.array([self.flat_hazard] + [p[1] for p in pts]))

    @classmethod
    def from_cds_spread(cls, spread_bps: float, recovery: float = 0.4) -> "HazardCurve":
        """Flat hazard from a CDS spread via the credit triangle ``h ~= s / LGD``."""
        lgd = 1.0 - recovery
        return cls(recovery=recovery, flat_hazard=(spread_bps * 1e-4) / lgd)

    @property
    def lgd(self) -> float:
        return 1.0 - self.recovery

    def hazard(self, t: np.ndarray | float) -> np.ndarray:
        return np.interp(np.asarray(t, dtype=float), self._t, self._h)

    def survival(self, t: np.ndarray | float) -> np.ndarray:
        """Survival probability ``S(t) = exp(-int_0^t h du)`` (trapezoidal)."""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        # Integrate the piecewise curve up to each t on a fine internal grid.
        out = np.empty_like(t)
        for i, ti in enumerate(t):
            if ti <= 0:
                out[i] = 1.0
                continue
            grid = np.linspace(0.0, ti, 200)
            out[i] = np.exp(-np.trapezoid(self.hazard(grid), grid))
        return out

    def marginal_pd(self, times: np.ndarray) -> np.ndarray:
        """Marginal default probs ``S(t_{k-1}) - S(t_k)`` per interval; len n_steps."""
        s = self.survival(times)
        return s[:-1] - s[1:]


def cva(ee_discounted: np.ndarray, times: np.ndarray, hazard: HazardCurve) -> float:
    """Unilateral CVA from a discounted EE profile and a hazard curve."""
    times = np.asarray(times, dtype=float)
    pd = hazard.marginal_pd(times)  # (n_steps,)
    ee_mid = 0.5 * (ee_discounted[:-1] + ee_discounted[1:])  # (n_steps,)
    return float(hazard.lgd * np.sum(ee_mid * pd))
