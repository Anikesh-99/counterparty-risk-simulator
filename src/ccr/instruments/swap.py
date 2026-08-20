"""Vanilla fixed-for-floating interest-rate swap (dual-curve).

Repriced analytically at every simulated node: the fixed leg discounts on the
OIS curve, the floating leg sums forward rates (off the simulated forecast curve)
each discounted on OIS. Only periods not yet started at the valuation node are
included, so the swap amortizes to zero at maturity -- the classic exposure hump.

Simplification: the in-flight accrual period (one straddling the valuation node)
is dropped rather than modelled with a stored fixing. On a typical exposure grid
this is a small, well-understood approximation.
"""

from __future__ import annotations

import numpy as np

from ..models.factors import FactorPaths
from .base import Instrument


class InterestRateSwap(Instrument):
    """Fixed-for-floating IRS.

    Parameters
    ----------
    notional:
        Trade notional.
    fixed_rate:
        Fixed coupon ``K`` (annualised).
    start, maturity:
        Accrual start and end as year fractions from today (start may be 0).
    freq:
        Payment/accrual period length in years (e.g. 0.5 for semiannual).
    pay_fixed:
        True -> we pay fixed / receive float (payer); False -> receiver.
    name:
        Optional label.
    """

    def __init__(
        self,
        notional: float,
        fixed_rate: float,
        start: float,
        maturity: float,
        freq: float = 0.5,
        pay_fixed: bool = True,
        name: str = "swap",
    ) -> None:
        if maturity <= start:
            raise ValueError("maturity must exceed start.")
        if freq <= 0:
            raise ValueError("freq must be positive.")
        self.notional = float(notional)
        self.fixed_rate = float(fixed_rate)
        self.start = float(start)
        self.maturity = float(maturity)
        self.freq = float(freq)
        self.pay_fixed = bool(pay_fixed)
        self.name = name
        self.sign = 1.0 if pay_fixed else -1.0
        self._periods = self._build_schedule()

    def _build_schedule(self) -> list[tuple[float, float, float]]:
        """Return accrual periods as ``(accrual_start, payment_date, year_fraction)``."""
        n = max(1, int(round((self.maturity - self.start) / self.freq)))
        edges = np.linspace(self.start, self.maturity, n + 1)
        return [(edges[i], edges[i + 1], edges[i + 1] - edges[i]) for i in range(n)]

    def mtm(self, factors: FactorPaths) -> np.ndarray:
        grid = factors.grid
        n_paths = factors.n_paths
        V = np.zeros((n_paths, grid.n_points))
        eps = 1e-9
        for k in range(grid.n_points):
            t = grid.times[k]
            fixed_acc = np.zeros(n_paths)  # sum tau_i * P(t, T_i)
            float_acc = np.zeros(n_paths)  # sum tau_i * fwd_i * P(t, T_i)
            for t0, t1, tau in self._periods:
                if t1 <= t + eps:  # already paid
                    continue
                if t0 < t - eps:  # in-flight accrual: dropped (approximation)
                    continue
                p = factors.discount_bond(k, t1)  # OIS discount, (n_paths,)
                fwd = factors.forecast_forward(k, t0, t1)  # (n_paths,)
                fixed_acc += tau * p
                float_acc += tau * fwd * p
            V[:, k] = self.notional * (float_acc - self.fixed_rate * fixed_acc)
        return self.sign * V

    def par_rate(self, factors: FactorPaths, k: int = 0) -> np.ndarray:
        """Par swap rate implied at node ``k`` (useful for calibration/tests)."""
        t = factors.grid.times[k]
        eps = 1e-9
        annuity = np.zeros(factors.n_paths)
        float_acc = np.zeros(factors.n_paths)
        for t0, t1, tau in self._periods:
            if t1 <= t + eps or t0 < t - eps:
                continue
            p = factors.discount_bond(k, t1)
            annuity += tau * p
            float_acc += tau * factors.forecast_forward(k, t0, t1) * p
        return float_acc / annuity
