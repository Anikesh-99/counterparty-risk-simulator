"""European equity option, repriced along paths with the Black-Scholes formula.

At each node the option is valued from the *simulated* state: spot from the GBM
path and discounting from the simulated OIS bond ``P(t_k, T_expiry)`` (stochastic
rates). After expiry the option has settled and contributes zero MtM; at the
expiry node its value is the (undiscounted) intrinsic payoff.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from ..models.factors import FactorPaths
from .base import Instrument


class EuropeanEquityOption(Instrument):
    """European call or put on the simulated equity.

    Parameters
    ----------
    strike, expiry:
        Option strike and expiry (year fraction from today).
    sigma:
        Black-Scholes volatility used for repricing.
    div_yield:
        Continuous dividend yield of the underlying.
    is_call:
        True for a call, False for a put.
    long:
        True if we hold the option (MtM >= 0); False if we wrote it (MtM <= 0).
    notional:
        Number of contracts / multiplier.
    name:
        Optional label.
    """

    def __init__(
        self,
        strike: float,
        expiry: float,
        sigma: float,
        div_yield: float = 0.0,
        is_call: bool = True,
        long: bool = True,
        notional: float = 1.0,
        name: str = "equity_option",
    ) -> None:
        if expiry <= 0:
            raise ValueError("expiry must be positive.")
        if sigma <= 0:
            raise ValueError("sigma must be positive.")
        self.strike = float(strike)
        self.expiry = float(expiry)
        self.sigma = float(sigma)
        self.div_yield = float(div_yield)
        self.is_call = bool(is_call)
        self.sign = 1.0 if long else -1.0
        self.notional = float(notional)
        self.name = name

    def _bs_price(self, S: np.ndarray, P: np.ndarray, tau: float) -> np.ndarray:
        """Black-Scholes value using discount bond ``P`` = P(t, expiry) and time ``tau``."""
        # Forward price of the equity: F = S e^{-q tau} / P(t,T).
        fwd = S * np.exp(-self.div_yield * tau) / P
        vol_sqrt = self.sigma * np.sqrt(tau)
        d1 = (np.log(fwd / self.strike) + 0.5 * vol_sqrt**2) / vol_sqrt
        d2 = d1 - vol_sqrt
        if self.is_call:
            undisc = fwd * norm.cdf(d1) - self.strike * norm.cdf(d2)
        else:
            undisc = self.strike * norm.cdf(-d2) - fwd * norm.cdf(-d1)
        return P * undisc

    def _intrinsic(self, S: np.ndarray) -> np.ndarray:
        if self.is_call:
            return np.maximum(S - self.strike, 0.0)
        return np.maximum(self.strike - S, 0.0)

    def mtm(self, factors: FactorPaths) -> np.ndarray:
        grid = factors.grid
        n_paths = factors.n_paths
        V = np.zeros((n_paths, grid.n_points))
        eps = 1e-9
        for k in range(grid.n_points):
            t = grid.times[k]
            tau = self.expiry - t
            S = factors.spot(k)
            if tau > eps:
                P = factors.discount_bond(k, self.expiry)
                V[:, k] = self._bs_price(S, P, tau)
            elif tau > -eps:  # at expiry: settle to intrinsic
                V[:, k] = self._intrinsic(S)
            else:  # after expiry: option gone
                V[:, k] = 0.0
        return self.sign * self.notional * V
