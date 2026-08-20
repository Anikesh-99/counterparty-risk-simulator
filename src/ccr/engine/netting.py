"""Netting set: aggregate signed trade MtM, then apply a collateral strategy.

Netting sums signed MtM across trades *before* flooring at zero -- offsetting
positions reduce exposure, which is the economic purpose of a netting agreement.
"""

from __future__ import annotations

import numpy as np

from ..instruments.base import Instrument
from ..models.factors import FactorPaths
from .collateral import Collateral, Uncollateralized


class NettingSet:
    """A group of trades under one legal netting agreement.

    Parameters
    ----------
    trades:
        Instruments in the set.
    collateral:
        Collateral strategy (defaults to :class:`Uncollateralized`).
    name:
        Optional label (e.g. counterparty name).
    """

    def __init__(
        self,
        trades: list[Instrument],
        collateral: Collateral | None = None,
        name: str = "netting_set",
    ) -> None:
        if not trades:
            raise ValueError("A netting set needs at least one trade.")
        self.trades = trades
        self.collateral = collateral or Uncollateralized()
        self.name = name

    def net_mtm(self, factors: FactorPaths) -> np.ndarray:
        """Signed netted MtM cube, shape ``(n_paths, n_points)``."""
        total = np.zeros((factors.n_paths, factors.grid.n_points))
        for trade in self.trades:
            total += trade.mtm(factors)
        return total

    def exposure(self, factors: FactorPaths) -> np.ndarray:
        """Collateralized positive exposure cube ``E(path, node) >= 0``."""
        net = self.net_mtm(factors)
        return self.collateral.exposure(net, factors.grid)

    def negative_exposure(self, factors: FactorPaths) -> np.ndarray:
        """Negative exposure ``min(net_mtm, 0)`` (for ENE / DVA); uncollateralized."""
        return np.minimum(self.net_mtm(factors), 0.0)
