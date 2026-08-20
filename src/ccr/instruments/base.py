"""Instrument interface.

Every trade reprices to a *signed* mark-to-market cube of shape
``(n_paths, n_points)`` from the reporting party's perspective (positive = the
counterparty owes us). Instruments never floor their value at zero -- exposure
flooring happens on the netted sum in the engine layer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..models.factors import FactorPaths


class Instrument(ABC):
    """Base class for repriceable trades."""

    name: str

    @abstractmethod
    def mtm(self, factors: FactorPaths) -> np.ndarray:
        """Signed mark-to-market at every (path, grid node); shape (n_paths, n_points)."""
        raise NotImplementedError
