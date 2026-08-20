"""Collateral (CSA) strategies mapping netted MtM to collateralized exposure.

A margin agreement posts/returns collateral to chase the net MtM, but with a lag:
collateral held at node ``k`` reflects the requirement as of ``k - MPoR`` (the
margin period of risk). Because exposure is measured *now* while collateral is
set *then*, collateralized exposure is the adverse move over the MPoR window --
small but non-zero, and the residual risk a CVA desk charges for.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..models.time_grid import TimeGrid


class Collateral(ABC):
    """Maps a netted MtM cube to a (positive) collateralized exposure cube."""

    @abstractmethod
    def exposure(self, net_mtm: np.ndarray, grid: TimeGrid) -> np.ndarray:
        raise NotImplementedError


class Uncollateralized(Collateral):
    """No margin agreement: exposure is simply ``max(net_mtm, 0)``."""

    def exposure(self, net_mtm: np.ndarray, grid: TimeGrid) -> np.ndarray:
        return np.maximum(net_mtm, 0.0)


class CSA(Collateral):
    """Variation-margin CSA with threshold, minimum transfer amount, and MPoR.

    Parameters
    ----------
    threshold:
        Unsecured amount ``H`` below which no collateral is called (use ``np.inf``
        to disable collateral entirely). Applied symmetrically.
    min_transfer_amount:
        Calls smaller than the MTA do not occur.
    mpor_years:
        Margin period of risk in years (e.g. 10/252 for 10 business days).

    Notes
    -----
    The required collateral against a net MtM ``V`` is ``max(V - H, 0)`` when we
    are in the money and ``min(V + H, 0)`` when out of the money; the MTA gates
    small *changes*. Collateral held at node ``k`` is the required amount computed
    at the MPoR-lagged node, so exposure ``= max(V_now - C_lagged, 0)``.
    """

    def __init__(
        self,
        threshold: float = 0.0,
        min_transfer_amount: float = 0.0,
        mpor_years: float = 10.0 / 252.0,
    ) -> None:
        if threshold < 0:
            raise ValueError("threshold must be non-negative.")
        if min_transfer_amount < 0:
            raise ValueError("min_transfer_amount must be non-negative.")
        if mpor_years < 0:
            raise ValueError("mpor_years must be non-negative.")
        self.threshold = float(threshold)
        self.mta = float(min_transfer_amount)
        self.mpor_years = float(mpor_years)

    def _required(self, v: np.ndarray) -> np.ndarray:
        """Collateral the counterparty must have posted to us given net MtM ``v``.

        Positive = they posted to us (we are in the money beyond the threshold);
        negative = we posted to them.
        """
        h = self.threshold
        return np.where(v > h, v - h, np.where(v < -h, v + h, 0.0))

    def exposure(self, net_mtm: np.ndarray, grid: TimeGrid) -> np.ndarray:
        lag = grid.mpor_lagged_index(self.mpor_years)  # (n_points,)
        required = self._required(net_mtm)  # (n_paths, n_points)

        # Held collateral, applying the MTA to gate small transfers between the
        # last (lagged) settlement and now. With MTA=0 this is just required[lag].
        held = required[:, lag].copy()
        if self.mta > 0:
            # Only move collateral when the required change exceeds the MTA.
            target = required[:, lag]
            # previous held tracked forward across nodes
            prev = np.zeros(net_mtm.shape[0])
            for k in range(grid.n_points):
                change = target[:, k] - prev
                move = np.where(np.abs(change) >= self.mta, change, 0.0)
                prev = prev + move
                held[:, k] = prev

        return np.maximum(net_mtm - held, 0.0)
