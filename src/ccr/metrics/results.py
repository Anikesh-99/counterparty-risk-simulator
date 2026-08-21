"""ExposureResult: the typed output consumed by notebooks and the dashboard."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ExposureResult:
    """Exposure profiles and headline scalars for one netting set.

    Attributes
    ----------
    times:
        Grid year fractions, shape (n_points,).
    ee, ene, pfe:
        Discounted EE, ENE, and PFE profiles, each shape (n_points,).
    epe:
        Expected Positive Exposure (time-weighted average of EE).
    max_pfe:
        Peak PFE across the grid (the risk-limit headline number).
    cva:
        Counterparty CVA (wrong-way-adjusted when a WWR model is configured).
    dva:
        Debit Valuation Adjustment (our own default benefit); 0 if unilateral.
    bcva:
        Bilateral CVA = cva - dva (the net XVA charge).
    pfe_level:
        Confidence level used for PFE (e.g. 0.975).
    n_paths:
        Number of Monte Carlo paths.
    """

    times: np.ndarray
    ee: np.ndarray
    ene: np.ndarray
    pfe: np.ndarray
    epe: float
    max_pfe: float
    cva: float
    pfe_level: float
    n_paths: int
    dva: float = 0.0
    bcva: float = 0.0

    def profile_frame(self) -> pd.DataFrame:
        """Exposure profiles as a tidy DataFrame indexed by time."""
        return pd.DataFrame(
            {"EE": self.ee, "ENE": self.ene, f"PFE_{self.pfe_level:.3f}": self.pfe},
            index=pd.Index(self.times, name="t"),
        )

    def summary(self) -> pd.Series:
        """Headline scalars as a Series."""
        return pd.Series(
            {
                "EPE": self.epe,
                "MaxPFE": self.max_pfe,
                "CVA": self.cva,
                "DVA": self.dva,
                "BCVA": self.bcva,
                "PFE_level": self.pfe_level,
                "n_paths": self.n_paths,
            }
        )

    def __repr__(self) -> str:
        return (
            f"ExposureResult(EPE={self.epe:,.0f}, MaxPFE={self.max_pfe:,.0f}, "
            f"CVA={self.cva:,.0f}, DVA={self.dva:,.0f}, BCVA={self.bcva:,.0f}, "
            f"PFE@{self.pfe_level:.1%}, paths={self.n_paths:,})"
        )
