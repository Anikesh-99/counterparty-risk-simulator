"""End-to-end orchestrator: ScenarioConfig -> ExposureResult.

Wires the layers together: build models from config, simulate correlated factors,
reprice + net + collateralize, then reduce the exposure cube to metrics. EE/ENE
are pathwise-discounted; PFE is reported on undiscounted exposure (a risk-limit
amount). CVA integrates discounted EE against the hazard curve.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ..metrics.cva import cva as compute_cva
from ..metrics.exposure import (
    expected_exposure,
    expected_negative_exposure,
    expected_positive_exposure,
    pfe,
)
from ..metrics.results import ExposureResult
from ..models import CorrelatedDriver, DiscountCurve, GBMEquity, HullWhite1F, TimeGrid
from ..models.factors import FactorPaths
from .netting import NettingSet

if TYPE_CHECKING:
    from ..config import ScenarioConfig


def simulate_factors(cfg: "ScenarioConfig") -> FactorPaths:
    """Build models from config and simulate correlated risk-factor paths."""
    m, s = cfg.market, cfg.sim
    disc = DiscountCurve(flat_rate=m.flat_rate, points=list(m.curve_points))
    hw = HullWhite1F(a=m.hw_a, sigma=m.hw_sigma, discount=disc)
    eq = GBMEquity(s0=m.equity_s0, sigma=m.equity_sigma, div_yield=m.equity_div_yield)
    grid = TimeGrid.regular(s.horizon, s.n_steps)

    corr = np.array([[1.0, m.rho_rate_equity], [m.rho_rate_equity, 1.0]])
    shocks = CorrelatedDriver(corr, seed=s.seed).draw(s.n_paths, grid.n_steps)
    rates = hw.simulate(grid, shocks[:, :, 0])
    equity = eq.simulate(grid, shocks[:, :, 1], rates)
    return FactorPaths(grid=grid, rates=rates, equity=equity, basis_spread=m.basis_spread)


@dataclass
class RunDiagnostics:
    """Full run output including the raw cubes (for plotting/inspection)."""

    result: ExposureResult
    factors: FactorPaths
    exposure: np.ndarray  # collateralized positive exposure cube
    net_mtm: np.ndarray  # signed netted MtM cube


def run_detailed(cfg: "ScenarioConfig") -> RunDiagnostics:
    """Run the simulation and return metrics plus the underlying cubes."""
    factors = simulate_factors(cfg)
    ns = NettingSet(cfg.portfolio, cfg.collateral, name=cfg.name)

    net_mtm = ns.net_mtm(factors)
    exposure = ns.collateral.exposure(net_mtm, factors.grid)
    neg_exposure = np.minimum(net_mtm, 0.0)
    discount = factors.rates.discount_0t  # pathwise (n_paths, n_points)
    times = factors.grid.times

    ee = expected_exposure(exposure, discount)  # discounted EE
    ene = expected_negative_exposure(neg_exposure, discount)
    pfe_profile = pfe(exposure, cfg.sim.pfe_level)  # undiscounted PFE

    result = ExposureResult(
        times=times,
        ee=ee,
        ene=ene,
        pfe=pfe_profile,
        epe=expected_positive_exposure(ee, times),
        max_pfe=float(pfe_profile.max()),
        cva=compute_cva(ee, times, cfg.hazard),
        pfe_level=cfg.sim.pfe_level,
        n_paths=cfg.sim.n_paths,
    )
    return RunDiagnostics(result=result, factors=factors, exposure=exposure, net_mtm=net_mtm)


def run(cfg: "ScenarioConfig") -> ExposureResult:
    """Run the full counterparty-risk simulation and return metrics."""
    return run_detailed(cfg).result
