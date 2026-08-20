"""Shared test fixtures: a helper to build simulated FactorPaths."""

from __future__ import annotations

import numpy as np
import pytest

from ccr.models import (
    CorrelatedDriver,
    DiscountCurve,
    GBMEquity,
    HullWhite1F,
    TimeGrid,
)
from ccr.models.factors import FactorPaths


def build_factors(
    *,
    flat_rate: float = 0.03,
    curve_points=None,
    a: float = 0.1,
    sigma_r: float = 0.01,
    s0: float = 100.0,
    sigma_s: float = 0.2,
    div_yield: float = 0.0,
    rho: float = 0.0,
    basis_spread: float = 0.0,
    horizon: float = 5.0,
    n_steps: int = 60,
    n_paths: int = 20000,
    seed: int = 0,
) -> FactorPaths:
    disc = DiscountCurve(flat_rate=flat_rate, points=curve_points or [])
    hw = HullWhite1F(a=a, sigma=sigma_r, discount=disc)
    eq = GBMEquity(s0=s0, sigma=sigma_s, div_yield=div_yield)
    grid = TimeGrid.regular(horizon, n_steps)
    corr = np.array([[1.0, rho], [rho, 1.0]])
    shocks = CorrelatedDriver(corr, seed=seed).draw(n_paths, grid.n_steps)
    rates = hw.simulate(grid, shocks[:, :, 0])
    equity = eq.simulate(grid, shocks[:, :, 1], rates)
    return FactorPaths(grid=grid, rates=rates, equity=equity, basis_spread=basis_spread)


@pytest.fixture
def factors_factory():
    return build_factors
