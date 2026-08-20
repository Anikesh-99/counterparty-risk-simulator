"""Typed scenario configuration -- the single source of truth for a run.

The library, CLI, and dashboard all build a :class:`ScenarioConfig` and hand it
to :func:`ccr.engine.simulation.run`, so they can never disagree about what a
scenario means.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .metrics.cva import HazardCurve

if TYPE_CHECKING:
    from .engine.collateral import Collateral
    from .instruments.base import Instrument


@dataclass
class MarketConfig:
    """Initial market state and model parameters."""

    flat_rate: float = 0.03
    curve_points: list[tuple[float, float]] = field(default_factory=list)
    hw_a: float = 0.1
    hw_sigma: float = 0.01
    equity_s0: float = 100.0
    equity_sigma: float = 0.20
    equity_div_yield: float = 0.0
    rho_rate_equity: float = 0.2
    basis_spread: float = 0.0


@dataclass
class SimConfig:
    """Monte Carlo settings."""

    n_paths: int = 20_000
    n_steps: int = 60
    horizon: float = 5.0
    seed: int | None = 0
    pfe_level: float = 0.975


@dataclass
class ScenarioConfig:
    """A complete run: portfolio + market + sim + collateral + credit."""

    portfolio: "list[Instrument]"
    market: MarketConfig = field(default_factory=MarketConfig)
    sim: SimConfig = field(default_factory=SimConfig)
    collateral: "Collateral | None" = None
    hazard: HazardCurve = field(default_factory=HazardCurve)
    name: str = "scenario"

    def __post_init__(self) -> None:
        if not self.portfolio:
            raise ValueError("Scenario needs at least one instrument.")
