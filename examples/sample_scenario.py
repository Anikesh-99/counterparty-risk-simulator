"""Example scenario: a small collateralized netting set.

Run with:  ccr examples/sample_scenario.py
"""

from ccr.config import MarketConfig, ScenarioConfig, SimConfig
from ccr.engine import CSA
from ccr.instruments import EuropeanEquityOption, InterestRateSwap
from ccr.metrics import HazardCurve

portfolio = [
    InterestRateSwap(
        notional=10_000_000,
        fixed_rate=0.025,
        start=0.0,
        maturity=5.0,
        freq=0.5,
        pay_fixed=True,
        name="IRS 5y payer",
    ),
    EuropeanEquityOption(
        strike=100.0,
        expiry=3.0,
        sigma=0.25,
        div_yield=0.0,
        is_call=True,
        long=True,
        notional=50_000,
        name="Equity call 3y",
    ),
]

scenario = ScenarioConfig(
    name="Demo counterparty (EPL Bank)",
    portfolio=portfolio,
    market=MarketConfig(
        flat_rate=0.03,
        hw_a=0.10,
        hw_sigma=0.012,
        equity_s0=100.0,
        equity_sigma=0.22,
        equity_div_yield=0.0,
        rho_rate_equity=0.30,
        basis_spread=0.0005,  # 5bp OIS/forecast tenor basis
    ),
    sim=SimConfig(n_paths=40_000, n_steps=60, horizon=5.0, seed=1, pfe_level=0.975),
    collateral=CSA(threshold=250_000, min_transfer_amount=50_000, mpor_years=10 / 252),
    hazard=HazardCurve.from_cds_spread(spread_bps=150, recovery=0.4),
)
