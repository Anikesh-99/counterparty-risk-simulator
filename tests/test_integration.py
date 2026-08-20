"""End-to-end pipeline tests: ScenarioConfig -> ExposureResult."""

import numpy as np

from ccr.config import MarketConfig, ScenarioConfig, SimConfig
from ccr.engine import CSA, run
from ccr.instruments import EuropeanEquityOption, InterestRateSwap


def _base_scenario(**sim_kwargs):
    portfolio = [
        InterestRateSwap(1e6, 0.02, 0.0, 5.0, 0.5, pay_fixed=True, name="irs"),
        EuropeanEquityOption(100.0, 3.0, 0.25, is_call=True, long=True, name="call"),
    ]
    market = MarketConfig(flat_rate=0.03, equity_s0=100.0, rho_rate_equity=0.3)
    sim = SimConfig(n_paths=8000, n_steps=40, horizon=5.0, seed=42, **sim_kwargs)
    return ScenarioConfig(portfolio=portfolio, market=market, sim=sim)


def test_end_to_end_produces_sensible_metrics():
    res = run(_base_scenario())
    assert res.times.shape == (41,)
    assert res.ee.shape == (41,)
    assert (res.ee >= -1e-6).all()
    assert (res.pfe >= res.ee - 1e-6).all()  # PFE envelope above EE
    assert res.epe > 0
    assert res.max_pfe > 0
    assert res.cva > 0
    assert res.ee[-1] < res.ee[len(res.ee) // 2]  # profile rolls down by the end


def test_collateral_lowers_cva():
    uncol = run(_base_scenario())
    col_scn = _base_scenario()
    col_scn.collateral = CSA(threshold=0.0, min_transfer_amount=0.0, mpor_years=10 / 252)
    col = run(col_scn)
    assert col.cva < uncol.cva


def test_reproducible_with_seed():
    a = run(_base_scenario())
    b = run(_base_scenario())
    assert np.allclose(a.ee, b.ee)
    assert a.cva == b.cva


def test_mc_error_shrinks_with_more_paths():
    # EPE spread across independent seeds should fall roughly as 1/sqrt(N).
    def epe_std(n, seeds):
        scn = _base_scenario()
        scn.sim.n_paths = n
        out = []
        for sd in seeds:
            scn.sim.seed = sd
            out.append(run(scn).epe)
        return np.std(out)

    seeds = list(range(6))
    std_low = epe_std(1000, seeds)
    std_high = epe_std(16000, seeds)
    # 16x more paths -> ~4x tighter; allow slack but require a clear reduction.
    assert std_high < 0.6 * std_low
