import numpy as np

from ccr.config import MarketConfig, ScenarioConfig, SimConfig
from ccr.engine import run
from ccr.instruments import EuropeanEquityOption
from ccr.metrics import HazardCurve, WrongWayModel, cva_pathwise


def _equity_scenario(beta, driver="equity"):
    # Long equity call: exposure rises with the equity price.
    portfolio = [EuropeanEquityOption(100.0, 4.0, 0.30, is_call=True, long=True,
                                      notional=100_000, name="call")]
    market = MarketConfig(flat_rate=0.03, equity_s0=100.0, equity_sigma=0.30)
    sim = SimConfig(n_paths=20000, n_steps=48, horizon=4.0, seed=3, pfe_level=0.975)
    hazard = HazardCurve.from_cds_spread(200, recovery=0.4)
    scn = ScenarioConfig(portfolio=portfolio, market=market, sim=sim, hazard=hazard)
    if beta is not None:
        scn.wrong_way = WrongWayModel(hazard, beta=beta, driver=driver)
    return scn


def test_zero_beta_matches_independent_cva():
    indep = run(_equity_scenario(beta=None))
    wwr0 = run(_equity_scenario(beta=0.0))
    # beta=0 -> per-path intensity == base -> pathwise CVA equals independent CVA
    # up to survival-integration granularity (grid vs HazardCurve's fine grid).
    assert np.isclose(wwr0.cva, indep.cva, rtol=1e-3)


def test_wrong_way_increases_cva_right_way_decreases():
    indep = run(_equity_scenario(beta=None)).cva
    # Exposure grows with equity; hazard up with equity (beta>0) => wrong-way.
    wrong = run(_equity_scenario(beta=1.0)).cva
    right = run(_equity_scenario(beta=-1.0)).cva
    assert wrong > indep > right


def test_cva_pathwise_reduces_to_independent_when_survival_uniform():
    n_paths, n_pts = 5000, 11
    exposure = np.abs(np.random.default_rng(0).standard_normal((n_paths, n_pts))) * 1e5
    discount = np.ones((n_paths, n_pts))
    # identical survival across paths
    surv = np.tile(np.exp(-0.03 * np.linspace(0, 5, n_pts)), (n_paths, 1))
    lgd = 0.6
    got = cva_pathwise(exposure, discount, surv, lgd)
    # independent equivalent using mean EE
    ee = exposure.mean(axis=0)
    ee_mid = 0.5 * (ee[:-1] + ee[1:])
    pd = surv[0, :-1] - surv[0, 1:]
    expected = lgd * np.sum(ee_mid * pd)
    assert np.isclose(got, expected, rtol=1e-9)