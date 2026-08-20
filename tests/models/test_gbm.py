import numpy as np

from ccr.models import CorrelatedDriver, DiscountCurve, GBMEquity, HullWhite1F, TimeGrid


def test_equity_forward_matches_riskneutral_growth():
    # Under zero vol on rates and flat curve, E[S_T] = S0 exp((r-q)T).
    r, q, s0, T = 0.03, 0.01, 100.0, 3.0
    disc = DiscountCurve(flat_rate=r)
    hw = HullWhite1F(a=0.1, sigma=0.0, discount=disc)  # deterministic rate == r
    eq = GBMEquity(s0=s0, sigma=0.2, div_yield=q)
    grid = TimeGrid.regular(T, 36)

    corr = np.array([[1.0, 0.0], [0.0, 1.0]])
    shocks = CorrelatedDriver(corr, seed=5).draw(120000, grid.n_steps)
    rates = hw.simulate(grid, shocks[:, :, 0])
    s = eq.simulate(grid, shocks[:, :, 1], rates)

    expected = s0 * np.exp((r - q) * T)
    assert abs(s[:, -1].mean() / expected - 1.0) < 0.01


def test_equity_lognormal_variance():
    r, s0, sig, T = 0.0, 50.0, 0.3, 2.0
    disc = DiscountCurve(flat_rate=r)
    hw = HullWhite1F(a=0.1, sigma=0.0, discount=disc)
    eq = GBMEquity(s0=s0, sigma=sig, div_yield=0.0)
    grid = TimeGrid.regular(T, 48)
    corr = np.eye(2)
    shocks = CorrelatedDriver(corr, seed=9).draw(150000, grid.n_steps)
    rates = hw.simulate(grid, shocks[:, :, 0])
    s = eq.simulate(grid, shocks[:, :, 1], rates)
    # Var[log S_T] = sig^2 T
    assert abs(np.log(s[:, -1]).var() - sig**2 * T) < 0.01
