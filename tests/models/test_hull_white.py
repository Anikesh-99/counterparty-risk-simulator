import numpy as np

from ccr.models import CorrelatedDriver, DiscountCurve, HullWhite1F, TimeGrid


def _sim(a=0.1, sigma=0.01, rate=0.03, curve=None, seed=0, n_paths=40000, horizon=5.0, n=20):
    disc = curve or DiscountCurve(flat_rate=rate)
    hw = HullWhite1F(a=a, sigma=sigma, discount=disc)
    grid = TimeGrid.regular(horizon, n)
    shocks = CorrelatedDriver(np.array([[1.0]]), seed=seed).draw(n_paths, grid.n_steps)[:, :, 0]
    return hw, grid, hw.simulate(grid, shocks)


def test_bond_at_t0_reprices_market_curve_exactly():
    # The defining Hull-White property: P(0,T) == market discount curve.
    disc = DiscountCurve(flat_rate=0.03, points=[(1.0, 0.025), (5.0, 0.035)])
    hw = HullWhite1F(a=0.1, sigma=0.015, discount=disc)
    r0 = np.array([disc.instantaneous_forward(0.0)])  # r(0) = f(0,0)
    T = np.array([0.5, 1.0, 2.0, 5.0, 10.0])
    model_bonds = hw.bond_price(0.0, T, r0)[0]
    assert np.allclose(model_bonds, disc.discount(T), atol=1e-10)


def test_short_rate_starts_at_forward_zero():
    disc = DiscountCurve(flat_rate=0.03)
    _, _, paths = _sim(curve=disc)
    assert np.allclose(paths.r[:, 0], disc.instantaneous_forward(0.0), atol=1e-12)


def test_expected_short_rate_tracks_alpha():
    # E[r_t] = alpha(t) since E[x_t] = 0.
    hw, grid, paths = _sim(n_paths=60000, seed=3)
    mean_r = paths.r.mean(axis=0)
    assert np.allclose(mean_r, hw.alpha(grid.times), atol=2e-4)


def test_pathwise_discount_unbiased_vs_market():
    # E[ exp(-int_0^T r) ] ~ P(0,T) for a flat curve.
    disc = DiscountCurve(flat_rate=0.03)
    _, grid, paths = _sim(curve=disc, n_paths=80000, seed=11, horizon=5.0, n=40)
    mc = paths.discount_0t.mean(axis=0)
    mkt = disc.discount(grid.times)
    # HW convexity makes E[D] slightly above P(0,T); tolerance covers MC + bias.
    assert np.allclose(mc, mkt, atol=3e-3)
