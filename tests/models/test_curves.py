import numpy as np

from ccr.models import DiscountCurve, ForecastCurve


def test_flat_curve_discount_matches_analytic():
    c = DiscountCurve(flat_rate=0.03)
    T = np.array([0.5, 1.0, 5.0, 10.0])
    assert np.allclose(c.discount(T), np.exp(-0.03 * T))


def test_flat_curve_instantaneous_forward_equals_flat_rate():
    c = DiscountCurve(flat_rate=0.025)
    T = np.array([0.25, 1.0, 7.0])
    assert np.allclose(c.instantaneous_forward(T), 0.025, atol=1e-6)


def test_points_curve_interpolates_zero_rates():
    c = DiscountCurve(flat_rate=0.01, points=[(1.0, 0.02), (5.0, 0.03)])
    # midpoint tenor interpolates linearly in zero space
    assert np.isclose(c.zero_rate(3.0), 0.025, atol=1e-9)
    assert np.isclose(c.discount(1.0), np.exp(-0.02 * 1.0))


def test_forecast_forward_positive_for_upward_basis():
    disc = DiscountCurve(flat_rate=0.03)
    fc = ForecastCurve(discount=disc, basis_spread=0.001)
    f = fc.forward_rate(1.0, 1.5)
    assert f > 0.03  # basis lifts the forward above the flat discount rate
