import numpy as np

from ccr.metrics import (
    HazardCurve,
    cva,
    expected_exposure,
    expected_positive_exposure,
    pfe,
)


def test_ee_and_pfe_match_numpy_on_toy_cube():
    rng = np.random.default_rng(0)
    cube = rng.random((10000, 5)) * 100.0
    ee = expected_exposure(cube)
    assert np.allclose(ee, cube.mean(axis=0))
    p = pfe(cube, 0.95)
    assert np.allclose(p, np.quantile(cube, 0.95, axis=0))


def test_pfe_dominates_ee():
    rng = np.random.default_rng(1)
    cube = np.abs(rng.standard_normal((20000, 8)))
    assert (pfe(cube, 0.975) >= expected_exposure(cube)).all()


def test_epe_of_constant_profile_is_that_constant():
    times = np.linspace(0, 5, 11)
    ee = np.full(11, 42.0)
    assert np.isclose(expected_positive_exposure(ee, times), 42.0)


def test_survival_probability_flat_hazard():
    h = HazardCurve(recovery=0.4, flat_hazard=0.02)
    assert np.isclose(h.survival(5.0)[0], np.exp(-0.02 * 5.0), atol=1e-4)
    assert np.isclose(h.lgd, 0.6)


def test_cva_matches_closed_form_for_flat_ee_and_hazard():
    # Analytic control: constant discounted EE, flat hazard.
    times = np.linspace(0, 5, 61)
    ee_const = 100_000.0
    ee = np.full_like(times, ee_const)
    h = HazardCurve(recovery=0.4, flat_hazard=0.03)
    # CVA = LGD * EE * (1 - S(T)) since sum of marginal PDs telescopes to 1 - S(T).
    expected = h.lgd * ee_const * (1.0 - h.survival(times[-1])[0])
    assert np.isclose(cva(ee, times, h), expected, rtol=1e-3)


def test_cds_spread_credit_triangle():
    h = HazardCurve.from_cds_spread(spread_bps=150, recovery=0.4)
    assert np.isclose(h.flat_hazard, 0.015 / 0.6, atol=1e-9)
