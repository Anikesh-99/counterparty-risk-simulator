import numpy as np

from ccr.metrics import HazardCurve, bilateral_cva, cva


def test_bilateral_reduces_to_unilateral_without_own_hazard():
    times = np.linspace(0, 5, 61)
    ee = np.linspace(0, 100_000, 61)
    ene = -np.linspace(0, 80_000, 61)
    h = HazardCurve(recovery=0.4, flat_hazard=0.03)
    c, d, b = bilateral_cva(ee, ene, times, h)
    assert d == 0.0
    assert np.isclose(b, c)
    assert np.isclose(c, cva(ee, times, h))  # matches the unilateral function


def test_dva_positive_and_bcva_is_cva_minus_dva():
    times = np.linspace(0, 5, 61)
    ee = np.full(61, 100_000.0)
    ene = np.full(61, -100_000.0)
    cpty = HazardCurve(recovery=0.4, flat_hazard=0.03)
    own = HazardCurve(recovery=0.4, flat_hazard=0.02)
    c, d, b = bilateral_cva(ee, ene, times, cpty, own)
    assert c > 0 and d > 0
    assert np.isclose(b, c - d)


def test_symmetric_credit_and_exposure_gives_near_zero_bcva():
    # Same hazard both sides and symmetric EE/ENE -> BCVA ~ 0.
    times = np.linspace(0, 5, 121)
    ee = np.full(121, 50_000.0)
    ene = np.full(121, -50_000.0)
    h = HazardCurve(recovery=0.4, flat_hazard=0.025)
    c, d, b = bilateral_cva(ee, ene, times, h, h)
    assert abs(b) < 1e-6 * c