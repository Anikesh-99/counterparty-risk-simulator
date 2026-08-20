import numpy as np
import pytest

from ccr.models import CorrelatedDriver


def test_shocks_have_target_correlation():
    rho = 0.4
    corr = np.array([[1.0, rho], [rho, 1.0]])
    drv = CorrelatedDriver(corr, seed=7)
    shocks = drv.draw(n_paths=200_000, n_steps=1)
    x = shocks[:, 0, 0]
    y = shocks[:, 0, 1]
    est = np.corrcoef(x, y)[0, 1]
    assert abs(est - rho) < 0.01
    assert abs(x.std() - 1.0) < 0.01
    assert abs(y.std() - 1.0) < 0.01


def test_reproducible_with_seed():
    corr = np.array([[1.0, 0.2], [0.2, 1.0]])
    a = CorrelatedDriver(corr, seed=1).draw(10, 3)
    b = CorrelatedDriver(corr, seed=1).draw(10, 3)
    assert np.array_equal(a, b)


def test_rejects_non_psd_matrix():
    bad = np.array([[1.0, 1.5], [1.5, 1.0]])  # |rho| > 1 -> not PSD
    with pytest.raises(ValueError):
        CorrelatedDriver(bad)
