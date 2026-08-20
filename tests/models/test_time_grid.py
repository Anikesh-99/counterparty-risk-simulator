import numpy as np
import pytest

from ccr.models import TimeGrid


def test_regular_grid_shape_and_dt():
    g = TimeGrid.regular(5.0, 10)
    assert g.n_points == 11
    assert g.n_steps == 10
    assert np.allclose(g.dt, 0.5)


def test_rejects_grid_not_starting_at_zero():
    with pytest.raises(ValueError):
        TimeGrid(np.array([0.5, 1.0, 2.0]))


def test_mpor_lag_maps_to_earlier_index():
    g = TimeGrid.regular(1.0, 10)  # step = 0.1y
    mpor = 0.2  # two steps back
    idx = g.mpor_lagged_index(mpor)
    # at k>=2 the lagged index should be k-2; early points clamp at 0
    assert idx[0] == 0 and idx[1] == 0
    assert idx[5] == 3
    assert idx[10] == 8


def test_mpor_zero_is_identity():
    g = TimeGrid.regular(2.0, 8)
    idx = g.mpor_lagged_index(0.0)
    assert np.array_equal(idx, np.arange(g.n_points))
