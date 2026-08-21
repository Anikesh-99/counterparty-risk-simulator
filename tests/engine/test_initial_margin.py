import numpy as np

from ccr.engine import CSA, NettingSet
from ccr.instruments import InterestRateSwap


def test_initial_margin_reduces_exposure_further_than_vm_only(factors_factory):
    f = factors_factory(flat_rate=0.04, n_paths=8000, horizon=5.0, n_steps=40)
    swap = InterestRateSwap(1e6, 0.02, 0.0, 5.0, 0.5)
    vm_only = NettingSet([swap], CSA(threshold=0.0, mpor_years=10 / 252)).exposure(f)
    with_im = NettingSet(
        [swap], CSA(threshold=0.0, mpor_years=10 / 252, initial_margin=True, im_quantile=0.99)
    ).exposure(f)
    assert with_im.mean() < vm_only.mean()
    assert (with_im >= -1e-9).all()


def test_higher_im_quantile_leaves_more_exposure(factors_factory):
    # A higher quantile => larger IM => less residual exposure.
    f = factors_factory(flat_rate=0.04, n_paths=8000, horizon=5.0, n_steps=40)
    swap = InterestRateSwap(1e6, 0.02, 0.0, 5.0, 0.5)
    ee_99 = NettingSet(
        [swap], CSA(threshold=0.0, mpor_years=10 / 252, initial_margin=True, im_quantile=0.99)
    ).exposure(f).mean()
    ee_90 = NettingSet(
        [swap], CSA(threshold=0.0, mpor_years=10 / 252, initial_margin=True, im_quantile=0.90)
    ).exposure(f).mean()
    assert ee_99 <= ee_90  # 99th pct IM is larger -> subtracts more -> less residual


def test_im_flag_off_is_unchanged(factors_factory):
    f = factors_factory(flat_rate=0.04, n_paths=4000, horizon=5.0, n_steps=40)
    swap = InterestRateSwap(1e6, 0.02, 0.0, 5.0, 0.5)
    a = NettingSet([swap], CSA(threshold=0.0, mpor_years=10 / 252, initial_margin=False)).exposure(f)
    b = NettingSet([swap], CSA(threshold=0.0, mpor_years=10 / 252)).exposure(f)
    assert np.allclose(a, b)