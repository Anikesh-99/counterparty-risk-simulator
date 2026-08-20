import numpy as np

from ccr.engine import NettingSet
from ccr.instruments import EuropeanEquityOption, InterestRateSwap


def test_offsetting_swaps_net_to_zero_exposure(factors_factory):
    f = factors_factory(n_paths=2000, horizon=5.0, n_steps=20)
    payer = InterestRateSwap(1e6, 0.03, 0.0, 5.0, 0.5, pay_fixed=True)
    receiver = InterestRateSwap(1e6, 0.03, 0.0, 5.0, 0.5, pay_fixed=False)
    ns = NettingSet([payer, receiver])
    assert np.allclose(ns.net_mtm(f), 0.0)
    assert np.allclose(ns.exposure(f), 0.0)


def test_netting_reduces_exposure_vs_standalone(factors_factory):
    # Two anti-correlated trades net to less exposure than the sum of exposures.
    f = factors_factory(flat_rate=0.04, n_paths=5000, horizon=5.0, n_steps=20)
    payer = InterestRateSwap(1e6, 0.02, 0.0, 5.0, 0.5, pay_fixed=True)
    receiver = InterestRateSwap(1e6, 0.06, 0.0, 5.0, 0.5, pay_fixed=False)
    ns_joint = NettingSet([payer, receiver])
    ee_joint = ns_joint.exposure(f).mean(axis=0)
    ee_sep = (
        NettingSet([payer]).exposure(f).mean(axis=0)
        + NettingSet([receiver]).exposure(f).mean(axis=0)
    )
    assert (ee_joint <= ee_sep + 1e-6).all()
    assert ee_joint.sum() < ee_sep.sum()


def test_exposure_is_nonnegative(factors_factory):
    f = factors_factory(n_paths=3000, horizon=4.0, n_steps=32)
    swap = InterestRateSwap(1e6, 0.02, 0.0, 4.0, 0.5)
    opt = EuropeanEquityOption(100.0, 3.0, 0.25, is_call=True)
    ns = NettingSet([swap, opt])
    assert (ns.exposure(f) >= -1e-9).all()
