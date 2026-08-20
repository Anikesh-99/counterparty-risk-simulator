import numpy as np

from ccr.engine import CSA, NettingSet, Uncollateralized
from ccr.instruments import InterestRateSwap
from ccr.models import TimeGrid


def test_csa_with_zero_mpor_and_infinite_threshold_equals_uncollateralized(factors_factory):
    # The structural invariant: no threshold effect, no lag -> identical exposure.
    f = factors_factory(flat_rate=0.04, n_paths=4000, horizon=5.0, n_steps=40)
    swap = InterestRateSwap(1e6, 0.02, 0.0, 5.0, 0.5)
    uncol = NettingSet([swap], Uncollateralized()).exposure(f)
    csa = NettingSet([swap], CSA(threshold=np.inf, mpor_years=0.0)).exposure(f)
    assert np.allclose(uncol, csa, atol=1e-9)


def test_perfect_collateral_zero_mpor_zero_threshold_kills_exposure(factors_factory):
    # Fully collateralized with no lag -> exposure collapses to ~0.
    f = factors_factory(flat_rate=0.04, n_paths=4000, horizon=5.0, n_steps=40)
    swap = InterestRateSwap(1e6, 0.02, 0.0, 5.0, 0.5)
    csa = NettingSet([swap], CSA(threshold=0.0, mpor_years=0.0)).exposure(f)
    assert np.allclose(csa, 0.0, atol=1e-9)


def test_collateral_reduces_aggregate_exposure(factors_factory):
    # A tight CSA slashes total exposure. Note it is NOT guaranteed pointwise:
    # collateralized exposure ~ max(V(t) - V(t-MPoR), 0), the MPoR *change*, which
    # near maturity can exceed the shrinking uncollateralized *level* at a few
    # nodes. The invariant is a large aggregate reduction.
    f = factors_factory(flat_rate=0.04, n_paths=6000, horizon=5.0, n_steps=40)
    swap = InterestRateSwap(1e6, 0.02, 0.0, 5.0, 0.5)
    ee_uncol = NettingSet([swap], Uncollateralized()).exposure(f).mean(axis=0)
    ee_csa = NettingSet([swap], CSA(threshold=0.0, mpor_years=10 / 252)).exposure(f).mean(axis=0)
    assert ee_csa.sum() < 0.2 * ee_uncol.sum()


def test_higher_threshold_gives_higher_exposure(factors_factory):
    f = factors_factory(flat_rate=0.04, n_paths=6000, horizon=5.0, n_steps=40)
    swap = InterestRateSwap(1e6, 0.02, 0.0, 5.0, 0.5)
    low = NettingSet([swap], CSA(threshold=0.0, mpor_years=10 / 252)).exposure(f).mean()
    high = NettingSet([swap], CSA(threshold=50_000.0, mpor_years=10 / 252)).exposure(f).mean()
    assert high >= low


def test_positive_mpor_leaves_residual_exposure(factors_factory):
    # With a lag, even zero-threshold collateral leaves the MPoR gap move.
    f = factors_factory(flat_rate=0.04, n_paths=6000, horizon=5.0, n_steps=60)
    swap = InterestRateSwap(1e6, 0.02, 0.0, 5.0, 0.5)
    ee = NettingSet([swap], CSA(threshold=0.0, mpor_years=20 / 252)).exposure(f).mean(axis=0)
    assert ee[1:-1].sum() > 0.0  # non-trivial residual in the interior
