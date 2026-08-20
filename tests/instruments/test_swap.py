import numpy as np

from ccr.instruments import InterestRateSwap


def test_par_swap_is_zero_value_at_t0(factors_factory):
    f = factors_factory(n_paths=2000, horizon=5.0, n_steps=20)
    swap = InterestRateSwap(
        notional=1_000_000, fixed_rate=0.05, start=0.0, maturity=5.0, freq=0.5
    )
    par = swap.par_rate(f, 0)[0]  # deterministic at t0
    swap_at_par = InterestRateSwap(
        notional=1_000_000, fixed_rate=par, start=0.0, maturity=5.0, freq=0.5
    )
    v0 = swap_at_par.mtm(f)[:, 0]
    assert np.allclose(v0, 0.0, atol=1e-6 * 1_000_000)


def test_swap_value_rolls_to_zero_at_maturity(factors_factory):
    f = factors_factory(n_paths=3000, horizon=5.0, n_steps=20)
    swap = InterestRateSwap(
        notional=1_000_000, fixed_rate=0.02, start=0.0, maturity=5.0, freq=0.5
    )
    v = swap.mtm(f)
    assert np.allclose(v[:, -1], 0.0, atol=1e-9)  # nothing left to pay at maturity


def test_payer_receiver_are_mirror_images(factors_factory):
    f = factors_factory(n_paths=3000, horizon=5.0, n_steps=20)
    payer = InterestRateSwap(1e6, 0.03, 0.0, 5.0, 0.5, pay_fixed=True)
    receiver = InterestRateSwap(1e6, 0.03, 0.0, 5.0, 0.5, pay_fixed=False)
    assert np.allclose(payer.mtm(f), -receiver.mtm(f))


def test_below_market_fixed_rate_gives_positive_payer_value(factors_factory):
    # Paying a low fixed rate while receiving float is valuable to the payer.
    f = factors_factory(flat_rate=0.04, n_paths=2000, horizon=5.0, n_steps=20)
    swap = InterestRateSwap(1e6, 0.01, 0.0, 5.0, 0.5, pay_fixed=True)
    assert swap.mtm(f)[:, 0].mean() > 0.0
