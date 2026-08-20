import numpy as np
from scipy.stats import norm

from ccr.instruments import EuropeanEquityOption


def _bs_reference(S, K, r, q, sigma, T, is_call=True):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if is_call:
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def test_call_at_t0_matches_black_scholes(factors_factory):
    r, q, s0, sig, K, T = 0.03, 0.01, 100.0, 0.25, 105.0, 3.0
    f = factors_factory(
        flat_rate=r, s0=s0, sigma_s=sig, div_yield=q, n_paths=1000, horizon=3.0, n_steps=12
    )
    opt = EuropeanEquityOption(strike=K, expiry=T, sigma=sig, div_yield=q, is_call=True)
    v0 = opt.mtm(f)[:, 0]  # deterministic at t0
    ref = _bs_reference(s0, K, r, q, sig, T, is_call=True)
    assert np.allclose(v0, ref, atol=1e-6)


def test_put_at_t0_matches_black_scholes(factors_factory):
    r, q, s0, sig, K, T = 0.03, 0.0, 100.0, 0.3, 95.0, 2.0
    f = factors_factory(
        flat_rate=r, s0=s0, sigma_s=sig, div_yield=q, n_paths=1000, horizon=2.0, n_steps=8
    )
    opt = EuropeanEquityOption(strike=K, expiry=T, sigma=sig, div_yield=q, is_call=False)
    v0 = opt.mtm(f)[:, 0]
    ref = _bs_reference(s0, K, r, q, sig, T, is_call=False)
    assert np.allclose(v0, ref, atol=1e-6)


def test_long_option_mtm_is_nonnegative(factors_factory):
    f = factors_factory(n_paths=5000, sigma_s=0.3, horizon=3.0, n_steps=36)
    opt = EuropeanEquityOption(strike=100.0, expiry=2.5, sigma=0.3, is_call=True, long=True)
    v = opt.mtm(f)
    assert (v >= -1e-9).all()


def test_put_call_parity_holds_across_paths(factors_factory):
    r, q, sig, K, T = 0.03, 0.0, 0.2, 100.0, 3.0
    f = factors_factory(flat_rate=r, sigma_s=sig, div_yield=q, n_paths=4000, horizon=3.0, n_steps=24)
    call = EuropeanEquityOption(K, T, sig, div_yield=q, is_call=True)
    put = EuropeanEquityOption(K, T, sig, div_yield=q, is_call=False)
    k = 6  # some node before expiry
    tau = T - f.grid.times[k]
    S = f.spot(k)
    P = f.discount_bond(k, T)
    lhs = call.mtm(f)[:, k] - put.mtm(f)[:, k]
    rhs = S * np.exp(-q * tau) - K * P
    assert np.allclose(lhs, rhs, atol=1e-6)


def test_option_settles_to_zero_after_expiry(factors_factory):
    f = factors_factory(n_paths=1000, horizon=5.0, n_steps=50)
    opt = EuropeanEquityOption(strike=100.0, expiry=2.0, sigma=0.2, is_call=True)
    v = opt.mtm(f)
    assert np.allclose(v[:, -1], 0.0)  # well after expiry
