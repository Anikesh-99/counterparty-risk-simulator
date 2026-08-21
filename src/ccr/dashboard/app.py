"""Streamlit dashboard for the counterparty-risk simulator.

Run with:  streamlit run src/ccr/dashboard/app.py

The dashboard is a thin shell: sidebar widgets build a ScenarioConfig, the run
button calls the engine, and the returned cubes drive the plots. No pricing or
risk logic lives here.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from ccr.config import MarketConfig, ScenarioConfig, SimConfig
from ccr.engine import CSA, Uncollateralized, run_detailed
from ccr.instruments import EuropeanEquityOption, InterestRateSwap
from ccr.metrics import HazardCurve, WrongWayModel

st.set_page_config(page_title="Counterparty Risk Simulator", layout="wide")
st.title("Monte Carlo Counterparty Risk Simulator")
st.caption("Correlated Hull-White rates + GBM equity -> EE / PFE / CVA on a collateralized netting set")


# --------------------------------------------------------------------------- #
# Sidebar: build the scenario
# --------------------------------------------------------------------------- #
def build_scenario() -> ScenarioConfig:
    sb = st.sidebar
    sb.header("Portfolio")

    portfolio = []
    if sb.checkbox("Interest-rate swap", value=True):
        with sb.expander("Swap terms", expanded=True):
            notional = st.number_input("Notional", 1e5, 1e9, 1e7, step=1e6, format="%.0f")
            fixed = st.number_input("Fixed rate", 0.0, 0.15, 0.025, step=0.005, format="%.4f")
            mat = st.slider("Maturity (y)", 1.0, 10.0, 5.0, 0.5)
            pay_fixed = st.radio("Direction", ["Pay fixed", "Receive fixed"]) == "Pay fixed"
        portfolio.append(
            InterestRateSwap(notional, fixed, 0.0, mat, 0.5, pay_fixed=pay_fixed, name="IRS")
        )

    if sb.checkbox("European equity option", value=True):
        with sb.expander("Option terms", expanded=True):
            strike = st.number_input("Strike", 1.0, 1000.0, 100.0, step=5.0)
            expiry = st.slider("Expiry (y)", 0.5, 10.0, 3.0, 0.5)
            ovol = st.slider("Option vol", 0.05, 1.0, 0.25, 0.01)
            is_call = st.radio("Type", ["Call", "Put"]) == "Call"
            long = st.radio("Position", ["Long", "Short"]) == "Long"
            oqty = st.number_input("Contracts", 1.0, 1e6, 5e4, step=1e4, format="%.0f")
        portfolio.append(
            EuropeanEquityOption(strike, expiry, ovol, is_call=is_call, long=long,
                                 notional=oqty, name="EqOption")
        )

    if not portfolio:
        st.sidebar.error("Add at least one trade.")
        st.stop()

    sb.header("Models")
    flat_rate = sb.slider("Flat zero rate", 0.0, 0.10, 0.03, 0.005)
    hw_a = sb.slider("HW mean reversion a", 0.01, 1.0, 0.10, 0.01)
    hw_sigma = sb.slider("HW vol sigma", 0.001, 0.05, 0.012, 0.001, format="%.3f")
    s0 = sb.number_input("Equity spot S0", 1.0, 1000.0, 100.0, step=5.0)
    eq_sigma = sb.slider("Equity vol", 0.05, 1.0, 0.22, 0.01)
    rho = sb.slider("Rate-equity correlation", -0.95, 0.95, 0.30, 0.05)
    basis = sb.slider("OIS/forecast basis (bp)", 0.0, 50.0, 5.0, 1.0) * 1e-4

    sb.header("Collateral (CSA)")
    collateralized = sb.checkbox("Collateralized", value=True)
    if collateralized:
        threshold = sb.number_input("Threshold", 0.0, 1e8, 2.5e5, step=5e4, format="%.0f")
        mta = sb.number_input("Min transfer amount", 0.0, 1e8, 5e4, step=1e4, format="%.0f")
        mpor_days = sb.slider("MPoR (business days)", 1, 40, 10)
        use_im = sb.checkbox("Initial margin (SIMM-lite)", value=False)
        im_q = sb.slider("IM quantile", 0.90, 0.999, 0.99, 0.005) if use_im else 0.99
        collateral = CSA(threshold, mta, mpor_days / 252.0,
                         initial_margin=use_im, im_quantile=im_q)
    else:
        collateral = Uncollateralized()

    sb.header("Credit")
    cds = sb.slider("Counterparty CDS (bp)", 10, 1000, 150, 10)
    recovery = sb.slider("Recovery", 0.0, 0.9, 0.4, 0.05)
    bilateral = sb.checkbox("Bilateral (include own default / DVA)", value=False)
    own_hazard = None
    if bilateral:
        own_cds = sb.slider("Own CDS (bp)", 10, 1000, 80, 10)
        own_hazard = HazardCurve.from_cds_spread(own_cds, recovery)

    sb.header("Wrong-way risk")
    use_wwr = sb.checkbox("Enable WWR", value=False)
    wrong_way = None
    cpty_hazard = HazardCurve.from_cds_spread(cds, recovery)
    if use_wwr:
        driver = sb.selectbox("Driver", ["equity", "rate"])
        beta = sb.slider("WWR strength beta (>0 wrong-way)", -3.0, 3.0, 1.0, 0.1)
        wrong_way = WrongWayModel(cpty_hazard, beta=beta, driver=driver)

    sb.header("Simulation")
    n_paths = sb.select_slider("Paths", [2000, 5000, 10000, 20000, 40000], value=10000)
    n_steps = sb.slider("Time steps", 12, 120, 60, 12)
    horizon = sb.slider("Horizon (y)", 1.0, 10.0, 5.0, 0.5)
    pfe_level = sb.slider("PFE level", 0.90, 0.999, 0.975, 0.005)
    seed = sb.number_input("Seed", 0, 10_000, 1, step=1)

    return ScenarioConfig(
        name="Dashboard scenario",
        portfolio=portfolio,
        market=MarketConfig(
            flat_rate=flat_rate, hw_a=hw_a, hw_sigma=hw_sigma, equity_s0=s0,
            equity_sigma=eq_sigma, rho_rate_equity=rho, basis_spread=basis,
        ),
        sim=SimConfig(n_paths=int(n_paths), n_steps=int(n_steps), horizon=horizon,
                      seed=int(seed), pfe_level=pfe_level),
        collateral=collateral,
        hazard=cpty_hazard,
        own_hazard=own_hazard,
        wrong_way=wrong_way,
    )


# --------------------------------------------------------------------------- #
# Main: run and render
# --------------------------------------------------------------------------- #
scenario = build_scenario()

diag = run_detailed(scenario)
res, factors = diag.result, diag.factors
t = res.times

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("EPE", f"{res.epe:,.0f}")
c2.metric(f"Max PFE @{res.pfe_level:.1%}", f"{res.max_pfe:,.0f}")
c3.metric("CVA", f"{res.cva:,.0f}")
c4.metric("DVA", f"{res.dva:,.0f}")
c5.metric("BCVA", f"{res.bcva:,.0f}")
c6.metric("Paths", f"{res.n_paths:,}")

# Exposure profile
fig = go.Figure()
fig.add_trace(go.Scatter(x=t, y=res.pfe, name=f"PFE {res.pfe_level:.1%}",
                         line=dict(color="#d62728", dash="dot")))
fig.add_trace(go.Scatter(x=t, y=res.ee, name="EE (discounted)", line=dict(color="#1f77b4")))
fig.add_trace(go.Scatter(x=t, y=res.ene, name="ENE (discounted)", line=dict(color="#2ca02c")))
fig.update_layout(title="Exposure profile", xaxis_title="Time (years)",
                  yaxis_title="Exposure", height=420, legend=dict(orientation="h"))
st.plotly_chart(fig, width="stretch")

left, right = st.columns(2)

# Exposure distribution at a chosen time
with left:
    st.subheader("Exposure distribution")
    k = st.slider("Time node", 1, len(t) - 1, min(len(t) // 3, len(t) - 1))
    st.caption(f"t = {t[k]:.2f}y")
    col = diag.exposure[:, k]
    hist = go.Figure(go.Histogram(x=col[col > 0], nbinsx=60, marker_color="#1f77b4"))
    hist.update_layout(height=340, xaxis_title="Exposure", yaxis_title="Paths",
                       margin=dict(t=10))
    st.plotly_chart(hist, width="stretch")

# CVA decomposition: discounted EE weighted by marginal default prob
with right:
    st.subheader("CVA contribution by time")
    pd_marg = scenario.hazard.marginal_pd(t)
    ee_mid = 0.5 * (res.ee[:-1] + res.ee[1:])
    contrib = scenario.hazard.lgd * ee_mid * pd_marg
    bar = go.Figure(go.Bar(x=t[1:], y=contrib, marker_color="#d62728"))
    bar.update_layout(height=340, xaxis_title="Time (years)",
                      yaxis_title="CVA contribution", margin=dict(t=10))
    st.plotly_chart(bar, width="stretch")

# Sample risk-factor paths
st.subheader("Sample simulated paths")
p1, p2 = st.columns(2)
idx = np.linspace(0, factors.n_paths - 1, 30).astype(int)
with p1:
    rf = go.Figure()
    for i in idx:
        rf.add_trace(go.Scatter(x=t, y=factors.rates.r[i], line=dict(width=0.6),
                                showlegend=False, opacity=0.5))
    rf.update_layout(title="Short rate", height=320, xaxis_title="t", margin=dict(t=30))
    st.plotly_chart(rf, width="stretch")
with p2:
    ef = go.Figure()
    for i in idx:
        ef.add_trace(go.Scatter(x=t, y=factors.equity[i], line=dict(width=0.6),
                                showlegend=False, opacity=0.5))
    ef.update_layout(title="Equity spot", height=320, xaxis_title="t", margin=dict(t=30))
    st.plotly_chart(ef, width="stretch")

with st.expander("Exposure profile data"):
    st.dataframe(res.profile_frame())
