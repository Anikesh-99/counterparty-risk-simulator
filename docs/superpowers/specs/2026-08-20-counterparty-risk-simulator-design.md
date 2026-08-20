# Counterparty Risk Simulator — Design

**Date:** 2026-08-20
**Status:** Implemented (v0.1)

## Goal

A realistic, vectorized Monte Carlo counterparty credit risk (CCR) engine in
Python. Simulate correlated market risk factors, reprice a derivatives portfolio
along every path, aggregate into collateralized netting sets, and compute the
exposure and credit-valuation metrics banks use to measure and price counterparty
default risk (EE, EPE, ENE, PFE, unilateral CVA).

## Scope (agreed)

- **Instruments:** interest-rate swaps (dual-curve) and European equity options.
- **Risk factors:** Hull–White 1-factor short rate + GBM equity, correlated.
- **Metrics:** EE / EPE / ENE, PFE (percentile), unilateral CVA.
- **Netting & collateral:** netting sets with a CSA (threshold, MTA, MPoR),
  classic lagged-collateral model.
- **Stack:** Python + NumPy/SciPy/pandas. Library + Streamlit dashboard + CLI.
- **Curve:** flat + optional (tenor, zero) points; parameters user-specified.
  Bootstrapping and market calibration are documented extensions.

## Architecture

One-directional pipeline of NumPy arrays shaped `(n_paths, n_points)`:

```
config (ScenarioConfig)
  -> models/     correlated Hull-White rates + GBM equity on a TimeGrid
  -> instruments/  analytic reprice each trade at every (path, node) -> signed MtM
  -> engine/     NettingSet (sum signed MtM) + Collateral (CSA/MPoR) -> exposure cube
  -> metrics/    EE/ENE/EPE, PFE quantile, CVA -> ExposureResult
  -> dashboard/  Streamlit shell; CLI for config-file runs
```

### Layers and key units

- **models/** — `TimeGrid` (with MPoR lag mapping), `DiscountCurve`/`ForecastCurve`,
  `CorrelatedDriver` (Cholesky-coupled shocks), `HullWhite1F` (exact OU simulation,
  affine bonds `P(t,T)=A e^{-B r}`), `GBMEquity` (drift uses the simulated short
  rate), `FactorPaths` (the seam handed to instruments; exposes stochastic OIS and
  forecast bonds and the pathwise discount factor).
- **instruments/** — `Instrument` ABC (`mtm(factors) -> signed cube`, never floors),
  `InterestRateSwap` (dual-curve: OIS discount, forecast-curve forwards from
  simulated bonds), `EuropeanEquityOption` (Black–Scholes at the simulated state,
  stochastic discounting via `P(t_k, T_expiry)`).
- **engine/** — `NettingSet` (sum-before-floor), `Collateral` strategy
  (`Uncollateralized`, `CSA`); collateral held at node `k` uses the MPoR-lagged
  requirement, so collateralized exposure ≈ `max(V(t) − V(t−MPoR), 0)`. Orchestrator
  `run` / `run_detailed`.
- **metrics/** — pathwise-discounted `expected_exposure`/`expected_negative_exposure`,
  `expected_positive_exposure`, `pfe` quantile, `HazardCurve` (flat/points; CDS
  credit-triangle), `cva`, and the `ExposureResult` output dataclass.

## Key modeling decisions

- **Exact-transition sampling** for HW and GBM — no discretization bias, coarse
  grids allowed.
- **Analytic forward repricing** (HW bonds, Black–Scholes) — avoids nested Monte
  Carlo; the whole engine is `O(paths)`.
- **Pathwise discounting** — exposure scaled by each path's `exp(-∫r)` before
  averaging (correct under stochastic rates).
- **Dual-curve via deterministic basis** — forecast bonds are simulated OIS bonds
  times `exp(-s (T-t))`, keeping the floating leg stochastic without a second factor.
- **MPoR-lagged collateral** — the crux; residual exposure is the MtM move over the
  margin period of risk.

## Testing strategy

Every numerical unit is pinned to an analytic benchmark: HW reprices the input
curve at t0 to 1e-10; option value at t0 equals Black–Scholes; par swap is ~0 at
t0; `CSA(MPoR=0, threshold=∞) ≡ Uncollateralized` exactly; CVA of a flat-EE,
constant-hazard case matches the closed form; EPE Monte Carlo error shrinks with
path count. Plus CLI and dashboard smoke tests. (47 tests.)

## Assumptions & documented extensions

- Unilateral CVA assumes **exposure/default independence** (no wrong-way risk).
- In-flight swap accrual periods are dropped (small approximation on typical grids).
- Extensions: bilateral CVA/DVA, initial margin (SIMM), stochastic basis, model
  calibration to swaptions/vols, curve bootstrapping, more instruments.
