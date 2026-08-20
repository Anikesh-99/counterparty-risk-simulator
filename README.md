# Counterparty Risk Simulator

A vectorized **Monte Carlo counterparty credit risk (CCR)** engine in Python. It
simulates correlated market risk factors, reprices a derivatives portfolio along
every path, and computes the exposure and credit-valuation metrics banks use to
measure and price counterparty default risk.

> ⚠️ **Status: under active development.** Architecture and scope below are the
> agreed design; implementation is in progress.

## What it does

- **Risk-factor simulation** — a **Hull–White 1-factor** short-rate model and a
  **Geometric Brownian Motion** equity process, evolved jointly under a
  correlation matrix (Cholesky-factorized shocks) over a configurable time grid.
- **Analytic repricing along paths** — **interest-rate swaps** and **European
  equity options** are revalued at every simulated date/state in closed form
  (no nested simulation), producing full mark-to-market cubes of shape
  `(paths × time steps)`.
- **Collateralized netting sets** — trades aggregate into netting sets with a
  **CSA** margin agreement (threshold, minimum transfer amount, and a
  **margin period of risk** so collateral realistically lags exposure).
- **Exposure & credit metrics**
  - Expected Exposure (**EE**), Expected Positive/Negative Exposure (**EPE/ENE**)
  - Potential Future Exposure (**PFE**) at a configurable confidence level
  - Unilateral **CVA** from a hazard-rate/CDS-implied default curve and LGD

## Tech

Python · NumPy · SciPy · pandas · Streamlit (interactive dashboard)

## Interface

- Importable library (`models/`, `instruments/`, `engine/`, `metrics/`)
- Streamlit dashboard with controls for the portfolio, confidence level, and CSA
  terms, plotting live exposure profiles, the PFE envelope, and CVA

## License

MIT
