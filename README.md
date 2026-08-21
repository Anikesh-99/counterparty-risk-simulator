# Counterparty Risk Simulator

A vectorized **Monte Carlo counterparty credit risk (CCR)** engine in Python. It
simulates correlated market risk factors, reprices a derivatives portfolio along
every path, aggregates trades into collateralized netting sets, and computes the
exposure and credit-valuation metrics banks use to measure and price counterparty
default risk.

```
Hull–White rates  ─┐
                   ├─►  reprice swaps & options  ─►  net + collateralize  ─►  EE · PFE · CVA
GBM equity        ─┘        along every path            (CSA, MPoR)
     (correlated)
```

## Highlights

- **Correlated multi-factor simulation** — a **Hull–White 1-factor** short rate and
  a **GBM** equity process, evolved jointly under a correlation matrix
  (Cholesky-coupled shocks), with the equity drift driven by the *simulated* rate.
- **Analytic path repricing** — **interest-rate swaps** (dual-curve: OIS discounting
  + forecast-curve forwards) and **European equity options** (Black–Scholes at the
  simulated state) revalued in closed form at every node — no nested Monte Carlo.
- **Collateralized netting sets** — a **CSA** with threshold, minimum transfer
  amount, and a **margin period of risk**, so collateral realistically lags exposure.
- **Industry metrics** — Expected Exposure (**EE/EPE/ENE**), Potential Future
  Exposure (**PFE**) at a chosen confidence, and **CVA / DVA / bilateral CVA** from
  hazard-rate / CDS-implied default curves.
- **Wrong-way risk** — a factor-tilted stochastic default intensity correlates
  default with exposure (mean-preserving, so β=0 reproduces the independent CVA
  exactly); β>0 with a matching driver inflates CVA the way real WWR does.
- **Initial margin (SIMM-lite)** — an IM add-on on top of variation margin, sized to
  a high quantile of residual exposure, clipping exposure to the far tail.
- **Correct-by-construction** — the Hull–White model reprices today's discount curve
  to 1e-10; every numerical unit is pinned to an analytic benchmark (**56 tests**).

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dashboard,notebook,dev]"
```

## Quickstart (library)

```python
from ccr.config import MarketConfig, ScenarioConfig, SimConfig
from ccr.engine import CSA, run
from ccr.instruments import InterestRateSwap, EuropeanEquityOption
from ccr.metrics import HazardCurve

scenario = ScenarioConfig(
    portfolio=[
        InterestRateSwap(10_000_000, 0.025, 0.0, 5.0, 0.5, pay_fixed=True),
        EuropeanEquityOption(100.0, 3.0, 0.25, is_call=True, notional=50_000),
    ],
    market=MarketConfig(flat_rate=0.03, equity_sigma=0.22, rho_rate_equity=0.30),
    sim=SimConfig(n_paths=40_000, n_steps=60, horizon=5.0, seed=1, pfe_level=0.975),
    collateral=CSA(threshold=250_000, min_transfer_amount=50_000, mpor_years=10/252),
    hazard=HazardCurve.from_cds_spread(spread_bps=150, recovery=0.4),
)

result = run(scenario)
print(result)                 # EPE, Max PFE, CVA
print(result.profile_frame()) # EE / ENE / PFE by time
```

## CLI

```bash
ccr examples/sample_scenario.py          # human-readable
ccr examples/sample_scenario.py --json   # machine-readable metrics
```

## Interactive dashboard

```bash
streamlit run src/ccr/dashboard/app.py
```

Sidebar controls for the portfolio, model parameters, CSA terms, credit curve, and
Monte Carlo settings; live plots of the exposure profile (EE / PFE / ENE), the
exposure distribution at a chosen horizon, CVA contribution over time, and sample
simulated rate/equity paths.

## Notebook

`notebooks/example_walkthrough.ipynb` walks through the exposure profile, the effect
of collateral, the netting benefit, and the exposure distribution with plots.

## Architecture

A one-directional pipeline of NumPy arrays shaped `(n_paths, n_points)`:

| Layer | Module | Responsibility |
|---|---|---|
| Models | `ccr.models` | Correlated Hull–White rates + GBM equity on a time grid; analytic bonds |
| Instruments | `ccr.instruments` | Signed mark-to-market of each trade at every (path, node) |
| Engine | `ccr.engine` | Netting (sum before floor) + collateral (CSA / MPoR) → exposure cube |
| Metrics | `ccr.metrics` | EE / EPE / ENE, PFE quantile, CVA → `ExposureResult` |
| Dashboard | `ccr.dashboard` | Streamlit shell over the library |

Design notes: `docs/superpowers/specs/2026-08-20-counterparty-risk-simulator-design.md`.

## What the metrics mean

| Metric | Definition |
|---|---|
| **EE(t)** | Mean discounted positive exposure across paths at time *t* |
| **EPE** | Time-weighted average of EE (the regulatory EAD input) |
| **ENE(t)** | Mean discounted negative exposure (our liability side) |
| **PFE(t)** | The α-percentile of exposure at *t* (peak-risk / limit metric) |
| **CVA** | LGD × Σ discounted-EE × marginal default probability — the price of default risk |
| **DVA** | The mirror of CVA on *your* default, using negative exposure — a benefit to you |
| **BCVA** | Bilateral CVA = CVA − DVA — the net XVA charge |

### Advanced XVA controls

```python
from ccr.metrics import HazardCurve, WrongWayModel
from ccr.engine import CSA

scenario.own_hazard = HazardCurve.from_cds_spread(80, recovery=0.4)      # -> DVA / BCVA
scenario.wrong_way  = WrongWayModel(scenario.hazard, beta=1.2, driver="equity")  # WWR
scenario.collateral = CSA(0, 0, 10/252, initial_margin=True, im_quantile=0.99)   # VM + IM
```

- **Bilateral CVA/DVA**: set `own_hazard`; each leg is conditioned on the other
  party surviving (first-to-default approximation).
- **Wrong-way risk**: set `wrong_way`; `beta>0` raises the counterparty's hazard when
  the driver (equity/rate) is high. Pick the driver/sign so hazard rises with
  exposure. `beta=0` recovers the independent CVA.
- **Initial margin**: `initial_margin=True` on the CSA adds an IM buffer sized to the
  `im_quantile` of residual (post-VM) exposure.

## Tests

```bash
pytest -q
```

Benchmarks include: Hull–White reprices the initial curve at t0 (1e-10), option
value at t0 equals Black–Scholes, par swap is ~0 at t0, `CSA(MPoR=0, threshold=∞)`
equals uncollateralized exactly, CVA of a flat-EE constant-hazard case matches the
closed form, and Monte Carlo error shrinks with path count.

## Assumptions & extensions

In-flight swap accrual periods are dropped (a small approximation on typical grids);
the WWR intensity tilt and SIMM-lite IM are simplified proxies of their full models.
Natural next extensions: full ISDA SIMM sensitivities, stochastic basis, model
calibration to swaptions / option vols, curve bootstrapping, and more instruments.

## License

MIT
