# Pryce Exotic Options Platform

Pryce is a Python research environment for calibrating the Heston stochastic volatility model and pricing a curated set of exotic options.  It combines a reproducible Monte Carlo engine, semi-analytic valuation routines, and turnkey data acquisition so that quantitative researchers can experiment with full pricing workflows end-to-end.

## Key Capabilities

- **Market data ingestion** – downloads option chains and underlying prices directly from Yahoo Finance with reproducible filtering logic.
- **Heston calibration** – fits the full Heston parameter vector to the observed implied volatility surface using numerical optimization.
- **Monte Carlo path generation** – simulates coupled asset/variance dynamics suitable for path-dependent payoffs.
- **Exotic payoffs** – includes Monte Carlo pricers for arithmetic Asian, up-and-in barrier, European chooser, and compound call-on-call options.
- **Deterministic APIs** – exposes the same pricing stack through both a command-line interface and an optional FastAPI service.

## Repository Layout

```
.
├── run_pryce.py            # Command-line entry point for calibration + pricing
├── api.py                  # FastAPI service exposing calibration and pricing endpoints
├── heston/                 # Core model components (characteristic fn, calibration, simulation)
├── exotics/                # Monte Carlo payoffs for supported exotic contracts
├── data/options_chain.py   # Yahoo Finance data access and preprocessing utilities
├── utils/iv.py             # Black–Scholes and implied volatility helpers
├── scripts/                # Convenience scripts (e.g. inspect option chains)
└── tests/                  # Pytest-based sanity checks for numerical routines
```

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| Python 3.10+ | Development is tested against Python 3.10 and 3.11. |
| C compiler toolchain | Required by SciPy/NumPy wheels on some platforms. |
| Internet access | Needed at runtime to pull market data from Yahoo Finance. |

The project depends on the following PyPI packages:

```
numpy
scipy
pandas
yfinance
fastapi
uvicorn[standard]
```

Install them into a virtual environment to isolate the toolchain from your system Python.

## Installation

```bash
# Clone the repository
 git clone https://github.com/<your-org>/pryce.git
 cd pryce

# (Recommended) create a virtual environment
 python -m venv .venv
 source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install Python dependencies
 pip install --upgrade pip
 pip install numpy scipy pandas yfinance fastapi "uvicorn[standard]"
```

If you plan to run the sample tests, install the additional tooling:

```bash
pip install pytest
```

## Obtaining Market Data

Pricing routines require an active internet connection because market inputs are pulled live from Yahoo Finance:

1. The current underlying price (`S0`) is sourced via [`yfinance.Ticker.fast_info`](https://aroussi.com/post/python-yahoo-finance).
2. Option chains are downloaded for the requested ticker/expiry, filtered, and converted into implied volatility observations (see `data/options_chain.py`).
3. Calibration dates must match official exchange expirations (format `YYYY-MM-DD`).

If you need to run fully offline, capture the JSON payloads returned by `data.options_chain.get_chain` and feed them back into the calibrator by modifying `heston/calibrate_heston.py`.

## Command-Line Workflow

`run_pryce.py` performs calibration and then prices one of the supported exotics.

```text
Usage: python run_pryce.py <TICKER> <EXPIRY YYYY-MM-DD> <STRIKE> <TYPE> [--json]
TYPE  : asian | barrier | chooser | compound
--json: emit machine-readable output with calibrated parameters
```

Example session for an arithmetic Asian call:

```bash
python run_pryce.py AAPL 2026-02-20 180 asian
```

Expected console output (abridged):

```
Calibrating Heston to AAPL 2026-02-20 ...
=== Result ===
AAPL Asian option (K=180.0, Expiry=2026-02-20)
Spot / T    : 190.123456 / 1.987654y
Model Price : 8.321456
Std. Error  : 0.214567
Params used : HestonParams(kappa=..., theta=..., sigma=..., rho=..., v0=...)
```

The Monte Carlo standard error is reported alongside the price to help you judge convergence. Re-run with the `--json` flag to retrieve the same information as a compact JSON object.

### Numerical Controls

Monte Carlo configuration (`n_paths`, `n_steps`, and random seeds) is hard-coded in `run_pryce.py`. Adjust those values to change precision or runtime trade-offs. All pricers consume the calibrated `HestonParams` dataclass emitted by `heston.calibrate_heston.calibrate_heston`.

## FastAPI Service (Optional)

Launch an HTTP API that mirrors the CLI functionality:

```bash
uvicorn api:app --reload
```

Key endpoints:

- `POST /price/exotic` – calibrates the model and returns the Monte Carlo price for any of the supported exotics.
- `GET /health` – lightweight readiness probe for infrastructure integrations.

See `api.py` for the full request/response schema, including JSON payloads for direct parameter injection.

## Validation

Run the included tests after installing `pytest`:

```bash
pytest
```

The suite covers implied-volatility inversions and core Heston path properties to ensure that numerical changes do not regress the calibration or pricing stack.

## Extending Pryce

1. Add new payoffs under `exotics/` and expose them through `run_pryce.py` and `api.py`.
2. Update `tests/` with regression checks tailored to the new contract.
3. Document any new CLI options or API payloads here so downstream users can follow reproducible steps.

## Licensing and Attribution

- Source code is released under the Apache 2.0 license.
- Conceptual design, payoff visualization structure, and educational framing are the intellectual property of **Yousef Elshora (2025)**. Reuse of the brand or educational material requires explicit permission.
