# api.py
import logging
from functools import lru_cache
from typing import Literal, Optional, Dict

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from heston.params import HestonParams
from heston.calibrate_heston import (
    CalibrationError,
    CalibrationInputError,
    CalibrationResult,
    calibrate_market,
)
from heston.vanilla_cf import heston_call
from exotics.asian import price_asian_call_mc
from exotics.kibarrier import price_barrier_ui_call_mc
from exotics.chooser import price_chooser_mc
from exotics.compound import price_compound_call_on_call_mc
import numpy as np

app = FastAPI(title="Pryce API", version="0.1.0")

logger = logging.getLogger("pryce.api")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- helpers ----------

def feller_ratio(p: HestonParams) -> float:
    return float(2.0 * p.kappa * p.theta / (p.sigma * p.sigma + 1e-16))

DEFAULT_WINDOW = 0.30


@lru_cache(maxsize=64)
def _cached_calibration(ticker: str, expiry: str, window: float) -> Dict:
    # Run calibration (returns fitted params + filtered market data)
    result: CalibrationResult = calibrate_market(ticker, expiry)
    p_star = result.params
    S0 = result.spot

    # Build a simple IV curve for plotting on UI (moneyness vs model IV)
    strikes = np.asarray(result.strikes, dtype=float)
    mny = strikes / S0
    # Filter to a window around ATM for the plot
    m = (mny >= (1 - window)) & (mny <= (1 + window))
    strikes, mny = strikes[m], mny[m]

    # Model IVs via Heston→BS inversion
    from utils.iv import implied_vol_call, bs_call_price
    from heston.vanilla_cf import heston_call
    T = float(result.maturity)
    # price → clamp → invert
    bs_lo = np.array([bs_call_price(S0, K, T, p_star.r, p_star.q, vol=1e-4) for K in strikes])
    bs_hi = np.array([bs_call_price(S0, K, T, p_star.r, p_star.q, vol=5.0) for K in strikes])
    prices = np.array([heston_call(S0, K, T, p_star) for K in strikes])
    prices = np.clip(prices, bs_lo + 1e-8, bs_hi - 1e-8)
    iv_model = np.array([implied_vol_call(px, S0, K, T, p_star.r, p_star.q) for px, K in zip(prices, strikes)])

    # Market IVs from calibration pipeline
    iv_mkt = np.asarray(result.market_ivs, dtype=float)[m]

    return {
        "ticker": ticker, "expiry": expiry, "S0": float(S0), "window": float(window),
        "params": dict(kappa=float(p_star.kappa), theta=float(p_star.theta), sigma=float(p_star.sigma),
                       rho=float(p_star.rho), v0=float(p_star.v0), r=float(p_star.r), q=float(p_star.q)),
        "diagnostics": {
            "feller": feller_ratio(p_star),
            "rmse": float(result.rmse),
            "vega_weighted_rmse": float(result.vega_weighted_rmse),
            "maturity": float(result.maturity),
        },
        "curve": {
            "moneyness": mny.tolist(),
            "iv_market": iv_mkt.tolist(),
            "iv_heston": iv_model.tolist()
        }
    }


def _get_calibration_payload(ticker: str, expiry: str, window: float = DEFAULT_WINDOW) -> Dict:
    return _cached_calibration(ticker.upper(), expiry, window)


def _calibration_or_http_error(ticker: str, expiry: str, window: float = DEFAULT_WINDOW) -> Dict:
    try:
        return _get_calibration_payload(ticker, expiry, window)
    except CalibrationInputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except CalibrationError as exc:
        logger.exception("Calibration failure for %s %s", ticker, expiry)
        raise HTTPException(status_code=502, detail="Calibration failed") from exc
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.exception("Unexpected calibration error for %s %s", ticker, expiry)
        raise HTTPException(status_code=500, detail="Unexpected calibration error") from exc

# ---------- routes ----------

@app.get("/api/calibrate")
def calibrate(
    ticker: str = Query(..., min_length=1),
    expiry: str = Query(..., regex=r"^\d{4}-\d{2}-\d{2}$"),
    window: float = Query(DEFAULT_WINDOW, ge=0.05, le=0.60)
):
    return _calibration_or_http_error(ticker, expiry, window)

@app.get("/api/price")
def price(
    type: Literal["asian", "barrier", "chooser", "compound"],
    ticker: str,
    expiry: str,
    K: float,
    T: Optional[float] = None,
    paths: int = Query(10000, ge=1000, le=50000),
    steps: int = Query(126, ge=32, le=504),
    seed: int = 42,
    barrier_mult: float = 1.2,   # for barrier demo: B = barrier_mult * K
    chooser_tau_frac: float = 0.25  # tau = chooser_tau_frac * T
):
    # get params from calibration cache (or run once)
    calib = _calibration_or_http_error(ticker, expiry)
    p = HestonParams(**calib["params"])
    S0 = float(calib["S0"])

    # time to expiry if not provided
    from heston.calibrate_heston import _T
    T_eff = float(T) if T is not None else _T(expiry)

    if type == "asian":
        px, se = price_asian_call_mc(S0, K, T_eff, p, n_steps=steps, n_paths=paths, seed=seed)
        viz = {"kind": "hist_avg"}  # frontend chooses how to draw
    elif type == "barrier":
        B = float(barrier_mult) * K
        px, se = price_barrier_ui_call_mc(S0, K, B, T_eff, p, n_steps=steps, n_paths=paths, seed=seed)
        viz = {"kind": "paths_with_barrier", "B": B}
    elif type == "chooser":
        tau = chooser_tau_frac * T_eff
        px, se = price_chooser_mc(S0, K, T_eff, tau, p, n_outer=2000, n_inner=200, n_steps=steps, seed=seed)
        viz = {"kind": "chooser_split", "tau": tau}
    elif type == "compound":
        # demo values for K1,T1; wire actual UI fields later
        K1, T1 = 10.0, 0.5 * T_eff
        from exotics.compound import price_compound_call_on_call_mc
        px, se = price_compound_call_on_call_mc(S0, K1, K, T1, T_eff, p,
                                                n_outer=2000, n_inner=200, n_steps=steps, seed=seed)
        viz = {"kind": "compound_distribution", "K1": K1, "T1": T1}
    else:
        raise HTTPException(status_code=400, detail="Unsupported type")

    return {
        "inputs": {"type": type, "ticker": ticker, "expiry": expiry, "K": K, "T": T_eff,
                   "paths": paths, "steps": steps, "seed": seed},
        "params_used": calib["params"],
        "results": {"price": float(px), "std_error": float(se), "feller": float(feller_ratio(p))},
        "viz": viz
    }

@app.get("/api/payoff")
def payoff(
    type: Literal["asian", "barrier", "chooser", "compound"],
    K: float,
    B: Optional[float] = None,
    tau: Optional[float] = None,
    K1: Optional[float] = None, K2: Optional[float] = None,
    T1: Optional[float] = None, T2: Optional[float] = None
):
    # Generate simple payoff curve over S grid for visualizer
    S_axis = np.linspace(0.1 * K, 2.0 * K, 200)
    if type == "asian":
        # visualizer uses S_T proxy; true asian needs S̄—this is pedagogical
        payoff = np.maximum(S_axis - K, 0.0)
        tex = r"( \bar S - K )^+"
        desc = "Arithmetic-average Asian call payoff shown vs a proxy S-axis (teaching view)."
    elif type == "barrier":
        if B is None: B = 1.2 * K
        payoff = np.where(S_axis >= K, S_axis - K, 0.0)  # displayed only; barrier condition is pathwise
        tex = r"\mathbb{1}_{\{\max_{t\le T} S_t \ge B\}} (S_T - K)^+"
        desc = "Up-and-In call: pays only if barrier was hit before expiry."
    elif type == "chooser":
        payoff = np.maximum(S_axis - K, K - S_axis)  # max(call, put)
        tex = r"\max\{(S_T - K)^+,\ (K - S_T)^+\}"
        desc = "Chooser: at τ, choose the more valuable of call or put."
    elif type == "compound":
        payoff = np.maximum(np.maximum(S_axis - K, 0.0) - (K1 if K1 else 0.0), 0.0)
        tex = r"\max(C(T_1;K_2,T_2) - K_1, 0)"
        desc = "Call on Call (schematic): payoff vs S-axis at T1 is shown schematically."
    else:
        raise HTTPException(status_code=400, detail="Unsupported type")

    return {
        "type": type,
        "formula_tex": tex,
        "description": desc,
        "plot": {"S_axis": S_axis.tolist(), "payoff": payoff.tolist()}
    }
