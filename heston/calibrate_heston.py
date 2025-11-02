import logging
import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from data.options_chain import get_chain
from heston.params import HestonParams
from heston.vanilla_cf import heston_call
from utils.iv import implied_vol_call, bs_call_price


logger = logging.getLogger("pryce.calibration")


class CalibrationError(RuntimeError):
    """Base class for calibration failures."""


class CalibrationInputError(CalibrationError):
    """Raised when market data is insufficient or malformed."""


@dataclass(frozen=True)
class CalibrationResult:
    ticker: str
    expiry: str
    spot: float
    maturity: float
    params: HestonParams
    strikes: Tuple[float, ...]
    market_ivs: Tuple[float, ...]
    rmse: float
    vega_weighted_rmse: float

def bs_vega(S0, K, T, r=0.0, q=0.0, vol=0.2):
    if vol <= 0 or T <= 0:
        return 0.0
    from math import log, sqrt, exp
    from math import erf
    d1 = (log(S0/K) + (r - q + 0.5*vol*vol)*T) / (vol*sqrt(T))
    nprime = (1.0/np.sqrt(2*np.pi)) * np.exp(-0.5*d1*d1)
    return S0 * np.exp(-q*T) * nprime * np.sqrt(T)

def soft_feller_penalty(kappa, theta, sigma):
    # Feller: 2 kappa theta > sigma^2 ; penalize violations smoothly
    gap = 2.0 * kappa * theta - sigma**2
    return 1e-4 * max(0.0, -gap) ** 2


def _T(expiry_str):
    from datetime import datetime, timezone, timedelta
    e = datetime.strptime(expiry_str, "%Y-%m-%d").replace(tzinfo=timezone.utc) + timedelta(hours=20)
    now = datetime.now(timezone.utc)
    dt_years = max((e - now).total_seconds(), 0.0) / (365 * 24 * 3600)
    return dt_years

def model_iv(S0, K, T, p: HestonParams):
    # Heston price → BS IV, with tiny clamp into BS price range
    c = heston_call(S0, K, T, p)
    lo = bs_call_price(S0, K, T, p.r, p.q, vol=1e-4)
    hi = bs_call_price(S0, K, T, p.r, p.q, vol=5.0)
    c = min(max(c, lo + 1e-10), hi - 1e-10)
    return float(implied_vol_call(c, S0, K, T, p.r, p.q))

def calibrate_market(ticker, expiry, p0=HestonParams(2.0, 0.04, 0.5, -0.7, 0.04, 0.01, 0.0)):
    # 1) Load chain + compute T
    S0, calls_df, _ = get_chain(ticker, expiry)  # calls only
    T = _T(expiry)
    if T < 0.02:
        raise CalibrationInputError("Expiry too short; quotes unreliable.")
    logger.info("Calibrating Heston to %s %s  T=%.4fy  S0=%s", ticker, expiry, T, S0)

    # 2) Minimal filtering: keep only finite, positive IVs (0 < IV < 5)
    strikes = calls_df['strike'].to_numpy(float)
    bid  = calls_df['bid'].to_numpy(float)
    ask  = calls_df['ask'].to_numpy(float)
    last = calls_df['lastPrice'].to_numpy(float)
    # --- moneyness filter (super simple) ---
    mny = strikes / S0
    mask_mny = (mny >= 0.70) & (mny <= 1.30)   # your 0.70 floor; symmetric top at 1.30
    kept = np.count_nonzero(mask_mny)
    dropped = strikes.size - kept
    logger.info("Moneyness filter: kept %d, dropped %d (range 0.70–1.30)", kept, dropped)

    strikes = strikes[mask_mny]
    bid     = bid[mask_mny]
    ask     = ask[mask_mny]
    last    = last[mask_mny]

    def pick_px(b, a, l):
        # 1) clean mid if both sides positive and not crossed
        if np.isfinite(b) and np.isfinite(a) and b > 0 and a > 0 and a >= b:
            return 0.5 * (b + a)
        # 2) else use last trade if positive
        if np.isfinite(l) and l > 0:
            return l
        # 3) else take the nonzero side if any
        if np.isfinite(a) and a > 0:
            return a
        if np.isfinite(b) and b > 0:
            return b
        return np.nan

    raw_px = np.array([pick_px(b, a, l) for b, a, l in zip(bid, ask, last)], float)

    # clamp into a BS-feasible price band before inversion
    bs_lo = np.array([bs_call_price(S0, K, T, 0.0, 0.0, vol=1e-4) for K in strikes])
    bs_hi = np.array([bs_call_price(S0, K, T, 0.0, 0.0, vol=5.0 ) for K in strikes])
    clamped_px = np.clip(raw_px, bs_lo + 1e-8, bs_hi - 1e-8)

    # invert to IVs with our own solver (no Yahoo IVs involved)
    market_ivs = np.array(
        [implied_vol_call(px, S0, K, T, 0.0, 0.0) if np.isfinite(px) else np.nan
        for px, K in zip(clamped_px, strikes)],
        dtype=float
    )

    # basic clean-up
    m = np.isfinite(market_ivs) & (market_ivs > 0.01) & (market_ivs < 5.0)
    strikes, market_ivs = strikes[m], market_ivs[m]

    if strikes.size < 3:
        raise CalibrationInputError("Not enough clean IV points; try a different expiry or widen the strike window.")

    # 3) Define objective: Heston -> price -> BS IV, compare to market IVs (vega-weighted)
    def objective(pvec):
        p = HestonParams(pvec[0], pvec[1], pvec[2], pvec[3], pvec[4], r=p0.r, q=p0.q)
        iv_model = np.array([model_iv(S0, K, T, p) for K in strikes], float)

        # mask invalid IVs but keep optimizer smooth
        msk = np.isfinite(iv_model)
        if msk.sum() < 3:
            return 1e6  # softer than 1e9

        vega = np.array([bs_vega(S0, K, T, p.r, p.q, iv)
                        for K, iv in zip(strikes[msk], market_ivs[msk])])
        vega = np.clip(vega, 1e-8, None)
        w = vega / vega.max()

        diff = iv_model[msk] - market_ivs[msk]
        mse = float(np.sum(w * diff * diff) / np.sum(w))

        # simple regularization: soft Feller + mild rho cap
        feller = 2.0 * p.kappa * p.theta / (p.sigma * p.sigma + 1e-12)
        feller_pen = 50.0 * max(0.0, 1.0 - feller)**2
        rho_pen = 1e-3 * max(0.0, abs(p.rho) - 0.98)**2

        return mse + feller_pen + rho_pen

    # 4) Optimize
    from scipy.optimize import minimize

    bounds = [
        (0.30, 5.0),     # kappa
        (0.04, 0.25),    # theta  (≤ ~45% long-run vol)
        (0.20, 1.20),    # sigma
        (-0.90, -0.10),  # rho    (don’t allow weak skew)
        (0.04, 0.25)     # v0
    ]
    res = minimize(objective,
                   x0=[p0.kappa, p0.theta, p0.sigma, p0.rho, p0.v0],
                   bounds=bounds,
                   method='L-BFGS-B')
    if not res.success:
        raise CalibrationError("Calibration failed: " + res.message)

    p_star = HestonParams(*res.x, r=p0.r, q=p0.q)

    # 5) Diagnostics
    iv_model = np.array([model_iv(S0, K, T, p_star) for K in strikes])
    err = iv_model - market_ivs
    if err.size:
        rmse = float(np.sqrt(np.mean(err**2)))
        logger.info("Fit RMSE (IV points): %.4f", rmse)
    else:
        rmse = float("nan")
        logger.info("Fit RMSE (IV points): n/a")

    # Vega-weighted RMSE (guard empty)
    w = np.array([bs_vega(S0, K, T, p_star.r, p_star.q, max(iv, 1e-4)) for K, iv in zip(strikes, market_ivs)])
    if w.size == 0:
        wrmse = float("nan")
        logger.info("Vega-weighted RMSE: n/a (no valid points)")
    else:
        w = w / max(1e-8, w.max())
        wrmse = float(np.sqrt(np.sum(w * err**2) / np.sum(w)))
        logger.info("Vega-weighted RMSE: %.4f", wrmse)

    logger.info("Fitted Heston params: %s", p_star)
    logger.info("Feller ratio = %.3f", 2 * p_star.kappa * p_star.theta / (p_star.sigma**2))
    # 6) Quick plot (opt-in for local debugging)
    if os.getenv("PRYCE_SHOW_PLOTS") == "1":
        try:
            import matplotlib.pyplot as plt

            moneyness = strikes / S0
            plt.figure()
            plt.scatter(moneyness, market_ivs, label="Market IV", s=18)
            plt.plot(moneyness, iv_model, label="Heston IV (fitted)")
            plt.xlabel("Moneyness (K/S0)")
            plt.ylabel("Implied Volatility")
            plt.title(f"{ticker} {expiry}  T={T:.4f}  S0={S0}")
            plt.legend()
            plt.show()
        except Exception as exc:  # pragma: no cover - best effort diagnostic
            logger.debug("Plot skipped: %s", exc)

    return CalibrationResult(
        ticker=ticker,
        expiry=expiry,
        spot=float(S0),
        maturity=float(T),
        params=p_star,
        strikes=tuple(float(x) for x in strikes.tolist()),
        market_ivs=tuple(float(x) for x in market_ivs.tolist()),
        rmse=rmse,
        vega_weighted_rmse=wrmse,
    )


# === Final wrappers ===

def calibrate_iv(ticker, expiry, p0=HestonParams(2.0, 0.04, 0.5, -0.7, 0.04, 0.01, 0.0)):
    return calibrate_market(ticker, expiry, p0=p0).params


def calibrate_heston(ticker, expiry):
    """Backward-compatible helper used by the CLI."""
    try:
        return calibrate_market(ticker, expiry).params
    except CalibrationError as e:
        logger.error("Calibration failed: %s", e)
        return None


if __name__ == "__main__":
    import sys
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)
    if len(sys.argv) < 3:
        raise SystemExit("Usage: python calibrate_heston.py <TICKER> <EXPIRY>")
    ticker = sys.argv[1].strip().upper()
    expiry = sys.argv[2].strip()
    p = calibrate_heston(ticker, expiry)
    if p is not None:
        print("\nReturned fitted parameters:")
        print(p)
    else:
        print("Calibration failed.")



