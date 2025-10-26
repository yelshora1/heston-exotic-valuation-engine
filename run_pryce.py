
import sys
import json
import numpy as np
import yfinance as yf

# Package imports (no path hacks; run from repo root or `python -m Pryce.run_pryce`)
from heston.calibrate_heston import calibrate_heston, _T
from heston.params import HestonParams
from exotics.asian import price_asian_call_mc
from exotics.kibarrier import price_barrier_ui_call_mc
from exotics.chooser import price_chooser_mc
from exotics.compound import price_compound_call_on_call_mc

def _spot_from_yf(ticker: str) -> float:
    tk = yf.Ticker(ticker)
    # Prefer fast_info if available; fallback to info
    px = getattr(tk, "fast_info", None)
    if px and getattr(px, "last_price", None):
        return float(px.last_price)
    info = tk.info
    if "currentPrice" in info and info["currentPrice"] is not None:
        return float(info["currentPrice"])
    if "regularMarketPrice" in info and info["regularMarketPrice"] is not None:
        return float(info["regularMarketPrice"])
    raise RuntimeError("Could not determine spot from yfinance.")

def main():
    if len(sys.argv) < 5:
        raise SystemExit(
            "Usage: python run_pryce.py <TICKER> <EXPIRY YYYY-MM-DD> <STRIKE> <TYPE> [--json]"
        )
    ticker = sys.argv[1].strip().upper()
    expiry = sys.argv[2].strip()
    K = float(sys.argv[3])
    opt_type = sys.argv[4].strip().lower()
    want_json = ("--json" in sys.argv[5:])

    # 1) Market inputs
    S0 = _spot_from_yf(ticker)
    T = _T(expiry)  # year fraction consistent with your calibrator

    # 2) Calibrate Heston (returns HestonParams)
    print(f"\nCalibrating Heston to {ticker} {expiry} ...")
    p = calibrate_heston(ticker, expiry)
    if p is None:
        raise SystemExit("Calibration failed; no parameters returned.")

    # 3) Route to the requested exotic
    if opt_type == "chooser":
        price, se = price_chooser_mc(
            S0, K, T, 0.25 * T, p, n_outer=3000, n_inner=250, n_steps=252, seed=1
        )
    elif opt_type == "compound":
        price, se = price_compound_call_on_call_mc(
            S0, 0.05 * S0, K, 0.5 * T, T, p, n_outer=3000, n_inner=250, n_steps=252, seed=2
        )
    elif opt_type == "barrier":
        price, se = price_barrier_ui_call_mc(
            S0, K, 1.2 * K, T, p, n_steps=252, n_paths=30000, seed=3
        )
    elif opt_type == "asian":
        price, se = price_asian_call_mc(
            S0, K, T, p, n_steps=252, n_paths=30000, seed=4
        )
    else:
        raise SystemExit("TYPE must be one of: asian | barrier | chooser | compound")

    # 4) Output
    if want_json:
        from dataclasses import asdict, is_dataclass
        params_dict = asdict(p) if is_dataclass(p) else p.__dict__
        out = {
            "ticker": ticker,
            "expiry": expiry,
            "K": K,
            "type": opt_type,
            "S0": S0,
            "T": T,
            "price": float(price),
            "se": float(se),
            "params": params_dict,
        }
        print(json.dumps(out, separators=(",", ":"), ensure_ascii=False))
    else:
        print("\n=== Result ===")
        print(f"{ticker} {opt_type.title()} option (K={K}, Expiry={expiry})")
        print(f"Spot / T    : {S0:.6f} / {T:.6f}y")
        print(f"Model Price : {price:.6f}")
        print(f"Std. Error  : {se:.6f}")
        print(f"Params used : {p}\n")

if __name__ == "__main__":
    main()
