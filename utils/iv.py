import numpy as np
from scipy.optimize import brentq
from math import log, sqrt, exp

def bs_call_price(S0, K, T, r = 0.0, q = 0.0, vol = 0.2):
    from math import erf
    if vol <= 0 or T <= 0:
        return max(S0*exp(-q*T) - K*exp(-r*T), 0.0)
    d1 = (log(S0/K) + (r - q + 0.5*vol*vol)*T) / (vol*sqrt(T))
    d2 = d1 - vol*sqrt(T)
    N = lambda x: 0.5 * (1.0 + erf(x / np.sqrt(2.0)))
    return S0 * exp(-q*T) * N(d1) - K * exp(-r*T) * N(d2)

def implied_vol_call(target, S0, K, T, r=0.0, q=0.0, lo = 1e-4, hi = 5.0):
    if not np.isfinite(target):
        raise ValueError("Target option price must be finite.")

    lower = bs_call_price(S0, K, T, r, q, lo)
    upper = bs_call_price(S0, K, T, r, q, hi)
    if not np.isfinite(lower) or not np.isfinite(upper):
        raise ValueError("Failed to compute Black-Scholes bracket for implied volatility.")

    # Guard against deep ITM/OTM quotes falling outside the numerical bracket
    eps = 1e-8
    if upper - lower < eps:
        return lo

    clamped_target = float(np.clip(target, lower + eps, upper - eps))

    def objective(vol: float) -> float:
        return bs_call_price(S0, K, T, r, q, vol) - clamped_target

    try:
        return brentq(objective, lo, hi)
    except ValueError as exc:
        raise ValueError(
            f"Implied volatility root finding failed for price {target:.6f}; "
            f"expected to be within [{lower:.6f}, {upper:.6f}]"
        ) from exc