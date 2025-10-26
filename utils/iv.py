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
    f = lambda vol: bs_call_price(S0, K, T, r, q, vol) - target
    return brentq(f, lo, hi)