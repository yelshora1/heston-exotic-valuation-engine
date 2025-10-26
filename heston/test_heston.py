# This script tests the Heston option pricer and implied volatility calculator.
# It assumes you are running from the Pryce directory (parent of heston/ and utils/).

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.options_chain import get_chain
import numpy as np

from heston.params import HestonParams
from heston.vanilla_cf import heston_call, heston_put
from utils.iv import implied_vol_call, bs_call_price

# Heston Model Parameters
params = HestonParams(
    kappa=2.0,    # Mean reversion speed
    theta=0.04,   # Long-run variance
    sigma=0.2,    # Volatility of variance (reduced from 0.5 for more realistic prices)
    rho=-0.7,     # Correlation
    v0=0.04,      # Initial variance
    r=0.01,       # Risk-free rate
    q=0.0         # Dividend yield
)

# Option Contract Params
S0 = 100.0   # Spot price
K = 100.0    # Strike price
T = 1.0      # Time to maturity (1 year)

# Price Options Using Heston Model
call_price = heston_call(S0, K, T, params)
put_price = heston_put(S0, K, T, params)

print(f"Heston Call Price: {call_price:.4f}")
print(f"Heston Put Price:  {put_price:.4f}")

# Check Black-Scholes price range for implied volatility calculation
bs_lo = bs_call_price(S0, K, T, params.r, params.q, vol=1e-4)
bs_hi = bs_call_price(S0, K, T, params.r, params.q, vol=5.0)
print(f"Black-Scholes Call Price Range: [{bs_lo:.4f}, {bs_hi:.4f}]")

if not (bs_lo < call_price < bs_hi):
    print("Warning: Heston call price is outside the Black-Scholes price range for vol in [0.0001, 5.0].")
    print("Implied volatility calculation will fail. Try adjusting your Heston parameters or integration range (_L in vanilla_cf.py).")
else:
    try:
        iv = implied_vol_call(call_price, S0, K, T, params.r, params.q)
        print(f"Implied Volatility (Call): {iv:.4f}")
    except Exception as e:
        print(f"Implied volatility calculation failed: {e}")

