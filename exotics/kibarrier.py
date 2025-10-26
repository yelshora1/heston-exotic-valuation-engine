import numpy as np
from heston.paths import simulate_heston_paths

def price_barrier_ui_call_mc(S0, K, B, T, p, n_steps=252, n_paths=20000, seed=None):
    S, _ = simulate_heston_paths(S0, p.v0, p.kappa, p.theta, p.sigma, p.rho,
                                 p.r, p.q, T, n_steps, n_paths, seed)
    hit = np.any(S >= B, axis=0)
    pay = np.where(hit, np.maximum(S[-1] - K, 0.0), 0.0)
    disc = np.exp(-p.r * T)
    return disc * pay.mean(), disc * pay.std(ddof=1)/np.sqrt(n_paths)
