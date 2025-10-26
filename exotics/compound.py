import numpy as np
from heston.paths import simulate_heston_paths

def price_compound_call_on_call_mc(S0, K1, K2, T1, T2, p,
                                   n_outer=3000, n_inner=200, n_steps=252, seed=None):
    steps_pre = int(round(n_steps * (T1/T2)))
    steps_post = n_steps - steps_pre
    S1, v1 = simulate_heston_paths(S0, p.v0, p.kappa, p.theta, p.sigma,
                                   p.rho, p.r, p.q, T1, steps_pre, n_outer, seed)
    S1, v1 = S1[-1], v1[-1]

    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 2**31-1, n_outer, dtype=np.int64)
    disc0_T1 = np.exp(-p.r * T1)
    disc_T1_T2 = np.exp(-p.r * (T2 - T1))
    vals_T1 = np.empty(n_outer)

    for i in range(n_outer):
        S_T, _ = simulate_heston_paths(S1[i], v1[i], p.kappa, p.theta, p.sigma,
                                       p.rho, p.r, p.q, T2 - T1, steps_post,
                                       n_inner, int(seeds[i]))
        ST = S_T[-1]
        call_pay = np.maximum(ST - K2, 0.0)
        C_T1 = disc_T1_T2 * call_pay.mean()
        vals_T1[i] = max(C_T1 - K1, 0.0)

    px = disc0_T1 * vals_T1.mean()
    se = disc0_T1 * vals_T1.std(ddof=1)/np.sqrt(n_outer)
    return px, se
