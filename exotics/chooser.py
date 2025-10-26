import numpy as np
from heston.paths import simulate_heston_paths

def price_chooser_mc(S0, K, T, tau, p,
                     n_outer=3000, n_inner=200, n_steps=252, seed=None):
    steps_pre = int(round(n_steps * (tau/T)))
    steps_post = n_steps - steps_pre
    S_tau, v_tau = simulate_heston_paths(S0, p.v0, p.kappa, p.theta, p.sigma,
                                         p.rho, p.r, p.q, tau, steps_pre, n_outer, seed)
    S_tau, v_tau = S_tau[-1], v_tau[-1]

    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 2**31-1, n_outer, dtype=np.int64)

    disc0_tau = np.exp(-p.r * tau)
    disc_tau_T = np.exp(-p.r * (T - tau))
    vals_tau = np.empty(n_outer)

    for i in range(n_outer):
        S_T, _ = simulate_heston_paths(S_tau[i], v_tau[i], p.kappa, p.theta, p.sigma,
                                       p.rho, p.r, p.q, T - tau, steps_post,
                                       n_inner, int(seeds[i]))
        ST = S_T[-1]
        call = np.maximum(ST - K, 0.0)
        put  = np.maximum(K - ST, 0.0)
        vals_tau[i] = max(disc_tau_T * call.mean(), disc_tau_T * put.mean())

    px = disc0_tau * vals_tau.mean()
    se = disc0_tau * vals_tau.std(ddof=1)/np.sqrt(n_outer)
    return px, se
