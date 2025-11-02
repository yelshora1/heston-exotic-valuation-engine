# heston/paths.py
import numpy as np

def simulate_heston_paths(S0, v0, kappa, theta, sigma, rho, r, q,
                          T, n_steps, n_paths, seed=None):
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    dt = T / n_steps
    S = np.empty((n_steps + 1, n_paths)); S[0] = S0
    v = np.empty((n_steps + 1, n_paths)); v[0] = max(v0, 0.0)
    sqrt_dt = np.sqrt(dt)
    for t in range(n_steps):
        z1 = rng.standard_normal(n_paths)
        z2 = rng.standard_normal(n_paths)
        z2 = rho * z1 + np.sqrt(max(0.0, 1 - rho * rho)) * z2
        v_pos = np.maximum(v[t], 0.0)
        sqrt_v = np.sqrt(v_pos)
        v_next = v[t] + kappa * (theta - v_pos) * dt + sigma * sqrt_v * sqrt_dt * z2
        v_next = np.maximum(v_next, 0.0)

        drift = (r - q - 0.5 * v_pos) * dt
        diffusion = sqrt_v * sqrt_dt * z1
        S_next = S[t] * np.exp(drift + diffusion)
        S[t + 1], v[t + 1] = S_next, v_next
    return S, v

# --- add this at the very bottom of heston/paths.py ---
if __name__ == "__main__":
    # Minimal self-test: simulate and print a few sanity stats
    from params import HestonParams
    import numpy as np

    # quick demo params (replace with your calibrated ones if you want)
    p = HestonParams(kappa=2.0, theta=0.04, sigma=0.6, rho=-0.6, v0=0.04, r=0.01, q=0.0)

    S0, T = 100.0, 1.0
    n_steps, n_paths, seed = 252, 30_000, 42

    S, v = simulate_heston_paths(
        S0=S0, v0=p.v0, kappa=p.kappa, theta=p.theta, sigma=p.sigma, rho=p.rho,
        r=p.r, q=p.q, T=T, n_steps=n_steps, n_paths=n_paths, seed=seed
    )
    ST = S[-1]

    print("Heston paths smoke test")
    print(f"  steps={n_steps}  paths={n_paths}  T={T}y  seed={seed}")
    print(f"  E[ST]≈{ST.mean():.4f}   std(ST)≈{ST.std(ddof=1):.4f}")
    print(f"  min(ST)={ST.min():.4f}  max(ST)={ST.max():.4f}")
    print(f"  v(t) negative count = {(v < 0).sum()} (should be 0 with truncation)")
