import numpy as np

from heston.paths import simulate_heston_paths


def test_simulate_heston_paths_reproducible():
    S0 = 100.0
    params = dict(kappa=2.0, theta=0.04, sigma=0.6, rho=-0.5, r=0.01, q=0.0, v0=0.04)
    T = 1.0
    n_steps = 128
    n_paths = 8000

    S, v = simulate_heston_paths(
        S0,
        params["v0"],
        params["kappa"],
        params["theta"],
        params["sigma"],
        params["rho"],
        params["r"],
        params["q"],
        T,
        n_steps,
        n_paths,
        seed=123,
    )

    assert S.shape == (n_steps + 1, n_paths)
    assert v.shape == (n_steps + 1, n_paths)
    assert np.all(S > 0.0)
    assert np.all(v >= 0.0)

    expected_mean = S0 * np.exp((params["r"] - params["q"]) * T)
    sample_mean = float(S[-1].mean())
    assert abs(sample_mean - expected_mean) < 1.0
