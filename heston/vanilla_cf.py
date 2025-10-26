import numpy as np
from .params import HestonParams
from .charfunc import heston_cf
from numpy.polynomial.legendre import leggauss

_N = 96  # number of quadrature points
_x, _w = leggauss(_N)
_L = 40  # fine for now

def _Pj(j: int, S0: float, K: float, T: float, p: HestonParams) -> float:
    lnK = np.log(K)
    u = 0.5 * (_x + 1) * _L
    w = 0.5 * _L * _w

    # --- normalization to prevent P1 divergence ---
    # φ(-i) under j=1; equals E[S_T]/S0 = exp((r-q)T) in RN measure.
    if j == 1:
        phi_mi = heston_cf(-1j, T, S0, p, j=1)  # complex scalar ~ exp((r-q)T)
        # numerical safety (shouldn’t trigger, but be safe)
        if abs(phi_mi) < 1e-14:
            phi_mi = np.exp((p.r - p.q)*T)

    def integrand(u_val: float):
        if abs(u_val) < 1e-10:
            return 0.0
        if j == 1:
            # Use j=1 CF with shift (u - i), AND normalize by φ(-i)
            phi = heston_cf(u_val - 1j, T, S0, p, j=1) / phi_mi
        else:
            # P2 uses j=2 CF with no shift, no normalization needed (φ(0)=1).
            phi = heston_cf(u_val, T, S0, p, j=2)
        return np.real(np.exp(-1j * u_val * lnK) * phi / (1j * u_val))

    vals = np.array([integrand(uv) for uv in u])
    integral = np.sum(w * vals)
    return 0.5 + (1/np.pi) * integral


def heston_call(S0: float, K: float, T: float, p: HestonParams) -> float:
    disc_r = np.exp(-p.r * T)
    disc_q = np.exp(-p.q * T)
    P1 = _Pj(1, S0, K, T, p)
    P2 = _Pj(2, S0, K, T, p)
    return S0 * disc_q * P1 - K * disc_r * P2

def heston_put(S0: float, K: float, T: float, p: HestonParams) -> float:
    disc_r = np.exp(-p.r * T)
    disc_q = np.exp(-p.q * T)
    return heston_call(S0, K, T, p) - S0 * disc_q + K * disc_r
