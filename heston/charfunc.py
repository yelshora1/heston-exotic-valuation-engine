import numpy as np
from .params import HestonParams

def heston_cf(u: complex, T: float, S0: float, p: HestonParams, j: int = 2) -> complex:
    """
    Heston characteristic function for ln S_T with j-variant:
      j=1 uses b1 = kappa - rho*sigma   (for P1)
      j=2 uses b2 = kappa               (for P2)
    """
    if u == 0:
        return 1.0

    x0 = np.log(S0)
    kappa, theta, sigma, rho, v0, r, q = p.kappa, p.theta, p.sigma, p.rho, p.v0, p.r, p.q

    iu = 1j * u
    b = kappa - rho * sigma if j == 1 else kappa

    d = np.sqrt((rho * sigma * iu - b)**2 + sigma**2 * (iu + u**2))
    if np.real(d) < 0:
        d = -d  # ensure Re(d) >= 0

    num = b - rho * sigma * iu - d
    den = b - rho * sigma * iu + d
    if np.abs(den) < 1e-16:
        den = 1e-16  # avoid div by zero
    g = num / den
    if abs(g) >= 1.0:
        g = g / (abs(g) + 1e-16) * (1 - 1e-12)  # ensure |g| < 1

    exp_negdT = np.exp(-d * T)
    one_minus_g = 1 - g
    one_minus_g_exp = 1 - g * exp_negdT
    if abs(one_minus_g_exp) < 1e-16:
        one_minus_g_exp = 1e-16  # avoid div by zero


    C = (r - q) * iu * T + (kappa * theta / sigma**2) * (
        (b - rho * sigma * iu - d) * T - 2 * (np.log1p(-g * exp_negdT) - np.log1p(-g))
    )
    D = ((b - rho * sigma * iu - d) / sigma**2) * ((1 - exp_negdT) / one_minus_g_exp)

    return np.exp(C + D * v0 + iu * x0)
