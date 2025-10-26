from dataclasses import dataclass

@dataclass(frozen=True)
class HestonParams:
    kappa: float #mean reversion
    theta: float #long-run variance
    sigma: float # vol of vol
    rho: float # corr(dW1, dW2)
    v0: float #init variance
    r: float = 0.0 #rate
    q: float = 0.0 # dividend/borrow
