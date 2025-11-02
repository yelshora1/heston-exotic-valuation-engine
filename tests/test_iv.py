import numpy as np

from utils.iv import bs_call_price, implied_vol_call


def test_implied_vol_call_clamps_into_bracket():
    S0, K, T, r, q = 100.0, 80.0, 0.5, 0.01, 0.0
    true_vol = 0.6
    price = bs_call_price(S0, K, T, r, q, true_vol)

    # Perturb the price above the nominal bracket to exercise clamping
    inflated_price = price * 1.05
    iv = implied_vol_call(inflated_price, S0, K, T, r, q)
    assert np.isfinite(iv)
    assert 0.0001 <= iv <= 5.0

    back_price = bs_call_price(S0, K, T, r, q, iv)
    assert abs(back_price - price) / price < 0.05
