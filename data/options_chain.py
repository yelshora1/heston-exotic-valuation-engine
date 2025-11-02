import yfinance as yf
import pandas as pd
from scipy.stats import norm
from datetime import datetime, timezone


def _spot_from_ticker(tk: yf.Ticker) -> float:
    fast = getattr(tk, "fast_info", None)
    if fast is not None and getattr(fast, "last_price", None):
        return float(fast.last_price)
    info = tk.info
    for key in ("currentPrice", "regularMarketPrice"):
        if key in info and info[key] is not None:
            return float(info[key])
    raise RuntimeError("Could not determine spot from yfinance metadata.")


def get_chain(ticker_symbol: str, selected_date: str, strike_window: int = 20):
    import yfinance as yf
    tk = yf.Ticker(ticker_symbol)
    spot = _spot_from_ticker(tk)
    chain = tk.option_chain(selected_date)
    calls_df, puts_df = chain.calls.copy(), chain.puts.copy()

    # IVs are already provided by yfinance in 'impliedVolatility'
    # Keep your exact windowing logic so results match the CLI output:
    atm = min(calls_df['strike'], key=lambda k: abs(k - spot))
    # after you compute 'atm'
    calls_df['dist'] = (calls_df['strike'] - atm).abs()
    puts_df['dist']  = (puts_df['strike']  - atm).abs()

    # pick the 14 closest strikes on each side (tweak n as you like)
    n = 14
    calls_df = calls_df.sort_values('dist').head(2*n).sort_values('strike').reset_index(drop=True)
    puts_df  = puts_df .sort_values('dist').head(2*n).sort_values('strike').reset_index(drop=True)

    # (optional) drop helper col
    calls_df.drop(columns=['dist'], inplace=True)
    puts_df.drop(columns=['dist'], inplace=True)

    return float(spot), calls_df, puts_df
