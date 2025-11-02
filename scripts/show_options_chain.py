#!/usr/bin/env python3
"""Interactive helper to inspect option chains via yfinance."""

from __future__ import annotations

import argparse

import pandas as pd
import yfinance as yf

from data.options_chain import _spot_from_ticker


def main() -> None:
    parser = argparse.ArgumentParser(description="Display a filtered options chain from yfinance")
    parser.add_argument("ticker", help="Equity ticker symbol (e.g., AAPL)")
    parser.add_argument(
        "--expiry",
        help="Expiration date (YYYY-MM-DD). If omitted, the first available expiry is used.",
    )
    parser.add_argument("--window", type=int, default=20, help="Strike window around ATM (default: 20)")
    args = parser.parse_args()

    ticker_symbol = args.ticker.strip().upper()
    tk = yf.Ticker(ticker_symbol)
    spot = _spot_from_ticker(tk)
    print(f"Spot price for {ticker_symbol}: {spot:.4f}\n")

    exp_dates = tk.options
    if not exp_dates:
        raise SystemExit("No options data found for the selected ticker.")

    if args.expiry:
        if args.expiry not in exp_dates:
            raise SystemExit(f"Expiration {args.expiry} is not available.")
        selected_date = args.expiry
    else:
        selected_date = exp_dates[0]
        print(f"Using first available expiration date: {selected_date}\n")

    chain = tk.option_chain(selected_date)
    calls_df, puts_df = chain.calls.copy(), chain.puts.copy()

    atm = min(calls_df["strike"], key=lambda strike: abs(strike - spot))
    window = args.window
    calls_df = calls_df.loc[(calls_df["strike"] >= atm - window) & (calls_df["strike"] <= atm + window)]
    puts_df = puts_df.loc[(puts_df["strike"] >= atm - window) & (puts_df["strike"] <= atm + window)]

    calls_df = calls_df.sort_values("strike", ascending=True).tail(20)
    puts_df = puts_df.sort_values("strike", ascending=False).head(20)

    pd.set_option("display.max_columns", None)
    print(f"{ticker_symbol} options chain for expiration date {selected_date}\n")
    print("Calls:\n", calls_df[["contractSymbol", "strike", "lastPrice", "bid", "ask", "impliedVolatility"]])
    print("\nPuts:\n", puts_df[["contractSymbol", "strike", "lastPrice", "bid", "ask", "impliedVolatility"]])
if __name__ == "__main__":
    main()
