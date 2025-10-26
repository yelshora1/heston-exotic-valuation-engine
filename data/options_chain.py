import yfinance as yf
import pandas as pd
from scipy.stats import norm
from datetime import datetime, timezone

if __name__ == "__main__":

    ticker_symbol = input("Enter the ticker symbol (e.g., AAPL): ").strip().upper()
    tk = yf.Ticker(ticker_symbol)
    current_price = tk.info['currentPrice']
    print(f"The Current price of {ticker_symbol} is {current_price}\n")

    exp_dates = tk.options
    print("Available expiration dates:", exp_dates, "\n")
    if not exp_dates:
        raise SystemExit("No Options data found.")

    if selected_date := input("Enter the expiration date (YYYY-MM-DD) or press Enter to select the first available date: ").strip():
        if selected_date not in exp_dates:
            raise SystemExit("Invalid expiration date selected.")
    else:
        selected_date = exp_dates[0]
        print(f"Fetching options for expiration date...: {selected_date}\n")

    chain = tk.option_chain(selected_date)
    calls_df, puts_df = chain.calls.copy(), chain.puts.copy()

    atm = min(calls_df['strike'], key=lambda k: abs(k - current_price))
    calls_df = calls_df.loc[(calls_df['strike'] >= atm - 20) & (calls_df['strike'] <= atm + 20)]
    puts_df = puts_df.loc[(puts_df['strike'] >= atm - 20) & (puts_df['strike'] <= atm + 20)]

    calls_df = calls_df.sort_values('strike', ascending=True).tail(20)
    puts_df = puts_df.sort_values('strike', ascending=False).head(20)

    print(f"{ticker_symbol} Options chain for expiration date {selected_date}")
    print("Calls:\n", calls_df[['contractSymbol', 'strike', 'lastPrice', 'bid', 'ask', 'impliedVolatility']])
    print("\nPuts:\n", puts_df[['contractSymbol', 'strike', 'lastPrice', 'bid', 'ask', 'impliedVolatility']])

def get_chain(ticker_symbol: str, selected_date: str, strike_window: int = 20):
    import yfinance as yf
    tk = yf.Ticker(ticker_symbol)
    spot = tk.info['currentPrice']
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
