import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.externals.array_api_compat.torch import result_type

TICKER = "SPY"
START = "2015-01-01"
SHORT_WIN = 50    # it shows what happen now, 50 days past
LONG_WIN = 200     # long term, it is about one stock year, long term dierection of the market
COST = 0.001

def load_prices(TICKER, START):
    try:
        import yfinance as yf
        df = yf.download(TICKER, start= START, auto_adjust=True, progress=False)       #auto_adjust - prices were corrected by share's split and dividends
        prices = df["Close"].squeeze().dropna()
        if len(prices) > LONG_WIN:
            return prices
    except Exception as e:
        print("Błąd", e)
    print("Lack of acces to the data, im using synthetic data")
    np.random.seed(42)
    dates = pd.bdate_range(START, periods= 3500)   # bdate = business date
    returns = np.random.normal(0.0004, 0.012, len(dates))   #(mean, std, length)
    return pd.Series(100 * np.exp(np.cumsum(returns)), index = dates)    # Black-Schols' model to estimate options


def backtest(prices):
    df = pd.DataFrame({"price" : prices})
    df["ret"] = df["price"].pct_change()
    df["sma_short"] = df["price"].rolling(SHORT_WIN).mean()  # moving average
    df["sma_long"] = df["price"].rolling(LONG_WIN).mean()   # moving average
    df["signal"] = (df["sma_short"] > df["sma_long"]).astype(int)  # we recive 1 or 0 despite if we want to have money or shares
    df["position"] = df["signal"].shift(1).fillna(0)    # look-ahead bias, we recieve information 1 day later
    trades = df["position"].diff().abs().fillna(0)
    df["strat_ret"] = df["position"] * df["ret"] - trades * COST

    df = df.dropna(subset=["sma_long"]).copy()
    df["equity_strat"] = (1 + df["strat_ret"]).cumprod()  #cumulative product
    df["equity_bh"] = (1 + df["ret"]).cumprod()
    return df, int(trades.loc[df.index].sum())

def metrics(returns, equity):
    years = len(returns)/ 252                       #Volatility measures how wildly the portfolio bounces up and down
    cagr = equity.iloc[-1] ** (1/years) - 1    # Compound Annual Growth Rate
    vol = returns.std() * np.sqrt(252)              #Volatility measures how wildly the portfolio bounces up and down
    sharpe = returns.mean()/ returns.std() * np.sqrt(252)   #the Sharpe ratio is return divided by risk. It shows how much you earn per unit of risk
    max_dd = (equity / equity.cummax() - 1).min()
    return {"CAGR": f"{cagr:.2%}", "Zmienność": f"{vol:.2%}",
            "Sharpe": f"{sharpe:.2f}", "Max drawdown": f"{max_dd:.2%}"}  # Max drowdown = the biggest lost

if __name__ == "__main__":
    prices = load_prices(TICKER, START)
    df, n_trades = backtest(prices)

    results = pd.DataFrame({"Strategia SMA": metrics(df["strat_ret"], df["equity_strat"]),
                            "Kup i trzymaj": metrics(df["ret"], df["equity_bh"]),})
    print(f"\n{TICKER}: {df.index[0].date()} - {df.index[-1].date()}, transakcji: {n_trades}\n")
    print(results)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize =(11, 8), sharex = True)
    ax1.plot(df["price"], label = "Cena", lw=1)
    ax1.plot(df["sma_short"], label = f"SMA {SHORT_WIN}", lw=1)
    ax1.plot(df["sma_long"], label = f"SMA {LONG_WIN}", lw =1)
    ax1.set_title(f"{TICKER} - cena i średnie kroczące"); ax1.legend()
    ax2.plot(df["equity_strat"], label="Strategia SMA")
    ax2.plot(df["equity_bh"], label = "Kup i trzymaj")
    ax2.set_title("Wartość portfela (start = 1)"); ax2.legend()
    plt.tight_layout()
    plt.savefig("wyniki.png", dpi=120)
    plt.show()
