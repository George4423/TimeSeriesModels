import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from math import sqrt

# ───────────────── Loader ─────────────────
@st.cache_data
def load_price_data(ticker, start, end):
    s = yf.download(ticker, start=start, end=end,
                    progress=False)["Close"].squeeze()
    s.name = ticker
    return s

# ───────────────── Strategies ─────────────────
def golden_cross(pr, fast=50, slow=200):
    return pd.Series(
        (pr.rolling(fast).mean() > pr.rolling(slow).mean()).astype(int),
        index=pr.index
    ).diff().fillna(0)

def ma_crossover(pr, short=20, long=100):
    return pd.Series(
        (pr.rolling(short).mean() > pr.rolling(long).mean()).astype(int),
        index=pr.index
    ).diff().fillna(0)

def rsi_strategy(pr, per=14, low=30, high=70):
    d = pr.diff()
    up = d.clip(lower=0).rolling(per).mean()
    dn = -d.clip(upper=0).rolling(per).mean()
    rsi = 100 - 100 / (1 + up / dn)
    pos = (rsi < low).astype(int) - (rsi > high).astype(int)
    return pos.diff().fillna(0)

def backtest(prices, trades):
    r = prices.pct_change().fillna(0)
    strat = r * trades.shift().fillna(0)
    return {
        "Sharpe": round(sqrt(252) * strat.mean() / strat.std(ddof=0), 3),
        "P&L %": round(strat.sum() * 100, 2),
        "VaR 95%": round(np.percentile(strat.dropna(), 5), 4),
        "cum": (1 + strat).cumprod(),
        "daily": strat
    }

# ───────────────── Streamlit UI ─────────────────
st.set_page_config(page_title="Quick Backtester", layout="wide")
st.title("📈 Quick Backtester (Close Prices)")

with st.sidebar:
    ticker = st.text_input("Ticker", "AAPL")
    start  = st.date_input("Start", value=pd.to_datetime("2015-01-01"))
    end    = st.date_input("End",   value=pd.to_datetime("today"))
    strat_name = st.selectbox("Strategy", ("Golden Cross", "MA Crossover", "RSI"))

    if strat_name == "Golden Cross":
        fast = st.number_input("Fast SMA", 5, 200, 50)
        slow = st.number_input("Slow SMA", 20, 400, 200)
    elif strat_name == "MA Crossover":
        short = st.number_input("Short MA", 5, 100, 20)
        long  = st.number_input("Long MA",  20, 400, 100)
    else:
        per  = st.number_input("RSI Period", 5, 50, 14)
        low  = st.number_input("Oversold", 10, 50, 30)
        high = st.number_input("Overbought", 50, 90, 70)

    run = st.button("Run Backtest ▶")

if run:
    with st.spinner("Downloading & crunching…"):
        price = load_price_data(ticker, start, end)

        if strat_name == "Golden Cross":
            trades = golden_cross(price, fast, slow)
        elif strat_name == "MA Crossover":
            trades = ma_crossover(price, short, long)
        else:
            trades = rsi_strategy(price, per, low, high)

        res = backtest(price, trades)

    st.metric("Sharpe Ratio", res["Sharpe"])
    st.metric("Total P&L (%)", f"{res['P&L %']}%")
    st.metric("VaR 95%", res["VaR 95%"])

    st.line_chart(res["cum"].rename("Cumulative Returns"))
    st.area_chart(res["daily"].rename("Daily Strategy Returns"))

    st.download_button("Download daily returns (CSV)",
                       data=res["daily"].to_csv().encode(),
                       file_name=f"{ticker}_{strat_name.replace(' ','_')}_returns.csv",
                       mime="text/csv")
else:
    st.info("Set parameters on the left and press **Run Backtest ▶**")
