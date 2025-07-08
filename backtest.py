import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def load_ohlc(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(
        ticker,
        start=start,
        end=end,
        progress=False,
        auto_adjust=False,
        multi_level_index=False,
    )
    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = df.columns.get_level_values(0)
        lvl1 = df.columns.get_level_values(1)
        ohlcv = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
        df.columns = lvl0 if ohlcv.issubset(lvl0) else lvl1
    if "Close" not in df.columns and "Adj Close" in df.columns:
        df = df.rename(columns={"Adj Close": "Close"})
    need = ["Open", "High", "Low", "Close", "Volume"]
    missing = list(set(need) - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns from Yahoo Finance: {missing} – got {list(df.columns)}")
    return df[need].dropna().astype("float64")


def detect_pivots(series: pd.Series, window: int = 2):
    roll_max = series.rolling(window * 2 + 1, center=True).max()
    roll_min = series.rolling(window * 2 + 1, center=True).min()
    is_top = (series == roll_max).fillna(False)
    is_bot = (series == roll_min).fillna(False)
    return is_top, is_bot


def compute_rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0).rolling(length).mean()
    dn = -delta.clip(upper=0).rolling(length).mean()
    return 100 - 100 / (1 + up / dn)

def bull_bear_engulfing(df):
    prev_o, prev_c = df["Open"].shift(1), df["Close"].shift(1)
    o, c = df["Open"], df["Close"]
    bull = (prev_c < prev_o) & (c > o) & (o <= prev_c) & (c >= prev_o)
    bear = (prev_c > prev_o) & (c < o) & (o >= prev_c) & (c <= prev_o)
    sig = pd.Series(0, index=df.index)
    sig.loc[bull] = 1
    sig.loc[bear] = -1
    return sig


def piercing_pattern(df):
    prev_o, prev_c = df["Open"].shift(1), df["Close"].shift(1)
    mid = (prev_o + prev_c) / 2
    bull = (
        (prev_c < prev_o)
        & (df["Open"] < df["Low"].shift(1))
        & (df["Close"] > mid)
        & (df["Close"] < prev_o)
    )
    sig = pd.Series(0, index=df.index)
    sig.loc[bull] = 1
    return sig


def dark_cloud_cover(df):
    prev_o, prev_c = df["Open"].shift(1), df["Close"].shift(1)
    mid = (prev_o + prev_c) / 2
    bear = (
        (prev_c > prev_o)
        & (df["Open"] > df["High"].shift(1))
        & (df["Close"] < mid)
        & (df["Close"] > prev_o)
    )
    sig = pd.Series(0, index=df.index)
    sig.loc[bear] = -1
    return sig


def rsi_divergence(
    df: pd.DataFrame,
    rsi: pd.Series,
    pivot_window: int = 5,
    overbought: float = 70,
    oversold: float = 30,
) -> pd.Series:
    price_top, price_bot = detect_pivots(df["High"], pivot_window)[0], detect_pivots(
        df["Low"], pivot_window
    )[1]
    rsi_top, rsi_bot = detect_pivots(rsi, pivot_window)

    top_idx = df.index[(price_top) & (rsi_top) & (rsi > overbought)]
    bot_idx = df.index[(price_bot) & (rsi_bot) & (rsi < oversold)]

    sig = pd.Series(0, index=df.index)

    last_rsi = last_price = None
    for i in top_idx:
        if last_rsi is not None and rsi[i] > last_rsi and df.loc[i, "High"] < last_price:
            sig[i] = -1
        last_rsi, last_price = rsi[i], df.loc[i, "High"]

    last_rsi = last_price = None
    for i in bot_idx:
        if last_rsi is not None and rsi[i] < last_rsi and df.loc[i, "Low"] > last_price:
            sig[i] = 1
        last_rsi, last_price = rsi[i], df.loc[i, "Low"]

    return sig


def pivot_filter(sig, price_top, price_bot):
    out = pd.Series(0, index=sig.index)
    out.loc[(sig == 1) & price_bot] = 1
    out.loc[(sig == -1) & price_top] = -1
    return out

st.set_page_config("📌 Signal‑Spotter (Daily)", layout="wide")
st.title("Signal‑Spotter 📌 – Candlestick & RSI Divergence")

with st.sidebar:
    st.markdown("### Parameters")
    ticker = st.text_input("Ticker", value="AAPL", max_chars=10)
    start = st.date_input("Start", value=pd.to_datetime("2023-01-01"))
    end = st.date_input("End", value=pd.to_datetime("today"))
    pivot_window = st.slider("Pivot window", 2, 10, value=5)
    rsi_len = st.slider("RSI length", 5, 50, value=50)
    st.markdown("### Select Signals to Display")
    signal_opts = {
        "Bull/Bear Engulfing": "engulf",
        "Piercing": "pierce",
        "Dark‑Cloud‑Cover": "dcc",
        "RSI Divergence": "rsidiv",
    }
    chosen = st.multiselect("Signals", list(signal_opts.keys()), default=list(signal_opts.keys()))
    run = st.button("📊 Show Signals")

if run:
    if not chosen:
        st.error("Please select **at least one** signal to plot.")
        st.stop()

    with st.spinner("Loading & analysing …"):
        df = load_ohlc(ticker, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
        rsi = compute_rsi(df["Close"], rsi_len)
        price_top, price_bot = detect_pivots(df["High"], pivot_window)[0], detect_pivots(
            df["Low"], pivot_window
        )[1]

        sigs = {}
        sigs["engulf"] = pivot_filter(bull_bear_engulfing(df), price_top, price_bot)
        sigs["pierce"] = pivot_filter(piercing_pattern(df), price_top, price_bot)
        sigs["dcc"] = pivot_filter(dark_cloud_cover(df), price_top, price_bot)
        sigs["rsidiv"] = rsi_divergence(df, rsi, pivot_window, 70, 30)

        colour_map = {
            "engulf_buy": ("green", "Bull Engulf"),
            "engulf_sell": ("red", "Bear Engulf"),
            "pierce_buy": ("lime", "Piercing"),
            "dcc_sell": ("maroon", "Dark Cloud"),
            "rsidiv_buy": ("blue", "RSI Div Bull"),
            "rsidiv_sell": ("orange", "RSI Div Bear"),
        }

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3],
            subplot_titles=(f"{ticker} Price", "RSI"),
        )

        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name="Price",
            ),
            row=1,
            col=1,
        )

        def add_marker(mask, y_col, color, name):
            pts = df.loc[mask]
            fig.add_trace(
                go.Scatter(
                    x=pts.index,
                    y=pts[y_col],
                    mode="markers",
                    marker=dict(size=10, color=color, line=dict(width=1, color="black")),
                    name=name,
                    hoverinfo="name",
                ),
                row=1,
                col=1,
            )

        for key, code in signal_opts.items():
            if key not in chosen:
                continue
            s = sigs[code]
            if code == "engulf":
                add_marker(s == 1, "Low", *colour_map["engulf_buy"])
                add_marker(s == -1, "High", *colour_map["engulf_sell"])
            elif code == "pierce":
                add_marker(s == 1, "Low", *colour_map["pierce_buy"])
            elif code == "dcc":
                add_marker(s == -1, "High", *colour_map["dcc_sell"])
            elif code == "rsidiv":
                add_marker(s == 1, "Low", *colour_map["rsidiv_buy"])
                add_marker(s == -1, "High", *colour_map["rsidiv_sell"])

        fig.add_trace(
            go.Scatter(x=df.index, y=rsi, mode="lines", name="RSI"),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=[70] * len(df), mode="lines", line=dict(dash="dash"), name="70 OB", hoverinfo="skip"),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=[30] * len(df), mode="lines", line=dict(dash="dash"), name="30 OS", hoverinfo="skip"),
            row=2,
            col=1,
        )

        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
        fig.update_layout(
            title=f"{ticker} — Selected Signals", xaxis_rangeslider_visible=False, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), height=900,
        )

            st.plotly_chart(fig, use_container_width=True)

