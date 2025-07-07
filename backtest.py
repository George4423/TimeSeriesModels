import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ────────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_ohlc(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Download full OHLCV data from Yahoo! Finance."""
    data = yf.download(ticker, start=start, end=end, progress=False)
    # Make sure we keep only the columns we need and fill missing values if any
    data = data[["Open", "High", "Low", "Close", "Volume"]].dropna()
    return data


def resample_ohlc(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    """Resample an OHLCV DataFrame to Daily / Weekly / Monthly."""
    if tf == "Daily":
        return df.copy()

    rule = {"Weekly": "W", "Monthly": "M"}[tf]
    agg = {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }
    rs = df.resample(rule).agg(agg).dropna()
    rs.index.name = df.index.name
    return rs


# ────────────────────────────────────────────────────────────────────────────────
# Candlestick‑pattern detectors (vectorised, TA‑Lib‑free)
# ────────────────────────────────────────────────────────────────────────────────

def bull_bear_engulfing(df: pd.DataFrame) -> pd.Series:
    """Return +1 (bull), -1 (bear), 0 (none) for engulfing patterns."""
    prev_open = df["Open"].shift(1)
    prev_close = df["Close"].shift(1)
    o, c = df["Open"], df["Close"]

    # Bullish engulfing – previous red, current green engulfing body
    bull = (
        prev_close < prev_open  # previous bearish
        ) & (
        c > o  # current bullish
        ) & (
        o <= prev_close
        ) & (
        c >= prev_open
    )

    # Bearish engulfing – previous green, current red engulfing body
    bear = (
        prev_close > prev_open  # previous bullish
        ) & (
        c < o  # current bearish
        ) & (
        o >= prev_close
        ) & (
        c <= prev_open
    )

    sig = pd.Series(0, index=df.index)
    sig[bull] = 1
    sig[bear] = -1
    return sig


def piercing_pattern(df: pd.DataFrame) -> pd.Series:
    """Bullish piercing pattern (+1) / none (0)."""
    prev_open = df["Open"].shift(1)
    prev_close = df["Close"].shift(1)
    midpoint = (prev_open + prev_close) / 2

    bull_pierce = (
        prev_close < prev_open  # previous bearish
        ) & (
        df["Open"] < df["Low"].shift(1)  # gap down
        ) & (
        df["Close"] > midpoint  # close above body midpoint
        ) & (
        df["Close"] < prev_open  # but below previous open
    )

    sig = pd.Series(0, index=df.index)
    sig[bull_pierce] = 1
    return sig


def dark_cloud_cover(df: pd.DataFrame) -> pd.Series:
    """Bearish dark‑cloud‑cover pattern (‑1) / none (0)."""
    prev_open = df["Open"].shift(1)
    prev_close = df["Close"].shift(1)
    midpoint = (prev_open + prev_close) / 2

    bear_dcc = (
        prev_close > prev_open  # previous bullish
        ) & (
        df["Open"] > df["High"].shift(1)  # gap up
        ) & (
        df["Close"] < midpoint  # close below body midpoint
        ) & (
        df["Close"] > prev_open  # but above prev open
    )

    sig = pd.Series(0, index=df.index)
    sig[bear_dcc] = -1
    return sig


# ────────────────────────────────────────────────────────────────────────────────
# RSI divergence (naïve implementation)
# ────────────────────────────────────────────────────────────────────────────────

def rsi_divergence(df: pd.DataFrame, rsi_len: int = 14, lookback: int = 10) -> pd.Series:
    """Detect simple bullish / bearish RSI divergence.

    +1 → bullish divergence (price lower‑low but RSI higher‑low)
    ‑1 → bearish divergence (price higher‑high but RSI lower‑high)
    0  → none
    """
    # RSI
    delta = df["Close"].diff()
    up = delta.clip(lower=0).rolling(rsi_len).mean()
    down = -delta.clip(upper=0).rolling(rsi_len).mean()
    rsi = 100 - 100 / (1 + up / down)

    # Windows to compare
    price_change = df["Close"] - df["Close"].shift(lookback)
    rsi_change = rsi - rsi.shift(lookback)

    bull = (price_change < 0) & (rsi_change > 0)
    bear = (price_change > 0) & (rsi_change < 0)

    sig = pd.Series(0, index=df.index)
    sig[bull] = 1
    sig[bear] = -1
    return sig

# ────────────────────────────────────────────────────────────────────────────────
# Streamlit App
# ────────────────────────────────────────────────────────────────────────────────
st.set_page_config("📌 Pattern‑Spotter", layout="wide")
st.title("Pattern‑Spotter 📌 – RSI Divergence & Candlesticks")

with st.sidebar:
    ticker = st.text_input("Ticker", value="AAPL", max_chars=10)
    start = st.date_input("Start", value=pd.to_datetime("2015-01-01"))
    end = st.date_input("End", value=pd.to_datetime("today"))
    timeframe = st.selectbox("Time‑Frame", ("Daily", "Weekly", "Monthly"))
    lookback = st.slider("RSI divergence lookback bars", 3, 30, value=10)
    rsi_len = st.slider("RSI length", 5, 30, value=14)
    run = st.button("📊 Show Signals")

if run:
    with st.spinner("Loading & analysing …"):
        raw = load_ohlc(ticker, start, end)
        df = resample_ohlc(raw, timeframe)

        # --- Detect patterns ---------------------------------------------------
        sig_engulf = bull_bear_engulfing(df)
        sig_pierce = piercing_pattern(df)
        sig_dcc = dark_cloud_cover(df)
        sig_rsi   = rsi_divergence(df, rsi_len=rsi_len, lookback=lookback)

        # --- Plotting with Plotly ---------------------------------------------
        fig = go.Figure()
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name="Price",
            )
        )

        # Helper to add markers -------------------------------------------------
        def add_marker(mask: pd.Series, y_col: str, color: str, name: str):
            pts = df.loc[mask]
            fig.add_trace(
                go.Scatter(
                    x=pts.index,
                    y=pts[y_col],
                    mode="markers",
                    marker=dict(symbol="circle", size=10, color=color, line=dict(width=1, color="black")),
                    name=name,
                )
            )

        add_marker(sig_engulf == 1, "Low", "green", "Bull Engulf")
        add_marker(sig_engulf == -1, "High", "red", "Bear Engulf")
        add_marker(sig_pierce == 1, "Low", "lime", "Piercing")
        add_marker(sig_dcc == -1, "High", "maroon", "Dark Cloud")
        add_marker(sig_rsi == 1, "Low", "blue", "RSI Div Bull")
        add_marker(sig_rsi == -1, "High", "orange", "RSI Div Bear")

        # Layout tweaks --------------------------------------------------------
        fig.update_layout(
            xaxis_rangeslider_visible=False,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            title=f"{ticker} – {timeframe} Signals",
            height=700,
        )

    st.plotly_chart(fig, use_container_width=True)

    # Show data table optionally ---------------------------------------------
    with st.expander("🗃️ Raw data & signals"):
        signals = pd.DataFrame({
            "Bull_Engulf": (sig_engulf == 1).astype(int),
            "Bear_Engulf": (sig_engulf == -1).astype(int),
            "Piercing": (sig_pierce == 1).astype(int),
            "Dark_Cloud": (sig_dcc == -1).astype(int),
            "RSI_Div_Bull": (sig_rsi == 1).astype(int),
            "RSI_Div_Bear": (sig_rsi == -1).astype(int),
        })
        st.dataframe(pd.concat([df, signals], axis=1))
else:
    st.info("⏪ Fill the parameters on the left and press **Show Signals** to begin.")
