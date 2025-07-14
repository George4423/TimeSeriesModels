from __future__ import annotations

import io
from typing import Tuple, List

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from pandas.tseries.offsets import BDay
from scipy.stats import loguniform, norm, uniform
from sklearn.metrics import accuracy_score, mean_squared_error, make_scorer
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

START_DATE = "2015-01-01"
TRAIN_FRAC = 0.8
SEED = 42
N_LAGS = 10
FORECAST = 5
N_BOOT = 1000
WIN_TECH = 10

def sanitize(df: pd.DataFrame) -> pd.DataFrame:
    return df.replace([np.inf, -np.inf], np.nan).dropna()

def get_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)[
        "Open High Low Close Volume".split()
    ]
    df["log_ret"] = np.log(df["Close"]).diff()
    return sanitize(df)


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[f"SMA_{WIN_TECH}"] = df["Close"].rolling(WIN_TECH).mean()
    delta = df["Close"].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    rs = up.rolling(WIN_TECH).mean() / down.rolling(WIN_TECH).mean()
    df[f"RSI_{WIN_TECH}"] = 100 - (100 / (1 + rs))
    return sanitize(df)


def prepare_supervised(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    cols_ret = [f"lag_{i}" for i in range(N_LAGS, 0, -1)]
    X_ret = np.column_stack([df["log_ret"].shift(i) for i in range(1, N_LAGS + 1)])
    X = pd.DataFrame(X_ret, columns=cols_ret, index=df.index)
    X = pd.concat([X, df[[f"SMA_{WIN_TECH}", f"RSI_{WIN_TECH}"]]], axis=1)
    y = df["log_ret"].copy()
    return sanitize(pd.concat([X, y], axis=1)).iloc[:, :-1], sanitize(pd.concat([X, y], axis=1))["log_ret"]

def dir_acc(y_true, y_pred):
    return accuracy_score(y_true > 0, y_pred > 0)


def train_svm(X_train: pd.DataFrame, y_train: pd.Series):
    # adaptive CV splits
    n_splits = min(5, max(2, len(X_train) // 50))
    tscv = TimeSeriesSplit(n_splits=n_splits)
    pipe = Pipeline([("scaler", StandardScaler()), ("svr", SVR(kernel="rbf"))])
    param_dist = {
        "svr__C": loguniform(1e-1, 1e3),
        "svr__gamma": loguniform(1e-4, 1e1),
        "svr__epsilon": uniform(1e-4, 0.1),
    }
    search = RandomizedSearchCV(
        pipe,
        param_dist,
        n_iter=50,
        cv=tscv,
        scoring=make_scorer(dir_acc, greater_is_better=True),
        random_state=SEED,
        n_jobs=-1,
        verbose=0,
        error_score="raise",
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_

def simulate_paths(
    model,
    last_row: pd.Series,
    lag_cols: List[str],
    resid: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    paths = np.empty((N_BOOT, FORECAST))
    lag_idx = np.array([last_row.index.get_loc(c) for c in lag_cols])
    feat0 = last_row.values.astype(float)

    for b in range(N_BOOT):
        feat = feat0.copy()
        for h in range(FORECAST):
            mu = model.predict(feat.reshape(1, -1))[0]
            eps = rng.choice(resid)
            step = mu + eps
            paths[b, h] = step
            feat[lag_idx[:-1]] = feat[lag_idx[1:]]  # roll lags
            feat[lag_idx[-1]] = step
    return paths


def wilson_ci(k: int, n: int, conf: float):
    p = k / n
    z = norm.ppf(1 - (1 - conf) / 2)
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return center - half, center + half

st.set_page_config(page_title="Probabilistic SVM Direction & Volatility Forecast", layout="wide")
st.title("📈 Probabilistic SVM – Direction & Volatility")

with st.sidebar:
    st.header("Parameters")
    ticker = st.text_input("Ticker (Yahoo Finance)", value="ES=F")
    st.markdown(f"**Start date:** `{START_DATE}` (fixed)")
    end_date = st.date_input("End date", value=pd.to_datetime("2025-07-10"))
    ci = st.slider("Confidence interval", 0.80, 0.99, 0.90, 0.01)

    run_btn = st.button("Run model 🚀", type="primary")

@st.cache_data(show_spinner=False)
def load_and_engineer(ticker: str, start: str, end: str) -> pd.DataFrame:
    return add_technical_indicators(get_data(ticker, start, end))

if run_btn:
    if pd.to_datetime(end_date) <= pd.to_datetime(START_DATE):
        st.error("End date must be after 2015‑01‑01.")
        st.stop()

    with st.spinner("Downloading data & training model …"):
        df = load_and_engineer(ticker, START_DATE, str(end_date))

        X, y = prepare_supervised(df)
        split = int(len(X) * TRAIN_FRAC)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]

        model, best_params = train_svm(X_train, y_train)

        y_pred_test = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        da = accuracy_score(y_test > 0, y_pred_test > 0)

        resid = y_train.values - model.predict(X_train)
        lag_cols = [c for c in X.columns if c.startswith("lag_")]
        last_row = X.iloc[-1]
        rng = np.random.default_rng(SEED)
        log_paths = simulate_paths(model, last_row, lag_cols, resid, rng)

        start = (df.index[-1] + BDay(1)).normalize()
        f_dates = pd.bdate_range(start, periods=FORECAST)
        last_price = df["Close"].iloc[-1]

    st.subheader("Model performance on test set")
    m1, m2 = st.columns(2)
    m1.metric("RMSE (log‑ret)", f"{rmse:.6f}")
    m2.metric("Directional Accuracy", f"{da:.2%}")
    st.caption(f"Best hyper‑parameters: {best_params}")

    prob_records, vol_records = [], []
    cum_log_paths = log_paths.cumsum(axis=1)

    for i, d in enumerate(f_dates):
        step = log_paths[:, i]
        k_up = int((step > 0).sum())
        p_up = k_up / N_BOOT
        low_ci, high_ci = wilson_ci(k_up, N_BOOT, ci)
        prob_records.append({
            "Date": d.date().isoformat(),
            "P(up)": p_up,
            f"CI_low_{int(ci*100)}%": low_ci,
            f"CI_high_{int(ci*100)}%": high_ci,
        })

        sigma = step.std(ddof=0)
        cum_step = cum_log_paths[:, i]
        ret_5, ret_95 = np.percentile(cum_step, [5, 95])
        price_5 = last_price * np.exp(ret_5)
        price_95 = last_price * np.exp(ret_95)
        vol_records.append({
            "Date": d.date().isoformat(),
            "σ(log_ret)": sigma,
            "ret_5%": ret_5,
            "ret_95%": ret_95,
            "price_low_5%": price_5,
            "price_high_95%": price_95,
        })

    prob_df = pd.DataFrame(prob_records)
    st.subheader("Probability of positive daily log‑return")
    st.table(prob_df.set_index("Date"))
    st.line_chart(prob_df.set_index("Date")["P(up)"])

    vol_df = pd.DataFrame(vol_records).set_index("Date")
    st.subheader("Forecasted volatility / price range (5‑95 % band)")
    st.table(vol_df[["σ(log_ret)", "ret_5%", "ret_95%", "price_low_5%", "price_high_95%"]])

    median_price = last_price * np.exp(np.percentile(cum_log_paths, 50, axis=0))
    p5 = last_price * np.exp(np.percentile(cum_log_paths, 5, axis=0))
    p95 = last_price * np.exp(np.percentile(cum_log_paths, 95, axis=0))

    fan_df = pd.DataFrame({
        "median": median_price,
        "low_5%": p5,
        "high_95%": p95,
    }, index=f_dates)

    st.subheader("Fan chart – expected price band (5‑95 %)")
    st.line_chart(fan_df)

    buf = io.BytesIO()
    np.save(buf, log_paths)
    buf.seek(0)
    st.download_button(
        "⬇️ Download simulated log‑return paths (.npy)",
        data=buf,
        file_name=f"{ticker.replace('=','')}_log_paths.npy",
    )

    st.subheader("Historical Close price")
    st.line_chart(df["Close"])
