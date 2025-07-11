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

def get_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(
        ticker,
        start=start,
        end=end,
        progress=False,
        auto_adjust=False,
    )["Open High Low Close Volume".split()]

    df.dropna(inplace=True)
    df["log_ret"] = np.log(df["Close"]).diff()
    return df.dropna()


def add_technical_indicators(df: pd.DataFrame, win: int = 14) -> pd.DataFrame:
    df = df.copy()
    df[f"SMA_{win}"] = df["Close"].rolling(win).mean()
    delta = df["Close"].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    rs = up.rolling(win).mean() / down.rolling(win).mean()
    df[f"RSI_{win}"] = 100 - (100 / (1 + rs))
    df["vol"] = df["log_ret"].rolling(win).std(ddof=0)
    volume_safe = df["Volume"].clip(lower=1)
    log_vol = np.log(volume_safe)
    df["volume_z"] = (
        (log_vol - log_vol.rolling(252).mean()) / log_vol.rolling(252).std(ddof=0)
    )

    return df.dropna()


def prepare_supervised(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    cols_ret = [f"lag_{i}" for i in range(N_LAGS, 0, -1)]
    X_ret = np.column_stack([df["log_ret"].shift(i) for i in range(1, N_LAGS + 1)])
    X = pd.DataFrame(X_ret, columns=cols_ret, index=df.index)

    tech_cols = ["Open", "SMA_14", "RSI_14", "volume_z"]
    X = pd.concat([X, df[tech_cols]], axis=1)

    y = df["log_ret"].copy()
    data = pd.concat([X, y], axis=1).dropna()
    data.columns = data.columns.map(str)
    return data.iloc[:, :-1], data["log_ret"]

def dir_acc(y_true, y_pred):
    return accuracy_score(y_true > 0, y_pred > 0)


def train_svm(X_train: pd.DataFrame, y_train: pd.Series):
    tscv = TimeSeriesSplit(n_splits=5)
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
            feat[lag_idx[:-1]] = feat[lag_idx[1:]]
            feat[lag_idx[-1]] = step
    return paths


def wilson_ci(k: int, n: int, conf: float):
    p = k / n
    z = norm.ppf(1 - (1 - conf) / 2)
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return center - half, center + half

st.set_page_config(page_title="Probabilistic SVM Direction Forecast", layout="wide")
st.title("📈 Probabilistic SVM Direction Forecast")

with st.sidebar:
    st.header("Parameters")

    ticker = st.text_input("Ticker (Yahoo Finance)", value="ES=F")

    st.markdown(f"**Start date:** `{START_DATE}` (fixed)")

    end_date = st.date_input("End date", value=pd.to_datetime("2025-07-10"))

    ci = st.slider(
        "Confidence interval", min_value=0.80, max_value=0.99, value=0.90, step=0.01
    )

    run_btn = st.button("Run model 🚀", type="primary")

@st.cache_data(show_spinner=False)
def load_and_engineer(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = get_data(ticker, start, end)
    df = add_technical_indicators(df)
    return df

if run_btn:
    if pd.to_datetime(end_date) <= pd.to_datetime(START_DATE):
        st.error("End date must be after 2015‑01‑01.")
        st.stop()

    with st.spinner("Downloading data & training model ..."):
        df = load_and_engineer(ticker, START_DATE, str(end_date))

        X, y = prepare_supervised(df)
        split = int(len(X) * TRAIN_FRAC)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]

        model, best_params = train_svm(X_train, y_train)

        rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
        da = accuracy_score(y_test > 0, model.predict(X_test) > 0)

        resid = y_train.values - model.predict(X_train)
        lag_cols = [c for c in X.columns if c.startswith("lag_")]
        last_row = X.iloc[-1]
        rng = np.random.default_rng(SEED)
        log_paths = simulate_paths(model, last_row, lag_cols, resid, rng)

        start = (df.index[-1] + BDay(1)).normalize()
        f_dates = pd.bdate_range(start, periods=FORECAST)

    st.subheader("Model performance on test set")
    col1, col2 = st.columns(2)
    col1.metric("RMSE (log‑ret)", f"{rmse:.6f}")
    col2.metric("Directional Accuracy", f"{da:.2%}")
    st.caption(f"Best hyper‑parameters: {best_params}")

    records = []
    for i, d in enumerate(f_dates):
        step = log_paths[:, i]
        k_up = int((step > 0).sum())
        p_up = k_up / N_BOOT
        low, high = wilson_ci(k_up, N_BOOT, ci)
        records.append({
            "Date": d.date().isoformat(),
            "P(up)": p_up,
            f"CI_low_{int(ci*100)}%": low,
            f"CI_high_{int(ci*100)}%": high,
        })

    prob_df = pd.DataFrame(records)
    st.subheader("Probability of positive daily log‑return")
    st.table(prob_df.set_index("Date"))
    st.line_chart(prob_df.set_index("Date")["P(up)"])

    buf = io.BytesIO()
    np.save(buf, log_paths)
    buf.seek(0)
    st.download_button(
        "Download simulated log‑return paths (.npy)",
        data=buf,
        file_name=f"{ticker.replace('=','')}_log_paths.npy",
    )

    st.subheader("Historical Close price")
    st.line_chart(df["Close"])
