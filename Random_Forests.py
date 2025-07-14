from __future__ import annotations

import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from pandas.tseries.offsets import BDay
from scipy.stats import loguniform, norm, uniform, randint
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(
    page_title="Probabilistic Random Forest Forecast",
    page_icon="🌳",
    layout="centered",
    initial_sidebar_state="auto",
)

st.title("🌳 Πιθανότητα θετικής απόδοσης (Random Forest)")
st.markdown(
    """
    Δώσε τον **Ticker** (σύμβολο Yahoo Finance) και ημερομηνία λήξης.
    Το μοντέλο εκπαιδεύεται σε ημερήσια δεδομένα από **1 Ιανουαρίου 2015**⋯ ή,
    αν το σύμβολο ξεκίνησε αργότερα, από την πρώτη διαθέσιμη μέρα.

    Το αποτέλεσμα είναι η πιθανότητα (με διάστημα εμπιστοσύνης Wilson)
    να κλείσει με θετική απόδοση τις επόμενες 5 trading days.
    """
)


def get_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, progress=False)[["Close"]]
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
    return df.dropna()


def prepare_supervised(
    df: pd.DataFrame,
    n_lags: int = 10,
    use_tech: bool = True,
    tech_lags: int = 0,
):
    cols_ret = [f"lag_{i}" for i in range(n_lags, 0, -1)]
    X_ret = np.column_stack([df["log_ret"].shift(i) for i in range(1, n_lags + 1)])
    X = pd.DataFrame(X_ret, columns=cols_ret, index=df.index)

    if use_tech:
        tech_cols = ["SMA_14", "RSI_14"]
        X_tech = df[tech_cols].copy()
        if tech_lags > 0:
            lagged = {
                f"{col}_lag{i}": df[col].shift(i)
                for col in tech_cols
                for i in range(1, tech_lags + 1)
            }
            X_tech = pd.concat([X_tech, pd.DataFrame(lagged, index=df.index)], axis=1)
        X = pd.concat([X, X_tech], axis=1)

    y = df["log_ret"].copy()
    data = pd.concat([X, y], axis=1).dropna()
    data.columns = data.columns.map(str)
    return data.iloc[:, :-1], data["log_ret"]


def train_rf(X_train: pd.DataFrame, y_train: pd.Series, *, seed: int = 42):
    """Train a Random‑Forest Regressor with time‑series cross‑validation."""
    tscv = TimeSeriesSplit(n_splits=5)
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),  # scaling not strictly necessary but harmless
            ("rf", RandomForestRegressor(random_state=seed, n_jobs=-1)),
        ]
    )

    # Hyper‑parameter search space
    param_dist = {
        "rf__n_estimators": randint(200, 601),
        "rf__max_depth": randint(3, 31),
        "rf__min_samples_split": randint(2, 16),
        "rf__min_samples_leaf": randint(1, 9),
        "rf__max_features": uniform(0.4, 0.6),  # 0.4 → 1.0
    }

    search = RandomizedSearchCV(
        pipe,
        param_dist,
        n_iter=40,
        cv=tscv,
        scoring="neg_root_mean_squared_error",
        random_state=seed,
        n_jobs=-1,
        verbose=0,
    )

    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_


def wilson_ci(k: int, n: int, conf: float):
    p = k / n
    z = norm.ppf(1 - (1 - conf) / 2)
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return center - half, center + half


def simulate_paths(model, last_row, lag_cols, resid, *, n_steps=5, n_boot=1000, seed=42):
    paths = np.empty((n_boot, n_steps))
    lag_idx = np.array([last_row.index.get_loc(c) for c in lag_cols])
    base_feat = last_row.values.astype(float)
    rng = np.random.default_rng(seed)

    for b in range(n_boot):
        feat = base_feat.copy()
        for h in range(n_steps):
            mu = model.predict(feat.reshape(1, -1))[0]
            eps = rng.choice(resid)
            paths[b, h] = mu + eps
            feat[lag_idx[:-1]] = feat[lag_idx[1:]]
            feat[lag_idx[-1]] = mu
    return paths


def forecast_prob(df: pd.DataFrame, *, forecast=5, ci=0.9):
    X, y = prepare_supervised(df, n_lags=10, use_tech=True)
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model, best_params = train_rf(X_train, y_train)

    rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))

    resid = y_train.values - model.predict(X_train)
    lag_cols = [c for c in X.columns if c.startswith("lag_")]
    last_row = X.iloc[-1]
    paths = simulate_paths(model, last_row, lag_cols, resid, n_steps=forecast)

    start_date = df.index[-1] + BDay(1)
    pred_dates = pd.bdate_range(start_date, periods=forecast)

    result_rows = []
    for i, d in enumerate(pred_dates):
        step = paths[:, i]
        k_up = int((step > 0).sum())
        p_up = k_up / len(step)
        low, high = wilson_ci(k_up, len(step), ci)
        result_rows.append(
            {
                "date": d.date(),
                "P(up)": p_up,
                f"{int(ci*100)}%_low": low,
                f"{int(ci*100)}%_high": high,
            }
        )

    return pd.DataFrame(result_rows), rmse, best_params


with st.sidebar:
    st.header("Ρυθμίσεις")
    ticker = st.text_input("Ticker", value="MNQ=F")
    end_date = st.date_input("End date", value=date.today())
    run_btn = st.button("▶️ Run")
    ci_value = st.slider("Confidence level", 0.80, 0.99, 0.90, 0.01)

if run_btn:
    with st.spinner("Λήψη δεδομένων & εκπαίδευση μοντέλου…"):
        df_raw = get_data(ticker, start="2015-01-01", end=str(end_date))
        if df_raw.empty:
            st.error("Δεν βρέθηκαν δεδομένα για αυτό το σύμβολο.")
            st.stop()
        df = add_technical_indicators(df_raw)

        results_df, rmse_val, best_params = forecast_prob(df, ci=ci_value)

    st.subheader("Αποτελέσματα")
    st.write(f"Test RMSE (log‑ret): **{rmse_val:.6f}**")
    with st.expander("Βέλτιστες υπερ‑παράμετροι"):
        st.json({k: (int(v) if isinstance(v, (int, np.integer)) else float(v)) for k, v in best_params.items()}, expanded=False)

    st.table(
        results_df.style.format(
            {
                "P(up)": "{:.2%}",
                f"{int(ci_value*100)}%_low": "{:.2%}",
                f"{int(ci_value*100)}%_high": "{:.2%}",
            }
        )
    )

    st.markdown("—")
    st.caption("© 2025 Probabilistic RF Demo — μόνο για εκπαιδευτική χρήση")
