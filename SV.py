from __future__ import annotations

import warnings
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from pandas.tseries.offsets import BDay
from scipy.stats import loguniform, norm, uniform
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings("ignore")

# ─────────────────────────────── UI CONFIG ────────────────────────────────

st.set_page_config(
    page_title="Probabilistic SVM Forecast + Volatility",
    page_icon="📈",
    layout="centered",
    initial_sidebar_state="auto",
)

st.title("📈 Πιθανότητα θετικής απόδοσης & Ημερήσια Μεταβλητότητα (SVM)")
st.markdown(
    """
    Δώσε τον **Ticker** (σύμβολο Yahoo Finance) και ημερομηνία λήξης.
    Το μοντέλο εκπαιδεύεται σε ημερήσια δεδομένα από **1 Ιανουαρίου 2015**⋯ ή,
    αν το σύμβολο ξεκίνησε αργότερα, από την πρώτη διαθέσιμη μέρα.

    Το αποτέλεσμα είναι:
    * η πιθανότητα να κλείσει με **θετική απόδοση** τις επόμενες 5 trading days (με Wilson CI)
    * η **ημερήσια μεταβλητότητα** (τυπική απόκλιση % μεταβολής τιμής) από 1.000 προσομοιωμένες διαδρομές
    """
)

# ──────────────────────── DATA & FEATURE ENGINEERING ───────────────────────

def get_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download Close & compute log‑returns."""
    df = yf.download(ticker, start=start, end=end, progress=False)[["Close"]]
    df.dropna(inplace=True)
    df["log_ret"] = np.log(df["Close"]).diff()
    return df.dropna()


def add_technical_indicators(df: pd.DataFrame, win: int = 14) -> pd.DataFrame:
    """Add SMA & RSI (14‑day)."""
    df = df.copy()
    df[f"SMA_{win}"] = df["Close"].rolling(win).mean()

    delta = df["Close"].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    rs = up.rolling(win).mean() / down.rolling(win).mean()
    df[f"RSI_{win}"] = 100 - (100 / (1 + rs))
    return df.dropna()


# ---------------------------------------------------------------------------
# -------------------------- SUPERVISED MATRIX ------------------------------
# ---------------------------------------------------------------------------

def prepare_supervised(
    df: pd.DataFrame,
    *,
    n_lags: int = 10,
    tech_lags: int = 10,
):
    """Return X, y with mandatory lagged log‑ret & SMA/RSI.

    Target is **next‑day** log‑return (shift −1) so the model truly forecasts one
    step ahead.
    """
    # 1) log‑return lags
    cols_ret = [f"lag_{i}" for i in range(n_lags, 0, -1)]
    X_ret = np.column_stack([df["log_ret"].shift(i) for i in range(1, n_lags + 1)])
    X = pd.DataFrame(X_ret, columns=cols_ret, index=df.index)

    # 2) technicals + THEIR lags
    tech_cols = ["SMA_14", "RSI_14"]
    X_tech = df[tech_cols].copy()

    lagged = {
        f"{col}_lag{i}": df[col].shift(i)
        for col in tech_cols
        for i in range(1, tech_lags + 1)
    }
    X_tech = pd.concat([X_tech, pd.DataFrame(lagged, index=df.index)], axis=1)

    # Combine features
    X = pd.concat([X, X_tech], axis=1)

    # ---------------------- TARGET: next‑day log‑return ---------------------
    y = df["log_ret"].shift(-1)

    data = pd.concat([X, y], axis=1).dropna()
    data.columns = data.columns.map(str)
    return data.iloc[:, :-1], data.iloc[:, -1]


# ────────────────────────────── MODEL TRAINING ─────────────────────────────

def train_svm(X_train: pd.DataFrame, y_train: pd.Series, *, seed: int = 42):
    tscv = TimeSeriesSplit(n_splits=5)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svr", SVR(kernel="rbf")),
    ])
    param_dist = {
        "svr__C": loguniform(1e-1, 1e3),
        "svr__gamma": loguniform(1e-4, 1e1),
        "svr__epsilon": uniform(1e-4, 0.1),
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


# ────────────────────────────── UTILITIES ──────────────────────────────────

def wilson_ci(k: int, n: int, conf: float):
    p = k / n
    z = norm.ppf(1 - (1 - conf) / 2)
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return center - half, center + half


def simulate_paths(
    model,
    last_row: pd.Series,
    lag_cols: list[str],
    resid: np.ndarray,
    *,
    n_steps: int = 5,
    n_boot: int = 1000,
    seed: int = 42,
):
    """Monte‑Carlo simulation of **next‑day** log‑return paths (bootstrap residuals)."""
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
            # roll log‑return lags ahead by 1
            feat[lag_idx[:-1]] = feat[lag_idx[1:]]
            feat[lag_idx[-1]] = mu
    return paths


# ────────────────────────────── FORECASTING ────────────────────────────────

def forecast_prob(df: pd.DataFrame, *, forecast: int = 5, ci: float = 0.9):
    """Return probability/volatility table, RMSE & hyper‑params."""
    X, y = prepare_supervised(df, n_lags=10, tech_lags=10)

    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model, best_params = train_svm(X_train, y_train)
    rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))

    # ------------------------- Monte‑Carlo simulation -----------------------
    resid = y_train.values - model.predict(X_train)
    lag_cols = [c for c in X.columns if c.startswith("lag_")]
    last_row = X.iloc[-1]
    paths = simulate_paths(model, last_row, lag_cols, resid, n_steps=forecast)

    start_date = df.index[-1] + BDay(1)
    pred_dates = pd.bdate_range(start_date, periods=forecast)

    rows = []
    for i, d in enumerate(pred_dates):
        step = paths[:, i]
        pct_step = np.exp(step) - 1  # convert to % price change
        sigma = pct_step.std()

        k_up = int((step > 0).sum())
        p_up = k_up / len(step)
        low, high = wilson_ci(k_up, len(step), ci)

        rows.append(
            {
                "date": d.date(),
                "P(up)": p_up,
                f"{int(ci*100)}%_low": low,
                f"{int(ci*100)}%_high": high,
                "volatility": sigma,
            }
        )

    return pd.DataFrame(rows), rmse, best_params


# ──────────────────────────── STREAMLIT APP ───────────────────────────────

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
        st.json({k: float(v) for k, v in best_params.items()}, expanded=False)

    fmt = {
        "P(up)": "{:.2%}",
        f"{int(ci_value*100)}%_low": "{:.2%}",
        f"{int(ci_value*100)}%_high": "{:.2%}",
        "volatility": "{:.2%}",
    }
    st.table(results_df.style.format(fmt))

    st.markdown("### Ημερήσια μεταβλητότητα από προσομοίωση")
    st.line_chart(results_df.set_index("date")["volatility"])

    st.markdown("—")
    st.caption("© 2025 Probabilistic SVM Demo — μόνο για εκπαιδευτική χρήση")
