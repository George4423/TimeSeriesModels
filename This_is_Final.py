from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from datetime import date
from pandas.tseries.offsets import BDay
from scipy.stats import loguniform, norm, uniform
from sklearn.metrics import accuracy_score, mean_squared_error, make_scorer
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

st.set_page_config(
    page_title="Probabilistic SVM Forecast (v2)",
    page_icon="📈",
    layout="centered",
    initial_sidebar_state="auto",
)

st.title("📈 Πιθανότητες & τιμές‑στόχοι (SVM v2)")
st.markdown(
    """
    Το μοντέλο εκπαιδεύεται σε ημερήσια δεδομένα από **1 Ιαν 2015** και προσομοιώνει
    1000 μονοπάτια λογαριθμικών αποδόσεων για τις επόμενες 5 συνεδριάσεις.

    Για κάθε ημέρα εμφανίζονται:

    * **P(up)** – πιθανότητα θετικής απόδοσης  
    * **MinClose / MaxClose** – ελάχιστο & μέγιστο κλείσιμο από τις προσομοιώσεις  
    * **P(≥ x %)** – πιθανότητα η απόδοση να ξεπεράσει το κατώφλι x,  
      π.χ. `8.50 % (5804.88)`  
    """
)

def get_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(
        ticker, start=start, end=end, progress=False, auto_adjust=False
    )[["Close", "Volume"]]
    df.dropna(inplace=True)
    df["log_ret"] = np.log(df["Close"]).diff()
    return df.dropna()


def add_technical_indicators(df: pd.DataFrame, win: int = 14) -> pd.DataFrame:
    df = df.copy()
    df[f"SMA_{win}"] = df["Close"].rolling(win).mean()
    delta = df["Close"].diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    rs = up.rolling(win).mean() / down.rolling(win).mean()
    df[f"RSI_{win}"] = 100 - (100 / (1 + rs))
    df["vol"] = df["log_ret"].rolling(win).std(ddof=0)

    volume_safe = df["Volume"].clip(lower=1)
    log_vol = np.log(volume_safe)
    df["volume_z"] = (
        (log_vol - log_vol.rolling(252).mean())
        / log_vol.rolling(252).std(ddof=0)
    )
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
        base_tech_cols = ["SMA_14", "RSI_14"]
        tech_cols = base_tech_cols + ["vol", "volume_z"]

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


def dir_acc(y_true, y_pred):
    return accuracy_score(y_true > 0, y_pred > 0)


def train_svm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    n_splits: int = 5,
    seed: int = 42,
):
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
        random_state=seed,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_


def simulate_paths(
    model,
    last_row,
    lag_cols,
    resid,
    *,
    n_steps: int = 5,
    n_boot: int = 1000,
    seed: int = 42,
):
    paths = np.empty((n_boot, n_steps))
    lag_idx = np.array([last_row.index.get_loc(c) for c in lag_cols])
    base_feat = last_row.values.astype(float)
    rng = np.random.default_rng(seed)

    for b in range(n_boot):
        feat = base_feat.copy()
        for h in range(n_steps):
            mu = model.predict(feat.reshape(1, -1))[0]
            eps = rng.choice(resid)
            step = mu + eps
            paths[b, h] = step
            feat[lag_idx[:-1]] = feat[lag_idx[1:]]
            feat[lag_idx[-1]] = step
    return paths

def forecast_prob(
    df: pd.DataFrame,
    *,
    forecast: int = 5,
    thresholds: list[float] | None = None,
    n_boot: int = 1000,
):
    if thresholds is None:
        thresholds = [0.005, 0.01, 0.02]

    X, y = prepare_supervised(df, n_lags=10, use_tech=True)
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model, best_params = train_svm(X_train, y_train)
    rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
    da = dir_acc(y_test, model.predict(X_test))

    resid = y_train.values - model.predict(X_train)
    lag_cols = [c for c in X.columns if c.startswith("lag_")]
    last_row = X.iloc[-1]
    paths = simulate_paths(
        model,
        last_row,
        lag_cols,
        resid,
        n_steps=forecast,
        n_boot=n_boot,
        seed=42,
    )

    last_close_raw = df["Close"].iloc[-1]
    last_close = float(np.asarray(last_close_raw).ravel()[0])

    start_date = df.index[-1] + BDay(1)
    pred_dates = pd.bdate_range(start_date, periods=forecast)

    result_rows = []
    for i, d in enumerate(pred_dates):
        step = paths[:, i]
        pct_step = np.exp(step) - 1
        price_step = last_close * np.exp(step)

        row = {
            "date": d.date(),
            "P(up)": (step > 0).mean(),
            "MinClose": price_step.min(),
            "MaxClose": price_step.max(),
        }
        for thr in thresholds:
            p_val = (pct_step >= thr).mean()
            close_thr = last_close * (1 + thr)
            row[f"P≥{thr*100:.2f}%"] = f"{p_val:.2%} ({close_thr:.2f})"
        result_rows.append(row)

    res = pd.DataFrame(result_rows)
    return res, rmse, da, best_params

with st.sidebar:
    st.header("Ρυθμίσεις")
    ticker = st.text_input("Ticker", value="ES=F")
    end_date = st.date_input("End date", value=date.today())
    thr_input = st.text_input("Thresholds % (comma)", value="0.5,1,2")
    run_btn = st.button("▶️ Run")

if run_btn:
    with st.spinner("Λήψη δεδομένων & εκπαίδευση μοντέλου…"):
        df_raw = get_data(ticker, start="2015-01-01", end=str(end_date))
        if df_raw.empty:
            st.error("Δεν βρέθηκαν δεδομένα για αυτό το σύμβολο.")
            st.stop()
        df = add_technical_indicators(df_raw)

        try:
            thr_list = [
                float(x.strip()) / 100
                for x in thr_input.split(",") if x.strip() != ""
            ]
        except ValueError:
            st.error("Δώσε αριθμούς χωρισμένους με κόμμα (π.χ. 0.5,1,2)")
            st.stop()

        results_df, rmse_val, da_val, best_params = forecast_prob(
            df,
            thresholds=thr_list,
        )

    st.subheader("Αποτελέσματα")
    st.write(
        f"Test RMSE (log‑ret): **{rmse_val:.6f}** &nbsp;&nbsp;|&nbsp;&nbsp;"
        f"Directional Accuracy: **{da_val:.2%}**"
    )
    with st.expander("Βέλτιστες υπερ‑παράμετροι"):
        st.json({k: float(v) for k, v in best_params.items()}, expanded=False)

    st.table(results_df)

    st.markdown("—")
    st.caption("© 2025 Probabilistic SVM v2 — only for insights and NOT TO TRUST")


