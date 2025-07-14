from __future__ import annotations

# ─────────────────────────────── imports ──
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from datetime import date
from pandas.tseries.offsets import BDay
from scipy.stats import loguniform, uniform
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# ───────────────────────── Streamlit config ──
st.set_page_config(
    page_title="Probabilistic SVM Forecast",
    page_icon="📈",
    layout="centered",
    initial_sidebar_state="auto",
)

st.title("📈 Πιθανότητες & τιμές‑στόχοι (SVM)")
st.markdown(
    """
    Το μοντέλο εκπαιδεύεται σε ημερήσια δεδομένα από **1 Ιαν 2015** και προσομοιώνει
    1000 μονοπάτια λογαριθμικών αποδόσεων για τις επόμενες 5 συνεδριάσεις.

    Για κάθε ημέρα εμφανίζονται:

    * **P(up)** – πιθανότητα θετικής απόδοσης  
    * **MinClose / MaxClose** – ελάχιστο & μέγιστο κλείσιμο από τις προσομοιώσεις  
    * **P(≥ x %)** – πιθανότητα η απόδοση να ξεπεράσει το κατώφλι x  
    * **Close@x %** – τιμή κλεισίματος που αντιστοιχεί στο κατώφλι x
    """
)

# ─────────────────────────── helpers ──
def get_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download data & return single‑level Close + log returns."""
    raw = yf.download(ticker, start=start, end=end, progress=False)

    # αντιμετώπιση MultiIndex (Close, ticker)
    if isinstance(raw.columns, pd.MultiIndex):
        close_series = raw["Close"].iloc[:, 0]   # πρώτο (μοναδικό) ticker
    else:
        close_series = raw["Close"]

    df = close_series.to_frame(name="Close")
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

def train_svm(X_train: pd.DataFrame, y_train: pd.Series, *, seed: int = 42):
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
        n_iter=40,
        cv=tscv,
        scoring="neg_root_mean_squared_error",
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
            paths[b, h] = mu + eps
            feat[lag_idx[:-1]] = feat[lag_idx[1:]]
            feat[lag_idx[-1]] = mu
    return paths

# ───────────────────────── core routine ──
def forecast_prob(
    df: pd.DataFrame,
    *,
    forecast: int = 5,
    thresholds: list[float] | None = None,  # π.χ. [0.005, 0.01]
):
    if thresholds is None:
        thresholds = [0.005, 0.01, 0.02]

    X, y = prepare_supervised(df, n_lags=10, use_tech=True)
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model, best_params = train_svm(X_train, y_train)
    rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))

    resid = y_train.values - model.predict(X_train)
    lag_cols = [c for c in X.columns if c.startswith("lag_")]
    last_row = X.iloc[-1]
    paths = simulate_paths(model, last_row, lag_cols, resid,
                           n_steps=forecast, n_boot=1000)

    # safe scalar extraction
    last_close_raw = df["Close"].iloc[-1]
    last_close = float(np.asarray(last_close_raw).ravel()[0])

    start_date = df.index[-1] + BDay(1)
    pred_dates = pd.bdate_range(start_date, periods=forecast)

    result_rows = []
    for i, d in enumerate(pred_dates):
        step = paths[:, i]                       # log‑returns
        pct_step = np.exp(step) - 1              # simple %
        price_step = last_close * np.exp(step)   # closing prices

        row = {
            "date": d.date(),
            "P(up)": (step > 0).mean(),
            "MinClose": price_step.min(),
            "MaxClose": price_step.max(),
        }
        for thr in thresholds:
            row[f"P(≥{thr*100:.2f}%)"] = (pct_step >= thr).mean()
            row[f"Close@{thr*100:.2f}%"] = last_close * (1 + thr)
        result_rows.append(row)

    return pd.DataFrame(result_rows), rmse, best_params

# ───────────────────────── Streamlit UI ──
with st.sidebar:
    st.header("Ρυθμίσεις")
    ticker = st.text_input("Ticker", value="MNQ=F")
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

        # parse thresholds
        try:
            thr_list = [
                float(x.strip()) / 100
                for x in thr_input.split(",") if x.strip() != ""
            ]
        except ValueError:
            st.error("Δώσε αριθμούς χωρισμένους με κόμμα (π.χ. 0.5,1,2)")
            st.stop()

        results_df, rmse_val, best_params = forecast_prob(df, thresholds=thr_list)

    st.subheader("Αποτελέσματα")
    st.write(f"Test RMSE (log‑ret): **{rmse_val:.6f}**")
    with st.expander("Βέλτιστες υπερ‑παράμετροι"):
        st.json({k: float(v) for k, v in best_params.items()}, expanded=False)

    # dynamic formatting
    fmt = {}
    for col in results_df.columns:
        if col.startswith("P("):
            fmt[col] = "{:.2%}"
        elif col in ("MinClose", "MaxClose") or col.startswith("Close@"):
            fmt[col] = "{:.2f}"
    st.table(results_df.style.format(fmt))

    st.markdown("—")
    st.caption("© 2025 Probabilistic SVM Demo — μόνο για εκπαιδευτική χρήση")
