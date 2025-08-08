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

# ===========================
# Streamlit UI
# ===========================
st.set_page_config(
    page_title="Probabilistic SVM Forecast (v3)",
    page_icon="📈",
    layout="centered",
    initial_sidebar_state="auto",
)

st.title("📈 Πιθανότητες & τιμές-στόχοι (SVM v3)")
st.markdown(
    """
    Το μοντέλο εκπαιδεύεται σε ημερήσια δεδομένα από **1 Ιαν 2015** και
    προσομοιώνει 1000 μονοπάτια λογαριθμικών αποδόσεων για τις επόμενες *k* συνεδριάσεις.

    **Νέο (v3):**
    - Στόχος **επόμενης ημέρας** (χωρίς διαρροή χαρακτηριστικών)
    - Εξωγενή χαρακτηριστικά **VIX / VIX3M** (επίπεδο, αλλαγές, κλίση)
    - Realized vol: **Parkinson** & **ATR%**
    - Block-bootstrap υπολοίπων στη προσομοίωση
    """
)

# ===========================
# Data & Feature Engineering
# ===========================
def get_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Fetch OHLCV and compute log returns.
    """
    df = yf.download(
        ticker,
        start=start,
        end=end,
        progress=False,
        auto_adjust=False,
    )[["Open", "High", "Low", "Close", "Volume"]]

    df.dropna(inplace=True)
    df["log_ret"] = np.log(df["Close"]).diff()
    return df.dropna()


def add_technical_indicators(df: pd.DataFrame, win: int = 14) -> pd.DataFrame:
    """
    Basic SMA/RSI + rolling std of log returns and volume z-score.
    """
    df = df.copy()

    # SMA
    df[f"SMA_{win}"] = df["Close"].rolling(win).mean()

    # RSI
    delta = df["Close"].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    rs = up.rolling(win).mean() / down.rolling(win).mean()
    df[f"RSI_{win}"] = 100 - (100 / (1 + rs))

    # Realized volatility of log returns
    df["vol"] = df["log_ret"].rolling(win).std(ddof=0)

    # Volume z-score (log-volume vs 1y mean/std)
    volume_safe = df["Volume"].clip(lower=1)
    log_vol = np.log(volume_safe)
    df["volume_z"] = (
        (log_vol - log_vol.rolling(252).mean())
        / log_vol.rolling(252).std(ddof=0)
    )

    return df


def add_realized_vol(df: pd.DataFrame, win: int = 14) -> pd.DataFrame:
    """
    Add Parkinson volatility sqrt-mean and ATR% features.
    (pandas-native rolling to avoid AttributeErrors)
    """
    df = df.copy()

    # Parkinson volatility uses High/Low range
    hl = np.log(df["High"] / df["Low"]).replace([np.inf, -np.inf], np.nan)
    parkinson_var = (hl ** 2) / (4 * np.log(2))
    df["parkinson_vol"] = parkinson_var.rolling(win).mean().pow(0.5)

    # True Range & ATR% (keep as pandas Series so .rolling works)
    prev_close = df["Close"].shift()
    tr_components = pd.concat([
        (df["High"] - df["Low"]).abs(),
        (df["High"] - prev_close).abs(),
        (df["Low"] - prev_close).abs(),
    ], axis=1)
    tr = tr_components.max(axis=1)
    df["atr_pct"] = tr.rolling(win).mean() / df["Close"]

    return df


def add_exogenous_vix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Join VIX and VIX3M features (level, log change, z-score, slope).
    Safe if VIX data is missing or partially available.
    """
    start_date = df.index.min().date()
    end_date = (df.index.max() + pd.Timedelta(days=1)).date()
    vix = yf.download(
        ["^VIX", "^VIX3M"],
        start=start_date,
        end=end_date,
        progress=False,
        auto_adjust=False,
    )

    if vix is None or vix.empty:
        return df

    # If 'Close' level exists (typical), extract it
    if "Close" in vix:
        vix = vix["Close"]

    # Handle MultiIndex columns (ticker level)
    if isinstance(vix, pd.DataFrame) and isinstance(vix.columns, pd.MultiIndex):
        vix = vix.droplevel(0, axis=1)

    if isinstance(vix, pd.Series):
        vix = vix.to_frame(name="^VIX")

    if vix is None or vix.empty:
        return df  # no VIX data, skip

    rename_map = {}
    for col in vix.columns:
        up = str(col).upper()
        if up in ("^VIX", "VIX"):
            rename_map[col] = "VIX"
        elif up in ("^VIX3M", "VIX3M"):
            rename_map[col] = "VIX3M"
    vix = vix.rename(columns=rename_map)

    # Align on main index and ffill to handle non-overlapping holidays
    vix = vix.reindex(df.index).ffill()

    # Derived features
    cols_present = list(vix.columns)
    if "VIX" in cols_present:
        vix["VIX_logchg"] = np.log(vix["VIX"]).diff()
        vix["VIX_z"] = (vix["VIX"] - vix["VIX"].rolling(252).mean()) / vix["VIX"].rolling(252).std(ddof=0)
        if "VIX3M" in cols_present:
            with np.errstate(divide="ignore", invalid="ignore"):
                vix["VIX_slope"] = vix["VIX3M"] / vix["VIX"] - 1.0
        out = df.join(vix)
        out = out.dropna(subset=["VIX"])
        return out

    # If we don't have a VIX column (only VIX3M or something odd), just join whatever we have.
    return df.join(vix)


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cyclical encodings for day-of-week and month-of-year.
    """
    df = df.copy()
    dow = df.index.dayofweek  # 0..6
    month = df.index.month     # 1..12

    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    df["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)
    return df


# ===========================
# Supervised dataset
# ===========================
def prepare_supervised(
    df: pd.DataFrame,
    n_lags: int = 10,
    use_tech: bool = True,
    tech_lags: int = 0,
    extra_cols: list[str] | None = None,
):
    """
    Build features X_t to predict y_{t+1} (next-day log return).
    This avoids leakage from indicators computed with Close_t into y_t.
    """
    # Lagged log returns (t-1 ... t-n)
    X_ret = np.column_stack([df["log_ret"].shift(i) for i in range(1, n_lags + 1)])
    X = pd.DataFrame(X_ret, columns=[f"lag_{i}" for i in range(n_lags, 0, -1)], index=df.index)

    # Technical/exogenous columns available at time t
    base_tech_cols = ["SMA_14", "RSI_14", "vol", "volume_z"] if use_tech else []
    extra_cols = extra_cols or []
    cols = [c for c in base_tech_cols + extra_cols if c in df.columns]

    if cols:
        X_tech = df[cols].copy()
        if tech_lags > 0:
            lagged = {
                f"{col}_lag{i}": df[col].shift(i)
                for col in cols
                for i in range(1, tech_lags + 1)
            }
            X_tech = pd.concat([X_tech, pd.DataFrame(lagged, index=df.index)], axis=1)
        X = pd.concat([X, X_tech], axis=1)

    # Target: next-day log return
    y = df["log_ret"].shift(-1)

    data = pd.concat([X, y], axis=1).dropna()
    data.columns = data.columns.map(str)
    return data.iloc[:, :-1], data.iloc[:, -1]


# ===========================
# Scorers & Training
# ===========================
def dir_acc(y_true, y_pred):
    return accuracy_score(y_true > 0, y_pred > 0)

def rmse_neg(y_true, y_pred):
    # negative RMSE for "maximize" scorers
    return -np.sqrt(mean_squared_error(y_true, y_pred))

def combo_scorer(y_true, y_pred):
    # Simple blend: maximize DA, penalize RMSE
    da = accuracy_score(y_true > 0, y_pred > 0)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return da - 0.5 * rmse


def train_svm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    n_splits: int = 5,
    n_iter: int = 100,
    seed: int = 42,
    scoring: str = "DA",  # "DA" | "RMSE" | "Combo"
):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    pipe = Pipeline([("scaler", StandardScaler()), ("svr", SVR(kernel="rbf"))])
    param_dist = {
        "svr__C": loguniform(1e-1, 1e3),
        "svr__gamma": loguniform(1e-4, 1e1),
        "svr__epsilon": uniform(1e-4, 0.1),
    }

    if scoring == "DA":
        scorer = make_scorer(dir_acc, greater_is_better=True)
    elif scoring == "RMSE":
        scorer = make_scorer(rmse_neg, greater_is_better=True)
    else:
        scorer = make_scorer(combo_scorer, greater_is_better=True)

    search = RandomizedSearchCV(
        pipe,
        param_dist,
        n_iter=n_iter,
        cv=tscv,
        scoring=scorer,
        random_state=seed,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_


# ===========================
# Simulation utilities
# ===========================
def simulate_paths(
    model,
    last_row: pd.Series,
    lag_cols: list[str],
    resid: np.ndarray,
    *,
    n_steps: int = 5,
    n_boot: int = 1000,
    seed: int = 42,
    use_block_bootstrap: bool = True,
    block_len: int = 5,
):
    """
    Simulate future log-return paths by rolling lag features forward.
    Non-lag features are kept at their last observed values.
    Residuals can be sampled IID or via short block bootstrap.
    """
    paths = np.empty((n_boot, n_steps))
    lag_idx = np.array([last_row.index.get_loc(c) for c in lag_cols])
    base_feat = last_row.values.astype(float)
    rng = np.random.default_rng(seed)

    # Precompute possible block starts
    if use_block_bootstrap and len(resid) > block_len:
        max_start = len(resid) - block_len
    else:
        use_block_bootstrap = False  # fallback to IID

    for b in range(n_boot):
        feat = base_feat.copy()

        if use_block_bootstrap:
            starts_needed = int(np.ceil(n_steps / block_len))
            starts = rng.integers(0, max_start + 1, size=starts_needed)
            eps_stream = np.concatenate([resid[s:s+block_len] for s in starts])[:n_steps]
        else:
            eps_stream = rng.choice(resid, size=n_steps, replace=True)

        for h in range(n_steps):
            mu = model.predict(feat.reshape(1, -1))[0]
            step = mu + eps_stream[h]
            paths[b, h] = step

            # roll lag features forward
            feat[lag_idx[:-1]] = feat[lag_idx[1:]]
            feat[lag_idx[-1]] = step

    return paths


def wilson_ci(k: int, n: int, conf: float):
    p = k / n
    z = norm.ppf(1 - (1 - conf) / 2)
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return center - half, center + half


# ===========================
# Forecast wrapper
# ===========================
def forecast_prob(
    df: pd.DataFrame,
    *,
    forecast: int = 5,
    thresholds: list[float] | None = None,
    ci: float = 0.90,
    n_boot: int = 1000,
    n_lags: int = 10,
    tech_lags: int = 2,
    use_vix: bool = True,
    use_realized_vol: bool = True,
    use_calendar: bool = True,
    n_splits: int = 5,
    n_iter: int = 100,
    scoring: str = "DA",
    use_block_bootstrap: bool = True,
    block_len: int = 5,
):
    if thresholds is None:
        thresholds = [0.005, 0.01, 0.02]

    # Base tech indicators
    df_feat = add_technical_indicators(df)

    # Extra features
    if use_realized_vol:
        df_feat = add_realized_vol(df_feat)
    if use_vix:
        df_feat = add_exogenous_vix(df_feat)
    if use_calendar:
        df_feat = add_calendar_features(df_feat)

    # Build supervised
    extra_cols = []
    # realized vol
    for c in ["parkinson_vol", "atr_pct"]:
        if c in df_feat.columns:
            extra_cols.append(c)
    # VIX set
    for c in ["VIX", "VIX_logchg", "VIX_z", "VIX_slope"]:
        if c in df_feat.columns:
            extra_cols.append(c)
    # Calendar
    for c in ["dow_sin", "dow_cos", "month_sin", "month_cos"]:
        if c in df_feat.columns:
            extra_cols.append(c)

    X, y = prepare_supervised(
        df_feat,
        n_lags=n_lags,
        use_tech=True,
        tech_lags=tech_lags,
        extra_cols=extra_cols,
    )

    # Train/test split (time-based)
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    # Model
    model, best_params = train_svm(
        X_train,
        y_train,
        n_splits=n_splits,
        n_iter=n_iter,
        scoring=scoring,
    )

    # Metrics on test
    y_pred_test = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    da = dir_acc(y_test, y_pred_test)

    # Residuals from train for simulation noise
    resid = (y_train.values - model.predict(X_train))

    # Simulate future
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
        use_block_bootstrap=use_block_bootstrap,
        block_len=block_len,
    )

    # Convert to prices
    last_close_raw = df_feat["Close"].iloc[-1]
    last_close = float(np.asarray(last_close_raw).ravel()[0])

    start_date = df_feat.index[-1] + BDay(1)
    pred_dates = pd.bdate_range(start_date, periods=forecast)

    result_rows = []
    for i, d in enumerate(pred_dates):
        step = paths[:, i]              # log-returns
        pct_step = np.exp(step) - 1.0
        price_step = last_close * np.exp(step)

        k_up = int((step > 0).sum())
        low_ci, high_ci = wilson_ci(k_up, len(step), ci)

        row = {
            "date": d.date(),
            "P(up)": k_up / len(step),
            f"CI-{ci:.0%}-low": low_ci,
            f"CI-{ci:.0%}-high": high_ci,
            "MinClose": price_step.min(),
            "MaxClose": price_step.max(),
        }
        for thr in thresholds:
            row[f"P(≥{thr*100:.2f}%)"] = (pct_step >= thr).mean()
            row[f"Close@{thr*100:.2f}%"] = last_close * (1 + thr)
        result_rows.append(row)

    return pd.DataFrame(result_rows), rmse, da, best_params


# ===========================
# Sidebar controls
# ===========================
with st.sidebar:
    st.header("Ρυθμίσεις")

    ticker = st.text_input("Ticker", value="ES=F")
    end_date = st.date_input("End date", value=date.today())

    st.markdown("**Features**")
    use_vix = st.checkbox("Χρήση VIX / VIX3M", value=True)
    use_realized_vol = st.checkbox("Realized vol (Parkinson, ATR%)", value=True)
    use_calendar = st.checkbox("Ημερολογιακά χαρακτηριστικά", value=True)

    st.markdown("**Lags**")
    n_lags = st.slider("Lagged returns (n_lags)", 5, 30, 15, step=1)
    tech_lags = st.slider("Tech/Exogenous lags (tech_lags)", 0, 5, 2, step=1)

    st.markdown("**CV & Search**")
    n_splits = st.slider("TimeSeriesSplit folds", 3, 10, 5, step=1)
    n_iter = st.slider("RandomizedSearch iterations", 20, 250, 120, step=10)
    scoring_choice = st.selectbox("Scoring", ["Directional Accuracy", "RMSE", "Combo"], index=0)
    scoring_map = {"Directional Accuracy": "DA", "RMSE": "RMSE", "Combo": "Combo"}

    st.markdown("**Forecast & Simulation**")
    forecast_h = st.slider("Forecast horizon (days)", 1, 10, 5, step=1)
    n_boot = st.slider("Simulations (bootstrap paths)", 200, 5000, 1000, step=100)
    use_block_bootstrap = st.checkbox("Block-bootstrap residuals", value=True)
    block_len = st.slider("Block length", 2, 10, 5, step=1)

    thr_input = st.text_input("Thresholds % (comma)", value="0.5,1,2")
    ci_val = st.slider("Wilson CI level", min_value=0.80, max_value=0.99, value=0.90, step=0.01, format="%.2f")

    run_btn = st.button("▶️ Run")


# ===========================
# Run pipeline
# ===========================
if run_btn:
    with st.spinner("Λήψη δεδομένων & εκπαίδευση μοντέλου…"):
        df_raw = get_data(ticker, start="2015-01-01", end=str(end_date))
        if df_raw.empty:
            st.error("Δεν βρέθηκαν δεδομένα για αυτό το σύμβολο.")
            st.stop()

        # Parse thresholds
        try:
            thr_list = [
                float(x.strip()) / 100
                for x in thr_input.split(",") if x.strip() != ""
            ]
        except ValueError:
            st.error("Δώσε αριθμούς χωρισμένους με κόμμα (π.χ. 0.5,1,2)")
            st.stop()

        results_df, rmse_val, da_val, best_params = forecast_prob(
            df=df_raw,
            forecast=forecast_h,
            thresholds=thr_list,
            ci=ci_val,
            n_boot=n_boot,
            n_lags=n_lags,
            tech_lags=tech_lags,
            use_vix=use_vix,
            use_realized_vol=use_realized_vol,
            use_calendar=use_calendar,
            n_splits=n_splits,
            n_iter=n_iter,
            scoring=scoring_map[scoring_choice],
            use_block_bootstrap=use_block_bootstrap,
            block_len=block_len,
        )

    st.subheader("Αποτελέσματα")
    st.write(
        f"Test RMSE (log-ret): **{rmse_val:.6f}**"
        f" &nbsp;&nbsp;|&nbsp;&nbsp; Directional Accuracy: **{da_val:.2%}**"
    )
    with st.expander("Βέλτιστες υπερ-παράμετροι"):
        # cast to float where possible for clean display
        pretty_params = {}
        for k, v in best_params.items():
            try:
                pretty_params[k] = float(v)
            except Exception:
                pretty_params[k] = v
        st.json(pretty_params, expanded=False)

    # Format results table
    fmt = {}
    for col in results_df.columns:
        if col.startswith("P(") or col.startswith("CI-"):
            fmt[col] = "{:.2%}"
        elif col in ("MinClose", "MaxClose") or col.startswith("Close@"):
            fmt[col] = "{:.2f}"
    st.table(results_df.style.format(fmt))

    st.markdown("—")
    st.caption("© 2025 Probabilistic SVM v3 — only for insights and **NOT investment advice**")

