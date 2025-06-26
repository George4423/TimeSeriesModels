import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def download_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end, progress=False)
    return data[['Close']]

def add_lag_features(prices, n_lags):
    df = prices.copy()
    for lag in range(1, n_lags + 1):
        df[f'lag_{lag}'] = df['Close'].shift(lag)
    return df.dropna()

def split_train_test(df, train_fraction=0.6):
    split = int(len(df) * train_fraction)
    return df.iloc[:split], df.iloc[split:]

def train_random_forest(X, y, n_estimators=100, seed=42):
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=seed)
    model.fit(X, y)
    return model

def forecast_future(model, recent_values, n_steps, n_lags):
    preds = []
    history = recent_values.copy()
    for _ in range(n_steps):
        X_input = history[-n_lags:].reshape(1, -1)
        next_pred = model.predict(X_input)[0]
        preds.append(next_pred)
        history = np.append(history, next_pred)
    return preds

def plot_results(history, test_true, test_pred, future_pred, rmse):
    plt.figure(figsize=(12, 6))
    plt.plot(history.index, history['Close'], label='Historical')
    plt.plot(test_true.index, test_true, label='Actual (Test)')
    plt.plot(test_true.index, test_pred, label='RF on Test')
    plt.plot(future_pred.index, future_pred, 'o--', label='5‑Day Forecast')
    plt.title(f'Random Forest Forecast (RMSE = {rmse:.4f})')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    ticker = '^GSPC'
    data = download_data(ticker, '2025-01-01', '2025-06-25')

    n_lags = 10
    df_lags = add_lag_features(data, n_lags)

    train_df, test_df = split_train_test(df_lags, train_fraction=0.6)
    X_train = train_df[[f'lag_{i}' for i in range(1, n_lags+1)]]
    y_train = train_df['Close']
    X_test  = test_df[[f'lag_{i}' for i in range(1, n_lags+1)]]
    y_test  = test_df['Close']

    rf = train_random_forest(X_train, y_train)

    y_pred_test = rf.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    print(f'Random Forest Test RMSE: {rmse:.4f}')

    last_prices = data['Close'].values[-n_lags:]
    future_preds = forecast_future(rf, last_prices, n_steps=5, n_lags=n_lags)
    future_dates = pd.bdate_range(start=data.index.max() + pd.Timedelta(days=1), periods=5)
    forecast_series = pd.Series(future_preds, index=future_dates)

    print('\n5‑Day Forecast:')
    print(forecast_series)

    plot_results(data, y_test, y_pred_test, forecast_series, rmse)

if __name__ == '__main__':
    main()
