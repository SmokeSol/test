"""
FULL PRODUCTION CODE:
Multi-Factor Trading System for BTC Signal Generation on Binance
----------------------------------------------------------------
- Fetches historical data for BTC/USDT from Binance (via CCXT)
- Calculates technical indicators (RSI, MACD, ATR, etc.)
- Trains an LSTM model to predict short-term price movement
- (Commented out) Statistical arbitrage overlay (BTC-ETH spread)
- (Commented out) xAI Grok integration for news sentiment
- Aggregates factors to produce a trade signal: "LONG", "SHORT", or "FLAT"
- No order execution is performed (signals only)
----------------------------------------------------------------
Replit / Local Usage:
    1) Ensure Python 3.8+ environment.
    2) pip install -r requirements.txt
    3) python main.py
----------------------------------------------------------------
Author: ChatGPT
License: CC0 (No rights reserved)
"""

import os
import time
import logging
import traceback
from datetime import datetime, timedelta
from typing import List, Tuple

import ccxt
import pandas as pd
import numpy as np

# ----------------------------
# Uncomment these lines if you wish to integrate xAI Grok for news sentiment later.
# from xai import GrokClient
# ----------------------------

import torch
import torch.nn as nn
import torch.optim as optim

# To compute technical indicators (install via pip install ta)
import ta

# ------------------------------------------------------------------------------
# 1. LOGGING CONFIGURATION
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

# ------------------------------------------------------------------------------
# 2. GLOBAL SETTINGS / CONFIGURATION
# ------------------------------------------------------------------------------

# Create a CCXT client for Binance (public endpoints only)
BINANCE = ccxt.binance({
    "enableRateLimit": True,
    "options": {
        "defaultType": "future"  # Change to "spot" if you prefer spot data
    }
})

# For now, we are only trading BTC/USDT
SYMBOL = "BTC/USDT"

# How far back to fetch historical data (in days) for initial training
HISTORICAL_DAYS = 30
TIMEFRAME = "5m"  # Using 5-minute candles

# Model training parameters
EPOCHS = 2            # For demonstration; increase for production use
BATCH_SIZE = 32
PREDICTION_HORIZON = 3  # Number of bars ahead (5m each) to predict
DEVICE = torch.device("cpu")  # Use "cuda" if GPU is available

# Real-time loop configuration
LOOP_SLEEP_SEC = 60  # New data is fetched every minute

# ------------------------------------------------------------------------------
# 3. MACHINE LEARNING MODEL DEFINITION (LSTM)
# ------------------------------------------------------------------------------

class LSTMModel(nn.Module):
    """
    A simple LSTM model for time-series regression.
    Input shape: (batch_size, seq_length, num_features)
    Output: scalar predicted return (percentage change)
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 1):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(DEVICE)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(DEVICE)

        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]  # Use the last time step's output
        out = self.fc(out)
        return out

# ------------------------------------------------------------------------------
# 4. DATA FUNCTIONS
# ------------------------------------------------------------------------------

def fetch_ohlcv(symbol: str, timeframe: str, since=None, limit=500) -> pd.DataFrame:
    """
    Fetch historical OHLCV data from Binance using CCXT.
    Returns a DataFrame with columns: [timestamp, open, high, low, close, volume]
    """
    try:
        data = BINANCE.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
        df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms')
        df.set_index("timestamp", inplace=True)
        return df
    except Exception as e:
        logging.error(f"Failed to fetch OHLCV for {symbol}: {e}")
        return pd.DataFrame()


def build_features_and_labels(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """
    Given a DataFrame with columns [open, high, low, close, volume],
    compute technical indicators and a target which is the future return
    over 'horizon' bars (percentage change).
    """
    df = df.copy()

    # Calculate technical indicators
    df["rsi"] = ta.momentum.rsi(df["close"], window=14)
    df["macd"] = ta.trend.macd_diff(df["close"])
    df["atr"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=14)

    # Calculate future return as the target (percentage change)
    df["future_close"] = df["close"].shift(-horizon)
    df["target"] = (df["future_close"] - df["close"]) / df["close"]

    df.dropna(inplace=True)
    return df


def prepare_sequences(df: pd.DataFrame, seq_len: int = 16, feature_cols: List[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a feature-engineered DataFrame into sequences for the LSTM.
    feature_cols: list of columns to use as inputs.
    Returns:
      X: shape (num_samples, seq_len, num_features)
      y: shape (num_samples, 1)
    """
    if feature_cols is None:
        raise ValueError("feature_cols must be specified.")

    data = df[feature_cols + ["target"]].values
    X_list, y_list = [], []

    for i in range(len(data) - seq_len):
        X_list.append(data[i: i + seq_len, :-1])
        y_list.append(data[i + seq_len - 1, -1])
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32).reshape(-1, 1)
    return X, y

# ------------------------------------------------------------------------------
# 5. TRAINING & INFERENCE FUNCTIONS
# ------------------------------------------------------------------------------

def train_model(df: pd.DataFrame, seq_len: int = 16) -> Tuple[LSTMModel, List[str]]:
    """
    Train the LSTM model on the provided DataFrame.
    Returns the trained model and the list of feature columns used.
    """
    feature_cols = ["close", "volume", "rsi", "macd", "atr"]  # Example features
    X, y = prepare_sequences(df, seq_len=seq_len, feature_cols=feature_cols)

    split_idx = int(len(X) * 0.8)
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_val, y_val     = X[split_idx:], y[split_idx:]

    X_train_t = torch.tensor(X_train, device=DEVICE)
    y_train_t = torch.tensor(y_train, device=DEVICE)
    X_val_t   = torch.tensor(X_val, device=DEVICE)
    y_val_t   = torch.tensor(y_val, device=DEVICE)

    model = LSTMModel(input_dim=X.shape[2], hidden_dim=64, num_layers=1).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    train_dataset_size = X_train_t.shape[0]
    num_batches = (train_dataset_size // BATCH_SIZE) + 1

    logging.info("Starting model training...")
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        for b in range(num_batches):
            start = b * BATCH_SIZE
            end = start + BATCH_SIZE
            batch_x = X_train_t[start:end]
            batch_y = y_train_t[start:end]

            if len(batch_x) == 0:
                continue

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_t)
            val_loss = criterion(val_preds, y_val_t).item()

        logging.info(f"Epoch [{epoch+1}/{EPOCHS}] - Train Loss: {epoch_loss/num_batches:.6f}, Val Loss: {val_loss:.6f}")

    logging.info("Model training complete.")
    return model, feature_cols


def predict_next_return(model: LSTMModel, recent_df: pd.DataFrame, feature_cols: List[str], seq_len: int = 16) -> float:
    """
    Predict the next bar's return (percentage change) using the trained model.
    Returns a float (e.g., 0.01 means +1%).
    """
    if len(recent_df) < seq_len:
        return 0.0

    input_df = recent_df.iloc[-seq_len:].copy()
    x_input = input_df[feature_cols].values
    x_tensor = torch.tensor(x_input, dtype=torch.float32, device=DEVICE).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        output = model(x_tensor)
        pred = output.item()
    return pred

# ------------------------------------------------------------------------------
# 6. STATISTICAL ARBITRAGE & SENTIMENT PLACEHOLDER FUNCTIONS
# ------------------------------------------------------------------------------

# The following function for BTC-ETH spread is now commented out since we are trading BTC only.
# def compute_btc_eth_spread(df_btc: pd.DataFrame, df_eth: pd.DataFrame) -> float:
#     """
#     Compute a Z-score of the BTC/ETH price ratio as a measure for statistical arbitrage.
#     """
#     df_merged = pd.DataFrame(index=df_btc.index)
#     df_merged["btc_close"] = df_btc["close"]
#     df_merged["eth_close"] = df_eth["close"]
#     df_merged.dropna(inplace=True)
#     df_merged["ratio"] = df_merged["btc_close"] / df_merged["eth_close"]
#     mean_ratio = df_merged["ratio"].rolling(window=50).mean()
#     std_ratio = df_merged["ratio"].rolling(window=50).std()
#     df_merged["z_score"] = (df_merged["ratio"] - mean_ratio) / (std_ratio + 1e-9)
#     return df_merged["z_score"].iloc[-1]

def get_news_sentiment_from_xai(prompt: str) -> float:
    """
    Placeholder for xAI Grok API call to get news sentiment.
    Uncomment and modify the code below when integrating.
    """
    # ---------------------------
    # Example integration with xAI Grok:
    #
    # from xai import GrokClient
    # client = GrokClient(api_key="YOUR_XAI_API_KEY")
    # response = client.chat.completions.create(
    #     model="grok-beta",
    #     messages=[{"role": "user", "content": prompt}]
    # )
    # # Extract sentiment score from the response (implement your parsing logic)
    # sentiment = parse_sentiment(response)
    # return sentiment
    # ---------------------------
    return 0.0  # Neutral sentiment placeholder

# ------------------------------------------------------------------------------
# 7. SIGNAL AGGREGATION FUNCTION (BTC Only)
# ------------------------------------------------------------------------------

def generate_signal(btc_return_pred: float, news_sentiment: float) -> str:
    """
    Combine factors to produce a final trade signal for BTC only.
    - btc_return_pred: predicted short-term return for BTC.
    - news_sentiment: numerical sentiment value (e.g., from xAI Grok).
    Returns: "LONG", "SHORT", or "FLAT"
    """
    # Use only the BTC model's prediction.
    net_return = btc_return_pred

    # (Optional) Incorporate news sentiment:
    sentiment_factor = 0.0
    # Uncomment the next line to apply sentiment weighting:
    # sentiment_factor = news_sentiment * 0.01

    final_score = net_return + sentiment_factor

    if final_score > 0.0015:
        return "LONG"
    elif final_score < -0.0015:
        return "SHORT"
    else:
        return "FLAT"

# ------------------------------------------------------------------------------
# 8. MAIN PIPELINE
# ------------------------------------------------------------------------------

def main():
    logging.info("Starting multi-factor trading system for BTC (signals only).")

    # ---------------------------
    # 8.1 Fetch historical BTC data
    # ---------------------------
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=HISTORICAL_DAYS)
    since_ts = int(start_time.timestamp() * 1000)

    logging.info(f"Fetching historical BTC data for the past {HISTORICAL_DAYS} days...")
    df_btc = fetch_ohlcv(SYMBOL, TIMEFRAME, since=since_ts, limit=1000)

    if df_btc.empty:
        logging.error("Failed to fetch sufficient historical BTC data. Exiting.")
        return

    # ---------------------------
    # 8.2 Feature Engineering for BTC
    # ---------------------------
    df_btc = build_features_and_labels(df_btc, horizon=PREDICTION_HORIZON)

    if len(df_btc) < 50:
        logging.error("Not enough data after feature engineering. Exiting.")
        return

    # ---------------------------
    # 8.3 Train the BTC Model
    # ---------------------------
    logging.info("Training BTC model...")
    btc_model, btc_features = train_model(df_btc)

    # ---------------------------
    # 8.4 Real-Time Signal Generation Loop
    # ---------------------------
    logging.info("Entering real-time signal generation loop...")
    seq_len = 16  # Must match the training sequence length

    while True:
        try:
            loop_start = time.time()

            # 1) Fetch latest BTC data (using more candles than seq_len to ensure enough data)
            new_btc = fetch_ohlcv(SYMBOL, TIMEFRAME, limit=seq_len * 2)

            if new_btc.empty or len(new_btc) < seq_len:
                logging.warning("Not enough BTC data in real-time fetch to form a sequence. Waiting for next loop.")
            else:
                # 2) Recompute features on the latest BTC data
                df_btc_live = build_features_and_labels(new_btc, horizon=PREDICTION_HORIZON)

                if len(df_btc_live) < seq_len:
                    logging.warning("Not enough live data after feature engineering. Waiting for next loop.")
                else:
                    # 3) Predict the next short-term return for BTC
                    btc_pred = predict_next_return(btc_model, df_btc_live, btc_features, seq_len)

                    # 4) (Commented Out) Statistical Arbitrage Overlay:
                    # Since we are trading BTC only, we are not computing a BTC-ETH spread.
                    # spread_z = compute_btc_eth_spread(new_btc.iloc[-100:], new_eth.iloc[-100:])

                    # 5) Get news sentiment from xAI Grok (currently a placeholder)
                    news_sentiment = get_news_sentiment_from_xai("Latest crypto news sentiment for BTC")

                    # 6) Generate final signal based solely on BTC prediction (and optional sentiment)
                    signal = generate_signal(btc_pred, news_sentiment)
                    logging.info(f"Signal: {signal} | BTC_pred={btc_pred:.5f}")

            # Sleep until the next loop iteration
            elapsed = time.time() - loop_start
            sleep_time = max(0, LOOP_SLEEP_SEC - elapsed)
            time.sleep(sleep_time)

        except KeyboardInterrupt:
            logging.info("User interrupted. Exiting loop.")
            break
        except Exception as e:
            logging.error(f"Error in real-time loop: {e}")
            traceback.print_exc()
            time.sleep(10)

if __name__ == "__main__":
    main()
