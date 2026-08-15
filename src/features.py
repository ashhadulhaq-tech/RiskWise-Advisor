"""
Module A.2 - Feature Engineering
==================================
Job: turn raw price data (open/high/low/close/volume) into "features" -
numeric signals that actually carry predictive information. A raw closing
price by itself tells a model almost nothing useful; trends, momentum and
volatility do.

WHY THESE FEATURES:
- Moving averages (MA)      -> is the price trending up or down?
- RSI (Relative Strength)   -> is the stock "overbought" or "oversold"?
- MACD                      -> momentum / trend-change signal
- Volatility (rolling std)  -> how risky/choppy has this stock been lately?
- Daily return               -> the day-to-day % change, the basic unit of movement
- Volume change              -> is trading interest rising or falling?

TARGET (what we're predicting):
- target_direction: 1 if tomorrow's close > today's close, else 0
  (binary "up or down" - easier to predict reliably than the exact price)
- target_return: tomorrow's % return (used for regression / magnitude)
"""

import pandas as pd
import numpy as np
import os

from config import DATA_DIR, TICKERS, MIN_ROWS_REQUIRED, get_logger

logger = get_logger(__name__)


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # --- Trend features ---
    df["ma_5"] = df["close"].rolling(5).mean()
    df["ma_10"] = df["close"].rolling(10).mean()
    df["ma_20"] = df["close"].rolling(20).mean()
    df["ma_ratio_5_20"] = df["ma_5"] / df["ma_20"]  # >1 means short-term trend is above long-term

    # --- Momentum: RSI (Relative Strength Index, 14-day standard) ---
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # --- Momentum: MACD (12-day EMA minus 26-day EMA) ---
    ema_12 = df["close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema_12 - ema_26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    # --- Volatility (risk proxy) ---
    df["daily_return"] = df["close"].pct_change()
    df["volatility_10"] = df["daily_return"].rolling(10).std()
    df["volatility_20"] = df["daily_return"].rolling(20).std()

    # --- Volume signal ---
    df["volume_change"] = df["volume"].pct_change()
    df["volume_ma_10"] = df["volume"].rolling(10).mean()

    # --- Price range (how much it moved intraday) ---
    df["high_low_range"] = (df["high"] - df["low"]) / df["close"]

    # --- Targets (what the model will learn to predict) ---
    df["target_return"] = df["close"].shift(-1) / df["close"] - 1
    df["target_direction"] = (df["target_return"] > 0).astype(int)

    return df


def build_features(ticker: str) -> pd.DataFrame:
    raw_path = os.path.join(DATA_DIR, f"{ticker}.csv")
    if not os.path.exists(raw_path):
        raise FileNotFoundError(
            f"No raw data file for '{ticker}' at {raw_path}. "
            f"Run data_pipeline.py first to fetch it."
        )

    df = pd.read_csv(raw_path)
    if df.empty:
        raise ValueError(f"[{ticker}] Raw data file is empty.")

    df = add_technical_indicators(df)

    # drop rows with NaNs created by rolling windows (first ~20 rows) and
    # the very last row (no "tomorrow" to predict for it)
    df_clean = df.dropna().reset_index(drop=True)

    if len(df_clean) < MIN_ROWS_REQUIRED:
        raise ValueError(
            f"[{ticker}] Only {len(df_clean)} usable rows after cleaning "
            f"(need at least {MIN_ROWS_REQUIRED}). Stock may be too new or "
            f"data too sparse to model reliably."
        )

    out_path = os.path.join(DATA_DIR, f"{ticker}_features.csv")
    df_clean.to_csv(out_path, index=False)
    logger.info(f"[{ticker}] Features built: {df_clean.shape[0]} usable rows, "
                f"{df_clean.shape[1]} columns. Saved to {out_path}")
    return df_clean


if __name__ == "__main__":
    for t in TICKERS:
        try:
            build_features(t)
        except Exception as e:
            logger.error(f"[{t}] Failed to build features, skipping: {e}")
