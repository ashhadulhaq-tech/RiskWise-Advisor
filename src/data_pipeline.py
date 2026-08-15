"""
Module A - Data Pipeline
=========================
Job: get historical stock price data (Open, High, Low, Close, Volume) for a
given ticker (e.g. "AAPL") and save it as a CSV in data/.

WHY: every model we build later needs clean historical data as input.
This is the foundation of the whole app.

HOW IT WORKS:
- Primary method: yfinance, which pulls real data from Yahoo Finance for free.
- Fallback: if there's no internet connection (like in this sandbox), we
  generate a realistic synthetic price series instead, so the rest of the
  pipeline can still be built and tested. On your own laptop (with internet),
  it will always use the real data automatically.
"""

import pandas as pd
import numpy as np
import os

from config import DATA_DIR, TICKERS, get_logger

logger = get_logger(__name__)


def fetch_real_data(ticker: str, period: str = "3y") -> pd.DataFrame:
    """Try to pull real historical data using yfinance."""
    import yfinance as yf
    df = yf.download(ticker, period=period, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {ticker}")
    df = df.reset_index()
    df.columns = [c if isinstance(c, str) else c[0] for c in df.columns]
    df = df.rename(columns={"Date": "date", "Open": "open", "High": "high",
                             "Low": "low", "Close": "close", "Volume": "volume"})
    return df[["date", "open", "high", "low", "close", "volume"]]


def generate_synthetic_data(ticker: str, days: int = 750, seed: int = None) -> pd.DataFrame:
    """
    Create a realistic-looking synthetic price series using a random walk
    with drift + volatility clustering, so offline testing still behaves
    like real market data (trends, dips, noise).
    """
    rng = np.random.default_rng(seed if seed is not None else abs(hash(ticker)) % (2**32))
    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=days)

    # base drift + volatility differ slightly per "ticker" so stocks look distinct
    drift = rng.uniform(0.0001, 0.0006)
    vol = rng.uniform(0.012, 0.03)

    returns = rng.normal(drift, vol, size=days)
    # add volatility clustering (GARCH-like effect) so it doesn't look too clean
    vol_regime = np.abs(rng.normal(1, 0.3, size=days)).clip(0.5, 2.5)
    returns = returns * vol_regime

    price = 100 * np.exp(np.cumsum(returns))
    close = price
    open_ = close * (1 + rng.normal(0, 0.003, size=days))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.004, size=days)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.004, size=days)))
    volume = rng.integers(1_000_000, 8_000_000, size=days)

    df = pd.DataFrame({
        "date": dates, "open": open_, "high": high,
        "low": low, "close": close, "volume": volume
    })
    return df


def get_stock_data(ticker: str, period: str = "3y", use_synthetic_if_offline: bool = True) -> pd.DataFrame:
    """Main entry point: try real data first, fall back to synthetic if offline."""
    try:
        df = fetch_real_data(ticker, period)
        logger.info(f"[{ticker}] Loaded REAL data from Yahoo Finance ({len(df)} rows).")
    except Exception as e:
        if not use_synthetic_if_offline:
            raise
        logger.warning(f"[{ticker}] Could not reach Yahoo Finance ({e}). Using SYNTHETIC data instead.")
        df = generate_synthetic_data(ticker)

    if df is None or df.empty:
        raise ValueError(f"[{ticker}] No data could be obtained (real or synthetic).")

    path = os.path.join(DATA_DIR, f"{ticker}.csv")
    df.to_csv(path, index=False)
    logger.info(f"[{ticker}] Saved to {path}")
    return df


if __name__ == "__main__":
    for t in TICKERS:
        try:
            get_stock_data(t)
        except Exception as e:
            logger.error(f"[{t}] Failed to fetch data, skipping: {e}")
