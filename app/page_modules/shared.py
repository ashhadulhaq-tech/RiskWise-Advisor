"""
app/page_modules/shared.py - Shared cached wrappers
=======================================================
Kept in one place so every page uses the same cache instances — Streamlit's
cache decorators are keyed by function identity, so re-defining these
per-page would silently create separate, redundant caches.
"""
import os
import streamlit as st
import pandas as pd

from config import MODEL_DIR, features_parquet_path, get_logger
from model import predict_next_day, load_model
from risk_engine import recommend_stocks

logger = get_logger(__name__)


@st.cache_resource(show_spinner=False)
def cached_load_model(ticker: str):
    return load_model(ticker)


@st.cache_data(ttl=86400, show_spinner=False)
def cached_predict(ticker: str):
    return predict_next_day(ticker)


@st.cache_data(ttl=86400, show_spinner=False)
def cached_recommend(risk_category: str, tickers: tuple):
    return recommend_stocks(risk_category, list(tickers))


@st.cache_data(ttl=86400, show_spinner=False)
def cached_read_features(ticker: str):
    return pd.read_parquet(features_parquet_path(ticker))


def ensure_ticker_ready(ticker: str) -> bool:
    """
    Make sure a ticker's data/model exist before the UI tries to use them.
    Returns False (and shows a friendly message) instead of letting a raw
    FileNotFoundError reach the user — the crash found in the evaluation.
    """
    feat_path = features_parquet_path(ticker)
    model_path = os.path.join(MODEL_DIR, f"{ticker}_model.pkl")
    if os.path.exists(feat_path) and os.path.exists(model_path):
        return True

    st.warning(f"Data or model for **{ticker}** isn't ready yet — this stock "
               f"will be skipped this session. Run the setup pipeline "
               f"(`scripts/update_data.py`) to include it.")
    logger.warning(f"[{ticker}] Missing data/model files — skipped in UI.")
    return False
