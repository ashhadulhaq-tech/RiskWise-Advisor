"""
config.py - Single source of truth for shared settings.

WHY THIS FILE EXISTS:
The evaluation found TICKERS hardcoded independently in app.py, model.py,
and risk_engine.py - a classic DRY violation where updating the list in one
place silently breaks consistency elsewhere. Everything now imports from here.
"""

import os
import logging

# ---------------------------------------------------------------------------
# Paths (all relative to project root, resolved absolutely so it works
# regardless of which directory a script is run from)
# ---------------------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(ROOT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
RAW_UPLOADS_DIR = os.path.join(PROJECT_ROOT, "raw_uploads")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(RAW_UPLOADS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Tickers - THE ONE PLACE this list is defined
# ---------------------------------------------------------------------------
# Real PSX (Pakistan Stock Exchange) tickers, sourced from a manually
# downloaded historical data file (personal/academic use, per PSX's terms —
# see raw_uploads/ and psx_ingest.py). Replaces the earlier US demo tickers
# (AAPL, TSLA, etc.) now that real PSX data is available.
TICKERS = ["AICL", "FFC", "LUCK", "MEBL", "OGDC", "SYS"]

# The market index itself — ingested and processed the same way as a
# regular stock (so it gets a features.parquet with daily_return etc.),
# but explicitly excluded from get_available_tickers() below since it's
# a benchmark to compare stocks against, not a stock to recommend or
# predict direction for.
MARKET_INDEX_TICKER = "KSE100"

# Full names for display purposes (optional, used in the UI where helpful)
TICKER_NAMES = {
    "AICL": "Adamjee Insurance Company Limited",
    "FFC": "Fauji Fertilizer Company Limited",
    "LUCK": "Lucky Cement Limited",
    "MEBL": "Meezan Bank Limited",
    "OGDC": "Oil & Gas Development Company Limited",
    "SYS": "Systems Limited",
}

# Sector/industry classification — static reference metadata, sourced and
# verified against public company records, not derived from price data.
# This is descriptive context shown in the UI (requirement: "what other
# sources besides price are used") — it is NOT fed into the ML model as a
# numeric feature, since one stock per sector here isn't enough to build a
# real sector-index comparison; see README for the honest scope of this.
TICKER_SECTORS = {
    "AICL": "Insurance",
    "FFC": "Fertilizer & Chemicals",
    "LUCK": "Cement",
    "MEBL": "Commercial Banking (Islamic)",
    "OGDC": "Oil & Gas Exploration",
    "SYS": "Information Technology / Software",
}

# ---------------------------------------------------------------------------
# Risk categorization - FIXED absolute bands (annualized volatility),
# not relative percentiles. See risk_engine.py for how this is used.
# Thresholds are standard rough-industry cutoffs: <15% ann. vol = calm
# blue-chip-like behavior, 15-30% = moderate, >30% = high-growth/high-risk.
# ---------------------------------------------------------------------------
VOLATILITY_BANDS = {
    "low": (0.0, 0.15),
    "moderate": (0.15, 0.30),
    "high": (0.30, float("inf")),
}

TRADING_DAYS_PER_YEAR = 252

# which volatility bands each risk-tolerance category is ALLOWED to see
RISK_CATEGORY_ALLOWED_BANDS = {
    "Conservative": {"low"},
    "Moderate": {"low", "moderate"},
    "Aggressive": {"low", "moderate", "high"},
}

MIN_ROWS_REQUIRED = 30  # minimum rows of history needed before we trust a stock's stats

# ---------------------------------------------------------------------------
# CAPM inputs - for the expected-return / alpha calculation in risk_engine.py.
# RISK_FREE_RATE is a placeholder representative of Pakistan's high-rate
# environment (roughly in line with recent KIBOR/T-bill yields) - update this
# if you have a current, cited figure to use instead. This is deliberately
# configurable in one place rather than buried in a formula.
# ---------------------------------------------------------------------------
RISK_FREE_RATE = 0.11  # 11% annualized, placeholder — see note above

# Long-term outlook projection horizons, in trading days (~21/month).
# Used by risk_engine.project_future_price() — trend extrapolation, NOT
# an ML forecast. See that function's docstring for the honesty caveat.
PROJECTION_HORIZONS_MONTHS = [1, 3, 6, 12]
TRADING_DAYS_PER_MONTH = 21

# ---------------------------------------------------------------------------
# psxdata fetch settings - used by src/psx_api_fetch.py. IMPORTANT: psxdata
# scrapes PSX's public site to provide free programmatic access; PSX's own
# Terms of Use technically prohibit automated scraping. This project uses it
# anyway as a deliberate, informed trade-off (documented in README) in favor
# of automation over the manual-download approach used previously. Keep
# REQUEST_DELAY_SECONDS conservative to avoid hammering PSX's servers.
# ---------------------------------------------------------------------------
HISTORY_YEARS = 10              # how far back to backfill on first run
REQUEST_DELAY_SECONDS = 1.0     # pause between per-ticker API calls, be a polite citizen
FETCH_RETRY_ATTEMPTS = 3


def raw_parquet_path(ticker: str) -> str:
    return os.path.join(DATA_DIR, f"{ticker}.parquet")


def features_parquet_path(ticker: str) -> str:
    return os.path.join(DATA_DIR, f"{ticker}_features.parquet")


def get_available_tickers() -> list:
    """
    Scans data/ for every ticker that has been fully processed (has a
    _features.parquet file) and returns them, sorted. This is what lets the
    app scale from 6 tickers to all ~100 KSE-100 constituents with ZERO code
    changes — fetch more data via psx_api_fetch.py and this list grows
    automatically. Falls back to the static TICKERS list if data/ is empty
    (e.g. before the pipeline has ever run).
    """
    if not os.path.isdir(DATA_DIR):
        return list(TICKERS)
    found = [
        f[: -len("_features.parquet")]
        for f in os.listdir(DATA_DIR)
        if f.endswith("_features.parquet") and f != f"{MARKET_INDEX_TICKER}_features.parquet"
    ]
    return sorted(found) if found else list(TICKERS)

# ---------------------------------------------------------------------------
# Logging - replaces bare print() calls with real, leveled logging
# ---------------------------------------------------------------------------
def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:  # avoid duplicate handlers on re-import
        logger.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                                 datefmt="%Y-%m-%d %H:%M:%S")

        console = logging.StreamHandler()
        console.setFormatter(fmt)
        logger.addHandler(console)

        file_handler = logging.FileHandler(os.path.join(LOG_DIR, "app.log"))
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
    return logger
