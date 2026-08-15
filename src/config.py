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

# Full names for display purposes (optional, used in the UI where helpful)
TICKER_NAMES = {
    "AICL": "Adamjee Insurance Company Limited",
    "FFC": "Fauji Fertilizer Company Limited",
    "LUCK": "Lucky Cement Limited",
    "MEBL": "Meezan Bank Limited",
    "OGDC": "Oil & Gas Development Company Limited",
    "SYS": "Systems Limited",
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
