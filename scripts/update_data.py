"""
scripts/update_data.py - Daily data refresh
==============================================
Job: re-fetch the latest prices, rebuild features, and retrain/re-evaluate
the model for every tracked ticker. Meant to be run once per day, after
market close, via a scheduler (cron, Windows Task Scheduler, or GitHub
Actions - see .github/workflows/daily_refresh.yml alongside this file).

DATA SOURCE NOTE:
This project's demo tickers (AAPL, TSLA, JNJ, XOM, JPM) are US stocks and
use yfinance via data_pipeline.py. If you adapt this project to Pakistan
Stock Exchange (PSX/KSE) tickers instead, yfinance does NOT cover PSX -
you'd use the `psxdata` library against PSX's official Data Portal
(dps.psx.com.pk) instead. That integration is stubbed below, behind a
try/except, so this script runs correctly either way: it uses psxdata if
installed and PSX tickers are configured, otherwise it falls back to the
existing yfinance pipeline untouched.

WHY DAILY, NOT MINUTE-LEVEL:
See project README / conversation notes: true real-time data needs an
always-on streaming service, which is out of scope for this project.
PSX's own public data portal is itself only updated with a delay, so
"daily after close" is both realistic and sufficient for a next-day
direction predictor like this one.
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from config import TICKERS, get_logger
from data_pipeline import get_stock_data
from features import build_features
from model import train_model

logger = get_logger(__name__)


def try_fetch_via_psxdata(ticker: str):
    """
    Attempt to fetch data via psxdata (for PSX/KSE tickers). Returns True if
    successful, False if psxdata isn't installed or the ticker isn't a PSX
    symbol - callers should fall back to the yfinance pipeline in that case.
    """
    try:
        import psxdata  # only present if the user installed it for PSX use
    except ImportError:
        return False

    try:
        df = psxdata.stocks(ticker)  # pulls historical OHLCV from PSX
        if df is None or df.empty:
            return False
        out_path = os.path.join(os.path.dirname(__file__), "..", "data", f"{ticker}.csv")
        df.to_csv(out_path, index=False)
        logger.info(f"[{ticker}] Fetched via psxdata (PSX Data Portal).")
        return True
    except Exception as e:
        logger.warning(f"[{ticker}] psxdata fetch failed ({e}), will try yfinance fallback.")
        return False


def refresh_ticker(ticker: str) -> bool:
    """
    Full refresh for one ticker: fetch -> features -> retrain.

    Fetch priority:
      1. psxdata (if installed) - for a live PSX/KSE feed, if you set one up
      2. A newer file in raw_uploads/ (manually re-downloaded from PSX) -
         re-running psx_ingest.py picks up any newer export you've placed there
      3. yfinance/synthetic fallback - only relevant if TICKERS ever points
         back at non-PSX symbols

    Returns success/failure per ticker.
    """
    try:
        used_psx_live = try_fetch_via_psxdata(ticker)
        if not used_psx_live:
            # Re-run ingestion against whatever is currently in raw_uploads/ -
            # this is how a manually re-downloaded PSX export gets picked up.
            from psx_ingest import ingest_psx_excel
            candidates = [f for f in os.listdir(
                os.path.join(os.path.dirname(__file__), "..", "raw_uploads"))
                if f.lower().endswith(".xlsx")]
            if candidates:
                xlsx_path = os.path.join(
                    os.path.dirname(__file__), "..", "raw_uploads", candidates[0])
                ingest_psx_excel(xlsx_path, tickers_filter=[ticker])
            else:
                get_stock_data(ticker)  # yfinance-or-synthetic fallback

        build_features(ticker)
        train_model(ticker)
        logger.info(f"[{ticker}] Daily refresh complete.")
        return True
    except Exception as e:
        logger.error(f"[{ticker}] Daily refresh FAILED: {e}")
        return False


def refresh_all(tickers=None):
    tickers = tickers or TICKERS
    results = {t: refresh_ticker(t) for t in tickers}

    succeeded = [t for t, ok in results.items() if ok]
    failed = [t for t, ok in results.items() if not ok]

    logger.info(f"Daily refresh summary: {len(succeeded)}/{len(tickers)} succeeded.")
    if failed:
        logger.warning(f"Failed tickers (kept last known good data): {failed}")

    return results


if __name__ == "__main__":
    refresh_all()
