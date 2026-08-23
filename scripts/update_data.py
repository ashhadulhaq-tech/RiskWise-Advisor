"""
scripts/update_data.py - Automated PSX data pipeline (first-run + daily)
============================================================================
This is the ONE script that implements the full requested behavior:

  FIRST RUN (no local data yet):
    -> fetch the current KSE-100 constituent list from PSX (via psxdata)
    -> for each ticker, download up to HISTORY_YEARS (default 10) years of
       daily OHLCV history
    -> save as Parquet, one file per ticker
    -> build features + train a model for each

  EVERY SUBSEQUENT RUN:
    -> for each ticker already on disk, find the latest date already stored
    -> fetch ONLY the missing days since then (not the whole history again)
    -> append + de-duplicate + re-sort, save back to Parquet
    -> rebuild features + retrain (fast — seconds per ticker)
    -> any ticker newly added to the KSE-100 list since last run is
       automatically backfilled with full history, same as a first run

Run this via cron / Windows Task Scheduler / GitHub Actions
(.github/workflows/daily_refresh.yml) once a day, ideally after PSX market
close. The Streamlit app (app/app.py) always reads whatever's currently on
disk, so it picks up whatever this script produced with zero extra wiring.

IMPORTANT: run scripts/verify_data_source.py FIRST, before this script,
the very first time you set this up — see that file's docstring for why.
"""

import sys
import os
import time
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from config import (TICKERS, MARKET_INDEX_TICKER, get_available_tickers, raw_parquet_path,
                     features_parquet_path, REQUEST_DELAY_SECONDS, get_logger)
import psx_api_fetch
from features import build_features
from model import train_model
import macro_experiment

logger = get_logger(__name__)


def _load_existing_raw(ticker: str) -> pd.DataFrame:
    """Load whatever raw OHLCV parquet already exists for a ticker, or an
    empty frame if this is the first time we've ever fetched it. Always
    forces the date column to a proper datetime dtype — don't assume the
    storage round-trip preserved it, to avoid silent str-vs-Timestamp
    comparison bugs downstream."""
    path = raw_parquet_path(ticker)
    if os.path.exists(path):
        try:
            df = pd.read_parquet(path)
            df["date"] = pd.to_datetime(df["date"])
            return df
        except Exception as e:
            logger.warning(f"[{ticker}] Existing parquet file unreadable "
                            f"({e}) — treating as first run for this ticker.")
    return pd.DataFrame(columns=psx_api_fetch.STANDARD_COLUMNS)


def _save_raw(ticker: str, df: pd.DataFrame) -> None:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates(subset="date", keep="last").reset_index(drop=True)
    df.to_parquet(raw_parquet_path(ticker), index=False)


def fetch_ticker_incremental(ticker: str) -> tuple:
    """
    The core per-ticker logic: figure out whether this is a first-run
    (full HISTORY_YEARS backfill) or an incremental update (only new days
    since the last stored date), fetch accordingly, merge, and save.

    Returns (status, n_new_rows) where status is one of:
      "first_run", "updated", "already_current", "no_data", "failed"
    Never raises — failures are caught and logged so one bad ticker
    (delisted, renamed, temporarily unavailable) never aborts the batch.
    """
    try:
        existing = _load_existing_raw(ticker)

        if existing.empty:
            # First run for this ticker — full historical backfill
            new_data = psx_api_fetch.fetch_history(ticker)  # start=None -> HISTORY_YEARS back
            if new_data.empty:
                logger.warning(f"[{ticker}] First-run backfill returned no data "
                                f"(possibly delisted, suspended, or not covered by the source).")
                return ("no_data", 0)
            _save_raw(ticker, new_data)
            logger.info(f"[{ticker}] First-run backfill: {len(new_data)} rows "
                        f"({new_data['date'].min().date()} to {new_data['date'].max().date()}).")
            return ("first_run", len(new_data))

        # Incremental: only fetch what's missing since the last stored date
        last_date = pd.to_datetime(existing["date"]).max()
        next_day = last_date + pd.Timedelta(days=1)
        today_ts = pd.Timestamp.today().normalize()

        if next_day > today_ts:
            return ("already_current", 0)

        start = next_day.strftime("%Y-%m-%d")
        end = today_ts.strftime("%Y-%m-%d")

        new_data = psx_api_fetch.fetch_history(ticker, start=start, end=end)
        if new_data.empty:
            # Not necessarily an error — could just be a weekend/holiday
            # gap with genuinely nothing new to report yet.
            return ("already_current", 0)

        merged = pd.concat([existing, new_data], ignore_index=True)
        n_before = len(existing)
        _save_raw(ticker, merged)
        n_after_new = len(pd.read_parquet(raw_parquet_path(ticker))) - n_before
        logger.info(f"[{ticker}] Incremental update: +{n_after_new} new rows "
                    f"(now {last_date.date()} onward through {new_data['date'].max().date()}).")
        return ("updated", n_after_new)

    except Exception as e:
        logger.error(f"[{ticker}] Fetch failed: {e}. Existing local data (if any) left untouched.")
        return ("failed", 0)


def refresh_ticker(ticker: str) -> bool:
    """Fetch (first-run or incremental) + rebuild features + retrain for one ticker."""
    status, n_rows = fetch_ticker_incremental(ticker)

    if status in ("no_data", "failed"):
        return False

    if status == "already_current" and not os.path.exists(features_parquet_path(ticker)):
        # Edge case: raw data exists but features/model never got built
        # (e.g. a previous run crashed between steps) — still build them.
        pass
    elif status == "already_current":
        logger.info(f"[{ticker}] Already up to date, no new rows — skipping retrain.")
        return True

    try:
        build_features(ticker)
        train_model(ticker)
        try:
            macro_experiment.run_experiment(ticker)
        except Exception as e:
            # Non-fatal — the macro experiment needs the external Excel
            # file and requires the ticker's price data to overlap with
            # the ~24-month macro window. Missing/failing this should
            # never block the core price-prediction refresh.
            logger.warning(f"[{ticker}] Macro experiment skipped: {e}")
        logger.info(f"[{ticker}] Refresh complete ({status}, +{n_rows} rows).")
        return True
    except Exception as e:
        logger.error(f"[{ticker}] Feature/model build failed after successful fetch: {e}")
        return False


def refresh_market_index() -> bool:
    """
    Fetch + rebuild features for the KSE-100 index itself (not a stock —
    no model is trained for it, since we never predict direction for the
    index, only use its returns as the market benchmark for beta/CAPM in
    risk_engine.py). Uses the same incremental fetch logic as any ticker.

    This exists specifically because refresh_all() previously updated the
    6 stocks daily but never touched the index — meaning beta/correlation
    would silently go stale even while stock predictions stayed current.
    Caught and fixed after being asked how daily updates actually work.
    """
    status, n_rows = fetch_ticker_incremental(MARKET_INDEX_TICKER)

    if status in ("no_data", "failed"):
        logger.error(f"[{MARKET_INDEX_TICKER}] Index refresh failed — beta/correlation "
                     f"will keep using whatever index data was last successfully fetched.")
        return False

    if status == "already_current" and os.path.exists(features_parquet_path(MARKET_INDEX_TICKER)):
        logger.info(f"[{MARKET_INDEX_TICKER}] Index already up to date, no new rows.")
        return True

    try:
        build_features(MARKET_INDEX_TICKER)
        logger.info(f"[{MARKET_INDEX_TICKER}] Index refresh complete ({status}, +{n_rows} rows).")
        return True
    except Exception as e:
        logger.error(f"[{MARKET_INDEX_TICKER}] Feature build failed after successful fetch: {e}")
        return False


def refresh_all(tickers=None):
    """
    Full pipeline run. If `tickers` isn't given, fetches the live KSE-100
    list from PSX; falls back to config.TICKERS (the original 6) if that
    call fails (e.g. offline dev environment, or psxdata's scrape target
    changed) so this never hard-crashes just because the constituent-list
    lookup is unavailable. ALSO refreshes the KSE-100 index itself (see
    refresh_market_index above) — this is not optional, since beta and
    correlation throughout the app depend on it staying current.
    """
    if tickers is None:
        try:
            tickers = psx_api_fetch.get_kse100_tickers()
        except Exception as e:
            logger.warning(f"Couldn't fetch live KSE-100 list ({e}). "
                            f"Falling back to the {len(TICKERS)} configured tickers.")
            tickers = TICKERS

    if not tickers:
        logger.error("No tickers to refresh.")
        return {}

    index_ok = refresh_market_index()

    results = {}
    for i, ticker in enumerate(tickers):
        results[ticker] = refresh_ticker(ticker)
        if i < len(tickers) - 1:
            time.sleep(REQUEST_DELAY_SECONDS)

    succeeded = [t for t, ok in results.items() if ok]
    failed = [t for t, ok in results.items() if not ok]

    logger.info(f"Refresh summary: {len(succeeded)}/{len(tickers)} stocks succeeded. "
                f"Market index ({MARKET_INDEX_TICKER}): {'OK' if index_ok else 'FAILED — beta/correlation may be stale'}.")
    if failed:
        logger.warning(f"Failed/unavailable tickers (existing data, if any, left untouched): {failed}")

    return results


if __name__ == "__main__":
    refresh_all()
