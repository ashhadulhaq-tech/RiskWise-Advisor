"""
src/psx_api_fetch.py - Programmatic PSX data fetching
========================================================
Job: fetch historical + incremental OHLCV data for PSX-listed stocks via
the `psxdata` library, and the current KSE-100 constituent list. This is
the ONLY file in the project that talks to psxdata / the network for
market data — features.py, model.py, and risk_engine.py never import it
directly (kept separate per design requirement: fetching vs. ML/analysis
are different concerns with different failure modes).

IMPORTANT — READ BEFORE RELYING ON THIS FILE:
`psxdata` obtains data by scraping PSX's public website. PSX's own Terms
of Use technically prohibit automated/programmatic scraping of their site
without a license. This project uses psxdata anyway as a deliberate,
informed trade-off in favor of full automation — see README "Data source
& licensing" for the full reasoning. If that trade-off stops being
acceptable, `src/psx_ingest.py` (manual Excel download + ingest) still
works as a fully ToS-compliant fallback.

ALSO IMPORTANT — TESTING STATUS:
This file was written against psxdata's documented public API
(https://github.com/mtauha/psxdata) but has NOT been executed against a
live network by the person who wrote it (no network access to PSX from
that environment). Run scripts/verify_data_source.py FIRST, before
anything else, to confirm this actually works end-to-end on your machine.
"""

import time
import pandas as pd

from config import HISTORY_YEARS, REQUEST_DELAY_SECONDS, FETCH_RETRY_ATTEMPTS, get_logger

logger = get_logger(__name__)

# Columns we standardize everything to, matching the rest of the pipeline
# (same shape psx_ingest.py already produces from manual Excel exports).
STANDARD_COLUMNS = ["date", "open", "high", "low", "close", "volume"]


def _import_psxdata():
    """Import psxdata lazily so the rest of the app can run/be imported
    even in environments where psxdata isn't installed (e.g. this project's
    own dev sandbox, or if someone only wants the manual-Excel fallback)."""
    try:
        import psxdata
        return psxdata
    except ImportError as e:
        raise ImportError(
            "psxdata is not installed. Run: pip install psxdata\n"
            "(It's in requirements.txt — make sure you've run "
            "`pip install -r requirements.txt`.)"
        ) from e


def get_kse100_tickers() -> list:
    """
    Returns the CURRENT KSE-100 index constituent list from PSX, via
    psxdata. This is fetched live, not hardcoded, since index membership
    changes over time (companies are added/removed periodically).
    """
    psxdata = _import_psxdata()
    try:
        constituents = psxdata.indices("KSE100")
        # psxdata returns a DataFrame; the ticker/symbol column name is
        # documented as part of its public API but confirm it matches what
        # you see when you actually run this — adjust the column name below
        # if psxdata's real output differs (see verify_data_source.py).
        if hasattr(constituents, "columns"):
            symbol_col = next(
                (c for c in constituents.columns if c.lower() in ("symbol", "ticker")),
                constituents.columns[0],
            )
            tickers = constituents[symbol_col].astype(str).str.upper().tolist()
        else:
            tickers = [str(t).upper() for t in constituents]
        logger.info(f"Fetched {len(tickers)} KSE-100 constituents from psxdata.")
        return sorted(set(tickers))
    except Exception as e:
        logger.error(f"Failed to fetch KSE-100 constituent list: {e}")
        raise


def _standardize(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize whatever column names/casing psxdata returns into our
    standard schema, and enforce basic sanity (no negative prices, etc.)."""
    rename_map = {}
    for col in df.columns:
        low = col.strip().lower()
        if low in ("date", "datetime", "timestamp"):
            rename_map[col] = "date"
        elif low in ("open",):
            rename_map[col] = "open"
        elif low in ("high",):
            rename_map[col] = "high"
        elif low in ("low",):
            rename_map[col] = "low"
        elif low in ("close", "price"):
            rename_map[col] = "close"
        elif low in ("volume", "vol"):
            rename_map[col] = "volume"
    df = df.rename(columns=rename_map)

    missing = set(STANDARD_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"psxdata response is missing expected columns: {missing} "
                          f"(got: {list(df.columns)}) — psxdata's output format may "
                          f"have changed; check verify_data_source.py output.")

    df = df[STANDARD_COLUMNS].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates(subset="date", keep="last").reset_index(drop=True)

    # Basic sanity checks — don't silently accept obviously broken rows
    n_before = len(df)
    df = df[(df["open"] > 0) & (df["high"] > 0) & (df["low"] > 0) & (df["close"] > 0)]
    df = df[df["high"] >= df["low"]]
    if len(df) < n_before:
        logger.warning(f"Dropped {n_before - len(df)} rows failing basic OHLC sanity checks.")

    return df.reset_index(drop=True)


def fetch_history(ticker: str, start: str = None, end: str = None) -> pd.DataFrame:
    """
    Fetch historical OHLCV for one ticker between start and end (YYYY-MM-DD
    strings). If start is None, defaults to HISTORY_YEARS back from today —
    used for first-run full backfill. Retries on transient failures;
    returns an empty DataFrame (not an exception) for a ticker that's
    delisted/unavailable, so callers can skip it and continue the batch.
    """
    psxdata = _import_psxdata()

    if end is None:
        end = pd.Timestamp.today().strftime("%Y-%m-%d")
    if start is None:
        start = (pd.Timestamp.today() - pd.DateOffset(years=HISTORY_YEARS)).strftime("%Y-%m-%d")

    last_error = None
    for attempt in range(1, FETCH_RETRY_ATTEMPTS + 1):
        try:
            df = psxdata.stocks(ticker, start=start, end=end)
            if df is None or df.empty:
                logger.warning(f"[{ticker}] No data returned for {start}..{end} "
                                f"(possibly delisted, suspended, or not yet listed in this range).")
                return pd.DataFrame(columns=STANDARD_COLUMNS)
            return _standardize(df)
        except Exception as e:
            last_error = e
            wait = attempt * 2
            logger.warning(f"[{ticker}] Fetch attempt {attempt}/{FETCH_RETRY_ATTEMPTS} "
                            f"failed ({e}); retrying in {wait}s..." if attempt < FETCH_RETRY_ATTEMPTS
                            else f"[{ticker}] Fetch failed after {attempt} attempts: {e}")
            if attempt < FETCH_RETRY_ATTEMPTS:
                time.sleep(wait)

    logger.error(f"[{ticker}] Giving up after {FETCH_RETRY_ATTEMPTS} attempts. "
                 f"Last error: {last_error}. Skipping this ticker for this run — "
                 f"existing local data (if any) is left untouched.")
    return pd.DataFrame(columns=STANDARD_COLUMNS)


def fetch_all(tickers: list, start: str = None, end: str = None,
              delay_seconds: float = None) -> dict:
    """
    Fetch a batch of tickers one at a time (deliberately sequential, not
    parallel, to stay a polite, low-load citizen of a scraped free source).
    Returns {ticker: DataFrame}. A ticker whose fetch fails/returns nothing
    gets an empty DataFrame in the result rather than being silently
    dropped, so callers can distinguish "fetched, zero new rows" from
    "never attempted."
    """
    delay = delay_seconds if delay_seconds is not None else REQUEST_DELAY_SECONDS
    results = {}
    for i, ticker in enumerate(tickers):
        results[ticker] = fetch_history(ticker, start=start, end=end)
        if i < len(tickers) - 1:
            time.sleep(delay)
    return results
