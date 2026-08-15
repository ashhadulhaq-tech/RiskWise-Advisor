"""
src/psx_ingest.py - PSX historical data ingestion
====================================================
Job: read a manually-downloaded PSX historical data Excel file (downloaded
by the user personally from dps.psx.com.pk, per PSX's terms of use — see
README for the licensing discussion) and convert it into the same
data/{TICKER}.csv format that data_pipeline.py produces from yfinance.

This lets the REST of the pipeline (features.py, model.py, risk_engine.py,
app.py) work completely unchanged - they don't know or care whether a
CSV originally came from yfinance or from a manual PSX download.

EXPECTED FILE FORMAT (matches the file provided for this project):
An .xlsx workbook with one sheet per ticker, sheet names like
"AICL_01072016_27072026", each sheet having columns:
    Symbol | Date | Open | High | Low | Close | Volume
"""

import os
import re
import pandas as pd

from config import DATA_DIR, RAW_UPLOADS_DIR, TICKERS, get_logger

logger = get_logger(__name__)

REQUIRED_COLUMNS = {"Symbol", "Date", "Open", "High", "Low", "Close", "Volume"}


def _extract_ticker_from_sheet_name(sheet_name: str) -> str:
    """'AICL_01072016_27072026' -> 'AICL'. Falls back to the part before the first underscore."""
    match = re.match(r"^([A-Za-z0-9]+)_", sheet_name)
    return match.group(1).upper() if match else sheet_name.split("_")[0].upper()


def ingest_psx_excel(xlsx_path: str, tickers_filter: list = None) -> dict:
    """
    Read every per-ticker sheet in the workbook and write it out as a
    standard data/{TICKER}.csv file. Skips the summary sheet (no Symbol
    column) and any sheet that doesn't match the expected column set.

    Returns {ticker: rows_written} for tickers successfully ingested.
    """
    if not os.path.exists(xlsx_path):
        raise FileNotFoundError(f"PSX data file not found: {xlsx_path}")

    xls = pd.ExcelFile(xlsx_path)
    results = {}

    for sheet_name in xls.sheet_names:
        df = xls.parse(sheet_name)

        if not REQUIRED_COLUMNS.issubset(set(df.columns)):
            logger.info(f"Skipping sheet '{sheet_name}' — not a per-ticker "
                        f"data sheet (missing expected columns).")
            continue

        ticker = _extract_ticker_from_sheet_name(sheet_name)
        if tickers_filter and ticker not in tickers_filter:
            logger.info(f"Skipping ticker '{ticker}' (sheet '{sheet_name}') — "
                        f"not in the configured ticker list.")
            continue

        out = df.rename(columns={
            "Date": "date", "Open": "open", "High": "high",
            "Low": "low", "Close": "close", "Volume": "volume",
        })[["date", "open", "high", "low", "close", "volume"]].copy()

        out["date"] = pd.to_datetime(out["date"])
        out = out.sort_values("date").drop_duplicates(subset="date").reset_index(drop=True)

        n_before = len(out)
        out = out.dropna()
        if len(out) < n_before:
            logger.warning(f"[{ticker}] Dropped {n_before - len(out)} rows with missing values.")

        if out.empty:
            logger.error(f"[{ticker}] No usable rows after cleaning — skipped.")
            continue

        out_path = os.path.join(DATA_DIR, f"{ticker}.csv")
        out.to_csv(out_path, index=False)
        results[ticker] = len(out)
        logger.info(f"[{ticker}] Ingested {len(out)} rows "
                    f"({out['date'].min().date()} to {out['date'].max().date()}) -> {out_path}")

    missing = set(tickers_filter or TICKERS) - set(results.keys())
    if missing:
        logger.warning(f"Tickers configured but NOT found in the PSX file: {missing}")

    return results


if __name__ == "__main__":
    # picks up the first .xlsx found in raw_uploads/ by default
    candidates = [f for f in os.listdir(RAW_UPLOADS_DIR) if f.lower().endswith(".xlsx")]
    if not candidates:
        raise FileNotFoundError(
            f"No .xlsx file found in {RAW_UPLOADS_DIR}. Place your downloaded "
            f"PSX historical data file there first."
        )
    xlsx_path = os.path.join(RAW_UPLOADS_DIR, candidates[0])
    logger.info(f"Ingesting {xlsx_path} ...")
    results = ingest_psx_excel(xlsx_path, tickers_filter=TICKERS)
    logger.info(f"Ingestion complete: {results}")
