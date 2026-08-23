"""
src/external_data.py - Non-price data: sectors, macro, commodities, fundamentals
====================================================================================
Parses raw_uploads/External_Data_Systematically_Organized.xlsx into clean
pandas objects. This is the ONLY file that reads that workbook — kept
separate from the core price/ML pipeline (same separation-of-concerns
principle as psx_api_fetch.py being the only file touching psxdata).

WHAT'S IN THE SOURCE FILE, AND HONEST LIMITS OF EACH PIECE:
- Stock_Industry / Factor_Mapping: static reference data, complete for all
  6 tickers, no time-series alignment issues. Safe to use anywhere.
- Macro_Data / Sector_Data: REAL but MONTHLY, and only ~24 months
  (Aug 2024-Jul 2026) vs. the 10 years of daily price history this project
  otherwise has. Any model that uses these must be evaluated ONLY on that
  same ~24-month window, compared fairly against a price-only baseline on
  the identical window — see src/macro_experiment.py. Sector_Data's
  NASDAQ100 column is present but entirely empty in the source file.
- Company_Financials: annual EPS/Revenue/Net Profit, but ONLY for AICL,
  OGDC, MEBL — SYS/LUCK/FFC are blank in the source (explicitly, honestly
  left blank rather than invented, per the source file's own README sheet).
  Too sparse (4 points/ticker) to be a per-day ML feature; used here only
  for a real, simple P/E snapshot — see compute_pe_ratios().
"""

import os
import pandas as pd
import numpy as np

from config import RAW_UPLOADS_DIR, features_parquet_path, get_logger

logger = get_logger(__name__)

EXTERNAL_FILE_NAME = "External_Data_Systematically_Organized.xlsx"


def _find_external_file() -> str:
    path = os.path.join(RAW_UPLOADS_DIR, EXTERNAL_FILE_NAME)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"External data file not found at {path}. Place "
            f"'{EXTERNAL_FILE_NAME}' in raw_uploads/ first."
        )
    return path


def load_stock_industry() -> pd.DataFrame:
    """Ticker, Company, Industry, Sector, and the 2 hypothesized external
    factors most relevant to each stock. Static reference data."""
    df = pd.read_excel(_find_external_file(), sheet_name="Stock_Industry")
    df.columns = [c.strip() for c in df.columns]
    return df


def load_factor_mapping() -> pd.DataFrame:
    """Per-ticker economic hypotheses: which external factor, what
    relationship is expected, and why. Pure documentation/rationale —
    no computation, just the reasoning behind what gets tested."""
    df = pd.read_excel(_find_external_file(), sheet_name="Factor_Mapping")
    df.columns = [c.strip() for c in df.columns]
    return df.dropna(how="all")


def load_macro_data() -> pd.DataFrame:
    """Monthly macroeconomic series: SBP policy rate, CPI, USD/PKR, KIBOR,
    T-bill and PIB yields. ~24 months of coverage — see module docstring."""
    df = pd.read_excel(_find_external_file(), sheet_name="Macro_Data")
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m")
    return df.sort_values("Date").reset_index(drop=True)


def load_sector_data() -> pd.DataFrame:
    """Monthly commodity/global-market series: Brent, WTI, Coal, Natural
    Gas, Fertilizer price, NASDAQ100 (NASDAQ100 is present but empty in
    the source — kept as a column deliberately, so its absence is visible
    in the data, not silently dropped)."""
    df = pd.read_excel(_find_external_file(), sheet_name="Sector_Data")
    # Drop only genuinely unnamed/structural blank columns (e.g. a stray
    # trailing "Notes" column), never a named data column just because
    # it happens to be empty (NASDAQ100 must stay visible-but-empty).
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    if "Notes" in df.columns:
        df = df.drop(columns=["Notes"])
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date").reset_index(drop=True)


def load_company_financials() -> pd.DataFrame:
    """Annual EPS/Revenue/Net Profit per ticker where available (AICL,
    OGDC, MEBL only — see module docstring). Drops the trailing note row."""
    df = pd.read_excel(_find_external_file(), sheet_name="Company_Financials")
    df = df.dropna(subset=["Ticker"])
    df = df[df["Ticker"] != "Note:"]
    df["Date/Year"] = pd.to_numeric(df["Date/Year"], errors="coerce")
    return df.dropna(subset=["Date/Year"]).reset_index(drop=True)


def compute_pe_ratios(tickers: list, assumed_publication_lag_days: int = 90) -> pd.DataFrame:
    """
    Real, simple P/E ratio: current price / most recently PUBLISHED EPS.

    Honesty note on look-ahead: the source file gives fiscal YEAR, not an
    exact publication date. We assume results become public
    `assumed_publication_lag_days` after fiscal year-end (a reasonable
    approximation of typical PSX reporting lag, not a verified fact) —
    stated explicitly here and in the UI rather than silently assumed.
    For a simple "P/E as of today" snapshot (today = latest date in the
    price data), this only matters if the most recent fiscal year's
    results might not have been public yet by that date; we check for
    that explicitly below rather than assuming the latest EPS is always safe to use.
    """
    financials = load_company_financials()
    rows = []

    for t in tickers:
        stock_fin = financials[financials["Ticker"] == t].sort_values("Date/Year")
        if stock_fin.empty:
            rows.append({"ticker": t, "eps": None, "pe_ratio": None,
                         "fiscal_year": None, "note": "No fundamental data available for this stock."})
            continue

        try:
            feat_path = features_parquet_path(t)
            price_df = pd.read_parquet(feat_path)
            current_price = float(price_df["close"].iloc[-1])
            latest_date = pd.to_datetime(price_df["date"].iloc[-1])
        except Exception as e:
            rows.append({"ticker": t, "eps": None, "pe_ratio": None,
                         "fiscal_year": None, "note": f"No price data available: {e}"})
            continue

        # Walk backward through fiscal years to find the latest one whose
        # ASSUMED publication date is safely before latest_date.
        usable_row = None
        for _, r in stock_fin[::-1].iterrows():
            fiscal_year = int(r["Date/Year"])
            assumed_pub_date = pd.Timestamp(year=fiscal_year + 1, month=1, day=1) + \
                pd.Timedelta(days=assumed_publication_lag_days)
            if assumed_pub_date <= latest_date:
                usable_row = r
                break

        if usable_row is None or pd.isna(usable_row.get("EPS")):
            rows.append({"ticker": t, "eps": None, "pe_ratio": None,
                         "fiscal_year": None,
                         "note": "No fiscal year's EPS is safely known as of the latest price date."})
            continue

        eps = float(usable_row["EPS"])
        pe = round(current_price / eps, 2) if eps > 0 else None
        rows.append({
            "ticker": t, "eps": eps, "pe_ratio": pe,
            "fiscal_year": int(usable_row["Date/Year"]),
            "current_price": round(current_price, 2),
            "note": f"Using FY{int(usable_row['Date/Year'])} EPS "
                    f"(assumed public ~{assumed_publication_lag_days} days after fiscal year-end).",
        })

    return pd.DataFrame(rows)
