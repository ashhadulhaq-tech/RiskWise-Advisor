"""
src/macro_experiment.py - Does non-market data actually improve prediction?
==============================================================================
Builds a fair, honest comparison: a price-only model vs. a price+macro
model, trained and evaluated on the IDENTICAL, shorter window where macro
data actually exists (~24 months — see external_data.py's module
docstring for why). This directly tests the question the external data's
own README sheet posed: "Test whether non-market information improves
prediction."

WHY THIS IS A SEPARATE MODULE, NOT MERGED INTO model.py:
The main model.py trains on the full ~10-year history for the best
possible sample size. This experiment necessarily uses a much smaller
window (~500 trading days vs ~2500), so its results are NOT directly
comparable to model.py's headline accuracy numbers — mixing them would
misrepresent both. Keeping this separate, clearly labeled "experimental"
avoids that confusion.

METHODOLOGY:
- Monthly macro/sector values are forward-filled to daily frequency
  (standard approach — a rate/price stays "current" until the next
  update) and merged onto the stock's existing daily technical features,
  restricted to dates where BOTH exist.
- Same chronological train/test split and majority-class baseline
  comparison as the main model, applied consistently here too.
- Reports BOTH models' accuracy on the exact same test period, so any
  difference is attributable to the added features, not a different
  evaluation window.
- An accuracy improvement of a point or two on ~100 test days is not
  strong evidence of anything — this is stated explicitly in the output,
  not left for the reader to misjudge.
"""

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from config import MODEL_DIR, features_parquet_path, get_logger
from model import chronological_split, FEATURE_COLS as PRICE_FEATURE_COLS
import external_data as ext

logger = get_logger(__name__)

MACRO_FEATURE_COLS = [
    "SBP_Rate", "CPI_YoY", "USD_PKR", "KIBOR_3M", "KIBOR_6M", "KIBOR_12M",
    "TBill_3M", "TBill_6M", "TBill_12M", "PIB_5Y", "PIB_10Y",
]
SECTOR_FEATURE_COLS = ["Brent", "WTI", "Coal", "Natural_Gas", "Fertilizer_Price"]
# NASDAQ100 deliberately excluded — entirely empty in the source data,
# see external_data.py. Including an all-NaN column would just be noise.


def build_augmented_dataset(ticker: str) -> pd.DataFrame:
    """
    Merge a stock's daily technical features with forward-filled monthly
    macro + sector data, restricted to the overlapping date range only.
    """
    price_path = features_parquet_path(ticker)
    if not os.path.exists(price_path):
        raise FileNotFoundError(f"No price features for '{ticker}'. Run the main pipeline first.")

    price_df = pd.read_parquet(price_path)
    price_df["date"] = pd.to_datetime(price_df["date"])

    macro = ext.load_macro_data().rename(columns={"Date": "date"})
    sector = ext.load_sector_data().rename(columns={"Date": "date"}).drop(columns=["NASDAQ100"])

    # Forward-fill monthly data across a daily calendar spanning the
    # overlap, then merge onto the stock's actual trading days.
    overlap_start = max(macro["date"].min(), sector["date"].min(), price_df["date"].min())
    overlap_end = min(macro["date"].max(), sector["date"].max(), price_df["date"].max())

    if overlap_start >= overlap_end:
        raise ValueError(f"No overlapping date range between price data and macro/sector data for '{ticker}'.")

    daily_calendar = pd.DataFrame({"date": pd.date_range(overlap_start, overlap_end, freq="D")})
    macro_daily = daily_calendar.merge(macro, on="date", how="left").ffill()
    sector_daily = daily_calendar.merge(sector, on="date", how="left").ffill()

    merged = price_df[(price_df["date"] >= overlap_start) & (price_df["date"] <= overlap_end)].copy()
    merged = merged.merge(macro_daily, on="date", how="left")
    merged = merged.merge(sector_daily, on="date", how="left")
    merged = merged.dropna(subset=MACRO_FEATURE_COLS + SECTOR_FEATURE_COLS)

    return merged.reset_index(drop=True)


def _train_and_evaluate(df: pd.DataFrame, feature_cols: list, label: str) -> dict:
    train, test = chronological_split(df, test_size=0.2)
    X_train, y_train = train[feature_cols], train["target_direction"]
    X_test, y_test = test[feature_cols], test["target_direction"]

    if len(X_train) < 30 or len(X_test) < 10:
        raise ValueError(f"[{label}] Not enough rows to train/evaluate reliably "
                         f"({len(X_train)} train, {len(X_test)} test).")

    model = RandomForestClassifier(
        n_estimators=200, max_depth=5, min_samples_leaf=8,
        random_state=42, n_jobs=-1,
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    baseline_pred = np.full_like(y_test, y_train.mode()[0])

    return {
        "label": label,
        "train_rows": len(train),
        "test_rows": len(test),
        "test_period": f"{test['date'].min().date()} to {test['date'].max().date()}",
        "accuracy": round(accuracy_score(y_test, preds), 4),
        "baseline_accuracy": round(accuracy_score(y_test, baseline_pred), 4),
        "precision": round(precision_score(y_test, preds, zero_division=0), 4),
        "recall": round(recall_score(y_test, preds, zero_division=0), 4),
        "f1_score": round(f1_score(y_test, preds, zero_division=0), 4),
        "n_features": len(feature_cols),
    }


def run_experiment(ticker: str) -> dict:
    """
    The core comparison: price-only vs price+macro+sector, same window,
    same split methodology, same baseline comparison. Returns a dict
    with both results plus an honest interpretation note.
    """
    df = build_augmented_dataset(ticker)

    baseline_result = _train_and_evaluate(df, PRICE_FEATURE_COLS, "price_only")
    augmented_result = _train_and_evaluate(
        df, PRICE_FEATURE_COLS + MACRO_FEATURE_COLS + SECTOR_FEATURE_COLS, "price_plus_macro"
    )

    diff = augmented_result["accuracy"] - baseline_result["accuracy"]
    n_test = augmented_result["test_rows"]

    if abs(diff) < 0.02:
        interpretation = ("No meaningful difference — adding macro/commodity data did not "
                          "measurably change prediction accuracy on this stock, over this window.")
    elif diff > 0:
        interpretation = (f"Price+macro model scored {diff*100:.1f} points higher on this "
                          f"~{n_test}-day test period. With a sample this small, this is "
                          f"suggestive, not conclusive — it could plausibly be noise rather "
                          f"than a real, generalizable effect.")
    else:
        interpretation = (f"Price+macro model scored {abs(diff)*100:.1f} points LOWER than "
                          f"the price-only baseline on this window — adding these features "
                          f"did not help here, and may have added noise instead.")

    result = {
        "ticker": ticker,
        "overlap_window": baseline_result["test_period"],
        "price_only": baseline_result,
        "price_plus_macro": augmented_result,
        "accuracy_difference": round(diff, 4),
        "interpretation": interpretation,
    }

    out_path = os.path.join(MODEL_DIR, f"{ticker}_macro_experiment.json")
    import json
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    logger.info(f"[{ticker}] Macro experiment: price-only={baseline_result['accuracy']*100:.1f}% "
                f"vs price+macro={augmented_result['accuracy']*100:.1f}% "
                f"({'+' if diff >= 0 else ''}{diff*100:.1f}pp)")

    return result


if __name__ == "__main__":
    from config import get_available_tickers
    for t in get_available_tickers():
        try:
            run_experiment(t)
        except Exception as e:
            logger.error(f"[{t}] Macro experiment failed, skipping: {e}")
