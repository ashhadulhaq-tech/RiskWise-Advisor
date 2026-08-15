"""
Module B - Prediction Model
=============================
Job: train a model that predicts whether tomorrow's closing price will be
UP or DOWN, using the technical indicators built in Module A.2.

WHY RANDOM FOREST (not LSTM):
- Works very well on tabular/technical-indicator data (this kind of data,
  not raw sequences of images/text).
- Trains in seconds, not hours - important with your timeline.
- Gives "feature importance" for free, which is great for your report
  (you can literally show WHICH indicators the model relied on).
- Easy to explain in a viva: "it's many decision trees voting together."
LSTMs are more appropriate for raw sequential patterns and need far more
data + tuning to beat a well-tuned Random Forest on this kind of feature set.
If you have extra time at the end, swapping in an LSTM is a nice stretch goal,
but it is NOT required for a solid, defensible project.

CRITICAL DETAIL - WHY A TIME-BASED SPLIT:
Normal ML train/test splits shuffle data randomly. That is WRONG for stock
data: it would let the model "see the future" (train on Tuesday, test on
Monday). We must always train on the past and test on a later, unseen
period - this is called a walk-forward / chronological split.
"""

import pandas as pd
import numpy as np
import os
import json
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report)

from config import DATA_DIR, MODEL_DIR, TICKERS, get_logger

logger = get_logger(__name__)

# In-memory cache so a model .pkl is only read from disk once per process,
# not on every single prediction call (flagged in the evaluation as a
# performance weakness). Streamlit's own st.cache_resource wraps this
# again at the app layer for cross-request caching.
_MODEL_CACHE = {}

FEATURE_COLS = [
    "ma_5", "ma_10", "ma_20", "ma_ratio_5_20",
    "rsi_14", "macd", "macd_signal",
    "daily_return", "volatility_10", "volatility_20",
    "volume_change", "volume_ma_10", "high_low_range",
]


def chronological_split(df: pd.DataFrame, test_size: float = 0.2):
    """Split by TIME, not randomly. Train = earliest 80%, Test = most recent 20%."""
    split_idx = int(len(df) * (1 - test_size))
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]
    return train, test


def train_model(ticker: str, n_estimators: int = 300, max_depth: int = 6):
    path = os.path.join(DATA_DIR, f"{ticker}_features.csv")
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    train, test = chronological_split(df, test_size=0.2)

    X_train, y_train = train[FEATURE_COLS], train["target_direction"]
    X_test, y_test = test[FEATURE_COLS], test["target_direction"]

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=10,   # guards against overfitting on noisy market data
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    # --- Baseline to compare against: "always predict the majority class" ---
    baseline_pred = np.full_like(y_test, y_train.mode()[0])
    baseline_acc = accuracy_score(y_test, baseline_pred)

    metrics = {
        "ticker": ticker,
        "train_rows": len(train),
        "test_rows": len(test),
        "test_period": f"{test['date'].min().date()} to {test['date'].max().date()}",
        "accuracy": round(accuracy_score(y_test, preds), 4),
        "baseline_accuracy": round(baseline_acc, 4),   # <-- the honest benchmark
        "precision": round(precision_score(y_test, preds, zero_division=0), 4),
        "recall": round(recall_score(y_test, preds, zero_division=0), 4),
        "f1_score": round(f1_score(y_test, preds, zero_division=0), 4),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
    }

    # feature importance -> which signals the model actually relied on
    importances = dict(zip(FEATURE_COLS, model.feature_importances_.round(4)))
    importances = dict(sorted(importances.items(), key=lambda x: -x[1]))
    metrics["feature_importance"] = importances

    # save model + metrics
    joblib.dump(model, os.path.join(MODEL_DIR, f"{ticker}_model.pkl"))
    with open(os.path.join(MODEL_DIR, f"{ticker}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    top5 = ", ".join(f"{feat}={imp}" for feat, imp in list(importances.items())[:5])
    logger.info(
        f"[{ticker}] test={metrics['test_period']} ({metrics['test_rows']}d) | "
        f"acc={metrics['accuracy']*100:.1f}% vs baseline={metrics['baseline_accuracy']*100:.1f}% | "
        f"precision={metrics['precision']:.3f} recall={metrics['recall']:.3f} f1={metrics['f1_score']:.3f} | "
        f"top features: {top5}"
    )

    return model, metrics


def load_model(ticker: str):
    """Load a trained model, with a friendly error if it doesn't exist yet."""
    if ticker in _MODEL_CACHE:
        return _MODEL_CACHE[ticker]

    model_path = os.path.join(MODEL_DIR, f"{ticker}_model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No trained model found for '{ticker}'. Run model.py's "
            f"train_model('{ticker}') first."
        )
    model = joblib.load(model_path)
    _MODEL_CACHE[ticker] = model
    return model


def predict_next_day(ticker: str):
    """Load the saved model and predict tomorrow's direction from the latest row.

    Raises FileNotFoundError / ValueError with clear messages instead of
    letting a raw traceback reach the UI layer.
    """
    model = load_model(ticker)

    feat_path = os.path.join(DATA_DIR, f"{ticker}_features.csv")
    if not os.path.exists(feat_path):
        raise FileNotFoundError(
            f"No feature data found for '{ticker}'. Run features.py first."
        )

    df = pd.read_csv(feat_path)
    if df.empty:
        raise ValueError(f"[{ticker}] Feature file is empty — cannot predict.")

    latest = df.iloc[[-1]][FEATURE_COLS]

    pred = model.predict(latest)[0]
    prob_up = model.predict_proba(latest)[0][1]

    direction = "UP" if pred == 1 else "DOWN"
    logger.info(f"[{ticker}] Prediction for next trading day: {direction} "
                f"(confidence: {prob_up*100:.1f}% chance of UP)")
    return direction, prob_up


if __name__ == "__main__":
    all_metrics = []
    for t in TICKERS:
        try:
            _, m = train_model(t)
            all_metrics.append(m)
            predict_next_day(t)
        except Exception as e:
            logger.error(f"[{t}] Failed to train/predict, skipping: {e}")

    if all_metrics:
        avg_acc = np.mean([m["accuracy"] for m in all_metrics])
        avg_baseline = np.mean([m["baseline_accuracy"] for m in all_metrics])
        logger.info(f"SUMMARY — Average model accuracy: {avg_acc*100:.1f}% "
                    f"vs baseline: {avg_baseline*100:.1f}%")
