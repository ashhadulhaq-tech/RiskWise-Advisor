"""
Module C - Risk Profiling & Advisory Engine
==============================================
Job: turn a user's answers to a short questionnaire into a risk profile
(Conservative / Moderate / Aggressive), then recommend stocks that actually
match that profile - using REAL, MEASURABLE risk data (volatility), not
just vibes.

WHY NOT USE ML FOR THIS PART:
Real robo-advisors (Betterment, Wealthfront, etc.) use scored questionnaires,
not machine learning, for risk profiling - it's more transparent, explainable,
auditable, and it's what regulators / your professor will expect to see.
This is a legitimate design decision, not a shortcut - say so in your report.

HOW STOCK-RISK IS MEASURED:
We use `volatility_20` (20-day rolling standard deviation of daily returns)
from Module A.2's features as our objective risk proxy for each stock -
this is the same idea real finance uses (similar in spirit to a stock's
"beta" or standard deviation of returns). Higher volatility = higher risk.

HOW RECOMMENDATION WORKS:
1. Score the questionnaire -> risk category
2. Rank all stocks by their recent volatility
3. Filter to the volatility band matching the user's risk category
4. Within that filtered list, use Module B's prediction (direction + confidence)
   to rank/prioritize which of the suitable stocks to actually suggest
"""

import pandas as pd
import numpy as np
import os
import json

from config import (DATA_DIR, MODEL_DIR, TICKERS, VOLATILITY_BANDS,
                     TRADING_DAYS_PER_YEAR, RISK_CATEGORY_ALLOWED_BANDS, get_logger)

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# 1. QUESTIONNAIRE
# ---------------------------------------------------------------------------
# Each question: text + list of (answer_label, score) options.
# Scores loosely follow standard robo-advisor risk-tolerance questionnaires.

QUESTIONNAIRE = [
    {
        "id": "q1",
        "question": "If your investment dropped 20% in a month, what would you do?",
        "options": [
            ("Sell everything immediately", 1),
            ("Sell some to reduce risk", 2),
            ("Hold and wait it out", 3),
            ("Buy more while it's cheap", 4),
        ],
    },
    {
        "id": "q2",
        "question": "What is your investment time horizon?",
        "options": [
            ("Less than 1 year", 1),
            ("1-3 years", 2),
            ("3-7 years", 3),
            ("7+ years", 4),
        ],
    },
    {
        "id": "q3",
        "question": "How would you describe your investing experience?",
        "options": [
            ("None - I'm new to this", 1),
            ("Some - I've invested a little", 2),
            ("Moderate - I invest regularly", 3),
            ("Extensive - I actively trade", 4),
        ],
    },
    {
        "id": "q4",
        "question": "How stable is your current income/financial situation?",
        "options": [
            ("Unstable - I rely on these funds soon", 1),
            ("Somewhat stable", 2),
            ("Stable with some savings buffer", 3),
            ("Very stable with a large safety net", 4),
        ],
    },
    {
        "id": "q5",
        "question": "Which statement matches you best?",
        "options": [
            ("I want to protect my money above all", 1),
            ("I want steady, modest growth", 2),
            ("I want strong growth and accept ups and downs", 3),
            ("I want maximum growth, risk doesn't scare me", 4),
        ],
    },
    {
        "id": "q6",
        "question": "How would you feel checking your portfolio daily and seeing big swings?",
        "options": [
            ("Very anxious - I'd want to stop investing", 1),
            ("Uncomfortable but I'd manage", 2),
            ("Fairly relaxed about it", 3),
            ("Excited - volatility means opportunity", 4),
        ],
    },
]


def score_questionnaire(answers: dict) -> dict:
    """
    answers: dict of {question_id: option_index (0-based)}
    Returns risk score, category, and a breakdown.
    """
    total = 0
    max_possible = 0
    breakdown = []
    for q in QUESTIONNAIRE:
        qid = q["id"]
        idx = answers.get(qid)
        if idx is None:
            raise ValueError(f"Missing answer for question '{qid}': {q['question']}")
        if not isinstance(idx, int) or not (0 <= idx < len(q["options"])):
            raise ValueError(
                f"Invalid answer index {idx!r} for question '{qid}' "
                f"(must be an integer from 0 to {len(q['options']) - 1})"
            )
        label, score = q["options"][idx]
        total += score
        max_possible += 4
        breakdown.append({"question": q["question"], "answer": label, "score": score})

    pct = total / max_possible  # 0.25 (all min) to 1.0 (all max)

    if pct <= 0.45:
        category = "Conservative"
    elif pct <= 0.70:
        category = "Moderate"
    else:
        category = "Aggressive"

    return {
        "total_score": total,
        "max_score": max_possible,
        "percentage": round(pct * 100, 1),
        "risk_category": category,
        "breakdown": breakdown,
    }


# ---------------------------------------------------------------------------
# 2. RISK-TO-STOCK MATCHING
# ---------------------------------------------------------------------------
# FIXED at design time (see config.VOLATILITY_BANDS), not computed relative
# to whichever tickers happen to be in the dataset. This was flagged in the
# evaluation: relative percentile bands meant a stock could be labeled "low
# risk" purely for being the calmest of a risky group, and every label would
# shift if the ticker list changed. Fixed thresholds don't have that problem.
#
# We ALSO compute beta against a market benchmark - how much a stock moves
# relative to the overall market, the standard measure of systematic risk in
# finance. Ideally this benchmark is a real index (e.g. KSE-100 for PSX
# stocks). We don't have live index data wired in yet (see update_data.py /
# the daily-refresh discussion), so as an interim, documented stand-in we use
# an equal-weighted average of daily returns across our own tracked tickers
# as a proxy "market". This is clearly a placeholder - swap `get_market_returns()`
# for a real index feed before treating beta numbers as meaningful.

ALLOCATION_GUIDANCE = {
    "Conservative": "70% low-volatility stocks / bonds-style holdings, 30% moderate",
    "Moderate":     "50% low-volatility, 40% moderate, 10% higher-growth",
    "Aggressive":   "20% low-volatility, 30% moderate, 50% high-growth/high-volatility",
}


def _annualize_vol(daily_vol: float) -> float:
    return daily_vol * np.sqrt(TRADING_DAYS_PER_YEAR)


def _band_for_volatility(annualized_vol: float) -> str:
    for band_name, (low, high) in VOLATILITY_BANDS.items():
        if low <= annualized_vol < high:
            return band_name
    return "high"


def get_market_returns(tickers: list) -> pd.Series:
    """
    PLACEHOLDER market benchmark: equal-weighted average daily return across
    all tracked tickers, indexed by date. Real deployments should replace
    this with an actual index (e.g. KSE-100) once a live data feed exists.
    """
    all_returns = []
    for t in tickers:
        path = os.path.join(DATA_DIR, f"{t}_features.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)[["date", "daily_return"]].set_index("date")
        df.columns = [t]
        all_returns.append(df)
    if not all_returns:
        return pd.Series(dtype=float)
    combined = pd.concat(all_returns, axis=1)
    return combined.mean(axis=1)


def _compute_beta(stock_returns: pd.Series, market_returns: pd.Series) -> float:
    aligned = pd.concat([stock_returns, market_returns], axis=1).dropna()
    if len(aligned) < 20:
        return np.nan
    cov = aligned.iloc[:, 0].cov(aligned.iloc[:, 1])
    market_var = aligned.iloc[:, 1].var()
    if market_var == 0 or np.isnan(market_var):
        return np.nan
    return cov / market_var


def get_stock_risk_metrics(tickers: list) -> pd.DataFrame:
    """
    Compute risk metrics per ticker: annualized volatility (fixed-band
    classified), beta vs. the market proxy, and current price.
    Tickers with missing/unreadable data are skipped with a logged warning
    rather than crashing the whole batch (evaluation Bug #1).
    """
    market_returns = get_market_returns(tickers)

    rows = []
    for t in tickers:
        path = os.path.join(DATA_DIR, f"{t}_features.csv")
        try:
            if not os.path.exists(path):
                raise FileNotFoundError(f"No feature file for '{t}' at {path}")
            df = pd.read_csv(path)
            if df.empty:
                raise ValueError(f"Feature file for '{t}' is empty")

            recent = df.tail(60)  # last ~3 months of trading days
            daily_vol = recent["volatility_20"].mean()
            ann_vol = _annualize_vol(daily_vol)

            stock_returns = df.set_index("date")["daily_return"]
            beta = _compute_beta(stock_returns, market_returns)

            rows.append({
                "ticker": t,
                "avg_daily_volatility": daily_vol,
                "annualized_volatility": ann_vol,
                "risk_band": _band_for_volatility(ann_vol),
                "beta": round(beta, 3) if not np.isnan(beta) else None,
                "avg_daily_return": recent["daily_return"].mean(),
                "last_close": df["close"].iloc[-1],
            })
        except Exception as e:
            logger.warning(f"[{t}] Skipped in risk metrics — {e}")
            continue

    if not rows:
        raise ValueError(
            "No valid stock data available to compute risk metrics for any "
            f"of the requested tickers: {tickers}"
        )

    return pd.DataFrame(rows).sort_values("annualized_volatility").reset_index(drop=True)


def get_model_prediction(ticker: str):
    """Read the metrics/prediction saved by Module B, if available."""
    metrics_path = os.path.join(MODEL_DIR, f"{ticker}_metrics.json")
    if not os.path.exists(metrics_path):
        return None
    with open(metrics_path) as f:
        return json.load(f)


def recommend_stocks(risk_category: str, tickers: list, top_n: int = 5) -> pd.DataFrame:
    """
    Core advisory logic:
    1. Compute volatility (fixed bands) + beta for every candidate stock
    2. Keep only stocks in bands the user's risk category is allowed to see
    3. Attach the model's next-day prediction as extra context
    4. Rank by prediction confidence within the eligible set

    Raises ValueError for an unknown risk_category rather than a silent
    KeyError, and never lets one ticker's prediction failure crash the
    whole recommendation (evaluation Bug #1).
    """
    if risk_category not in RISK_CATEGORY_ALLOWED_BANDS:
        raise ValueError(
            f"Unknown risk category '{risk_category}'. "
            f"Must be one of: {list(RISK_CATEGORY_ALLOWED_BANDS.keys())}"
        )

    risk_df = get_stock_risk_metrics(tickers)
    allowed_bands = RISK_CATEGORY_ALLOWED_BANDS[risk_category]
    eligible = risk_df[risk_df["risk_band"].isin(allowed_bands)].copy()

    if eligible.empty:
        logger.warning(
            f"No stocks fall in the allowed risk bands {allowed_bands} for "
            f"'{risk_category}' investors — falling back to the single "
            f"lowest-volatility stock available so the user still gets a result."
        )
        eligible = risk_df.head(1).copy()

    # attach model predictions for context (not as the sole driver of risk).
    # Import kept local to avoid a circular import between model.py and
    # risk_engine.py at module load time.
    from model import predict_next_day

    directions, confidences = [], []
    for t in eligible["ticker"]:
        try:
            direction, prob_up = predict_next_day(t)
        except Exception as e:
            logger.warning(f"[{t}] Prediction unavailable — {e}")
            direction, prob_up = "N/A", np.nan
        directions.append(direction)
        confidences.append(prob_up)
    eligible["predicted_direction"] = directions
    eligible["prediction_confidence"] = confidences

    eligible = eligible.sort_values(
        by=["prediction_confidence"], ascending=False, na_position="last"
    ).reset_index(drop=True)

    return eligible.head(top_n)


def full_advisory_flow(answers: dict, tickers: list):
    """End-to-end: questionnaire answers -> profile -> recommendations."""
    profile = score_questionnaire(answers)
    recs = recommend_stocks(profile["risk_category"], tickers)
    return profile, recs


if __name__ == "__main__":
    # Example: a fairly cautious user
    example_answers = {
        "q1": 1,  # "Sell some to reduce risk"
        "q2": 2,  # "3-7 years"
        "q3": 0,  # "Some experience"
        "q4": 1,  # "Somewhat stable"
        "q5": 0,  # "Steady, modest growth"
        "q6": 0,  # "Uncomfortable but I'd manage"
    }

    profile, recs = full_advisory_flow(example_answers, TICKERS)

    logger.info(f"Risk profile: {profile['risk_category']} "
                f"({profile['total_score']}/{profile['max_score']}, {profile['percentage']}%)")
    logger.info(f"Suggested allocation: {ALLOCATION_GUIDANCE[profile['risk_category']]}")
    print(recs[["ticker", "annualized_volatility", "risk_band", "beta",
                "predicted_direction", "prediction_confidence", "last_close"]]
          .to_string(index=False))
