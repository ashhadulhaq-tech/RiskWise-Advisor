"""
tests/test_risk_engine.py
===========================
Covers the bugs actually found during evaluation, plus core scoring logic.
Run with: pytest tests/ -v   (from the project root, after the data
pipeline has been run at least once so feature CSVs exist).
"""

import sys
import os
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from risk_engine import (
    QUESTIONNAIRE, score_questionnaire, recommend_stocks,
    _band_for_volatility, _annualize_vol,
)
from config import TICKERS, VOLATILITY_BANDS


def _all_min_answers():
    return {q["id"]: 0 for q in QUESTIONNAIRE}


def _all_max_answers():
    return {q["id"]: len(q["options"]) - 1 for q in QUESTIONNAIRE}


class TestScoreQuestionnaire:
    def test_minimum_answers_yield_conservative(self):
        profile = score_questionnaire(_all_min_answers())
        assert profile["risk_category"] == "Conservative"
        assert profile["percentage"] == pytest.approx(25.0)

    def test_maximum_answers_yield_aggressive(self):
        profile = score_questionnaire(_all_max_answers())
        assert profile["risk_category"] == "Aggressive"
        assert profile["percentage"] == pytest.approx(100.0)

    def test_missing_answer_raises_value_error(self):
        answers = _all_min_answers()
        del answers["q1"]
        with pytest.raises(ValueError, match="Missing answer"):
            score_questionnaire(answers)

    def test_out_of_range_index_raises_value_error(self):
        answers = _all_min_answers()
        answers["q1"] = 99
        with pytest.raises(ValueError, match="Invalid answer index"):
            score_questionnaire(answers)

    def test_negative_index_raises_value_error(self):
        answers = _all_min_answers()
        answers["q1"] = -1
        with pytest.raises(ValueError, match="Invalid answer index"):
            score_questionnaire(answers)

    def test_non_integer_index_raises_value_error(self):
        answers = _all_min_answers()
        answers["q1"] = "not a number"
        with pytest.raises(ValueError, match="Invalid answer index"):
            score_questionnaire(answers)

    def test_empty_answers_raises(self):
        with pytest.raises(ValueError):
            score_questionnaire({})


class TestVolatilityBands:
    def test_band_boundaries(self):
        assert _band_for_volatility(0.05) == "low"
        assert _band_for_volatility(0.149) == "low"
        assert _band_for_volatility(0.15) == "moderate"
        assert _band_for_volatility(0.29) == "moderate"
        assert _band_for_volatility(0.30) == "high"
        assert _band_for_volatility(1.0) == "high"

    def test_annualize_vol_is_positive_scaling(self):
        assert _annualize_vol(0.01) > 0.01
        assert _annualize_vol(0.0) == 0.0


class TestRecommendStocks:
    """These require feature CSVs to exist (run data_pipeline.py + features.py first)."""

    def test_unknown_risk_category_raises(self):
        with pytest.raises(ValueError, match="Unknown risk category"):
            recommend_stocks("SuperRisky", TICKERS)

    def test_unknown_ticker_does_not_crash_whole_batch(self):
        recs = recommend_stocks("Moderate", TICKERS + ["NOT_A_REAL_TICKER"])
        assert not recs.empty
        assert "NOT_A_REAL_TICKER" not in recs["ticker"].values

    def test_all_unknown_tickers_raises_clear_error(self):
        with pytest.raises(ValueError):
            recommend_stocks("Moderate", ["FAKE1", "FAKE2"])

    def test_aggressive_sees_at_least_as_many_bands_as_conservative(self):
        cons = recommend_stocks("Conservative", TICKERS, top_n=10)
        aggr = recommend_stocks("Aggressive", TICKERS, top_n=10)
        assert set(cons["risk_band"]).issubset({"low"} | set(aggr["risk_band"]))

    def test_recommendations_have_expected_columns(self):
        recs = recommend_stocks("Moderate", TICKERS)
        expected = {"ticker", "annualized_volatility", "risk_band", "beta",
                    "predicted_direction", "prediction_confidence", "last_close"}
        assert expected.issubset(set(recs.columns))
