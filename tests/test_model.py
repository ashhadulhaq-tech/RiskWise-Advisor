"""
tests/test_model.py
======================
Verifies the parts of the model pipeline that are easy to silently get
wrong: the time-based split (critical for valid evaluation) and defensive
error handling around missing models/data.
"""

import sys
import os
import pytest
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from model import chronological_split, predict_next_day, load_model
from config import TICKERS


class TestChronologicalSplit:
    def test_split_preserves_row_order_and_count(self):
        df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=100), "x": range(100)})
        train, test = chronological_split(df, test_size=0.2)
        assert len(train) == 80
        assert len(test) == 20
        assert len(train) + len(test) == len(df)

    def test_test_set_is_strictly_later_than_train_set(self):
        """The single most important property: no lookahead leakage."""
        df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=100), "x": range(100)})
        train, test = chronological_split(df, test_size=0.2)
        assert train["date"].max() < test["date"].min()

    def test_different_test_sizes(self):
        df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=100), "x": range(100)})
        train, test = chronological_split(df, test_size=0.3)
        assert len(test) == 30


class TestPredictNextDay:
    def test_missing_model_raises_clear_error(self):
        with pytest.raises(FileNotFoundError, match="No trained model found"):
            predict_next_day("DEFINITELY_NOT_A_REAL_TICKER")

    def test_load_model_raises_clear_error_for_missing(self):
        with pytest.raises(FileNotFoundError):
            load_model("DEFINITELY_NOT_A_REAL_TICKER")

    @pytest.mark.parametrize("ticker", TICKERS)
    def test_prediction_returns_valid_direction_and_probability(self, ticker):
        """Requires model.py to have been run first for these tickers."""
        model_path = os.path.join(
            os.path.dirname(__file__), "..", "models", f"{ticker}_model.pkl"
        )
        if not os.path.exists(model_path):
            pytest.skip(f"No trained model for {ticker} — run model.py first.")

        direction, prob = predict_next_day(ticker)
        assert direction in ("UP", "DOWN")
        assert 0.0 <= prob <= 1.0
