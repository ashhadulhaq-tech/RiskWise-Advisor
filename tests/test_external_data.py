"""
tests/test_external_data.py
==============================
Covers external_data.py (sector/factor reference, P/E computation) and
macro_experiment.py (the price-only vs price+macro comparison). Requires
raw_uploads/External_Data_Systematically_Organized.xlsx to be present and
the main pipeline to have already been run (features exist for all tickers).
"""

import sys
import os
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import external_data as ext
from config import get_available_tickers


class TestExternalData:
    def test_stock_industry_covers_all_tickers(self):
        industry = ext.load_stock_industry()
        tickers = set(get_available_tickers())
        assert tickers.issubset(set(industry["Ticker"]))

    def test_factor_mapping_not_empty(self):
        factors = ext.load_factor_mapping()
        assert len(factors) > 0
        assert "Ticker" in factors.columns

    def test_macro_data_has_expected_columns(self):
        macro = ext.load_macro_data()
        for col in ["SBP_Rate", "CPI_YoY", "USD_PKR", "KIBOR_3M"]:
            assert col in macro.columns

    def test_sector_data_keeps_empty_nasdaq_column_visible(self):
        """NASDAQ100 is genuinely empty in the source — it should still
        appear as a column (not be silently dropped), so its absence is
        visible rather than hidden."""
        sector = ext.load_sector_data()
        assert "NASDAQ100" in sector.columns
        assert sector["NASDAQ100"].isna().all()

    def test_pe_ratios_only_computed_where_data_exists(self):
        pe_df = ext.compute_pe_ratios(get_available_tickers())
        has_pe = pe_df[pe_df["pe_ratio"].notna()]
        no_pe = pe_df[pe_df["pe_ratio"].isna()]
        # We know from the source file: AICL/OGDC/MEBL have data, SYS/LUCK/FFC don't
        assert set(has_pe["ticker"]).issubset({"AICL", "OGDC", "MEBL"})
        assert len(no_pe) >= 1  # at least the known-blank tickers show up honestly

    def test_pe_ratio_is_positive_where_computed(self):
        pe_df = ext.compute_pe_ratios(get_available_tickers())
        computed = pe_df[pe_df["pe_ratio"].notna()]
        assert (computed["pe_ratio"] > 0).all()


class TestMacroExperiment:
    """Requires the main pipeline (features.py) to have already run."""

    def test_experiment_runs_and_returns_both_models(self):
        import macro_experiment as mx
        tickers = get_available_tickers()
        if not tickers:
            pytest.skip("No tickers available — run the main pipeline first.")

        result = mx.run_experiment(tickers[0])
        assert "price_only" in result and "price_plus_macro" in result
        assert "interpretation" in result
        assert result["price_only"]["test_rows"] == result["price_plus_macro"]["test_rows"]

    def test_augmented_dataset_has_no_leaked_nan_macro_rows(self):
        import macro_experiment as mx
        tickers = get_available_tickers()
        if not tickers:
            pytest.skip("No tickers available — run the main pipeline first.")

        df = mx.build_augmented_dataset(tickers[0])
        for col in mx.MACRO_FEATURE_COLS + mx.SECTOR_FEATURE_COLS:
            assert df[col].notna().all(), f"{col} has unfilled gaps in the merged dataset"
