"""
app/page_modules/methodology.py - Page 4: Analysis & Methodology
=====================================================================
Moved from the old single-page app's Tab 4 — logic unchanged.
"""
import os
import json
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from config import MODEL_DIR, TICKER_NAMES, TICKER_SECTORS, get_available_tickers
from risk_engine import get_pairwise_correlation_matrix
import external_data as ext


def render_methodology():
    st.title("4. Analysis & Methodology")

    st.subheader("Data sources — what this app does and doesn't use")
    st.markdown(
        "**Main AI prediction model** (next-day direction, shown throughout the app):\n"
        "- ✅ **Historical daily price & volume (OHLCV)** for each stock — the only input\n"
        "- ❌ No news, sentiment, fundamentals, or macro data\n\n"
        "**Supplementary analysis** (this page only, kept separate from the main model):\n"
        "- ✅ **Sector/industry classification** — static context, not a model input\n"
        "- ✅ **Company fundamentals (EPS, P/E)** — real, but only where published data "
        "exists (3 of 6 stocks); used for a display metric, not fed into the ML model\n"
        "- ✅ **Macro & commodity data** (SBP rate, KIBOR, Brent, Coal, etc.) — real, "
        "used ONLY in the separate experimental comparison below, restricted to the "
        "~24-month window where this data actually exists\n"
        "- ❌ **No news or sentiment data** anywhere in this app\n\n"
        "The main model's scope (price/volume only) is a deliberate choice for "
        "reliability across the full 10-year history. The supplementary analysis "
        "below tests whether adding more data sources would help — honestly, "
        "including where it doesn't."
    )

    st.divider()
    st.subheader("Sector overview")
    tickers_now = get_available_tickers()
    try:
        industry_df = ext.load_stock_industry()
        industry_df = industry_df[industry_df["Ticker"].isin(tickers_now)]
        st.dataframe(industry_df, hide_index=True, use_container_width=True)
    except Exception as e:
        sector_rows = [{"Ticker": t, "Company": TICKER_NAMES.get(t, ""),
                         "Sector": TICKER_SECTORS.get(t, "N/A")} for t in tickers_now]
        st.dataframe(pd.DataFrame(sector_rows), hide_index=True, use_container_width=True)
        st.caption(f"(Using built-in sector data — external file unavailable: {e})")
    st.caption(
        "Each stock here represents a different sector — this dataset has one "
        "stock per sector, not a full sector index, so treat this as sector "
        "**context** for each stock, not a statistically representative "
        "sector-performance benchmark."
    )

    st.divider()
    st.subheader("Economic rationale — why these external factors, for these stocks")
    st.caption(
        "Before testing any external data, here's the actual economic reasoning "
        "for each stock — this wasn't a black-box search for correlations, it's "
        "hypothesis-driven."
    )
    try:
        factors_df = ext.load_factor_mapping()
        factors_df = factors_df[factors_df["Ticker"].isin(tickers_now)]
        st.dataframe(factors_df, hide_index=True, use_container_width=True)
    except Exception as e:
        st.warning(f"Couldn't load factor mapping: {e}")

    st.divider()
    st.subheader("Company fundamentals — real P/E ratio, where available")
    st.caption(
        "This app's AI model uses price/volume only (see above) — but where "
        "annual earnings (EPS) data exists, we can compute a genuine P/E "
        "ratio, something a pure price-history model can't do. **Only "
        "available for stocks with published financials in our source data** "
        "— shown honestly as unavailable for the rest, not estimated."
    )
    try:
        pe_df = ext.compute_pe_ratios(tickers_now)
        display_df = pe_df[["ticker", "eps", "pe_ratio", "fiscal_year", "note"]].copy()
        display_df.columns = ["Ticker", "EPS (Rs)", "P/E Ratio", "Fiscal Year", "Note"]
        st.dataframe(display_df, hide_index=True, use_container_width=True)
    except Exception as e:
        st.warning(f"Couldn't compute P/E ratios: {e}")

    st.divider()
    st.subheader("Experiment: does adding macro/commodity data improve prediction?")
    st.caption(
        "⚠️ **Important scope note**: real macro data (SBP rate, KIBOR, Brent, "
        "Coal, etc.) only covers ~24 months, vs. the 10 years of daily price "
        "history the main model uses. This experiment trains BOTH a price-only "
        "model and a price+macro model on that same shorter window, so the "
        "comparison is fair — but the sample is small, so treat results as "
        "suggestive, not definitive. This is intentionally kept separate from "
        "the main model's headline accuracy numbers shown elsewhere."
    )
    exp_ticker = st.selectbox("Choose a stock to see the comparison",
                               tickers_now, key="macro_exp_ticker")
    try:
        exp_path = os.path.join(MODEL_DIR, f"{exp_ticker}_macro_experiment.json")
        if os.path.exists(exp_path):
            with open(exp_path) as f:
                exp = json.load(f)
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Price-only accuracy", f"{exp['price_only']['accuracy']*100:.1f}%",
                           f"vs {exp['price_only']['baseline_accuracy']*100:.1f}% baseline")
            with c2:
                st.metric("Price + macro accuracy", f"{exp['price_plus_macro']['accuracy']*100:.1f}%",
                           f"{exp['accuracy_difference']*100:+.1f}pp vs price-only")
            st.info(exp["interpretation"])
            st.caption(f"Test window: {exp['overlap_window']} "
                       f"({exp['price_only']['test_rows']} trading days)")
        else:
            st.warning("No experiment results found for this stock yet — "
                       "run `python src/macro_experiment.py` to generate them.")
    except Exception as e:
        st.warning(f"Couldn't load macro experiment results: {e}")

    st.divider()
    st.subheader("How stocks move together (correlation)")
    st.caption(
        "Values close to **+1** mean two stocks tend to move in the same "
        "direction together (less diversification benefit if you hold both). "
        "Values close to **0** mean their movements are largely independent. "
        "Computed directly from these stocks' own price history — no external "
        "or proxy data needed for this specific metric."
    )
    try:
        corr = get_pairwise_correlation_matrix(tickers_now)
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
            colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
            text=corr.values.round(2), texttemplate="%{text}",
        ))
        fig_corr.update_layout(title="Stock-to-stock correlation matrix", height=420)
        st.plotly_chart(fig_corr, use_container_width=True)
    except Exception as e:
        st.warning(f"Couldn't compute correlation matrix: {e}")

    st.divider()
    st.subheader("Beta vs. the market")
    st.success(
        "**Beta shown throughout this app is computed against the real "
        "KSE-100 index** — actual historical daily KSE-100 prices, "
        "correlated against each stock's own returns. This is genuine "
        "market-relative risk, not an approximation."
    )

    st.divider()
    st.subheader("Model evaluation methodology, in one place")
    st.markdown(
        "- **Split**: chronological (train on earliest 80% of days, test on "
        "most recent unseen 20%) — never a random shuffle, which would leak "
        "future information into training\n"
        "- **Baseline comparison**: every accuracy number is shown next to "
        "what a naive majority-class guess would achieve, so the number is "
        "meaningful rather than cherry-picked\n"
        "- **Typical result**: ~48-52% accuracy, close to the ~50% baseline — "
        "consistent with the Efficient Market Hypothesis, and reported "
        "honestly rather than hidden or inflated\n"
        "- **Risk categorization**: fixed absolute annualized-volatility "
        "bands (Low <15%, Moderate 15-30%, High >30%), not a ranking "
        "relative to whichever other stocks happen to be in the app\n"
        "- **Long-term outlook**: trend extrapolation (historical CAGR + "
        "volatility-based range), explicitly NOT a new ML forecast — see "
        "the Stock Explorer page for the full reasoning"
    )
    st.caption(
        "This page exists specifically to put every methodological answer in "
        "one place ahead of a viva — sources, limitations, and evaluation "
        "approach, without having to hunt across pages."
    )
