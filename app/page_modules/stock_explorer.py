"""
app/page_modules/stock_explorer.py - Page 3: Stock Explorer
================================================================
Moved from the old single-page app's Tab 3 — logic unchanged.
"""
import os
import json
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from config import MODEL_DIR, TICKER_NAMES, TICKER_SECTORS, get_available_tickers, features_parquet_path
from risk_engine import project_future_price
from page_modules.shared import cached_predict, cached_read_features


def render_stock_explorer():
    st.title("3. Stock Explorer")
    st.subheader("Explore any stock's price history and AI prediction")

    ticker = st.selectbox(
        "Choose a stock", get_available_tickers(),
        format_func=lambda t: f"{t} — {TICKER_NAMES.get(t, '')}"
    )
    st.caption(f"**Sector:** {TICKER_SECTORS.get(ticker, 'N/A')}")

    feat_path = features_parquet_path(ticker)
    if not os.path.exists(feat_path):
        # In practice unreachable since get_available_tickers() only lists
        # tickers that already have a features file — kept as a defensive
        # guard in case that changes.
        st.error(f"Data not found for {ticker}. Run `scripts/update_data.py` "
                 f"to fetch and process it, then reload.")
        st.stop()

    try:
        df = cached_read_features(ticker)
    except Exception as e:
        st.error(f"Couldn't load data for {ticker}: {e}")
        st.stop()

    df["date"] = pd.to_datetime(df["date"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=df["close"], name="Close price", line=dict(color="#2563eb")))
    fig.add_trace(go.Scatter(x=df["date"], y=df["ma_20"], name="20-day moving average",
                              line=dict(color="#f59e0b", dash="dot")))
    fig.update_layout(title=f"{ticker} — Price History", xaxis_title="Date",
                       yaxis_title="Price (Rs)", height=420)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    m = None
    with col1:
        try:
            direction, prob_up = cached_predict(ticker)
            emoji = "🟢" if direction == "UP" else "🔴"
            st.metric("Next-day AI prediction", f"{emoji} {direction}", f"{prob_up*100:.1f}% confidence")
        except FileNotFoundError:
            st.warning(f"No trained model for {ticker} yet — run `model.py` to train one.")

    with col2:
        metrics_path = os.path.join(MODEL_DIR, f"{ticker}_metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                m = json.load(f)
            st.metric("Model accuracy (backtested)", f"{m['accuracy']*100:.1f}%",
                       f"vs {m['baseline_accuracy']*100:.1f}% baseline")
        else:
            st.warning("No evaluation metrics available yet for this stock.")

    if m is not None:
        with st.expander("📊 See full model evaluation (for your report)"):
            st.json(m)

        st.divider()
        st.subheader("What is this prediction based on?")
        st.caption(
            "**Data sources used**: only this stock's own historical price and "
            "trading volume (OHLCV) — no news, no sentiment, no company "
            "fundamentals (earnings/P/E), no macroeconomic data. Everything "
            "below is derived mathematically from price and volume history alone."
        )
        importances = m.get("feature_importance", {})
        if importances:
            top_features = dict(list(importances.items())[:8])
            max_val = max(top_features.values()) if top_features else 1

            st.markdown("**Which signals the model relies on most**")
            for feat, val in top_features.items():
                col1, col2, col3 = st.columns([2, 5, 1.3])
                with col1:
                    st.markdown(f"`{feat}`")
                with col2:
                    st.progress(val / max_val if max_val > 0 else 0)
                with col3:
                    st.markdown(f"**{val:.4f}**")
            st.caption(
                "Higher bars = the model leaned on that signal more when making "
                "its decisions. Common top signals: **daily_return/rsi_14** "
                "(recent momentum), **volatility_10/20** (recent choppiness), "
                "**macd/macd_signal** (trend-change signal), **volume_change** "
                "(shifts in trading interest)."
            )

        st.divider()
        st.subheader("Long-term outlook (trend-based projection)")
        st.caption(
            "⚠️ This is **not** an AI forecast — the model above only predicts "
            "next-day direction, evaluated honestly at ~50% accuracy (see above). "
            "This projects the stock's own historical average growth rate and "
            "volatility forward in time, showing what its price would plausibly "
            "be **if that historical trend continues** — a standard, transparent "
            "extrapolation, not a guarantee. The range widens the further out you "
            "look, which honestly reflects growing uncertainty, not a bug."
        )
        try:
            outlook = project_future_price(ticker)
            st.write(f"**Current price:** Rs {outlook['current_price']} · "
                     f"**Historical trend:** {outlook['trend_direction']} "
                     f"({outlook['historical_annual_return_pct']:+.1f}%/yr, "
                     f"{outlook['historical_annual_volatility_pct']:.1f}% annual volatility)")
            outlook_df = pd.DataFrame(outlook["projections"])
            outlook_df.columns = ["Months ahead", "Projected price (Rs)", "Low estimate (Rs)", "High estimate (Rs)"]
            st.dataframe(outlook_df, hide_index=True, use_container_width=True)
        except Exception as e:
            st.warning(f"Couldn't compute long-term outlook: {e}")

    st.caption(
        "This model predicts short-term price **direction**, not exact prices. "
        "Accuracy near 50% is expected and consistent with the Efficient Market "
        "Hypothesis — real markets are very difficult to predict short-term. "
        "This is disclosed intentionally as part of the project's honest evaluation."
    )
