"""
Module D - Streamlit App (the actual user-facing app)
=========================================================
Job: wire Modules A-C into a clickable web app:
  1. User fills out the risk questionnaire
  2. App shows their risk profile
  3. App shows recommended stocks for that profile
  4. User can click a stock to see its price history + AI prediction chart

HOW TO RUN THIS (on your own machine, after pip install):
    cd stockapp
    streamlit run app/app.py
It will open in your browser automatically at http://localhost:8501
"""

import sys
import os

# allow importing our src/ modules
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from config import DATA_DIR, MODEL_DIR, TICKERS, TICKER_NAMES, get_logger
from risk_engine import (
    QUESTIONNAIRE, score_questionnaire, recommend_stocks,
    ALLOCATION_GUIDANCE,
)
from model import predict_next_day, load_model
from data_pipeline import get_stock_data
from features import build_features

logger = get_logger(__name__)

st.set_page_config(page_title="AI Stock Advisor", page_icon="📈", layout="wide")


# ---------------------------------------------------------------------------
# Cached wrappers - fixes the performance issue flagged in the evaluation
# (models/data were being re-read from disk on every interaction).
# st.cache_resource is for non-serializable objects (the trained model);
# st.cache_data is for dataframes. TTL of 1 day matches the daily-refresh
# cadence the data pipeline is designed for.
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def cached_load_model(ticker: str):
    return load_model(ticker)


@st.cache_data(ttl=86400, show_spinner=False)
def cached_predict(ticker: str):
    return predict_next_day(ticker)


@st.cache_data(ttl=86400, show_spinner=False)
def cached_recommend(risk_category: str, tickers: tuple):
    return recommend_stocks(risk_category, list(tickers))


@st.cache_data(ttl=86400, show_spinner=False)
def cached_read_features(ticker: str):
    return pd.read_csv(os.path.join(DATA_DIR, f"{ticker}_features.csv"))


def ensure_ticker_ready(ticker: str) -> bool:
    """
    Make sure a ticker's data/model exist before the UI tries to use them.
    Returns False (and shows a friendly message) instead of letting a raw
    FileNotFoundError reach the user — the crash found in the evaluation.
    """
    feat_path = os.path.join(DATA_DIR, f"{ticker}_features.csv")
    model_path = os.path.join(MODEL_DIR, f"{ticker}_model.pkl")
    if os.path.exists(feat_path) and os.path.exists(model_path):
        return True

    st.warning(f"Data or model for **{ticker}** isn't ready yet — this stock "
               f"will be skipped this session. Run the setup pipeline "
               f"(`data_pipeline.py` → `features.py` → `model.py`) to include it.")
    logger.warning(f"[{ticker}] Missing data/model files — skipped in UI.")
    return False

# ---------------------------------------------------------------------------
# Session state setup (so the app remembers answers/profile between clicks)
# ---------------------------------------------------------------------------
if "profile" not in st.session_state:
    st.session_state.profile = None
if "recommendations" not in st.session_state:
    st.session_state.recommendations = None

st.title("📈 AI Stock Advisor")
st.caption(
    "A university project demo: risk-based stock recommendations combined "
    "with an AI price-direction model, using real Pakistan Stock Exchange "
    "(PSX) historical data. **Educational tool only — not real financial advice.**"
)

tab1, tab2, tab3 = st.tabs(["1. Risk Questionnaire", "2. Recommendations", "3. Stock Explorer"])

# ---------------------------------------------------------------------------
# TAB 1 — Questionnaire
# ---------------------------------------------------------------------------
with tab1:
    st.subheader("Tell us about your investing style")
    st.write("Answer honestly — there are no right or wrong answers, this just "
             "tailors which stocks we suggest.")

    answers = {}
    with st.form("risk_form"):
        for q in QUESTIONNAIRE:
            labels = [opt[0] for opt in q["options"]]
            choice = st.radio(q["question"], labels, key=q["id"], index=None)
            if choice is not None:
                answers[q["id"]] = labels.index(choice)

        submitted = st.form_submit_button("Get My Risk Profile")

    if submitted:
        if len(answers) < len(QUESTIONNAIRE):
            st.error("Please answer all questions before submitting.")
        else:
            try:
                profile = score_questionnaire(answers)
                st.session_state.profile = profile
                with st.spinner("Matching stocks to your profile..."):
                    available = [t for t in TICKERS if ensure_ticker_ready(t)]
                    if not available:
                        st.error("No stock data is available right now. Please "
                                 "run the data setup pipeline and try again.")
                        st.session_state.recommendations = None
                    else:
                        recs = cached_recommend(profile["risk_category"], tuple(available))
                        st.session_state.recommendations = recs
            except ValueError as e:
                # score_questionnaire / recommend_stocks raise ValueError for
                # genuinely invalid input - show it plainly instead of crashing
                st.error(f"Couldn't process your answers: {e}")
            except Exception as e:
                logger.error(f"Unexpected error building recommendations: {e}")
                st.error("Something went wrong generating your recommendations. "
                         "Please try again — if this keeps happening, the data "
                         "pipeline may need to be re-run.")

    if st.session_state.profile:
        p = st.session_state.profile
        st.success(f"**Your risk profile: {p['risk_category']}** "
                   f"(score {p['total_score']}/{p['max_score']}, {p['percentage']}%)")
        st.info(f"Suggested allocation style: {ALLOCATION_GUIDANCE[p['risk_category']]}")
        st.caption("👉 Head to the **Recommendations** tab to see your matched stocks.")

# ---------------------------------------------------------------------------
# TAB 2 — Recommendations
# ---------------------------------------------------------------------------
with tab2:
    if not st.session_state.profile:
        st.warning("Complete the questionnaire in Tab 1 first.")
    elif st.session_state.recommendations is None or st.session_state.recommendations.empty:
        st.error("No recommendations are available right now — the data pipeline "
                 "may not have run yet. See the README for setup steps.")
    else:
        p = st.session_state.profile
        recs = st.session_state.recommendations
        st.subheader(f"Recommended stocks for a {p['risk_category']} investor")

        cols = st.columns(len(recs))
        for i, row in recs.reset_index(drop=True).iterrows():
            with cols[i]:
                direction_emoji = "🟢" if row["predicted_direction"] == "UP" else ("🔴" if row["predicted_direction"] == "DOWN" else "⚪")
                st.metric(label=row["ticker"], value=f"Rs {row['last_close']:.2f}")
                st.caption(TICKER_NAMES.get(row["ticker"], ""))
                if row["predicted_direction"] == "N/A":
                    st.caption("⚪ Prediction unavailable for this stock right now")
                else:
                    st.write(f"{direction_emoji} AI predicts **{row['predicted_direction']}** "
                             f"({row['prediction_confidence']*100:.0f}% confidence)")
                st.caption(f"Risk band: **{row['risk_band']}** "
                           f"({row['annualized_volatility']*100:.0f}% annualized volatility)")
                if row["beta"] is not None:
                    st.caption(f"Beta vs. market: {row['beta']}")

        st.divider()
        st.caption(
            "⚠️ **How to read this**: risk band is a fixed classification (low / moderate / "
            "high) based on annualized volatility, not a ranking relative to other stocks "
            "shown here — so it stays meaningful even as the stock list changes. Beta shows "
            "how much a stock tends to move relative to the overall market (currently "
            "approximated from the tracked stocks themselves — see README for details). "
            "The AI prediction is a short-term directional signal, not a guarantee — see "
            "the Stock Explorer tab for the model's real backtested accuracy."
        )

# ---------------------------------------------------------------------------
# TAB 3 — Stock Explorer (price chart + model transparency)
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("Explore any stock's price history and AI prediction")
    ticker = st.selectbox(
        "Choose a stock", TICKERS,
        format_func=lambda t: f"{t} — {TICKER_NAMES.get(t, '')}"
    )

    feat_path = os.path.join(DATA_DIR, f"{ticker}_features.csv")
    if not os.path.exists(feat_path):
        st.info("Data not found for this stock — building it now (first-time setup)...")
        try:
            get_stock_data(ticker)
            build_features(ticker)
        except Exception as e:
            st.error(f"Couldn't prepare data for {ticker}: {e}")
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
        import json
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

    st.caption(
        "This model predicts short-term price **direction**, not exact prices. "
        "Accuracy near 50% is expected and consistent with the Efficient Market "
        "Hypothesis — real markets are very difficult to predict short-term. "
        "This is disclosed intentionally as part of the project's honest evaluation."
    )
