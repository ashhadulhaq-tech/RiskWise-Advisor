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

from config import (DATA_DIR, MODEL_DIR, TICKERS, TICKER_NAMES, TICKER_SECTORS,
                     get_available_tickers, features_parquet_path, get_logger)
from risk_engine import (
    QUESTIONNAIRE, score_questionnaire, recommend_stocks,
    ALLOCATION_GUIDANCE, encode_profile_code, decode_profile_code,
    get_pairwise_correlation_matrix, project_future_price,
)
from model import predict_next_day, load_model
import external_data as ext
import macro_experiment as mx

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
    return pd.read_parquet(features_parquet_path(ticker))


def ensure_ticker_ready(ticker: str) -> bool:
    """
    Make sure a ticker's data/model exist before the UI tries to use them.
    Returns False (and shows a friendly message) instead of letting a raw
    FileNotFoundError reach the user — the crash found in the evaluation.
    """
    feat_path = features_parquet_path(ticker)
    model_path = os.path.join(MODEL_DIR, f"{ticker}_model.pkl")
    if os.path.exists(feat_path) and os.path.exists(model_path):
        return True

    st.warning(f"Data or model for **{ticker}** isn't ready yet — this stock "
               f"will be skipped this session. Run the setup pipeline "
               f"(`scripts/update_data.py`) to include it.")
    logger.warning(f"[{ticker}] Missing data/model files — skipped in UI.")
    return False

# ---------------------------------------------------------------------------
# Session state setup (so the app remembers answers/profile between clicks)
# ---------------------------------------------------------------------------
if "profile" not in st.session_state:
    st.session_state.profile = None
if "recommendations" not in st.session_state:
    st.session_state.recommendations = None
if "profile_code" not in st.session_state:
    st.session_state.profile_code = None

st.title("📈 AI Stock Advisor")
st.caption(
    "A university project demo: risk-based stock recommendations combined "
    "with an AI price-direction model, using real Pakistan Stock Exchange "
    "(PSX) historical data. **Educational tool only — not real financial advice.**"
)

with st.sidebar:
    st.subheader("Data status")
    tickers_now = get_available_tickers()
    st.caption(f"{len(tickers_now)} stocks currently loaded.")

    newest_date = None
    for t in tickers_now[:1]:  # cheap check on one ticker rather than all
        try:
            df = pd.read_parquet(features_parquet_path(t))
            newest_date = pd.to_datetime(df["date"]).max().date()
        except Exception:
            pass
    if newest_date:
        st.caption(f"Most recent data on file: **{newest_date}**")

    st.caption(
        "Data auto-refreshes daily via a scheduled job (see README). "
        "To pull the latest PSX data manually right now, run "
        "`python scripts/update_data.py` from the project folder, then "
        "reload this page — this app always reads whatever's currently "
        "on disk, so it picks up new data automatically without any "
        "code changes."
    )

tab1, tab2, tab3, tab4 = st.tabs([
    "1. Risk Questionnaire", "2. Recommendations", "3. Stock Explorer", "4. Analysis & Methodology"
])

# ---------------------------------------------------------------------------
# TAB 1 — Questionnaire
# ---------------------------------------------------------------------------
with tab1:
    st.subheader("Already have a saved profile code?")
    st.caption(
        "Paste it below to skip the questionnaire. No account needed — this "
        "code is just an encoded copy of your answers, generated the first "
        "time you complete the questionnaire (see below once you submit)."
    )
    code_col1, code_col2 = st.columns([3, 1])
    with code_col1:
        pasted_code = st.text_input("Profile code", key="profile_code_input",
                                     placeholder="RW1-...", label_visibility="collapsed")
    with code_col2:
        restore_clicked = st.button("Restore profile", use_container_width=True)

    if restore_clicked:
        if not pasted_code:
            st.warning("Paste a profile code first.")
        else:
            try:
                restored_answers = decode_profile_code(pasted_code)
                profile = score_questionnaire(restored_answers)
                st.session_state.profile = profile
                with st.spinner("Matching stocks to your restored profile..."):
                    available = [t for t in get_available_tickers() if ensure_ticker_ready(t)]
                    if available:
                        st.session_state.recommendations = cached_recommend(
                            profile["risk_category"], tuple(available))
                st.success(f"Profile restored: **{profile['risk_category']}** "
                           f"— see the Recommendations tab.")
            except ValueError as e:
                st.error(f"Couldn't restore that profile: {e}")

    st.divider()
    st.subheader("Tell us about your investing style")
    st.write("15 short questions, answer honestly — there are no right or wrong "
             "answers, this just tailors which stocks we suggest.")

    answers = {}
    with st.form("risk_form"):
        for i, q in enumerate(QUESTIONNAIRE, start=1):
            labels = [opt[0] for opt in q["options"]]
            choice = st.radio(f"**{i}/{len(QUESTIONNAIRE)}.** {q['question']}",
                               labels, key=q["id"], index=None)
            if choice is not None:
                answers[q["id"]] = labels.index(choice)

        submitted = st.form_submit_button("Get My Risk Profile")

    if submitted:
        if len(answers) < len(QUESTIONNAIRE):
            st.error(f"Please answer all {len(QUESTIONNAIRE)} questions before submitting "
                     f"({len(answers)}/{len(QUESTIONNAIRE)} answered so far).")
        else:
            try:
                profile = score_questionnaire(answers)
                st.session_state.profile = profile
                st.session_state.profile_code = encode_profile_code(answers)
                with st.spinner("Matching stocks to your profile..."):
                    available = [t for t in get_available_tickers() if ensure_ticker_ready(t)]
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
        if st.session_state.get("profile_code"):
            st.text_input("💾 Save this code to restore your profile next time "
                           "(no need to redo the questionnaire):",
                           value=st.session_state.profile_code, disabled=True)
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
                st.caption(f"🏷️ {TICKER_SECTORS.get(row['ticker'], 'N/A')}")
                if row["predicted_direction"] == "N/A":
                    st.caption("⚪ Prediction unavailable for this stock right now")
                else:
                    st.write(f"{direction_emoji} AI predicts **{row['predicted_direction']}** "
                             f"({row['prediction_confidence']*100:.0f}% confidence)")
                st.caption(f"Risk band: **{row['risk_band']}** "
                           f"({row['annualized_volatility']*100:.0f}% annualized volatility)")
                if row["beta"] is not None:
                    st.caption(f"Beta: {row['beta']}  ·  Correlation w/ market: "
                               f"{row.get('correlation_with_market', 'N/A')}")
                if row.get("realized_annual_return") is not None:
                    st.caption(f"Historical return: {row['realized_annual_return']*100:.1f}%/yr")
                if row.get("expected_annual_return_capm") is not None:
                    st.caption(f"CAPM expected return: {row['expected_annual_return_capm']*100:.1f}%/yr")
                if row.get("alpha") is not None:
                    alpha_pct = row["alpha"] * 100
                    alpha_emoji = "📈" if alpha_pct > 0 else "📉"
                    st.caption(f"{alpha_emoji} Alpha: {alpha_pct:+.1f}% "
                               f"({'outperforming' if alpha_pct > 0 else 'underperforming'} "
                               f"risk-adjusted expectation)")

        st.divider()
        st.caption(
            "⚠️ **How to read this**: risk band is a fixed classification (low / moderate / "
            "high) based on annualized volatility, not a ranking relative to other stocks "
            "shown here — so it stays meaningful even as the stock list changes. Beta and "
            "correlation are computed against the **real KSE-100 index** — how much a stock "
            "tends to move relative to the actual market, and how closely it tends to move "
            "WITH it. **Alpha is a risk-adjusted "
            "performance signal (realized return vs. what CAPM says the stock's risk level "
            "should have earned) — it is NOT a fundamental valuation judgment.** True "
            "over/undervaluation would require earnings and balance-sheet data (P/E, P/B), "
            "which isn't available from price history alone — see README. The AI prediction "
            "is a short-term directional signal, not a guarantee — see the Stock Explorer tab "
            "for the model's real backtested accuracy."
        )

# ---------------------------------------------------------------------------
# TAB 3 — Stock Explorer (price chart + model transparency)
# ---------------------------------------------------------------------------
with tab3:
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
            fig_imp = go.Figure(go.Bar(
                x=list(top_features.values()), y=list(top_features.keys()),
                orientation="h", marker_color="#2563eb"
            ))
            fig_imp.update_layout(
                title="Which signals the model relies on most",
                xaxis_title="Relative importance", height=320,
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig_imp, use_container_width=True)
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

# ---------------------------------------------------------------------------
# TAB 4 — Analysis & Methodology (consolidated for viva/presentation use)
# ---------------------------------------------------------------------------
with tab4:
    st.subheader("Data sources — what this app does and doesn't use")
    st.markdown(
        "**Main AI prediction model** (next-day direction, shown throughout the app):\n"
        "- ✅ **Historical daily price & volume (OHLCV)** for each stock — the only input\n"
        "- ❌ No news, sentiment, fundamentals, or macro data\n\n"
        "**Supplementary analysis** (this tab only, kept separate from the main model):\n"
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
            import json
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
        "market-relative risk, not an approximation. (An earlier version "
        "of this app used an internal 6-stock proxy before real KSE-100 "
        "index data was available — that limitation is now resolved.)"
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
        "the Stock Explorer tab for the full reasoning"
    )
    st.caption(
        "This tab exists specifically to put every methodological answer in "
        "one place ahead of a viva — sources, limitations, and evaluation "
        "approach, without having to hunt across tabs."
    )
