"""
app/page_modules/recommendations.py - Page 2: Recommendations
==================================================================
Moved from the old single-page app's Tab 2 — logic unchanged. Reads
st.session_state.profile / .recommendations set by the Questionnaire page
(session state persists across pages in a Streamlit multi-page app).
"""
import streamlit as st

from config import TICKER_NAMES, TICKER_SECTORS


def render_recommendations():
    st.title("2. Recommendations")

    if not st.session_state.profile:
        st.warning("Complete the questionnaire on the **Risk Questionnaire** page first.")
        return
    if st.session_state.recommendations is None or st.session_state.recommendations.empty:
        st.error("No recommendations are available right now — the data pipeline "
                 "may not have run yet. See the README for setup steps.")
        return

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
        "which isn't available from price history alone — see the Analysis & Methodology "
        "page. The AI prediction is a short-term directional signal, not a guarantee — "
        "see the Stock Explorer page for the model's real backtested accuracy."
    )
