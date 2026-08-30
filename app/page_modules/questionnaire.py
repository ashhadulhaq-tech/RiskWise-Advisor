"""
app/page_modules/questionnaire.py - Page 1: Risk Questionnaire
===================================================================
Moved from the old single-page app's Tab 1 — logic is unchanged, only
restructured into a page function for the new multi-page layout.
"""
import streamlit as st

from config import get_available_tickers, get_logger
from risk_engine import (
    QUESTIONNAIRE, score_questionnaire, ALLOCATION_GUIDANCE,
    encode_profile_code, decode_profile_code,
)
from page_modules.shared import cached_recommend, ensure_ticker_ready

logger = get_logger(__name__)


def render_questionnaire():
    st.title("1. Risk Questionnaire")

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
                           f"— see the Recommendations page.")
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
        st.caption("👉 Head to the **Recommendations** page to see your matched stocks.")
