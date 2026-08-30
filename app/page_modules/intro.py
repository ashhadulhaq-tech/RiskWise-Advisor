"""
app/page_modules/intro.py - Landing/welcome page
====================================================
Shown before login — the "front door" of the app, giving an overview of
what it does and what to expect. This is what an examiner sees first.
"""
import streamlit as st


def render_intro():
    st.title("📈 AI Stock Advisor")
    st.subheader("Risk-based stock recommendations, powered by AI and real PSX data")

    st.markdown(
        "Welcome. This is a university project that combines two things:\n\n"
        "1. **An AI model** that predicts short-term price direction for "
        "Pakistan Stock Exchange (PSX) stocks, trained on real historical "
        "data spanning 2016–2026\n"
        "2. **A personal risk profiler** — a 15-question questionnaire that "
        "figures out how much investment risk you're actually comfortable "
        "with, then matches stock recommendations to that profile\n\n"
        "The same 6 stocks get recommended differently to a cautious "
        "investor than to a risk-tolerant one — that matching is the core "
        "idea behind this app."
    )

    st.divider()
    st.subheader("What you'll do")
    st.markdown(
        "- Answer a 15-question risk questionnaire\n"
        "- See stocks matched to your risk profile — with AI predictions, "
        "real KSE-100 beta, expected returns, and long-term outlook\n"
        "- Explore any stock's price history and see exactly what the AI "
        "prediction is actually based on\n"
        "- Review the full methodology, data sources, and honest "
        "limitations behind every number shown — all in one place"
    )

    st.divider()
    st.warning(
        "**Educational tool only — not real financial advice.** Built for "
        "a university project; every prediction and metric is disclosed "
        "with its real accuracy and limitations, not oversold."
    )

    st.divider()
    st.caption("Ready? Head to the **Login** page in the sidebar to continue.")
