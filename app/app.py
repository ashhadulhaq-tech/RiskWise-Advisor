"""
app/app.py - Entry point (multi-page, with a login gate)
=============================================================
Structure requested by the instructor: Intro page -> Login -> then the
actual app pages (Questionnaire, Recommendations, Stock Explorer,
Analysis & Methodology), instead of tabs on one page.

Uses Streamlit's native multi-page support (st.Page + st.navigation,
available from Streamlit 1.36+ — see requirements.txt). Page CONTENT
lives in app/page_modules/ — this file only wires pages together and
gates which ones are visible based on login state.

HOW TO RUN THIS (on your own machine, after pip install):
    cd stockapp
    streamlit run app/app.py
It will open in your browser automatically at http://localhost:8501
"""

import sys
import os

# allow importing our src/ modules and app/page_modules/
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.append(os.path.dirname(__file__))

import streamlit as st
import pandas as pd

from config import get_available_tickers, features_parquet_path, get_logger

from page_modules.intro import render_intro
from page_modules.login import render_login
from page_modules.questionnaire import render_questionnaire
from page_modules.recommendations import render_recommendations
from page_modules.stock_explorer import render_stock_explorer
from page_modules.methodology import render_methodology

logger = get_logger(__name__)

st.set_page_config(page_title="AI Stock Advisor", page_icon="📈", layout="wide")

# ---------------------------------------------------------------------------
# Session state (persists across pages within the same browser session)
# ---------------------------------------------------------------------------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "profile" not in st.session_state:
    st.session_state.profile = None
if "recommendations" not in st.session_state:
    st.session_state.recommendations = None
if "profile_code" not in st.session_state:
    st.session_state.profile_code = None

# ---------------------------------------------------------------------------
# Page list — gated by login state. Before login, only Welcome + Login are
# reachable (including via the sidebar nav, not just by hiding a button) —
# the actual app pages simply aren't in the list at all until authenticated.
# ---------------------------------------------------------------------------
if st.session_state.authenticated:
    pages = [
        st.Page(render_questionnaire, title="1. Risk Questionnaire", icon="📝", default=True),
        st.Page(render_recommendations, title="2. Recommendations", icon="⭐"),
        st.Page(render_stock_explorer, title="3. Stock Explorer", icon="🔍"),
        st.Page(render_methodology, title="4. Analysis & Methodology", icon="📊"),
    ]

    with st.sidebar:
        if st.button("🚪 Log out", use_container_width=True):
            st.session_state.authenticated = False
            st.rerun()

        st.divider()
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
else:
    pages = [
        st.Page(render_intro, title="Welcome", icon="🏠", default=True),
        st.Page(render_login, title="Login", icon="🔒"),
    ]

nav = st.navigation(pages)
nav.run()
