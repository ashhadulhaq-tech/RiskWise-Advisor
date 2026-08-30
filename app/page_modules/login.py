"""
app/page_modules/login.py - Simple login gate
==================================================
NOT real per-user authentication — a single shared username/password,
configurable via .streamlit/secrets.toml. This is deliberate: it gives the
app a genuine login SCREEN and gated flow (what was actually requested
— examiners expecting an "intro then login then app" structure), without
the added scope of registering a real OAuth provider for a university
project. Documented plainly here and in the README so this is never
mistaken for production-grade security. See README "Upgrading to real
login" for the path to Streamlit's built-in st.login() (Google/Microsoft
OAuth) if this project ever needs genuine per-user accounts.
"""
import streamlit as st

# Fallback credentials used ONLY if .streamlit/secrets.toml isn't
# configured — so the app still runs out of the box, with a visible
# notice rather than a silent, unlabeled default a viewer might miss.
DEFAULT_USERNAME = "student"
DEFAULT_PASSWORD = "riskwise2026"


def _get_credentials():
    """Returns (username, password, using_custom_secrets)."""
    try:
        return st.secrets["auth"]["username"], st.secrets["auth"]["password"], True
    except Exception:
        return DEFAULT_USERNAME, DEFAULT_PASSWORD, False


def render_login():
    st.title("🔒 Login")

    valid_username, valid_password, using_real_secrets = _get_credentials()

    if not using_real_secrets:
        st.info(
            f"No custom credentials configured — using the built-in demo "
            f"login. **Username:** `{DEFAULT_USERNAME}` · **Password:** "
            f"`{DEFAULT_PASSWORD}`. Set your own in `.streamlit/secrets.toml` "
            f"(see README) if you'd rather not show this default during "
            f"a presentation."
        )

    st.caption(
        "This is a simple login gate for demo/presentation purposes, not "
        "production-grade authentication — no real user accounts, no "
        "password hashing. That scope is intentional for a university "
        "project; see README for how to upgrade to real sign-in later."
    )

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log in")

    if submitted:
        if username == valid_username and password == valid_password:
            st.session_state.authenticated = True
            st.success("Logged in — redirecting...")
            st.rerun()
        else:
            st.error("Incorrect username or password.")
