"""
app/page_modules/login.py - Real multi-user sign-up + login
=================================================================
Accounts are stored in users/users.json, with salted+hashed passwords
(see src/auth_store.py for the hashing details and the honest note on
Streamlit Cloud's ephemeral filesystem — accounts persist for the life
of the running app instance, not across reboots/redeploys, unless
users.json is committed back to the repo).

The original shared demo account still works as a fallback:
    Username: student   Password: riskwise2026
"""
import streamlit as st

from auth_store import (
    create_user,
    verify_user,
    DEFAULT_USERNAME,
    DEFAULT_PASSWORD,
)


def render_login():
    st.title("🔒 Login")

    st.caption(
        "Real accounts, stored locally with salted/hashed passwords — not "
        "production-grade infrastructure (no real database, no email "
        "verification), which is intentional scope for a university "
        "project. See README for details."
    )

    login_tab, signup_tab = st.tabs(["Log in", "Sign up"])

    with login_tab:
        st.info(
            f"Don't want to sign up? The demo account still works — "
            f"**Username:** `{DEFAULT_USERNAME}` · **Password:** `{DEFAULT_PASSWORD}`"
        )
        with st.form("login_form"):
            username = st.text_input("Username", key="login_username")
            password = st.text_input("Password", type="password", key="login_password")
            submitted = st.form_submit_button("Log in")

        if submitted:
            if verify_user(username, password):
                st.session_state.authenticated = True
                st.session_state.username = username
                st.success("Logged in — redirecting...")
                st.rerun()
            else:
                st.error("Incorrect username or password.")

    with signup_tab:
        with st.form("signup_form"):
            new_username = st.text_input("Choose a username", key="signup_username")
            new_password = st.text_input("Choose a password", type="password", key="signup_password")
            confirm_password = st.text_input("Confirm password", type="password", key="signup_confirm")
            signup_submitted = st.form_submit_button("Sign up")

        if signup_submitted:
            if new_password != confirm_password:
                st.error("Passwords don't match.")
            else:
                success, message = create_user(new_username, new_password)
                if success:
                    st.session_state.authenticated = True
                    st.session_state.username = new_username
                    st.success("Account created — logging you in...")
                    st.rerun()
                else:
                    st.error(message)
