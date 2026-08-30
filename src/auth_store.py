"""
auth_store.py - Simple multi-user account storage.
=======================================================
Real per-user accounts: usernames + salted, hashed passwords, stored in a
JSON file (users/users.json). Passwords are NEVER stored in plaintext —
each one is hashed with a random per-user salt using hashlib's PBKDF2
implementation (stdlib only, no new dependency, so this can't reintroduce
a requirements.txt install failure).

HONEST LIMITATION (read this before relying on it for anything real):
Streamlit Community Cloud's filesystem is ephemeral — it resets whenever
the app reboots or redeploys (a new push to main, a manual reboot, going
to sleep from inactivity and waking back up, etc.). That means accounts
people sign up with will work for the lifetime of that running instance,
but will NOT survive a reboot/redeploy unless users.json itself is
committed to the repo afterward. This is fine for a university demo
(sign up, use the app, show it off in one sitting) but is NOT how you'd
want to run this for real, ongoing public use — a real deployment would
need a real database (e.g. Postgres) instead of a JSON file on local
disk. Documented here and in the README.

The original single shared demo account (student / riskwise2026) still
exists as a permanent fallback account, seeded automatically the first
time this file's storage is used, so a login always works even before
anyone has signed up.
"""
import os
import json
import hashlib
import binascii
from datetime import datetime, timezone

from config import USERS_FILE, get_logger

logger = get_logger(__name__)

DEFAULT_USERNAME = "student"
DEFAULT_PASSWORD = "riskwise2026"

PBKDF2_ITERATIONS = 200_000
MIN_PASSWORD_LENGTH = 6


def _hash_password(password: str, salt: bytes) -> str:
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS)
    return binascii.hexlify(dk).decode("utf-8")


def _new_salt() -> bytes:
    return os.urandom(16)


def _load_users() -> dict:
    if not os.path.exists(USERS_FILE):
        return {}
    try:
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        logger.warning("users.json unreadable/corrupt — starting from an empty user store.")
        return {}


def _save_users(users: dict) -> None:
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2)


def _ensure_default_account(users: dict) -> dict:
    """Seeds the original shared demo account if it's missing, so login
    always works even on a totally fresh users.json."""
    if DEFAULT_USERNAME not in users:
        salt = _new_salt()
        users[DEFAULT_USERNAME] = {
            "salt": binascii.hexlify(salt).decode("utf-8"),
            "password_hash": _hash_password(DEFAULT_PASSWORD, salt),
            "created": datetime.now(timezone.utc).isoformat(),
            "is_default_demo_account": True,
        }
        _save_users(users)
    return users


def username_exists(username: str) -> bool:
    users = _ensure_default_account(_load_users())
    return username.strip().lower() in {u.lower() for u in users.keys()}


def create_user(username: str, password: str):
    """Returns (success: bool, message: str)."""
    username = username.strip()
    if not username:
        return False, "Username can't be empty."
    if len(password) < MIN_PASSWORD_LENGTH:
        return False, f"Password must be at least {MIN_PASSWORD_LENGTH} characters."

    users = _ensure_default_account(_load_users())
    if username.lower() in {u.lower() for u in users.keys()}:
        return False, "That username is already taken."

    salt = _new_salt()
    users[username] = {
        "salt": binascii.hexlify(salt).decode("utf-8"),
        "password_hash": _hash_password(password, salt),
        "created": datetime.now(timezone.utc).isoformat(),
        "is_default_demo_account": False,
    }
    _save_users(users)
    logger.info(f"New account created: {username}")
    return True, "Account created."


def verify_user(username: str, password: str) -> bool:
    users = _ensure_default_account(_load_users())
    # case-insensitive username match, but stored/display form is preserved
    match = next((u for u in users.keys() if u.lower() == username.strip().lower()), None)
    if match is None:
        return False
    record = users[match]
    salt = binascii.unhexlify(record["salt"])
    return _hash_password(password, salt) == record["password_hash"]
