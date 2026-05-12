# auth/gmail_auth.py
# Per-user OAuth 2.0 flow for Gmail.
# Each user gets their own tokens/<id>.json.
# First call opens a browser once; every subsequent call silently refreshes.

import os
from pathlib import Path
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES           = ["https://www.googleapis.com/auth/gmail.readonly"]
CREDENTIALS_FILE = "credentials.json"   # downloaded from Google Cloud Console
TOKENS_DIR       = Path("tokens")       # one JSON file per user — gitignored


def get_gmail_service(id: str, user_email: str):
    """
    Returns an authenticated Gmail API service for a specific user.

    - First call for a user  → opens browser for Google consent, saves token
    - All subsequent calls   → silent token refresh, no browser needed
    - Token stored at        → tokens/<id>.json
    """
    TOKENS_DIR.mkdir(exist_ok=True)
    token_file = TOKENS_DIR / f"{id}.json"
    creds = None

    # Load cached token if it exists
    if token_file.exists():
        creds = Credentials.from_authorized_user_file(str(token_file), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            # Silent background refresh — no browser, no user interaction
            print(f"[AUTH] Refreshing token for {id}...")
            creds.refresh(Request())
        else:
            # First run only — opens a browser window for this user
            print(f"[AUTH] First-time OAuth for {id} ({user_email})")
            print(f"[AUTH] A browser will open — please log in as {user_email}")
            flow = InstalledAppFlow.from_client_secrets_file(
                CREDENTIALS_FILE,
                SCOPES
            )
            creds = flow.run_local_server(
                port=0,
                login_hint=user_email   # pre-fills Google login with this email
            )

        # Persist token for next run
        with open(token_file, "w") as f:
            f.write(creds.to_json())
        print(f"[AUTH] Token saved → {token_file}")

    return build("gmail", "v1", credentials=creds)


def revoke_token(id: str) -> None:
    """
    Delete a user's cached token file.
    Next crawl will require a new browser consent.
    """
    token_file = TOKENS_DIR / f"{id}.json"
    if token_file.exists():
        token_file.unlink()
        print(f"[AUTH] Token revoked for {id}")
    else:
        print(f"[AUTH] No token found for {id}")