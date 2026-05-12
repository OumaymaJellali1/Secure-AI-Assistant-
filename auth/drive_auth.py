# auth/drive_auth.py
# Per-user OAuth 2.0 for Google Drive — mirrors gmail_auth.py exactly.

import os
from pathlib import Path
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES           = ["https://www.googleapis.com/auth/drive.readonly"]
CREDENTIALS_FILE = "credentials.json"
TOKENS_DIR       = Path("tokens")


def get_drive_service(id: str, user_email: str):
    """
    Returns an authenticated Drive API service for a specific user.
    Token stored at tokens/<id>_drive.json (separate from Gmail token).
    """
    TOKENS_DIR.mkdir(exist_ok=True)
    token_file = TOKENS_DIR / f"{id}_drive.json"
    creds = None

    if token_file.exists():
        creds = Credentials.from_authorized_user_file(str(token_file), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            print(f"[DRIVE AUTH] Refreshing token for {id}...")
            creds.refresh(Request())
        else:
            print(f"[DRIVE AUTH] First-time OAuth for {id} ({user_email})")
            flow = InstalledAppFlow.from_client_secrets_file(
                CREDENTIALS_FILE, SCOPES
            )
            creds = flow.run_local_server(port=0, login_hint=user_email)

        with open(token_file, "w") as f:
            f.write(creds.to_json())
        print(f"[DRIVE AUTH] Token saved → {token_file}")

    return build("drive", "v3", credentials=creds)