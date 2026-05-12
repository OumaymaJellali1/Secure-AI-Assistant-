# connectors/gmail_crawler.py
# Crawls an entire Gmail mailbox for one user.
# Tags every chunk with id + owner_email so Qdrant can filter by owner.
# Hands off to your existing eml_chunker pipeline — nothing downstream changes.

import uuid
import base64
import re
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from auth.gmail_auth import get_gmail_service
from File_processing.cleaner import clean_email
from chunking.eml_chunker import _chunk_body, _build_metadata


def crawl_user(
    id: str,
    user_email: str,
    batch_size: int = 500,
    max_emails: int = None,
) -> list[dict]:
    """
    Fetch ALL emails for one user and return a flat list of chunks.
    Each chunk carries id + owner_email in its metadata,
    which is stored in Qdrant for access-control filtering at query time.

    Args:
        id     : unique identifier matching the users table
        user_email  : the user's Gmail address (used as OAuth login_hint)
        batch_size  : messages per API page (max 500)

    Returns:
        List of chunk dicts ready for normalize_chunks() → embed_chunks() → store()
    """
    service = get_gmail_service(id, user_email)
    all_chunks: list[dict] = []
    page_token = None
    total_fetched = 0

    print(f"\n[CRAWLER] Starting full crawl for {id} ({user_email})")

    while True:
        # Fetch a page of message IDs (no query filter = all mail)
        kwargs: dict = {"userId": "me", "maxResults": batch_size}
        if page_token:
            kwargs["pageToken"] = page_token

        response   = service.users().messages().list(**kwargs).execute()
        messages   = response.get("messages", [])
        page_token = response.get("nextPageToken")

        if not messages:
            break

        for msg_ref in messages:
            if max_emails and total_fetched >= max_emails:
              break
            msg = service.users().messages().get(
                userId="me",
                id=msg_ref["id"],
                format="full"
            ).execute()

            chunks = _message_to_chunks(msg, id, user_email)
            all_chunks.extend(chunks)

        total_fetched += len(messages)
        print(
            f"[CRAWLER] {id}: "
            f"{total_fetched} emails fetched → {len(all_chunks)} chunks so far"
        )

        if not page_token:
            break   # no more pages

    print(f"[CRAWLER] Done: {id} → {len(all_chunks)} total chunks")
    return all_chunks


# ── Internal helpers ──────────────────────────────────────────────────────────

def _message_to_chunks(
    msg: dict,
    id: str,
    user_email: str,
) -> list[dict]:
    """Convert one Gmail API message into chunks tagged with user identity."""
    headers = {
        h["name"]: h["value"]
        for h in msg["payload"].get("headers", [])
    }

    eml_metadata = {
        "source"      : f"{msg['id']}.gmail",
        "subject"     : headers.get("Subject", ""),
        "from"        : headers.get("From",    ""),
        "to"          : headers.get("To",      ""),
        "date"        : headers.get("Date",    ""),
        # ── SECURITY FIELDS — stored in Qdrant payload ──────────────
        "id"     : id,
        "owner_email" : user_email,
    }

    plain_text = _extract_body_text(msg["payload"])
    cleaned    = clean_email(plain_text)

    if not cleaned.strip():
        return []   # skip empty emails

    file_id = str(uuid.uuid4())
    meta    = _build_metadata_with_owner(eml_metadata, file_id)

    # Reuse your existing _chunk_body — unchanged
    return _chunk_body(cleaned, meta)


def _build_metadata_with_owner(eml_metadata: dict, file_id: str) -> dict:
    """
    Build a metadata dict in the shape eml_chunker expects,
    extended with id and owner_email for access control.
    """
    return {
        # Standard eml fields
        "source"      : eml_metadata.get("source",  ""),
        "file_id"     : file_id,
        "source_type" : "gmail",
        "eml_source"  : eml_metadata.get("source",  ""),
        "subject"     : eml_metadata.get("subject", ""),
        "from"        : eml_metadata.get("from",    ""),
        "to"          : eml_metadata.get("to",      ""),
        "date"        : eml_metadata.get("date",    ""),
        "attachments" : [],
        # Access-control fields
        "id"     : eml_metadata["id"],
        "owner_email" : eml_metadata["owner_email"],
    }


def _extract_body_text(payload: dict) -> str:
    """Recursively extract plain text from a Gmail MIME payload."""
    mime = payload.get("mimeType", "")

    if mime == "text/plain":
        data = payload.get("body", {}).get("data", "")
        if data:
            return base64.urlsafe_b64decode(data + "==").decode("utf-8", errors="ignore")

    if mime == "text/html":
        data = payload.get("body", {}).get("data", "")
        if data:
            html = base64.urlsafe_b64decode(data + "==").decode("utf-8", errors="ignore")
            try:
                from bs4 import BeautifulSoup
                return BeautifulSoup(html, "html.parser").get_text(
                    separator=" ", strip=True
                )
            except Exception:
                return re.sub(r"<[^>]+>", " ", html)

    # Recurse into multipart
    for part in payload.get("parts", []):
        result = _extract_body_text(part)
        if result:
            return result

    return ""