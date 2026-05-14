# connectors/gmail_crawler.py
# Crawls an entire Gmail mailbox for one user.
# Tags every chunk with id + owner_email so Qdrant can filter by owner.
# Uses the full extract_eml pipeline for proper parsing.

import uuid
import base64
import os
import sys
import tempfile
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from auth.gmail_auth import get_gmail_service
from File_processing.eml_handler import extract_eml
from chunking.eml_chunker import _chunk_body
from embedding.qdrant_store import _get_client, COLLECTION_NAME


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
        id         : unique identifier matching the users table
        user_email : the user's Gmail address (used as OAuth login_hint)
        batch_size : messages per API page (max 500)
        max_emails : optional cap on number of emails to fetch

    Returns:
        List of chunk dicts ready for normalize_chunks() → embed_chunks() → store()
    """
    service = get_gmail_service(id, user_email)
    all_chunks: list[dict] = []
    page_token = None
    total_fetched = 0

    print(f"\n[CRAWLER] Starting full crawl for {id} ({user_email})")

    while True:
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
        # ── Skip if already in Qdrant ──────────────────────────
            if _is_email_already_crawled(msg_ref["id"]):
              print(f"[CRAWLER]   Skipping (already crawled): {msg_ref['id']}")
              continue
            # ── fetch raw RFC-822 bytes ──────────────────────────
            msg = service.users().messages().get(
                userId="me",
                id=msg_ref["id"],
                format="raw",
            ).execute()

            chunks = _message_to_chunks(msg, id, user_email)
            all_chunks.extend(chunks)

        total_fetched += len(messages)
        print(
            f"[CRAWLER] {id}: "
            f"{total_fetched} emails fetched → {len(all_chunks)} chunks so far"
        )

        if not page_token or (max_emails and total_fetched >= max_emails):
            break

    print(f"[CRAWLER] Done: {id} → {len(all_chunks)} total chunks")
    return all_chunks


# ── Internal helpers ──────────────────────────────────────────────────────────

def _message_to_chunks(
    msg: dict,
    id: str,
    user_email: str,
) -> list[dict]:
    """
    Convert one Gmail API message into chunks using the full extract_eml pipeline.
    Handles reply stripping, forward stripping, inline tables,
    attachments (PDF/DOCX/PPTX/images), and CSV → Postgres.
    """

    # 1. Decode raw RFC-822 bytes
    raw = _decode_raw(msg)
    if not raw:
        print(f"[CRAWLER] Empty raw payload for message {msg.get('id')} — skipping")
        return []

    # 2. Write to temp .eml file
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".eml", delete=False) as tmp:
            tmp.write(raw)
            tmp_path = tmp.name

        # 3. Run through the full extract_eml pipeline
        result = extract_eml(tmp_path)

    except Exception as e:
        print(f"[CRAWLER] extract_eml failed for {msg.get('id')}: {e}")
        return []

    finally:
        # Cleanup temp .eml file
        if tmp_path and Path(tmp_path).exists():
            try:
                os.remove(tmp_path)
            except Exception:
                pass

        # Cleanup side-effect JSON that extract_eml writes
        if tmp_path:
            json_path = Path(Path(tmp_path).stem + "_parsed.json")
            if json_path.exists():
                try:
                    os.remove(json_path)
                except Exception:
                    pass

    # 4. Inject access-control + source fields into metadata
    result["metadata"].update({
        "id"         : id,
        "owner_email": user_email,
        "source"     : f"{msg['id']}.gmail",
        "source_type": "gmail",
        "eml_source" : f"{msg['id']}.gmail",
    })

    # 5. Build flat chunk list from structured extract_eml result
    return _build_chunks_from_result(result, id, user_email)

def _is_email_already_crawled(msg_id: str) -> bool:
    """
    Returns True if this Gmail message ID already has chunks in Qdrant.
    Emails don't change after received, so presence alone is enough —
    no need to compare a modified_time like Drive files.
    """
    try:
        client = _get_client()
        results, _ = client.scroll(
            collection_name = COLLECTION_NAME,
            scroll_filter   = {
                "must": [
                    {"key": "source", "match": {"value": f"{msg_id}.gmail"}}
                ]
            },
            limit        = 1,
            with_payload = False,
            with_vectors = False,
        )
        return len(results) > 0
    except Exception as e:
        print(f"[CRAWLER]   Qdrant check failed for {msg_id}: {e}")
    return False

def _decode_raw(msg: dict) -> bytes | None:
    """Decode the base64url-encoded raw RFC-822 bytes from a Gmail API response."""
    raw_data = msg.get("raw")
    if not raw_data:
        return None
    try:
        return base64.urlsafe_b64decode(raw_data + "==")
    except Exception as e:
        print(f"[CRAWLER] base64 decode error: {e}")
        return None


def _build_chunks_from_result(result: dict, id: str, user_email: str) -> list[dict]:
    """
    Turn extract_eml's structured output into a flat list of chunks
    ready for normalize_chunks() → embed_chunks() → store().

    Handles:
      - Body text chunks
      - Inline table chunks
      - Attachment chunks (PDF, DOCX, PPTX, images)
    """
    all_chunks = []
    base_meta  = result["metadata"]

    # ── BODY CHUNKS (text + inline tables) ───────────────────────
    for content_block in result.get("content", []):
        if not isinstance(content_block, dict):
            continue

        block_type = content_block.get("type")

        if block_type == "text":
            text = content_block.get("content", "").strip()
            if not text:
                continue
            meta   = _build_chunk_meta(base_meta, id, user_email)
            chunks = _chunk_body(text, meta)
            all_chunks.extend(chunks)

        elif block_type == "table":
            text = _table_to_text(content_block)
            if not text.strip():
                continue
            meta   = _build_chunk_meta(base_meta, id, user_email)
            chunks = _chunk_body(text, meta)
            all_chunks.extend(chunks)

    # ── ATTACHMENT CHUNKS ─────────────────────────────────────────
    for att in result.get("attachments", []):
        att_name   = att.get("name", "unknown")
        att_result = att.get("result", {})

        # Skip if handler returned nothing usable
        if not att_result or not isinstance(att_result, dict):
            continue

        # Different handlers return chunks under different keys:
        #   "chunks"  → pptx_chunker, scanned_pdf_chunker
        #   "content" → digital_pdf, image_handler  (may be a string OR list)
        raw_blocks = (
            att_result.get("chunks")
            or att_result.get("content")
            or []
        )

        # Normalize: image/txt handlers may return a plain string instead of a list
        if isinstance(raw_blocks, str):
            content_blocks = [{"type": "text", "content": raw_blocks}] if raw_blocks.strip() else []
        elif isinstance(raw_blocks, list):
            content_blocks = raw_blocks
        else:
            content_blocks = []

        for content_block in content_blocks:
            # Guard: skip bare strings that slipped through mixed lists
            if not isinstance(content_block, dict):
                continue

            text = (
                content_block.get("content")
                or content_block.get("text", "")
            )
            if not isinstance(text, str) or not text.strip():
                continue

            att_meta = _build_chunk_meta(
                base_meta, id, user_email,
                source_override=f"{att_name} (in {base_meta.get('source', '')})",
            )
            chunks = _chunk_body(text, att_meta)
            all_chunks.extend(chunks)

    return all_chunks


def _build_chunk_meta(
    base_meta: dict,
    id: str,
    user_email: str,
    source_override: str = None,
) -> dict:
    """Build a metadata dict in the shape eml_chunker expects."""
    return {
        "source"      : source_override or base_meta.get("source", ""),
        "file_id"     : str(uuid.uuid4()),
        "source_type" : "gmail",
        "eml_source"  : base_meta.get("eml_source", base_meta.get("source", "")),
        "subject"     : base_meta.get("subject", ""),
        "from"        : base_meta.get("from", ""),
        "to"          : base_meta.get("to", ""),
        "date"        : base_meta.get("date", ""),
        "attachments" : base_meta.get("attachments", []),
        # Access-control fields — carried through to Qdrant payload
        "id"          : id,
        "owner_email" : user_email,
    }


def _table_to_text(table_block: dict) -> str:
    """
    Serialize an inline table chunk from extract_eml into plain text
    so it can be passed to _chunk_body() for embedding.

    Example output:
        Headers: Name | Revenue | Region
        Row: Alice | 1000 | North
        Row: Bob   | 2000 | South
    """
    headers = table_block.get("headers", [])
    rows    = table_block.get("rows", [])

    lines = []
    if headers:
        lines.append("Headers: " + " | ".join(str(h) for h in headers))
    for row in rows:
        lines.append("Row: " + " | ".join(str(c) for c in row))

    return "\n".join(lines)