# connectors/drive_crawler.py
# Crawls a user's Google Drive, extracts text from supported files
# using the EXISTING ingest pipeline (same as Gmail/PDF/DOCX etc.)
# Returns chunks tagged with owner + access control metadata.

import uuid
import io
import os
import sys
import tempfile
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from embedding.qdrant_store import _get_client, COLLECTION_NAME
from auth.drive_auth import get_drive_service
from db.drive_permissions import upsert_file
from embedding.ingest import ingest_file, EXT_TO_SOURCE_TYPE


# ── Mime type mappings ────────────────────────────────────────────────────────

# Google Workspace files can't be downloaded directly — they must be exported
# We export them to the nearest equivalent format your pipeline already handles
EXPORT_MIME = {
    "application/vnd.google-apps.document"     : "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # Google Doc  → .docx
    "application/vnd.google-apps.spreadsheet"  : "text/plain",                                                               # Google Sheet → .txt
    "application/vnd.google-apps.presentation" : "application/vnd.openxmlformats-officedocument.presentationml.presentation", # Google Slide → .pptx
}

# All mime types we support (Workspace + native)
SUPPORTED_MIME = {
    *EXPORT_MIME.keys(),
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",   # .docx
    "application/vnd.openxmlformats-officedocument.presentationml.presentation", # .pptx
    "text/plain",             
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/bmp",
    "image/tiff",                                                    # .txt
}

# Mime → file extension (needed for temp file + ingest_file auto-detection)
MIME_TO_EXT = {
    "application/vnd.google-apps.document"     : ".docx",
    "application/vnd.google-apps.spreadsheet"  : ".txt",
    "application/vnd.google-apps.presentation" : ".pptx",
    "application/pdf"                          : ".pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"   : ".docx",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation" : ".pptx",
    "text/plain"                               : ".txt",
    "image/png"  : ".png",
    "image/jpeg" : ".jpg",
    "image/jpg"  : ".jpg",
    "image/bmp"  : ".bmp",
    "image/tiff" : ".tiff",
}


# ── Main crawler ──────────────────────────────────────────────────────────────

def _is_already_crawled(file_id: str, modified_time: str) -> bool:
    """
    Returns True if a file with this drive_file_id already exists in Qdrant
    AND its stored modified_time matches the Drive API value.
    If so, there's no need to re-crawl it.
    """
    try:
        client = _get_client()
        results, _ = client.scroll(
            collection_name = COLLECTION_NAME,
            scroll_filter   = {
                "must": [
                    {"key": "drive_file_id", "match": {"value": file_id}}
                ]
            },
            limit           = 1,
            with_payload    = True,
            with_vectors    = False,
        )
        if results:
            stored_modified = results[0].payload.get("modified_time")
            return stored_modified == modified_time
    except Exception as e:
        print(f"[DRIVE]   Qdrant check failed for {file_id}: {e}")
    return False

def crawl_drive_user(
    id         : str,
    user_email : str,
    max_files  : int = None,
) -> list[dict]:
    """
    Fetch all Drive files for one user and return a flat list of chunks.
    Each chunk carries owner_email, allowed_users, is_public in its metadata
    for Qdrant access-control filtering at query time.

    Args:
        id         : unique identifier matching the users table
        user_email : the user's Gmail/Drive address
        max_files  : optional cap for testing (e.g. 50)

    Returns:
        List of normalized chunk dicts ready for embed_chunks() → store()
    """
    service = get_drive_service(id, user_email)
    all_chunks: list[dict] = []
    page_token = None
    total_fetched = 0

    print(f"\n[DRIVE] Starting crawl for {id} ({user_email})")

    while True:
        # Fetch a page of files
        kwargs = {
            "pageSize" : 100,
            "fields"   : "nextPageToken, files(id, name, mimeType, modifiedTime)",
            "q"        : "trashed = false",
        }
        if page_token:
            kwargs["pageToken"] = page_token

        response   = service.files().list(**kwargs).execute()
        files      = response.get("files", [])
        page_token = response.get("nextPageToken")

        for f in files:
            # Stop if we hit the test limit
            if max_files and total_fetched >= max_files:
                break

            mime = f.get("mimeType", "")

            # Skip unsupported file types (videos, forms, maps etc.)
            if mime not in SUPPORTED_MIME:
                print(f"[DRIVE] Skipping unsupported type: {mime} ({f.get('name')})")
                continue

            modified_time = f.get("modifiedTime", "")
            if _is_already_crawled(file_id=f["id"], modified_time=modified_time):
                 print(f"[DRIVE]   Skipping (already crawled, unchanged): {f.get('name')}")
                 total_fetched += 1
                 continue
                

            chunks = _file_to_chunks(service, f, id, user_email, modified_time=modified_time)
            all_chunks.extend(chunks)
            total_fetched += 1

            print(
                f"[DRIVE] {id}: "
                f"{total_fetched} files processed → {len(all_chunks)} chunks so far"
            )

        if not page_token or (max_files and total_fetched >= max_files):
            break

    print(f"[DRIVE] Done: {id} → {len(all_chunks)} total chunks")
    return all_chunks


# ── Internal helpers ──────────────────────────────────────────────────────────

def _file_to_chunks(
    service    : object,
    f          : dict,
    id         : str,
    user_email : str,
    modified_time : str = "",
) -> list[dict]:
    """
    For one Drive file:
      1. Resolve permissions
      2. Save permissions to Postgres
      3. Download to a temp file
      4. Run through your existing ingest_file() pipeline
      5. Inject access control into every chunk's metadata
      6. Return chunks
    """
    from googleapiclient.http import MediaIoBaseDownload

    file_id   = f["id"]
    file_name = f.get("name", "unknown")
    mime      = f.get("mimeType", "")

    print(f"\n[DRIVE] Processing: {file_name} ({mime})")

    # ── 1. Resolve who can see this file ─────────────────────────
    is_public, allowed_users = _resolve_permissions(service, file_id, user_email)
    print(f"[DRIVE]   is_public={is_public}  allowed={allowed_users}")

    # ── 2. Save to Postgres ───────────────────────────────────────
    upsert_file(
        file_id     = file_id,
        name        = file_name,
        mime_type   = mime,
        owner_email = user_email,
        allowed_users = allowed_users,
        is_public   = is_public,
    )

    # ── 3. Get file extension for temp file ───────────────────────
    ext = MIME_TO_EXT.get(mime)
    if ext is None:
        print(f"[DRIVE]   No extension mapping for {mime} — skipping")
        return []

    # ── 4. Download to temp file ──────────────────────────────────
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp_path = tmp.name

        if mime in EXPORT_MIME:
            # Google Workspace files: export to equivalent format
            export_mime = EXPORT_MIME[mime]
            print(f"[DRIVE]   Exporting as {export_mime}...")
            response = service.files().export(
                fileId   = file_id,
                mimeType = export_mime,
            ).execute()
            # export() returns bytes or str depending on mime
            content = response if isinstance(response, bytes) else response.encode("utf-8")
            with open(tmp_path, "wb") as out:
                out.write(content)

        else:
            # Native files: download directly
            print(f"[DRIVE]   Downloading...")
            with open(tmp_path, "wb") as out:
                downloader = MediaIoBaseDownload(
                    out,
                    service.files().get_media(fileId=file_id)
                )
                done = False
                while not done:
                    _, done = downloader.next_chunk()

        # ── 5. Run through your existing ingest pipeline ──────────
        # ingest_file() auto-detects the chunker from the extension
        # This is exactly the same pipeline used for Gmail attachments,
        # uploaded PDFs, DOCX files etc. — nothing new.
        print(f"[DRIVE]   Ingesting via existing pipeline (ext={ext})...")
        chunks = ingest_file(tmp_path, validate=False)

    except Exception as e:
        print(f"[DRIVE]   Failed to process {file_name}: {e}")
        return []

    finally:
        # Always delete the temp file
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    if not chunks:
        print(f"[DRIVE]   No chunks produced for {file_name}")
        return []

    # ── 6. Inject access control into every chunk ─────────────────
    # ingest_file() doesn't know about Drive permissions,
    # so we add them here. These get stored in Qdrant payload
    # and used at query time to filter by user.
    for chunk in chunks:
        chunk["metadata"]["source"]        = file_name  # ← real filename
        chunk["metadata"]["owner_email"]   = user_email
        chunk["metadata"]["allowed_users"] = allowed_users
        chunk["metadata"]["is_public"]     = is_public
        chunk["metadata"]["drive_file_id"] = file_id
        chunk["metadata"]["id"]            = id
        chunk["metadata"]["modified_time"] = modified_time 

    print(f"[DRIVE]   ✓ {len(chunks)} chunks from {file_name}")
    return chunks


def _resolve_permissions(
    service    : object,
    file_id    : str,
    owner_email: str,
) -> tuple[bool, list[str]]:
    """
    Fetch Drive permissions for a file.
    Returns (is_public, list_of_allowed_emails).
    Owner is always included in allowed_users.
    """
    is_public     = False
    allowed_users = [owner_email]   # owner always has access

    try:
        perms = service.permissions().list(
            fileId = file_id,
            fields = "permissions(type, emailAddress, role)",
        ).execute().get("permissions", [])

        for p in perms:
            ptype = p.get("type")
            role  = p.get("role", "")

            # We only care about read access and above
            if role not in ("owner", "writer", "commenter", "reader"):
                continue

            if ptype == "anyone":
                # Public file — anyone with the link can see it
                is_public = True

            elif ptype in ("user", "group"):
                email = p.get("emailAddress", "")
                if email and email not in allowed_users:
                    allowed_users.append(email)

    except Exception as e:
        print(f"[DRIVE]   Could not fetch permissions for {file_id}: {e}")

    return is_public, allowed_users