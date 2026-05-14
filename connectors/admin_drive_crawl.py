# connectors/admin_drive_crawl.py
# Crawl ALL users' Google Drive and store chunks in Qdrant.
# Usage:
#   python connectors/admin_drive_crawl.py alice
#   python connectors/admin_drive_crawl.py alice --max 50   ← test mode

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from datetime import datetime, timezone
from auth.users import load_users, is_admin, update_crawl_status
from connectors.drive_crawler import crawl_drive_user
from embedding.normalizer import normalize_chunks, VALID_SOURCE_TYPES
from embedding.embedder import embed_chunks
from embedding.qdrant_store import store, create_collection
from db.drive_permissions import create_tables

VALID_SOURCE_TYPES.add("google_drive")


def run_drive_crawl(caller_id: str, max_files: int = None, dry_run: bool = False) -> None:
    if not is_admin(caller_id):
        print(f"[DRIVE CRAWL] Access denied: '{caller_id}' is not an admin.")
        sys.exit(1)

    print(f"[DRIVE CRAWL] Admin '{caller_id}' starting Drive crawl...")
    if max_files:
        print(f"[DRIVE CRAWL] TEST MODE: max {max_files} files per user")
    if dry_run:
        print(f"[DRIVE CRAWL] DRY RUN: no embedding, no storing — preview only")

    create_tables()
    create_collection()

    users = load_users()
    if not users:
        print("[DRIVE CRAWL] No users registered.")
        return

    for uid, user_info in users.items():
        email = user_info["email"]
        print(f"\n[DRIVE CRAWL] ── {uid} ({email}) ──")

        try:
            # 1. Crawl
            raw_chunks = crawl_drive_user(uid, email, max_files=max_files)

            if not raw_chunks:
                print(f"[DRIVE CRAWL] {uid}: no content found")
                continue

            # 2. Normalize
            normalized = raw_chunks

            if dry_run:
                # ── PREVIEW ONLY — no embed, no store ────────────
                print(f"\n{'='*60}")
                print(f"  PREVIEW: {len(normalized)} chunks from {uid}")
                print(f"{'='*60}")

                # Group chunks by file for readability
                by_file: dict = {}
                for chunk in normalized:
                    fname = chunk["metadata"].get("source", "unknown")
                    by_file.setdefault(fname, []).append(chunk)

                for fname, chunks in by_file.items():
                    print(f"\n  📄 FILE: {fname}")
                    print(f"     Chunks     : {len(chunks)}")
                    print(f"     owner      : {chunks[0]['metadata'].get('owner_email')}")
                    print(f"     allowed    : {chunks[0]['metadata'].get('allowed_users')}")
                    print(f"     is_public  : {chunks[0]['metadata'].get('is_public')}")
                    print(f"     source_type: {chunks[0]['metadata'].get('source_type')}")

                    for i, chunk in enumerate(chunks):
                        print(f"\n     --- Chunk {i+1} ---")
                        print(f"     type       : {chunk['type']}")
                        print(f"     token_count: {chunk['metadata'].get('token_count')}")
                        print(f"     content    : {chunk['content'][:300]}...")

                print(f"\n{'='*60}")
                print(f"  TOTAL: {len(normalized)} chunks from {len(by_file)} files")
                print(f"{'='*60}\n")

            else:
                # ── FULL PIPELINE ─────────────────────────────────
                

                embedded = embed_chunks(normalized)
                result   = _store_with_access_control(embedded)
                update_crawl_status(uid, "done", last_crawl=datetime.now(timezone.utc))
                print(f"[DRIVE CRAWL] {uid}: ✓ {result.get('upserted', 0)} chunks upserted")

        except Exception as exc:
            if not dry_run:
                update_crawl_status(uid, "failed")
            print(f"[DRIVE CRAWL] {uid}: ✗ FAILED — {exc}")


def _store_with_access_control(embedded: list[dict]) -> dict:
    """
    Store chunks but add access control fields to every Qdrant payload.
    Wraps qdrant_store.upsert_chunks and injects the extra fields.
    """
    from embedding.qdrant_store import _get_client, COLLECTION_NAME, UPSERT_BATCH
    from qdrant_client.models import PointStruct, SparseVector
    from embedding.qdrant_store import _chunk_id_to_uuid, _validate_for_upsert

    client  = _get_client()
    skipped = 0
    errors  = []
    total   = len(embedded)

    for batch_start in range(0, total, UPSERT_BATCH):
        batch  = embedded[batch_start : batch_start + UPSERT_BATCH]
        points = []

        for chunk in batch:
            issue = _validate_for_upsert(chunk)
            if issue:
                skipped += 1
                errors.append(issue)
                continue

            meta    = chunk.get("metadata", {})
            payload = {
                "type"          : chunk["type"],
                "content"       : chunk["content"],
                "source"        : meta.get("source"),
                "source_type"   : meta.get("source_type"),
                "section"       : meta.get("section"),
                "page"          : meta.get("page"),
                "token_count"   : meta.get("token_count"),
                # Access control
                "owner_email"   : meta.get("owner_email"),
                "allowed_users" : meta.get("allowed_users", []),
                "is_public"     : meta.get("is_public", False),
                "drive_file_id" : meta.get("drive_file_id"),
            }

            sv = chunk["sparse_vector"]
            points.append(PointStruct(
                id      = _chunk_id_to_uuid(chunk["chunk_id"]),
                vector  = {
                    "dense"  : chunk["dense_vector"],
                    "sparse" : SparseVector(
                        indices = sv["indices"],
                        values  = sv["values"],
                    ),
                },
                payload = payload,
            ))

        if points:
            client.upsert(
                collection_name = COLLECTION_NAME,
                points          = points,
                wait            = True,
            )

    return {"upserted": total - skipped, "skipped": skipped, "errors": errors}


if __name__ == "__main__":
    caller    = sys.argv[1] if len(sys.argv) > 1 else input("Your id: ").strip()
    max_files = int(sys.argv[3]) if "--max" in sys.argv else None
    dry_run   = "--dry-run" in sys.argv

    run_drive_crawl(caller, max_files=max_files, dry_run=dry_run)