# admin_crawl.py
# Run this to crawl ALL users' Gmail inboxes and store them in Qdrant.
# Only users with is_admin=True in the users table are allowed to run this.
#
# Usage:
#   python admin_crawl.py alice
#   python admin_crawl.py          ← prompts for id

import sys
from datetime import datetime, timezone
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
CREDENTIALS_FILE = os.path.join(os.path.dirname(__file__), "..", "credentials.json")
from auth.users import load_users, is_admin, update_crawl_status
from connectors.gmail_crawler import crawl_user
from embedding.normalizer import normalize_chunks, VALID_SOURCE_TYPES
from embedding.embedder import embed_chunks
from embedding.qdrant_store import store, create_collection

# Register gmail as a valid source type in your normalizer
VALID_SOURCE_TYPES.add("gmail")


def run_full_crawl(caller_id: str) -> None:
    """
    Crawl every registered user's Gmail and upsert chunks into Qdrant.
    Requires the caller to be an admin.
    """
    if not is_admin(caller_id):
        print(f"[CRAWL] Access denied: '{caller_id}' is not an admin.")
        sys.exit(1)

    print(f"[CRAWL] Admin '{caller_id}' starting full crawl...")

    users = load_users()
    if not users:
        print("[CRAWL] No users registered. Add users with: python auth/users.py add ...")
        return

    # Ensure Qdrant collection exists with id payload index
    create_collection()

    total_upserted = 0

    for id, user_info in users.items():
        email = user_info["email"]
        print(f"\n[TEST] Crawling {id} ({email}) — first 50 emails only")

        update_crawl_status(id, "running")

        try:
          # 1. Crawl only 50 emails
          raw_chunks = crawl_user(id, email, max_emails=10)

          if not raw_chunks:
            print(f"[TEST] {id}: no content found")
            update_crawl_status(id, "done", last_crawl=datetime.now(timezone.utc))
            continue

          # 2. Normalize
          normalized = normalize_chunks(raw_chunks, source_type="gmail")

          # 3. Embed
          embedded = embed_chunks(normalized)

          # 4. Store in Qdrant
          result = store(embedded)
          upserted = result.get("upserted", 0)

          update_crawl_status(id, "done", last_crawl=datetime.now(timezone.utc))
          print(f"[TEST] {id}: ✓ {upserted} chunks upserted from 50 emails")

        except Exception as exc:
         update_crawl_status(id, "failed")
         print(f"[TEST] {id}: ✗ FAILED — {exc}")

    """ email = user_info["email"]
        print(f"\n[CRAWL] ── {id} ({email}) ──")

        # Mark as running in Postgres
        update_crawl_status(id, "running")

        try:
            # 1. Crawl Gmail (browser opens once per user on first run)
            raw_chunks = crawl_user(id, email)

            if not raw_chunks:
                print(f"[CRAWL] {id}: no content — skipping")
                update_crawl_status(
                    id, "done", last_crawl=datetime.now(timezone.utc)
                )
                continue

            # 2. Normalize (reuses your existing normalizer)
            normalized = normalize_chunks(raw_chunks, source_type="gmail")

            # 3. Embed
            embedded = embed_chunks(normalized)

            # 4. Store in Qdrant — id is already in every chunk's payload
            result = store(embedded)
            upserted = result.get("upserted", 0)
            total_upserted += upserted

            # 5. Mark done with timestamp
            update_crawl_status(
                id, "done", last_crawl=datetime.now(timezone.utc)
            )
            print(f"[CRAWL] {id}: ✓ {upserted} chunks upserted")

        except Exception as exc:
            update_crawl_status(id, "failed")
            print(f"[CRAWL] {id}: ✗ FAILED — {exc}")

    print(f"\n[CRAWL] Complete. Total chunks upserted: {total_upserted}")

"""

if __name__ == "__main__":
    caller = sys.argv[1] if len(sys.argv) > 1 else input("Your id: ").strip()
    run_full_crawl(caller)