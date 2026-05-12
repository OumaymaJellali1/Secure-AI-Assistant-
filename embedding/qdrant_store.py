"""
qdrant_store.py
───────────────
Qdrant collection setup + upserter for hybrid RAG.

Two responsibilities:
  1. create_collection() — create the collection with the right vector config
  2. upsert_chunks()     — insert embedded chunks as Qdrant points

Collection design:
  - Two named vectors per point:
      "dense"  : 1024-dim cosine — BGE-M3 dense output
      "sparse" : sparse dot-product — BGE-M3 lexical_weights output
  - Payload  : all metadata fields → used for filtering at query time
  - No ColBERT — too heavy for storage at this stage

Docker usage (run before anything else):
  docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
"""

from __future__ import annotations

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    SparseVectorParams,
    Distance,
    SparseIndexParams,
    PointStruct,
    SparseVector,
)

import uuid


# ── CONFIG ────────────────────────────────────────────────────────

QDRANT_URL       = "http://localhost:6333"
COLLECTION_NAME  = "rag_chunks_voyage"
DENSE_DIM        = 1024
UPSERT_BATCH     = 64


# ── CLIENT SINGLETON ──────────────────────────────────────────────

_client = None

def _get_client() -> QdrantClient:
    global _client
    if _client is None:
        print(f"[QDRANT] Connecting to {QDRANT_URL}...")
        _client = QdrantClient(url=QDRANT_URL)
        print(f"[QDRANT] Connected.")
    return _client


# ── COLLECTION SETUP ──────────────────────────────────────────────

def create_collection(
    collection_name : str  = COLLECTION_NAME,
    recreate        : bool = False,
) -> None:
    client   = _get_client()
    existing = [c.name for c in client.get_collections().collections]

    if collection_name in existing:
        if recreate:
            print(f"[QDRANT] Deleting existing collection '{collection_name}'...")
            client.delete_collection(collection_name)
            print(f"[QDRANT] Deleted.")
        else:
            print(f"[QDRANT] Collection '{collection_name}' already exists — skipping creation.")
            return

    print(f"[QDRANT] Creating collection '{collection_name}'...")

    client.create_collection(
        collection_name = collection_name,
        vectors_config  = {
            "dense": VectorParams(
                size     = DENSE_DIM,
                distance = Distance.COSINE,
            ),
        },
        sparse_vectors_config = {
            "sparse": SparseVectorParams(
                index = SparseIndexParams(on_disk=False),
            ),
        },
    )

    print(f"[QDRANT] Collection '{collection_name}' created.")
    _create_payload_indexes(client, collection_name)


def _create_payload_indexes(client: QdrantClient, collection_name: str) -> None:
    """Create payload indexes for all filterable metadata fields."""
    from qdrant_client.models import PayloadSchemaType

    indexes = [
        ("source",      PayloadSchemaType.KEYWORD),
        ("source_type", PayloadSchemaType.KEYWORD),
        ("type",        PayloadSchemaType.KEYWORD),
        ("section",     PayloadSchemaType.KEYWORD),
        ("page",        PayloadSchemaType.INTEGER),
        ("document_id", PayloadSchemaType.KEYWORD),  # scope: single_doc
        ("uploaded_by", PayloadSchemaType.KEYWORD),  # scope: uploads_only
    ]

    for field_name, schema_type in indexes:
        client.create_payload_index(
            collection_name = collection_name,
            field_name      = field_name,
            field_schema    = schema_type,
        )
        print(f"[QDRANT]   Index created: {field_name} ({schema_type})")


# ── UPSERTER ──────────────────────────────────────────────────────

def upsert_chunks(
    embedded_chunks : list[dict],
    collection_name : str = COLLECTION_NAME,
) -> dict:
    """
    Insert embedded chunks into Qdrant as points.

    Each chunk must have:
      - chunk_id       : str
      - dense_vector   : list[float] of length 1024
      - sparse_vector  : {"indices": [...], "values": [...]}
      - type           : "text" | "table" | "image"
      - content        : str
      - metadata       : dict with all payload fields
                         (must include document_id and uploaded_by
                          for scope filtering to work)
    """
    if not embedded_chunks:
        print("[QDRANT] No chunks to upsert.")
        return {"upserted": 0, "skipped": 0, "errors": []}

    client  = _get_client()
    errors  = []
    skipped = 0
    total   = len(embedded_chunks)

    print(f"[QDRANT] Upserting {total} chunks → '{collection_name}' "
          f"(batch_size={UPSERT_BATCH})")

    total_batches = (total + UPSERT_BATCH - 1) // UPSERT_BATCH

    for batch_idx in range(total_batches):
        start = batch_idx * UPSERT_BATCH
        end   = min(start + UPSERT_BATCH, total)
        batch = embedded_chunks[start:end]

        points = []

        for chunk in batch:
            issue = _validate_for_upsert(chunk)
            if issue:
                errors.append(issue)
                skipped += 1
                print(f"[QDRANT] Skipping chunk {chunk.get('chunk_id','?')[:12]}: {issue}")
                continue

            # ── BUILD PAYLOAD ─────────────────────────────────────
            meta    = chunk.get("metadata", {})
            payload = {
                # chunk-level fields
                "type"        : chunk["type"],
                "content"     : chunk["content"],
                # metadata fields
                "source"      : meta.get("source"),
                "source_type" : meta.get("source_type"),
                "section"     : meta.get("section"),
                "page"        : meta.get("page"),
                "chunk_index" : meta.get("chunk_index"),
                "chunk_total" : meta.get("chunk_total"),
                "token_count" : meta.get("token_count"),
                "subject"     : meta.get("subject"),
                "from"        : meta.get("from"),
                "date"        : meta.get("date"),
                # scope-filter fields
                "document_id" : meta.get("document_id"),
                "uploaded_by" : meta.get("uploaded_by"),
            }

            # ── BUILD POINT ───────────────────────────────────────
            sv = chunk["sparse_vector"]

            point = PointStruct(
                id      = _chunk_id_to_uuid(chunk["chunk_id"]),
                vector  = {
                    "dense"  : chunk["dense_vector"],
                    "sparse" : SparseVector(
                        indices = sv["indices"],
                        values  = sv["values"],
                    ),
                },
                payload = payload,
            )
            points.append(point)

        if not points:
            continue

        try:
            client.upsert(
                collection_name = collection_name,
                points          = points,
                wait            = True,
            )
            print(f"[QDRANT]   Batch {batch_idx + 1}/{total_batches} upserted "
                  f"(chunks {start + 1}-{end})")

        except Exception as e:
            err_msg = f"Batch {batch_idx + 1} failed: {e}"
            errors.append(err_msg)
            skipped += len(points)
            print(f"[QDRANT] ERROR: {err_msg}")

    upserted = total - skipped
    _upsert_report(total, upserted, skipped, errors, collection_name, client)

    return {"upserted": upserted, "skipped": skipped, "errors": errors}


# ── HELPERS ───────────────────────────────────────────────────────

def _chunk_id_to_uuid(chunk_id: str) -> str:
    try:
        return str(uuid.UUID(chunk_id))
    except (ValueError, AttributeError):
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, str(chunk_id)))


def _validate_for_upsert(chunk: dict) -> str | None:
    if not chunk.get("chunk_id"):
        return "missing chunk_id"
    if not chunk.get("dense_vector"):
        return "missing dense_vector"
    if len(chunk.get("dense_vector", [])) != DENSE_DIM:
        return f"wrong dense_vector dim: {len(chunk.get('dense_vector', []))} (expected {DENSE_DIM})"
    sv = chunk.get("sparse_vector", {})
    if not sv.get("indices") or not sv.get("values"):
        return "missing or empty sparse_vector"
    if len(sv["indices"]) != len(sv["values"]):
        return "sparse_vector indices/values length mismatch"
    if not chunk.get("content", "").strip():
        return "empty content"
    return None


def _upsert_report(total, upserted, skipped, errors, collection_name, client):
    try:
        info  = client.get_collection(collection_name)
        count = info.points_count
    except Exception:
        count = "?"

    print(f"\n[QDRANT] ── Upsert Report ───────────────────────────────────")
    print(f"[QDRANT] Collection       : {collection_name}")
    print(f"[QDRANT] Input chunks     : {total}")
    print(f"[QDRANT] Upserted OK      : {upserted}")
    print(f"[QDRANT] Skipped / errors : {skipped}")
    print(f"[QDRANT] Total points now : {count}")
    if errors:
        print(f"\n[QDRANT] Errors ({len(errors)}):")
        for e in errors[:5]:
            print(f"         {e}")
        if len(errors) > 5:
            print(f"         ... and {len(errors) - 5} more")
    print(f"[QDRANT] ────────────────────────────────────────────────────\n")


# ── COLLECTION INFO ───────────────────────────────────────────────

def collection_info(collection_name: str = COLLECTION_NAME) -> None:
    client = _get_client()
    try:
        info = client.get_collection(collection_name)
    except Exception as e:
        print(f"[QDRANT] Could not get collection info: {e}")
        return

    vectors_count = getattr(info, "vectors_count", None)
    print(f"\n[QDRANT] ── Collection Info ─────────────────────────────────")
    print(f"[QDRANT] Name         : {collection_name}")
    print(f"[QDRANT] Points       : {info.points_count}")
    if vectors_count is not None:
        print(f"[QDRANT] Vectors      : {vectors_count}")
    print(f"[QDRANT] Status       : {info.status}")
    print(f"[QDRANT] ────────────────────────────────────────────────────\n")


# ── FULL PIPELINE HELPER ──────────────────────────────────────────

def store(
    embedded_chunks : list[dict],
    collection_name : str  = COLLECTION_NAME,
    recreate        : bool = False,
) -> dict:
    create_collection(collection_name=collection_name, recreate=recreate)
    result = upsert_chunks(embedded_chunks, collection_name=collection_name)
    collection_info(collection_name)
    return result


# ── ONE-TIME MIGRATION: add indexes to existing collection ────────

def migrate_add_scope_indexes(collection_name: str = COLLECTION_NAME) -> None:
    """
    Run this ONCE if your collection already exists and was created
    before document_id / uploaded_by indexes were added.

    Usage:
        python qdrant_store.py --migrate
    """
    from qdrant_client.models import PayloadSchemaType
    client = _get_client()
    for field, schema in [
        ("document_id", PayloadSchemaType.KEYWORD),
        ("uploaded_by", PayloadSchemaType.KEYWORD),
    ]:
        try:
            client.create_payload_index(collection_name, field, schema)
            print(f"[QDRANT] Migration: index created for '{field}'")
        except Exception as e:
            print(f"[QDRANT] Migration: '{field}' — {e}")


# ── CLI ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    args  = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = [a for a in sys.argv[1:] if a.startswith("--")]

    # python qdrant_store.py --migrate
    if "--migrate" in flags:
        migrate_add_scope_indexes()
        sys.exit(0)

    recreate = "--recreate" in flags

    if not args:
        print("Usage: python qdrant_store.py <file_path> [source_type] [--recreate]")
        print("       python qdrant_store.py --migrate   (add scope indexes to existing collection)")
        sys.exit(1)

    from ingest   import ingest_file
    from embedder import embed_chunks, embedding_health_check

    file_path   = args[0]
    source_type = args[1] if len(args) > 1 else None

    print(f"\n{'='*60}\n  RAG STORE PIPELINE\n{'='*60}")

    chunks = ingest_file(file_path, source_type=source_type, validate=True)

    ok = embedding_health_check()
    if not ok:
        print("[PIPELINE] Health check failed — aborting.")
        sys.exit(1)

    embedded = embed_chunks(chunks)
    if not embedded:
        print("[PIPELINE] No chunks embedded — aborting.")
        sys.exit(1)

    result = store(embedded, recreate=recreate)

    print(f"\n{'='*60}")
    print(f"  DONE  |  Upserted: {result['upserted']}  |  Skipped: {result['skipped']}")
    print(f"{'='*60}\n")