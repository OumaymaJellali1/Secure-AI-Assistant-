"""
api/routes/documents.py — Document upload, listing, deletion.

Wired to YOUR structure:
  embedding/ingest.py        → ingest_file()  (orchestrator)
  embedding/embedder.py      → embed_chunks()
  embedding/qdrant_store.py  → store()

Upload flow:
  1. Save file to uploads/
  2. Generate unique document_id
  3. ingest_file() → handles parsing + chunking based on extension
  4. Stamp document_id into chunks
  5. embed_chunks() → BGE-M3 dense + sparse vectors
  6. store() → upsert to Qdrant
  7. Save to documents table + document_permissions
"""
import os
import uuid
import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from sqlalchemy import text

from shared.db import engine
from api.auth import get_current_user
from api.schemas import DocumentOut, DocumentUploadResponse


# ── CONFIG ──────────────────────────────────────────────────────

UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", "uploads"))
UPLOAD_DIR.mkdir(exist_ok=True, parents=True)

# Match the extensions ingest.py supports (from EXT_TO_SOURCE_TYPE)
ALLOWED_EXTENSIONS = {
    ".pdf", ".docx", ".pptx", ".eml", ".txt",
    ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif",
}


router = APIRouter(prefix="/documents", tags=["documents"])


# ── DOCUMENTS METADATA TABLE ────────────────────────────────────

def _ensure_documents_table():
    """Create documents table if it doesn't exist (one-time)."""
    with engine.begin() as conn:
        conn.exec_driver_sql("""
            CREATE TABLE IF NOT EXISTS documents (
                document_id  TEXT PRIMARY KEY,
                filename     TEXT NOT NULL,
                uploaded_by  TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                file_path    TEXT,
                source_type  TEXT,
                chunks       INTEGER DEFAULT 0,
                uploaded_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            CREATE INDEX IF NOT EXISTS idx_documents_uploader 
                ON documents(uploaded_by, uploaded_at DESC);
        """)


# ── INGESTION RUNNER ────────────────────────────────────────────

def _run_ingestion(file_path: str, document_id: str) -> tuple[int, str]:
    """
    Run the 3-step pipeline:
      ingest_file → stamp document_id → embed_chunks → store
    
    Returns: (number_of_chunks, source_type)
    """
    # Imports for YOUR structure
    from embedding.ingest       import ingest_file, EXT_TO_SOURCE_TYPE
    from embedding.embedder     import embed_chunks
    from embedding.qdrant_store import store
    
    # Detect source type for metadata
    ext = Path(file_path).suffix.lower()
    source_type = EXT_TO_SOURCE_TYPE.get(ext, "unknown")
    
    # ── Step 1: parse + chunk + normalize ──────────────────────
    print(f"\n[UPLOAD 1/3] Ingesting {file_path}...")
    chunks = ingest_file(file_path, validate=True)
    
    if not chunks:
        raise HTTPException(400, "Ingestion produced 0 chunks (empty/unreadable file?)")
    
    print(f"[UPLOAD]    Got {len(chunks)} chunks")
    
    # ── Step 2: stamp document_id into each chunk's metadata ────
    # This is what enables permission filtering downstream
    for chunk in chunks:
        chunk.setdefault("metadata", {})["document_id"] = document_id
    
    # ── Step 3: embed ───────────────────────────────────────────
    print(f"[UPLOAD 2/3] Embedding {len(chunks)} chunks...")
    embedded = embed_chunks(chunks)
    
    if not embedded:
        raise HTTPException(500, "Embedding produced 0 vectors")
    
    # ── Step 4: upsert to Qdrant ────────────────────────────────
    print(f"[UPLOAD 3/3] Storing in Qdrant (document_id={document_id})...")
    result = store(embedded)
    
    upserted = result.get("upserted", 0)
    return upserted, source_type


# ── UPLOAD ──────────────────────────────────────────────────────

@router.post("", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    user_id: str = Depends(get_current_user),
):
    """Upload + parse + chunk + embed + index + grant ownership."""
    # 1. Validate extension
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            400,
            f"Extension {ext!r} not supported. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )
    
    # 2. Save file with unique name
    document_id = f"doc_{uuid.uuid4().hex[:12]}"
    safe_filename = f"{document_id}_{file.filename}"
    file_path = UPLOAD_DIR / safe_filename
    
    print(f"\n{'='*60}")
    print(f"[UPLOAD] User: {user_id}")
    print(f"[UPLOAD] File: {file.filename} → {file_path}")
    print(f"[UPLOAD] Document ID: {document_id}")
    print(f"{'='*60}")
    
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    # 3. Run ingestion pipeline (parse → embed → store)
    try:
        chunks_count, source_type = _run_ingestion(str(file_path), document_id)
    except HTTPException:
        # Clean up file on ingestion failure
        try: file_path.unlink()
        except: pass
        raise
    except Exception as e:
        try: file_path.unlink()
        except: pass
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"Ingestion failed: {e}")
    
    # 4. Save metadata + grant uploader access
    _ensure_documents_table()
    with engine.begin() as conn:
        # Save document metadata
        conn.execute(
            text("""
                INSERT INTO documents 
                    (document_id, filename, uploaded_by, file_path, source_type, chunks)
                VALUES (:did, :fn, :uid, :fp, :st, :ch)
                ON CONFLICT (document_id) DO UPDATE
                SET chunks = EXCLUDED.chunks
            """),
            {
                "did": document_id,
                "fn": file.filename,
                "uid": user_id,
                "fp": str(file_path),
                "st": source_type,
                "ch": chunks_count,
            },
        )
        
        # Grant uploader full access (Phase 1: only uploader can see it)
        conn.execute(
            text("""
                INSERT INTO document_permissions 
                    (user_id, document_id, permission_level, granted_by)
                VALUES (:uid, :did, 'admin', :uid)
                ON CONFLICT (user_id, document_id) DO NOTHING
            """),
            {"uid": user_id, "did": document_id},
        )
    
    print(f"\n[UPLOAD] ✓ Done: {chunks_count} chunks indexed")
    print(f"{'='*60}\n")
    
    return DocumentUploadResponse(
        document_id=document_id,
        filename=file.filename,
        chunks=chunks_count,
        message=f"Uploaded and indexed {chunks_count} chunks ({source_type})",
    )


# ── LIST ────────────────────────────────────────────────────────

@router.get("", response_model=list[DocumentOut])
def list_documents(user_id: str = Depends(get_current_user)):
    """List documents this user can access."""
    _ensure_documents_table()
    with engine.connect() as conn:
        rows = conn.execute(
            text("""
                SELECT d.document_id, d.filename, d.uploaded_at, d.chunks
                FROM documents d
                INNER JOIN document_permissions dp 
                    ON dp.document_id = d.document_id
                WHERE dp.user_id = :uid
                ORDER BY d.uploaded_at DESC
            """),
            {"uid": user_id},
        ).fetchall()
    
    return [
        DocumentOut(
            document_id=r[0],
            filename=r[1],
            uploaded_at=r[2],
            chunks=r[3],
            status="indexed",
        )
        for r in rows
    ]


# ── DELETE ──────────────────────────────────────────────────────

@router.delete("/{document_id}", status_code=204)
def delete_document(
    document_id: str,
    user_id: str = Depends(get_current_user),
):
    """Delete a document (only if user uploaded it)."""
    _ensure_documents_table()
    
    # Check ownership
    with engine.connect() as conn:
        row = conn.execute(
            text("""
                SELECT file_path FROM documents 
                WHERE document_id = :did AND uploaded_by = :uid
            """),
            {"did": document_id, "uid": user_id},
        ).fetchone()
    
    if not row:
        raise HTTPException(404, "Document not found or you don't own it")
    
    file_path = row[0]
    
    # Delete chunks from Qdrant (filter by document_id)
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import (
            Filter, FieldCondition, MatchValue, FilterSelector
        )
        from embedding.qdrant_store import COLLECTION_NAME, QDRANT_URL
        
        qclient = QdrantClient(url=QDRANT_URL)
        qclient.delete(
            collection_name=COLLECTION_NAME,
            points_selector=FilterSelector(filter=Filter(
                must=[FieldCondition(
                    key="document_id",
                    match=MatchValue(value=document_id),
                )]
            )),
        )
        print(f"[DELETE] ✓ Removed Qdrant chunks for {document_id}")
    except Exception as e:
        print(f"[DELETE] ⚠ Could not delete Qdrant chunks: {e}")
    
    # Delete from DB (cascades to permissions)
    with engine.begin() as conn:
        conn.execute(
            text("DELETE FROM document_permissions WHERE document_id = :did"),
            {"did": document_id},
        )
        conn.execute(
            text("DELETE FROM documents WHERE document_id = :did"),
            {"did": document_id},
        )
    
    # Delete physical file
    try:
        if file_path and Path(file_path).exists():
            Path(file_path).unlink()
    except Exception as e:
        print(f"[DELETE] ⚠ Could not delete file: {e}")
    
    return None