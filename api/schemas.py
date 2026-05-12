"""
api/schemas.py — Request/response models.

Pydantic models give us:
  • Auto validation (FastAPI rejects bad input)
  • Auto OpenAPI docs (visit /docs to see)
  • Type safety in your code
"""
from datetime import datetime
from typing import Optional, Literal
from pydantic import BaseModel, Field


# ── USERS ────────────────────────────────────────────────────────

class UserOut(BaseModel):
    id: str
    email: str | None = None
    display_name: str | None = None


# ── CONVERSATIONS ────────────────────────────────────────────────

class ConversationOut(BaseModel):
    session_id: str
    title: str | None = None
    is_active: bool
    created_at: datetime
    last_active: datetime
    turn_count: int


class ConversationCreate(BaseModel):
    title: str | None = None


class ConversationUpdate(BaseModel):
    title: str | None = None


class TurnOut(BaseModel):
    role: str = Field(..., examples=["user", "assistant"])
    content: str
    created_at: datetime | None = None


class ConversationDetail(BaseModel):
    session_id: str
    title: str | None
    is_active: bool
    created_at: datetime
    last_active: datetime
    turns: list[TurnOut]
    summary: str = ""


# ── QUERIES ──────────────────────────────────────────────────────

# Knowledge scope options:
#   "all_kb"       — uploads + full vector DB (default, existing behaviour)
#   "uploads_only" — restrict retrieval to documents uploaded by this user
#   "single_doc"   — restrict retrieval to one specific document_id
KnowledgeScope = Literal["all_kb", "uploads_only", "single_doc"]


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    stream: bool = False

    # ── Knowledge-scope controls ──────────────────────────────────
    scope: KnowledgeScope = Field(
        default="all_kb",
        description=(
            "Controls which knowledge sources are searched.\n"
            "  all_kb       → uploads + everything in the vector DB (default)\n"
            "  uploads_only → only documents uploaded by this user\n"
            "  single_doc   → only the document identified by `document_id`"
        ),
    )
    document_id: str | None = Field(
        default=None,
        description="Required when scope='single_doc'. The document_id to restrict search to.",
    )


class SourceOut(BaseModel):
    source: str
    page: int | None = None
    score: float | None = None
    document_id: str | None = None

class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceOut] = []
    original_query: str
    rewritten_query: str
    session_id: str
    total_latency_s: float | None = None
    scope: KnowledgeScope = "all_kb"
    document_id: str | None = None


# ── DOCUMENTS ────────────────────────────────────────────────────

class DocumentOut(BaseModel):
    document_id: str
    filename: str
    uploaded_at: datetime
    chunks: int | None = None
    status: str = "indexed"  # indexed | indexing | failed


class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    chunks: int
    message: str = "Document uploaded and indexed"