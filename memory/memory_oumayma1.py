"""
memory.py
─────────
4-layer memory system for RAG — with a Memory Tier Classifier and
PostgreSQL persistence (replacing in-RAM buffers and JSON files).

Memory Tiers (classified per turn):
  SHORT  — ephemeral turn context (current sub-topic, clarifications)
           TTL: ~30 minutes. Kept in DB, auto-expired.
  MID    — session/task context (current task, active entities, goals)
           TTL: ~24 hours. Compressed via rolling summary.
  LONG   — durable user facts (preferences, background, constraints)
           TTL: indefinite (until manually cleared or overwritten).

Layers:
  1. Conversation Buffer   — last N full turns kept verbatim (SHORT tier)
  2. Query Rewriting       — rewrites vague follow-up queries before search
  3. Conversation Summary  — compresses old turns into rolling summary (MID tier)
  4. User Memory           — persists key facts about the user (LONG tier)

PostgreSQL Schema (auto-created on first run):
  users               — user registry + metadata
  conversation_turns  — raw turn log with tier labels + expiry
  memory_summaries    — rolling summary per user (MID tier)
  user_facts          — long-term user facts (LONG tier)
  document_permissions— future: per-user document access control

Full pipeline with memory:
  query
    → Classifier: label this turn (SHORT / MID / LONG signals)
    → Layer 2: rewrite query using conversation history
    → retriever.search(rewritten_query)
    → reranker.rerank(rewritten_query, candidates)
    → Layer 1+3: inject buffer + summary into prompt context
    → generator.generate_with_memory(query, chunks, memory_context)
    → Layer 4: extract and persist user facts
    → update conversation buffer (auto-triggers Layer 3 if needed)

Usage:
  from memory import MemoryManager

  memory = MemoryManager(user_id="user_001", db_url="postgresql://...")

  result = memory.run(
      query     = "what about its limitations?",
      retriever = retriever,
      reranker  = reranker,
      generator = generator,
  )
  print(result["answer"])

PostgreSQL setup (run once):
  CREATE DATABASE ragdb;
  CREATE USER raguser WITH PASSWORD 'yourpassword';
  GRANT ALL PRIVILEGES ON DATABASE ragdb TO raguser;

  # Then let MemoryManager auto-create tables on first run, or run:
  python memory.py --setup
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Optional
import uuid
import psycopg2
import psycopg2.extras
from psycopg2.extras import RealDictCursor

# ── PATH SETUP ────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from groq import Groq

# ── CONFIG ────────────────────────────────────────────────────────────────────

MEMORY_MODEL       = "llama-3.1-8b-instant"

# Layer 1 (SHORT tier) — how many recent turns to keep verbatim
BUFFER_MAX_TURNS   = 8

# Layer 3 (MID tier) — compress when buffer exceeds this count
SUMMARY_THRESHOLD  = 6
SUMMARY_KEEP_TURNS = 3

# Layer 4 (LONG tier)
USER_MEMORY_MAX    = 20

# TTL per tier
SHORT_TTL_MINUTES  = 30
MID_TTL_HOURS      = 24
# LONG tier: no expiry (None)

# Groq retry config
MAX_RETRIES        = 3
RETRY_BASE_S       = 1.0
RETRY_JITTER       = 0.3

# Default DB URL — override via env or constructor
DEFAULT_DB_URL = os.environ.get(
    "MEMORY_DB_URL",
    "postgresql://raguser:yourpassword@localhost:5432/ragdb"
)

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


# ── MEMORY TIER ───────────────────────────────────────────────────────────────

class MemoryTier(str, Enum):
    """
    Three memory tiers with different TTLs and purposes.

    SHORT  — ephemeral: current exchange, clarifications, fillers.
             Expires in 30 minutes. Not summarized.
    MID    — session: current task, active topic, short-term goals.
             Expires in 24 hours. Fed into rolling summary (Layer 3).
    LONG   — durable: user preferences, background, constraints.
             Never expires. Fed into user facts store (Layer 4).
    """
    SHORT = "short"
    MID   = "mid"
    LONG  = "long"


# ── PROMPTS ───────────────────────────────────────────────────────────────────

REWRITE_PROMPT = """You are a query rewriter for a RAG system.

Given the conversation history below and the user's latest question, rewrite the question so it is fully self-contained and specific — as if the conversation history didn't exist.

Rules:
- Preserve the original meaning exactly
- Replace pronouns ("it", "they", "this", "that") with the specific entity they refer to
- Expand vague references ("the second one", "the previous method") into explicit terms
- If the question is already self-contained, return it unchanged
- Return ONLY the rewritten question — no explanation, no preamble
- Maximum 20 words. A single concise question only.

Conversation history:
{history}

Latest question: {query}

Rewritten question (max 20 words):"""


SUMMARY_PROMPT = """Summarize the following conversation segment into a concise paragraph.
Keep all important facts, decisions, topics discussed, and any entities mentioned.
Be specific — preserve names, numbers, and technical terms exactly.

Conversation:
{conversation}

Summary (one dense paragraph):"""


FACT_EXTRACTION_PROMPT = """Extract any memorable, reusable facts about the USER from this message.
Focus on: preferences, goals, constraints, background, expertise level, or anything that would help personalize future answers.
Ignore generic statements. Return an empty list if nothing is worth remembering.

Message: {message}

Return ONLY a JSON array of short strings (max 15 words each).
Example: ["prefers Python over R", "working on a RAG system", "uses Qdrant as vector DB"]
Return [] if nothing memorable."""


CLASSIFIER_PROMPT = """You are a memory tier classifier.

Recent context (last 2 turns):
{context}

Classify this user message: {message}

SHORT  — social filler only: greetings, thanks, ok, got it, yes/no 
         to a procedural question ("shall I continue?").
         No lasting informational value.

MID    — current task, topic being discussed, session-specific details.

LONG   — durable user facts: preferences, habits, constraints, tech choices.
         Includes single-word answers like "Python", "never", "always"
         when they answer a preference question.

Return ONLY: SHORT, MID, or LONG.

Tier:"""


# ── DATA CLASSES ──────────────────────────────────────────────────────────────

@dataclass
class Turn:
    role    : str
    content : str
    tier    : MemoryTier = MemoryTier.MID

    def to_text(self) -> str:
        return f"{self.role.upper()}: {self.content}"


@dataclass
class MemoryState:
    summary      : str
    buffer       : list[Turn]
    user_facts   : list[str]
    turn_count   : int


# ── GROQ HELPER ───────────────────────────────────────────────────────────────

def _call_groq(prompt: str, max_tokens: int = 300) -> str:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = groq_client.chat.completions.create(
                model      = MEMORY_MODEL,
                messages   = [{"role": "user", "content": prompt}],
                max_tokens = max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt == MAX_RETRIES:
                print(f"[MEMORY] Groq call failed after {MAX_RETRIES} attempts: {e}")
                return ""
            wait = RETRY_BASE_S * (2 ** (attempt - 1)) + random.uniform(0, RETRY_JITTER)
            print(f"[MEMORY] Groq error (attempt {attempt}/{MAX_RETRIES}), retrying in {wait:.1f}s...")
            time.sleep(wait)
    return ""


# ── POSTGRESQL BACKEND ────────────────────────────────────────────────────────

# DDL executed once to create all required tables.
# The document_permissions table is included now so it's ready when you add
# per-document access control — no migration needed later.

SCHEMA_SQL = """
-- ── Users ─────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS users (
    user_id     TEXT PRIMARY KEY,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata    JSONB       NOT NULL DEFAULT '{}'
);

-- ── Conversation turns (Layer 1 + classifier labels) ─────────────────────
-- Each raw turn is stored here with its tier label and an expiry timestamp.
-- SHORT turns expire in 30 min, MID in 24 h, LONG never (NULL).
-- The buffer query fetches only non-expired turns ordered by created_at.
CREATE TABLE IF NOT EXISTS sessions (
    session_id  TEXT        PRIMARY KEY,
    user_id     TEXT        NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    title       TEXT,
    device      TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_active TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    is_active   BOOLEAN     NOT NULL DEFAULT TRUE
);
CREATE TABLE IF NOT EXISTS conversation_turns (
    id          BIGSERIAL   PRIMARY KEY,
    user_id     TEXT        NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    session_id  TEXT        REFERENCES sessions(session_id) ON DELETE SET NULL,
    role        TEXT        NOT NULL CHECK (role IN ('user', 'assistant')),
    content     TEXT        NOT NULL,
    tier        TEXT        NOT NULL DEFAULT 'mid',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at  TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_turns_user_created
    ON conversation_turns (user_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_turns_expires
    ON conversation_turns (expires_at)
    WHERE expires_at IS NOT NULL;

-- ── Rolling summary (Layer 3 — MID tier) ─────────────────────────────────
-- One row per user. Updated in-place whenever compression triggers.
CREATE TABLE IF NOT EXISTS memory_summaries (
    user_id     TEXT        PRIMARY KEY REFERENCES users(user_id) ON DELETE CASCADE,
    summary     TEXT        NOT NULL DEFAULT '',
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ── User facts (Layer 4 — LONG tier) ─────────────────────────────────────
-- One row per fact per user. Deduplicated by content hash on insert.
CREATE TABLE IF NOT EXISTS user_facts (
    id          BIGSERIAL   PRIMARY KEY,
    user_id     TEXT        NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    fact        TEXT        NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (user_id, fact)
);

CREATE INDEX IF NOT EXISTS idx_facts_user
    ON user_facts (user_id, created_at DESC);

-- ── Document permissions (future — ready for your access control layer) ──
-- When you add documents that have per-user permissions, insert rows here.
-- The retriever can JOIN on this table to filter chunks before returning them.
--
-- permission_level: 'read' | 'write' | 'admin'
-- granted_by: the user_id of whoever granted the permission (audit trail)
CREATE TABLE IF NOT EXISTS document_permissions (
    id               BIGSERIAL   PRIMARY KEY,
    user_id          TEXT        NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    document_id      TEXT        NOT NULL,   -- matches your Qdrant payload field
    permission_level TEXT        NOT NULL DEFAULT 'read'
                                 CHECK (permission_level IN ('read', 'write', 'admin')),
    granted_by       TEXT        REFERENCES users(user_id),
    granted_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at       TIMESTAMPTZ,            -- NULL = permanent
    UNIQUE (user_id, document_id)
);

CREATE INDEX IF NOT EXISTS idx_perms_user
    ON document_permissions (user_id);

CREATE INDEX IF NOT EXISTS idx_perms_document
    ON document_permissions (document_id);
"""


class PostgresBackend:
    """
    Thin wrapper around a psycopg2 connection pool.

    All memory layers call methods on this object — none of them
    manage connections directly. Connection is lazy: opened on first use.

    Args:
        db_url : standard libpq connection string
                 e.g. "postgresql://user:pass@host:5432/dbname"
    """

    def __init__(self, db_url: str) -> None:
        self._db_url = db_url
        self._conn: Optional[psycopg2.extensions.connection] = None

    def connect(self) -> None:
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(self._db_url)
            self._conn.autocommit = False
        return self._conn

    def ensure_schema(self) -> None:
        """Create all tables if they don't exist (idempotent)."""
        conn = self.connect()
        with conn.cursor() as cur:
            cur.execute(SCHEMA_SQL)
        conn.commit()
        print("[MEMORY] PostgreSQL schema verified.")

    def ensure_user(self, user_id: str) -> None:
        """Insert user row if not present (idempotent)."""
        conn = self.connect()
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO users (user_id) VALUES (%s) ON CONFLICT DO NOTHING",
                (user_id,)
            )
        conn.commit()

    # ── Conversation turns ────────────────────────────────────────────────────

    def insert_turn(self, user_id, session_id, role, content, tier):
     expires_at = _tier_expiry(tier)
     conn = self.connect()
     with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO conversation_turns (user_id, session_id, role, content, tier, expires_at)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (user_id, session_id, role, content, tier.value, expires_at)  # ← session_id added
        )
     conn.commit()

    def get_recent_turns(self, user_id, session_id=None, n=8):
     conn = self.connect()
     now = _now()
     with conn.cursor(cursor_factory=RealDictCursor) as cur:
        if session_id:
            cur.execute(
                """
                SELECT role, content, tier
                FROM   conversation_turns
                WHERE  user_id    = %s
                  AND  session_id = %s          -- ← filter by session
                  AND  (expires_at IS NULL OR expires_at > %s)
                ORDER  BY created_at DESC
                LIMIT  %s
                """,
                (user_id, session_id, now, n)
            )
        else:
            cur.execute(  # fallback: all sessions (used by compress, count, etc.)
                """
                SELECT role, content, tier
                FROM   conversation_turns
                WHERE  user_id = %s
                  AND  (expires_at IS NULL OR expires_at > %s)
                ORDER  BY created_at DESC
                LIMIT  %s
                """,
                (user_id, now, n)
            )
        rows = cur.fetchall()
     return [
        Turn(role=r["role"], content=r["content"], tier=MemoryTier(r["tier"]))
        for r in reversed(rows)
    ]
    def count_active_turns(self, user_id: str) -> int:
        conn = self.connect()
        now  = _now()
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT COUNT(*) FROM conversation_turns
                WHERE  user_id  = %s
                  AND  (expires_at IS NULL OR expires_at > %s)
                """,
                (user_id, now)
            )
            return cur.fetchone()[0]

    def get_turns_to_compress(self, user_id: str, keep_last: int) -> tuple[list[Turn], list[Turn]]:
        """
        Return (to_compress, to_keep) — the older turns and the recent ones.
        to_compress: everything except the last `keep_last` active turns
        to_keep    : the last `keep_last` active turns
        """
        all_turns = self.get_recent_turns(user_id, n=1000)  # all active
        if len(all_turns) <= keep_last:
            return [], all_turns
        return all_turns[:-keep_last], all_turns[-keep_last:]

    def expire_compressed_turns(self, user_id: str, keep_last: int) -> None:
        """
        Mark all but the last `keep_last` active turns as immediately expired.
        This is called after compression so they won't appear in future queries.
        """
        conn = self.connect()
        now  = _now()
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE conversation_turns
                SET    expires_at = NOW()
                WHERE  user_id    = %s
                  AND  (expires_at IS NULL OR expires_at > %s)
                  AND  id NOT IN (
                      SELECT id FROM conversation_turns
                      WHERE  user_id   = %s
                        AND  (expires_at IS NULL OR expires_at > %s)
                      ORDER  BY created_at DESC
                      LIMIT  %s
                  )
                """,
                (user_id, now, user_id, now, keep_last)
            )
        conn.commit()

    def purge_expired_turns(self, user_id: Optional[str] = None) -> int:
        """
        Delete expired turns. Pass user_id to limit to one user,
        or None to purge globally (run as a maintenance job).
        Returns number of rows deleted.
        """
        conn = self.connect()
        now  = _now()
        with conn.cursor() as cur:
            if user_id:
                cur.execute(
                    "DELETE FROM conversation_turns WHERE user_id=%s AND expires_at <= %s",
                    (user_id, now)
                )
            else:
                cur.execute(
                    "DELETE FROM conversation_turns WHERE expires_at <= %s",
                    (now,)
                )
            count = cur.rowcount
        conn.commit()
        return count

    def clear_turns(self, user_id: str) -> None:
        conn = self.connect()
        with conn.cursor() as cur:
            cur.execute("DELETE FROM conversation_turns WHERE user_id=%s", (user_id,))
        conn.commit()

    # ── Summary ───────────────────────────────────────────────────────────────

    def get_summary(self, user_id: str) -> str:
        conn = self.connect()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT summary FROM memory_summaries WHERE user_id=%s",
                (user_id,)
            )
            row = cur.fetchone()
        return row[0] if row else ""

    def upsert_summary(self, user_id: str, summary: str) -> None:
        conn = self.connect()
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO memory_summaries (user_id, summary, updated_at)
                VALUES (%s, %s, NOW())
                ON CONFLICT (user_id)
                DO UPDATE SET summary=EXCLUDED.summary, updated_at=NOW()
                """,
                (user_id, summary)
            )
        conn.commit()

    def clear_summary(self, user_id: str) -> None:
        conn = self.connect()
        with conn.cursor() as cur:
            cur.execute("DELETE FROM memory_summaries WHERE user_id=%s", (user_id,))
        conn.commit()

    # ── User facts ────────────────────────────────────────────────────────────

    def get_facts(self, user_id: str, limit: int = USER_MEMORY_MAX) -> list[str]:
        conn = self.connect()
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT fact FROM user_facts
                WHERE  user_id = %s
                ORDER  BY created_at DESC
                LIMIT  %s
                """,
                (user_id, limit)
            )
            rows = cur.fetchall()
        return [r[0] for r in rows]

    def insert_facts(self, user_id: str, facts: list[str]) -> int:
        """Insert facts, ignoring duplicates. Returns count of new facts."""
        if not facts:
            return 0
        conn = self.connect()
        inserted = 0
        with conn.cursor() as cur:
            for fact in facts:
                cur.execute(
                    """
                    INSERT INTO user_facts (user_id, fact)
                    VALUES (%s, %s)
                    ON CONFLICT (user_id, fact) DO NOTHING
                    """,
                    (user_id, fact)
                )
                inserted += cur.rowcount
        conn.commit()
        return inserted

    def clear_facts(self, user_id: str) -> None:
        conn = self.connect()
        with conn.cursor() as cur:
            cur.execute("DELETE FROM user_facts WHERE user_id=%s", (user_id,))
        conn.commit()

    # ── Document permissions (ready for your future use) ──────────────────────

    def get_user_document_ids(self, user_id: str) -> list[str]:
        """
        Return list of document_ids the user has at least 'read' access to.
        Use this to build a Qdrant filter:

            filter = {"must": [{"key": "document_id", "match": {"any": doc_ids}}]}
        """
        conn = self.connect()
        now  = _now()
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT document_id FROM document_permissions
                WHERE  user_id          = %s
                  AND  permission_level IN ('read', 'write', 'admin')
                  AND  (expires_at IS NULL OR expires_at > %s)
                """,
                (user_id, now)
            )
            rows = cur.fetchall()
        return [r[0] for r in rows]

    def grant_permission(
        self,
        user_id         : str,
        document_id     : str,
        permission_level: str   = "read",
        granted_by      : Optional[str] = None,
        expires_at      : Optional[datetime] = None,
    ) -> None:
        conn = self.connect()
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO document_permissions
                    (user_id, document_id, permission_level, granted_by, expires_at)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (user_id, document_id)
                DO UPDATE SET
                    permission_level = EXCLUDED.permission_level,
                    granted_by       = EXCLUDED.granted_by,
                    granted_at       = NOW(),
                    expires_at       = EXCLUDED.expires_at
                """,
                (user_id, document_id, permission_level, granted_by, expires_at)
            )
        conn.commit()

    def revoke_permission(self, user_id: str, document_id: str) -> None:
        conn = self.connect()
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM document_permissions WHERE user_id=%s AND document_id=%s",
                (user_id, document_id)
            )
        conn.commit()

    def close(self) -> None:
        if self._conn and not self._conn.closed:
            self._conn.close()


# ── HELPERS ───────────────────────────────────────────────────────────────────

def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


def _tier_expiry(tier: MemoryTier) -> Optional[datetime]:
    if tier == MemoryTier.SHORT:
        return _now() + timedelta(minutes=SHORT_TTL_MINUTES)
    elif tier == MemoryTier.MID:
        return _now() + timedelta(hours=MID_TTL_HOURS)
    else:  # LONG
        return None  # never expires


# ── MEMORY TIER CLASSIFIER ────────────────────────────────────────────────────

class MemoryClassifier:
    """
    Classifies a user message into SHORT / MID / LONG memory tier.

    Strategy: fast rule-based pre-filter first (no Groq call needed for
    obvious cases), then LLM fallback for ambiguous messages.

    SHORT rules  — messages under 8 words, greetings, acknowledgments,
                   simple yes/no, immediate follow-ups.
    LONG rules   — explicit preference statements ("I prefer...", "I always
                   use...", "I work on..."), background assertions.
    MID fallback — everything else (task context, topic discussion, etc.)

    The LLM is only called when the rule-based check returns None.
    """

    # Patterns that strongly signal SHORT tier
    _SOCIAL_FILLER_PATTERNS = [
    r"^(yes|no|ok|okay|sure)\.?$",
    r"^(thanks|thank you|thx)[\s!.]*$",
    r"^(got it|i see|makes sense|understood)[\s.]*$",
    r"^(hi|hello|hey|good morning|good afternoon|good evening)[\s!.]*$",
    r"^(continue|go on|next|more)[\s.]*$",
    r"^(cool|great|nice|perfect|awesome)[\s!.]*$",
]

    # Patterns that strongly signal LONG tier
    _LONG_PATTERNS = [
        r"\b(i (always|usually|prefer|like|love|hate|never|tend to))\b",
        r"\b(my (background|experience|stack|setup|project|team|company))\b",
        r"\b(i am (a|an|the) \w+)\b",
        r"\b(i work (on|in|with|at|for))\b",
        r"\b(i use \w+)\b",
        r"\b(my preference|my goal|my constraint)\b",
        r"\b(i('m| am) (building|developing|working on|creating))\b",
    ]

    def classify(self, message: str, recent_turns: list[Turn] = None) -> MemoryTier:
     msg_lower = message.lower().strip()


    # ONLY explicit social fillers → SHORT
     for pattern in self._SOCIAL_FILLER_PATTERNS:
        if re.fullmatch(pattern, msg_lower, re.IGNORECASE):
            return MemoryTier.SHORT

    # LONG signals
     for pattern in self._LONG_PATTERNS:
        if re.search(pattern, msg_lower, re.IGNORECASE):
            return MemoryTier.LONG

    # fallback
     return self._llm_classify(message, recent_turns)


    def _llm_classify(self, message: str, recent_turns: list[Turn] = None) -> MemoryTier:
     context = ""
     if recent_turns:
        context = "\n".join(t.to_text() for t in recent_turns[-2:])

     raw = _call_groq(
        CLASSIFIER_PROMPT.format(
            message=message,
            context=context
        ),
        max_tokens=5,
     ).upper().strip()

     if raw.startswith("SHORT"):
        return MemoryTier.SHORT
     elif raw.startswith("LONG"):
        return MemoryTier.LONG
     else:
        return MemoryTier.MID

# ── LAYER 1+3: CONVERSATION MEMORY ───────────────────────────────────────────

class ConversationMemory:
    """
    Manages short-term (buffer) and medium-term (summary) conversation memory.
    Now backed by PostgreSQL — no in-RAM state.

    Layer 1 — Conversation Buffer:
        Fetches the last BUFFER_MAX_TURNS non-expired turns from the DB.
        These are injected verbatim into the prompt.

    Layer 3 — Rolling Summary:
        When active turn count exceeds SUMMARY_THRESHOLD, old turns are
        compressed into a summary stored in memory_summaries. Compressed
        turns are immediately expired in the DB.
    """

    def __init__(
        self,
        user_id          : str,
        db               : PostgresBackend,
        session_id       : Optional[str] = None,
        max_turns        : int = BUFFER_MAX_TURNS,
        summary_threshold: int = SUMMARY_THRESHOLD,
        keep_turns       : int = SUMMARY_KEEP_TURNS,
        classifier       : Optional[MemoryClassifier] = None,
    ) -> None:
        self.user_id           = user_id
        self.session_id        = session_id
        self.db                = db
        self.max_turns         = max_turns
        self.summary_threshold = summary_threshold
        self.keep_turns        = keep_turns
        self.classifier        = classifier or MemoryClassifier()

    # ── PUBLIC ────────────────────────────────────────────────────────────────

    def add_turn(self, role: str, content: str, tier: Optional[MemoryTier] = None) -> None:
        """
        Add one turn to the DB.

        If tier is not provided:
          - user turns: classified automatically
          - assistant turns: inherit MID (responses have session scope)

        After adding, checks if compression is needed.
        """
        if tier is None:
            if role == "user":
                recent = self.db.get_recent_turns(self.user_id, n=2)
                tier = self.classifier.classify(content, recent)
                print(f"[MEMORY] Turn classified as {tier.value.upper()}: {content[:60]!r}")
            else:
                tier = MemoryTier.MID  # assistant responses are session-scoped

        self.db.insert_turn(self.user_id, self.session_id, role, content, tier)


        # Check if we need to compress
        active_count = self.db.count_active_turns(self.user_id)
        if active_count > self.summary_threshold:
            self._compress()

    def get_prompt_context(self) -> str:
        """
        Build the memory section to inject into the LLM prompt.

        Returns formatted string with:
          - Summary of older turns (Layer 3, from DB)
          - Recent turns verbatim (Layer 1, from DB)
        """
        parts = []

        summary = self.db.get_summary(self.user_id)
        if summary:
            parts.append(
                f"[CONVERSATION SUMMARY — earlier exchanges]\n{summary}"
            )

        buffer = self.db.get_recent_turns(self.user_id, n=self.max_turns)
        if buffer:
            recent_text = "\n".join(t.to_text() for t in buffer)
            parts.append(
                f"[RECENT CONVERSATION — last {len(buffer)} turns]\n{recent_text}"
            )

        return "\n\n".join(parts)

    def get_recent_turns(self, n: int = 4) -> list[Turn]:
        return self.db.get_recent_turns(self.user_id, n=n)

    def clear(self) -> None:
        self.db.clear_turns(self.user_id)
        self.db.clear_summary(self.user_id)

    @property
    def summary(self) -> str:
        return self.db.get_summary(self.user_id)

    @property
    def buffer(self) -> list[Turn]:
        return self.db.get_recent_turns(self.user_id, n=self.max_turns)

    @property
    def total_turns(self) -> int:
        return self.db.count_active_turns(self.user_id)

    # ── PRIVATE ───────────────────────────────────────────────────────────────

    def _compress(self) -> None:
        """
        Layer 3 — Compress old turns into the rolling summary.
        Old turns are expired in the DB after summarization.
        """
        to_compress, _ = self.db.get_turns_to_compress(self.user_id, self.keep_turns)

        if not to_compress:
            return

        conv_text = "\n".join(t.to_text() for t in to_compress)
        print(f"[MEMORY] Compressing {len(to_compress)} turns into summary...")

        new_summary = _call_groq(
            SUMMARY_PROMPT.format(conversation=conv_text),
            max_tokens=400,
        )

        if new_summary:
            existing = self.db.get_summary(self.user_id)
            combined = f"{existing}\n\n{new_summary}".strip() if existing else new_summary
            self.db.upsert_summary(self.user_id, combined)
            self.db.expire_compressed_turns(self.user_id, self.keep_turns)
            print(f"[MEMORY] Summary updated ({len(combined)} chars).")
        else:
            print("[MEMORY] Summarization failed — keeping turns in buffer.")


# ── LAYER 2: QUERY REWRITER ───────────────────────────────────────────────────

class QueryRewriter:
    """
    Layer 2 — Rewrites follow-up queries into self-contained questions.
    Unchanged from the original — no DB interaction needed here.
    """

    def rewrite(self, query: str, recent_turns: list[Turn]) -> str:
        if not recent_turns:
            return query

        if not self._needs_rewrite(query):
            return query

        history_text = "\n".join(t.to_text() for t in recent_turns)
        print(f"[MEMORY] Rewriting query: {query!r}")

        rewritten = _call_groq(
            REWRITE_PROMPT.format(history=history_text, query=query),
            max_tokens=40,
        )

        if not rewritten or len(rewritten) < 3:
            print("[MEMORY] Rewrite failed — using original query.")
            return query

        for prefix in ("rewritten question:", "question:", "answer:", "query:"):
            if rewritten.lower().startswith(prefix):
                rewritten = rewritten[len(prefix):].strip()

        if len(rewritten.split()) > max(len(query.split()) * 3, 25):
            print(f"[MEMORY] Rewrite too long — using original query.")
            return query

        print(f"[MEMORY] Rewritten: {rewritten!r}")
        return rewritten

    @staticmethod
    def _needs_rewrite(query: str) -> bool:
        vague_patterns = [
            r"\bit\b", r"\bits\b", r"\bthey\b", r"\bthem\b", r"\btheir\b",
            r"\bthis\b", r"\bthat\b", r"\bthese\b", r"\bthose\b",
            r"\bthe (first|second|third|previous|latter|former|same|other)\b",
            r"\bwhich one\b", r"\bthe one\b", r"\babove\b", r"\bmentioned\b",
        ]
        query_lower = query.lower()
        return any(re.search(p, query_lower) for p in vague_patterns)


# ── LAYER 4: USER MEMORY ──────────────────────────────────────────────────────

class UserMemory:
    """
    Layer 4 — Persists key facts about the user across sessions.
    Now backed by PostgreSQL (user_facts table).
    """

    def __init__(
        self,
        user_id  : str,
        db       : PostgresBackend,
        max_facts: int = USER_MEMORY_MAX,
    ) -> None:
        self.user_id   = user_id
        self.db        = db
        self.max_facts = max_facts

        facts = self.db.get_facts(user_id, limit=max_facts)
        print(f"[MEMORY] User memory loaded: {len(facts)} facts for '{user_id}'")

    def extract_and_store(self, user_message: str) -> list[str]:
        raw = _call_groq(
            FACT_EXTRACTION_PROMPT.format(message=user_message),
            max_tokens=200,
        )
        new_facts = self._parse_facts(raw)

        if new_facts:
            inserted = self.db.insert_facts(self.user_id, new_facts)
            if inserted:
                print(f"[MEMORY] Stored {inserted} new fact(s).")

        return new_facts

    def get_prompt_context(self) -> str:
        facts = self.db.get_facts(self.user_id, limit=self.max_facts)
        if not facts:
            return ""
        facts_text = "\n".join(f"  - {f}" for f in facts)
        return f"[KNOWN USER CONTEXT]\n{facts_text}"

    def all_facts(self) -> list[str]:
        return self.db.get_facts(self.user_id, limit=self.max_facts)

    def clear(self) -> None:
        self.db.clear_facts(self.user_id)
        print(f"[MEMORY] Cleared all facts for user '{self.user_id}'")

    @staticmethod
    def _parse_facts(raw: str) -> list[str]:
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        try:
            facts = json.loads(raw)
            if isinstance(facts, list):
                return [str(f).strip() for f in facts if str(f).strip() and len(str(f)) < 100]
        except json.JSONDecodeError:
            pass
        return []


# ── MEMORY-AWARE PROMPT INJECTION ─────────────────────────────────────────────

def build_memory_context(
    conv_memory : ConversationMemory,
    user_memory : Optional[UserMemory] = None,
) -> str:
    parts = []

    if user_memory:
        user_ctx = user_memory.get_prompt_context()
        if user_ctx:
            parts.append(user_ctx)

    conv_ctx = conv_memory.get_prompt_context()
    if conv_ctx:
        parts.append(conv_ctx)

    return "\n\n".join(parts)


def generate_with_memory(
    query       : str,
    chunks      : list[dict],
    generator,
    conv_memory : ConversationMemory,
    user_memory : Optional[UserMemory] = None,
) -> dict:
    memory_ctx = build_memory_context(conv_memory, user_memory)

    if memory_ctx:
        memory_chunk = {
            "content"     : memory_ctx,
            "score"       : 1.0,
            "retrieval"   : "memory",
            "source"      : "conversation_memory",
            "source_type" : "memory",
            "type"        : "text",
            "section"     : "memory_context",
            "page"        : None,
            "chunk_index" : 0,
            "chunk_total" : None,
            "token_count" : len(memory_ctx) // 4,
        }
        augmented_chunks = [memory_chunk] + chunks
    else:
        augmented_chunks = chunks

    return generator.generate(query, augmented_chunks)

class SessionManager:
    """
    Manages session lifecycle — creation, lookup, and closing.
    Sessions are stored in the sessions table and referenced
    by conversation_turns via session_id FK.
    """

    def __init__(self, db: PostgresBackend) -> None:
        self.db = db

    def create_session(
        self,
        user_id : str,
        title   : Optional[str] = None,
        device  : Optional[str] = None,
    ) -> str:
        """Create a new session and return its session_id."""
        session_id = str(uuid.uuid4())
        conn = self.db.connect()
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO sessions (session_id, user_id, title, device)
                VALUES (%s, %s, %s, %s)
                """,
                (session_id, user_id, title, device)
            )
        conn.commit()
        print(f"[SESSION] Created session {session_id!r} for user {user_id!r}")
        return session_id

    def get_session(self, session_id: str) -> Optional[dict]:
        """Return session metadata or None if not found."""
        conn = self.db.connect()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM sessions WHERE session_id = %s",
                (session_id,)
            )
            row = cur.fetchone()
        return dict(row) if row else None

    def list_sessions(self, user_id: str) -> list[dict]:
        """Return all sessions for a user ordered by most recent."""
        conn = self.db.connect()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT session_id, title, device, created_at, 
                       last_active, is_active
                FROM   sessions
                WHERE  user_id = %s
                ORDER  BY last_active DESC
                """,
                (user_id,)
            )
            rows = cur.fetchall()
        return [dict(r) for r in rows]

    def update_last_active(self, session_id: str) -> None:
        """Call this every turn to keep last_active fresh."""
        conn = self.db.connect()
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE sessions 
                SET last_active = NOW()
                WHERE session_id = %s
                """,
                (session_id,)
            )
        conn.commit()

    def close_session(self, session_id: str) -> None:
        """Mark session as inactive."""
        conn = self.db.connect()
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE sessions 
                SET is_active = FALSE, last_active = NOW()
                WHERE session_id = %s
                """,
                (session_id,)
            )
        conn.commit()
        print(f"[SESSION] Closed session {session_id!r}")

    def update_title(self, session_id: str, title: str) -> None:
        """Update session title — useful after first turn."""
        conn = self.db.connect()
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE sessions SET title=%s WHERE session_id=%s",
                (title, session_id)
            )
        conn.commit()
# ── MAIN MEMORY MANAGER ───────────────────────────────────────────────────────

class MemoryManager:
    """
    Unified interface that wires all 4 memory layers + classifier together,
    backed by PostgreSQL.

    Usage:
        memory = MemoryManager(
            user_id = "user_001",
            db_url  = "postgresql://raguser:pass@localhost:5432/ragdb",
        )

        result = memory.run(
            query     = "what about its limitations?",
            retriever = retriever,
            reranker  = reranker,
            generator = generator,
        )
        print(result["answer"])

    Document permissions (future):
        memory.db.grant_permission("user_001", "doc_abc", "read")
        doc_ids = memory.db.get_user_document_ids("user_001")
        # Then pass doc_ids as a filter to your retriever
    """

    def __init__(
        self,
        user_id            : Optional[str]  = None,
        session_id : Optional[str] = None,
        db_url             : str            = DEFAULT_DB_URL,
        retrieval_pool     : int            = 20,
        top_n              : int            = 5,
        search_mode        : str            = "hybrid",
        buffer_max_turns   : int            = BUFFER_MAX_TURNS,
        summary_threshold  : int            = SUMMARY_THRESHOLD,
        summary_keep_turns : int            = SUMMARY_KEEP_TURNS,
    ) -> None:
        self.user_id        = user_id
        self.retrieval_pool = retrieval_pool
        self.top_n          = top_n
        self.search_mode    = search_mode

        # PostgreSQL backend (shared across all layers)
        self.db = PostgresBackend(db_url)
        self.db.ensure_schema()
        self.session_manager = SessionManager(self.db)
        # Create new session or resume existing
        if session_id:
            self.session_id = session_id
            print(f"[MEMORY] Resuming session {session_id!r}")
        else:
            self.session_id = self.session_manager.create_session(
                user_id = user_id or "anonymous",
                device  = "api",
            )

        if user_id:
            self.db.ensure_user(user_id)

        # Classifier (shared, stateless)
        self.classifier = MemoryClassifier()

        # Layer 1 + 3
        self.conv_memory = ConversationMemory(
            user_id           = user_id or "anonymous",
            db                = self.db,
            session_id        = self.session_id,
            max_turns         = buffer_max_turns,
            summary_threshold = summary_threshold,
            keep_turns        = summary_keep_turns,
            classifier        = self.classifier,
        )

        # Layer 2
        self.rewriter = QueryRewriter()

        # Layer 4
        self.user_memory: Optional[UserMemory] = None
        if user_id:
            self.user_memory = UserMemory(
                user_id   = user_id,
                db        = self.db,
            )

        print(f"\n[MEMORY] MemoryManager ready (PostgreSQL backend).")
        print(f"  user_id            = {user_id or 'anonymous'}")
        print(f"  buffer_max_turns   = {buffer_max_turns}")
        print(f"  summary_threshold  = {summary_threshold}")
        print(f"  user_memory        = {'enabled' if user_id else 'disabled'}")
        print(f"  retrieval_pool     = {retrieval_pool}")
        print(f"  top_n              = {top_n}")
        print(f"  search_mode        = {search_mode}\n")

    # ── MAIN ENTRY POINT ──────────────────────────────────────────────────────

    def run(
        self,
        query   : str,
        retriever,
        reranker,
        generator,
        filters : Optional[dict] = None,
    ) -> dict:
        """
        Full RAG pipeline turn with all memory layers active.

        Steps:
          1. Classify query tier (SHORT / MID / LONG)
          2. Rewrite query using recent conversation history  (Layer 2)
          3. Retrieve candidates using rewritten query
          4. Rerank candidates
          5. Generate answer with memory context              (Layers 1, 3, 4)
          6. Update conversation buffer in DB                 (Layer 1 → triggers 3)
          7. Extract and persist user facts if LONG tier      (Layer 4)
        """
        print(f"\n{'─'*60}")
        print(f"  [MEMORY] Processing: {query!r}")
        print(f"{'─'*60}")

        # ── STEP 1: CLASSIFY ──────────────────────────────────────────────────
        recent_turns = self.conv_memory.get_recent_turns(n=2)
        tier = self.classifier.classify(query, recent_turns)
        print(f"  [MEMORY] Tier: {tier.value.upper()}")

        # ── STEP 2: REWRITE QUERY ─────────────────────────────────────────────
        recent_turns = self.conv_memory.get_recent_turns(n=4)
        rewritten    = self.rewriter.rewrite(query, recent_turns)

        # ── STEP 3: RETRIEVE ──────────────────────────────────────────────────
        print(f"\n[1/3] Retrieving (pool={self.retrieval_pool}, mode={self.search_mode})...")
        print(f"      Query sent to Qdrant: {rewritten!r}")

        # Optionally build a permission filter for the retriever
        retriever_filters = filters or {}
        if self.user_id:
            doc_ids = self.db.get_user_document_ids(self.user_id)
            if doc_ids:
                # Merge with any caller-supplied filters
                retriever_filters["document_ids"] = doc_ids

        candidates = retriever.search(
            rewritten,
            mode    = self.search_mode,
            top_n   = self.retrieval_pool,
            filters = retriever_filters if retriever_filters else None,
        )

        if not candidates:
            print("  [No chunks retrieved]\n")
            self.conv_memory.add_turn("user",      query,     tier=tier)
            self.conv_memory.add_turn("assistant", "I could not find relevant information.",
                                      tier=MemoryTier.SHORT)
            return {
                "answer"          : "I could not find relevant information in the knowledge base.",
                "sources"         : [],
                "used_chunks"     : [],
                "model"           : "N/A",
                "tokens"          : {"prompt": 0, "completion": 0, "total": 0},
                "context_tokens"  : 0,
                "no_answer"       : True,
                "latency_s"       : 0.0,
                "rewritten_query" : rewritten,
                "memory_tier"     : tier.value,
            }

        # ── STEP 4: RERANK ────────────────────────────────────────────────────
        print(f"\n[2/3] Reranking → top {self.top_n}...")
        chunks = reranker.rerank(rewritten, candidates, top_n=self.top_n)

        # ── STEP 5: GENERATE WITH MEMORY ─────────────────────────────────────
        print(f"\n[3/3] Generating with memory context...")
        result = generate_with_memory(
            query       = query,
            chunks      = chunks,
            generator   = generator,
            conv_memory = self.conv_memory,
            user_memory = self.user_memory,
        )

        # ── STEP 6: UPDATE CONVERSATION BUFFER ───────────────────────────────
        # User turn inherits classified tier; assistant response gets SHORT
        # if the query was SHORT (not worth summarizing), MID otherwise.
        assistant_tier = MemoryTier.SHORT if tier == MemoryTier.SHORT else MemoryTier.MID
        self.conv_memory.add_turn("user",      query,             tier=tier)
        self.conv_memory.add_turn("assistant", result["answer"],  tier=assistant_tier)

        # ── STEP 7: EXTRACT USER FACTS (LONG tier only) ───────────────────────
        # Only extract facts from messages the classifier flagged as LONG.
        # This avoids wasting Groq calls on ephemeral chit-chat.
        if self.user_memory and tier == MemoryTier.LONG:
            self.user_memory.extract_and_store(query)

        result["rewritten_query"] = rewritten
        result["memory_tier"]     = tier.value
        return result

    # ── CONVENIENCE METHODS ───────────────────────────────────────────────────

    def rewrite_query(self, query: str) -> str:
        return self.rewriter.rewrite(query, self.conv_memory.get_recent_turns(n=4))

    def classify_query(self, query: str) -> MemoryTier:
        return self.classifier.classify(query)

    def get_prompt_context(self) -> str:
        return build_memory_context(self.conv_memory, self.user_memory)

    def add_turn(self, role: str, content: str, tier: Optional[MemoryTier] = None) -> None:
        self.conv_memory.add_turn(role, content, tier=tier)

    def get_state(self) -> MemoryState:
        return MemoryState(
            summary    = self.conv_memory.summary,
            buffer     = self.conv_memory.buffer,
            user_facts = self.user_memory.all_facts() if self.user_memory else [],
            turn_count = self.conv_memory.total_turns,
        )

    def print_state(self) -> None:
        state = self.get_state()
        print(f"\n{'='*60}")
        print(f"  MEMORY STATE  ({state.turn_count} total active turns)")
        print(f"{'='*60}")

        if state.summary:
            print(f"\n  [Layer 3 — Summary]\n  {state.summary[:300]}{'...' if len(state.summary) > 300 else ''}")
        else:
            print(f"\n  [Layer 3 — Summary] empty")

        print(f"\n  [Layer 1 — Buffer] {len(state.buffer)} turns")
        for t in state.buffer:
            preview = t.content[:80].replace("\n", " ")
            print(f"    [{t.tier.value.upper():5}] {t.role.upper()}: {preview}{'...' if len(t.content) > 80 else ''}")

        if state.user_facts:
            print(f"\n  [Layer 4 — User Facts] {len(state.user_facts)} facts")
            for f in state.user_facts:
                print(f"    - {f}")
        else:
            print(f"\n  [Layer 4 — User Facts] none yet")

        print(f"{'='*60}\n")

    def clear_session(self) -> None:
        """Clear conversation turns + summary. Keep user facts (LONG tier)."""
        self.conv_memory.clear()
        print("[MEMORY] Session cleared. User facts preserved.")

    def clear_all(self) -> None:
        """Clear everything including user facts."""
        self.conv_memory.clear()
        if self.user_memory:
            self.user_memory.clear()
        print("[MEMORY] All memory cleared.")

    def purge_expired(self) -> int:
        """Delete all expired turns from the DB. Run periodically."""
        count = self.db.purge_expired_turns()
        print(f"[MEMORY] Purged {count} expired turns.")
        return count

    def close(self) -> None:
        self.db.close()


# ── INTERACTIVE REPL WITH MEMORY ─────────────────────────────────────────────

def main():
    """
    REPL with full 4-layer memory + PostgreSQL backend + classifier.

    Usage:
      python memory.py
      python memory.py --user_id john --top_n 5 --pool 20 --mode hybrid
      python memory.py --setup   (only creates DB schema, then exits)

    Commands during REPL:
      memory   → print current memory state
      clear    → clear session memory (keep user facts)
      clearall → clear everything
      purge    → delete expired turns from DB
      exit     → quit
    """
    args    = sys.argv[1:]
    user_id = None
    top_n   = 5
    pool    = 20
    mode    = "hybrid"
    setup   = False
    db_url  = DEFAULT_DB_URL

    i = 0
    while i < len(args):
        if args[i] == "--user_id" and i + 1 < len(args):
            user_id = args[i + 1]; i += 2
        elif args[i] == "--top_n" and i + 1 < len(args):
            top_n = int(args[i + 1]); i += 2
        elif args[i] == "--pool" and i + 1 < len(args):
            pool = int(args[i + 1]); i += 2
        elif args[i] == "--mode" and i + 1 < len(args):
            mode = args[i + 1]; i += 2
        elif args[i] == "--db_url" and i + 1 < len(args):
            db_url = args[i + 1]; i += 2
        elif args[i] == "--setup":
            setup = True; i += 1
        else:
            i += 1

    if setup:
        print("[MEMORY] Running schema setup...")
        db = PostgresBackend(db_url)
        db.ensure_schema()
        db.close()
        print("[MEMORY] Schema setup complete. Exiting.")
        return

    print(f"\n{'='*60}")
    print(f"  RAG + MEMORY REPL  (PostgreSQL backend)")
    print(f"{'='*60}")
    print(f"  Loading pipeline (one time)...")

    from retrieve.retriever           import Retriever
    from reranking.reranker           import Reranker
    from answer_generation.generator  import Generator

    retriever = Retriever()
    reranker  = Reranker()
    generator = Generator()

    memory = MemoryManager(
        user_id        = user_id,
        db_url         = db_url,
        retrieval_pool = pool,
        top_n          = top_n,
        search_mode    = mode,
    )

    print(f"\n{'='*60}")
    print(f"  READY. Type your question and press Enter.")
    print(f"  Commands: memory | clear | clearall | purge | exit")
    print(f"{'='*60}\n")

    try:
        while True:
            try:
                question = input("Q> ").strip()
            except EOFError:
                break

            if not question:
                continue

            if question.lower() in {"exit", "quit", ":q"}:
                break

            if question.lower() == "memory":
                memory.print_state()
                continue

            if question.lower() == "clear":
                memory.clear_session()
                continue

            if question.lower() == "clearall":
                memory.clear_all()
                continue

            if question.lower() == "purge":
                memory.purge_expired()
                continue

            try:
                result = memory.run(
                    query     = question,
                    retriever = retriever,
                    reranker  = reranker,
                    generator = generator,
                )

                print(f"\n{'─'*60}")
                print(f"  ANSWER  [tier={result['memory_tier']}]")
                print(f"{'─'*60}")
                print(result["answer"])

                if result.get("rewritten_query") and result["rewritten_query"] != question:
                    print(f"\n  [Query rewritten → {result['rewritten_query']!r}]")

                if result.get("sources"):
                    print(f"\n  Sources cited: {len(result['sources'])}")
                    for idx, src in enumerate(result["sources"], 1):
                        loc = src.get("source", "?")
                        if src.get("page"):
                            loc += f", page {src['page']}"
                        print(f"    [{idx}] {loc}")

                print(f"\n  [latency={result['latency_s']}s | "
                      f"tokens={result['tokens']['total']} | "
                      f"no_answer={result['no_answer']} | "
                      f"tier={result['memory_tier']}]\n")

            except KeyboardInterrupt:
                print("\n[Interrupted — type 'exit' to quit]\n")
                continue
            except Exception as e:
                print(f"\n[ERROR] {e}\n")
                import traceback
                traceback.print_exc()
                continue

    except KeyboardInterrupt:
        pass
    finally:
        retriever.close()
        memory.close()
        print("\n[MEMORY REPL] Goodbye.\n")


if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception as e:
        print(f"\n[FATAL ERROR] {type(e).__name__}: {e}")
        traceback.print_exc()
        input("\nPress Enter to exit...")