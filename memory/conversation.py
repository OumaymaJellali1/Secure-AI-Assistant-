"""
memory/conversation.py — Layers 1 + 3.

Layer 1: Buffer
  • Recent turns stored as individual rows in conversation_turns
  • Atomic inserts (no read-modify-write race conditions)
  • Last N turns fetched on demand from DB

Layer 3: Summary
  • One row per session in session_summaries
  • Updated when buffer exceeds SUMMARY_THRESHOLD
  • Capped at SUMMARY_MAX_WORDS via re-summarization
  • Old turns are deleted from buffer after compression (saved in summary)
"""
from dataclasses import dataclass
from typing import Optional
from sqlalchemy import text

from shared.db import engine
from memory.llm import call_groq, SUMMARY_PROMPT, CONDENSE_PROMPT


# ── CONFIG ────────────────────────────────────────────────────────
BUFFER_MAX_TURNS    = 8       # max turns kept verbatim in buffer
SUMMARY_THRESHOLD   = 6       # compress when buffer exceeds this
SUMMARY_KEEP_TURNS  = 3       # keep this many recent turns after compression
SUMMARY_MAX_WORDS   = 500     # hard cap on summary length


@dataclass
class Turn:
    role: str           # "user" or "assistant"
    content: str
    created_at: Optional[str] = None

    def to_text(self) -> str:
        return f"{self.role.upper()}: {self.content}"

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


class ConversationMemory:
    """
    Per-session buffer + summary, both stored in Postgres.
    Use one instance per active conversation.
    """

    def __init__(
        self,
        session_id: str,
        user_id: str,
        max_turns: int = BUFFER_MAX_TURNS,
        summary_threshold: int = SUMMARY_THRESHOLD,
        keep_turns: int = SUMMARY_KEEP_TURNS,
        max_summary_words: int = SUMMARY_MAX_WORDS,
    ):
        self.session_id = session_id
        self.user_id = user_id
        self.max_turns = max_turns
        self.summary_threshold = summary_threshold
        self.keep_turns = keep_turns
        self.max_summary_words = max_summary_words

    # ── PUBLIC API ──────────────────────────────────────────────────

    def add_turn(self, role: str, content: str) -> None:
        """Add a turn. Triggers compression if buffer exceeds threshold."""
        if role not in ("user", "assistant"):
            raise ValueError(f"role must be 'user' or 'assistant', got {role!r}")

        # Atomic insert
        with engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO conversation_turns (session_id, user_id, role, content)
                    VALUES (:sid, :uid, :role, :content)
                """),
                {
                    "sid":     self.session_id,
                    "uid":     self.user_id,
                    "role":    role,
                    "content": content,
                },
            )
            # Update session activity
            conn.execute(
                text("UPDATE sessions SET last_active = NOW() WHERE session_id = :sid"),
                {"sid": self.session_id},
            )

        # Check if compression is needed
        if self._buffer_size() > self.summary_threshold:
            self._compress()

    def get_buffer(self, n: Optional[int] = None) -> list[Turn]:
        """Return the last N turns as Turn objects (chronological order)."""
        limit = n if n is not None else self.max_turns
        with engine.connect() as conn:
            rows = conn.execute(
                text("""
                    SELECT role, content, created_at
                    FROM conversation_turns
                    WHERE session_id = :sid
                    ORDER BY created_at DESC
                    LIMIT :n
                """),
                {"sid": self.session_id, "n": limit},
            ).fetchall()
        # Reverse to chronological order (oldest first)
        return [
            Turn(role=r[0], content=r[1], created_at=str(r[2]))
            for r in reversed(rows)
        ]

    def get_summary(self) -> str:
        """Return the rolling summary for this session."""
        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT summary FROM session_summaries WHERE session_id = :sid"),
                {"sid": self.session_id},
            ).fetchone()
        return row[0] if row else ""

    def get_context(self) -> dict:
        """Return summary + buffer in one DB-efficient call."""
        return {
            "summary": self.get_summary(),
            "buffer":  self.get_buffer(),
        }

    def get_prompt_context(self) -> str:
        """Format summary + buffer for prompt injection."""
        ctx = self.get_context()
        parts = []

        if ctx["summary"]:
            parts.append(f"[CONVERSATION SUMMARY]\n{ctx['summary']}")

        if ctx["buffer"]:
            recent = "\n".join(t.to_text() for t in ctx["buffer"])
            parts.append(f"[RECENT CONVERSATION — last {len(ctx['buffer'])} turns]\n{recent}")

        return "\n\n".join(parts)

    def clear(self) -> None:
        """Wipe this session's buffer and summary (keeps the session row)."""
        with engine.begin() as conn:
            conn.execute(
                text("DELETE FROM conversation_turns WHERE session_id = :sid"),
                {"sid": self.session_id},
            )
            conn.execute(
                text("DELETE FROM session_summaries WHERE session_id = :sid"),
                {"sid": self.session_id},
            )

    # ── INTERNAL ────────────────────────────────────────────────────

    def _buffer_size(self) -> int:
        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT COUNT(*) FROM conversation_turns WHERE session_id = :sid"),
                {"sid": self.session_id},
            ).fetchone()
        return row[0] if row else 0

    def _compress(self) -> None:
        """
        Compress oldest turns into summary.

        Steps:
          1. Take all buffer turns except last keep_turns
          2. Summarize them with LLM
          3. Merge into existing summary
          4. If summary > MAX_WORDS → re-summarize to condense
          5. Delete the compressed turns from buffer
        """
        # Identify which turns to compress
        with engine.connect() as conn:
            rows = conn.execute(
                text("""
                    SELECT id, role, content
                    FROM conversation_turns
                    WHERE session_id = :sid
                    ORDER BY created_at ASC
                """),
                {"sid": self.session_id},
            ).fetchall()

        if len(rows) <= self.keep_turns:
            return  # nothing to compress

        to_compress = rows[:-self.keep_turns]
        compress_ids = [r[0] for r in to_compress]
        conv_text = "\n".join(f"{r[1].upper()}: {r[2]}" for r in to_compress)

        # Get existing summary
        existing = self.get_summary()

        # Summarize
        print(f"[MEMORY] Compressing {len(to_compress)} turns...")
        new_summary = call_groq(
            SUMMARY_PROMPT.format(
                existing_summary=existing or "(none yet)",
                conversation=conv_text,
            ),
            max_tokens=600,
        )

        # FALLBACK: if summarization fails, keep turns in buffer (don't lose data)
        if not new_summary:
            print("[MEMORY] Summarization failed — keeping turns in buffer.")
            return

        # Cap summary growth via re-summarization
        if len(new_summary.split()) > self.max_summary_words:
            print(f"[MEMORY] Summary over {self.max_summary_words} words — condensing...")
            condensed = call_groq(
                CONDENSE_PROMPT.format(
                    max_words=self.max_summary_words,
                    summary=new_summary,
                ),
                max_tokens=700,
            )
            if condensed:
                new_summary = condensed

        # Save summary + delete compressed turns (atomic)
        with engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO session_summaries (session_id, summary, updated_at)
                    VALUES (:sid, :sum, NOW())
                    ON CONFLICT (session_id) DO UPDATE
                    SET summary = EXCLUDED.summary, updated_at = NOW()
                """),
                {"sid": self.session_id, "sum": new_summary},
            )
            conn.execute(
                text("DELETE FROM conversation_turns WHERE id = ANY(:ids)"),
                {"ids": compress_ids},
            )

        print(f"[MEMORY] Summary updated ({len(new_summary)} chars).")