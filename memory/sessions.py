"""
memory/sessions.py — Manages conversation lifecycle.

Conversations 😊 sessions) are first-class entities. Users can:
  • List their conversations
  • Resume any past conversation
  • Auto-title from first message
  • Archive (mark inactive)
  • Delete (cascades to all related data)
"""
import uuid
from datetime import datetime
from typing import Optional
from sqlalchemy import text

from shared.db import engine
from memory.llm import call_groq, TITLE_PROMPT


class SessionManager:
    """
    Manages conversation (session) lifecycle.
    A session is one continuous conversation thread for a user.
    """

    # ── CREATION & RESUMPTION ───────────────────────────────────────

    def create(self, user_id: str, title: Optional[str] = None) -> str:
        """Create a new conversation. Returns session_id."""
        session_id = f"sess_{uuid.uuid4().hex[:12]}"
        with engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO sessions (session_id, user_id, title)
                    VALUES (:sid, :uid, :title)
                """),
                {"sid": session_id, "uid": user_id, "title": title},
            )
        print(f"[SESSION] Created '{session_id}' for user '{user_id}'")
        return session_id

    def get(self, session_id: str) -> Optional[dict]:
        """Return session metadata or None."""
        with engine.connect() as conn:
            row = conn.execute(
                text("""
                    SELECT session_id, user_id, title, is_active,
                           created_at, last_active
                    FROM sessions WHERE session_id = :sid
                """),
                {"sid": session_id},
            ).fetchone()
        return dict(row._mapping) if row else None

    def list_for_user(
        self,
        user_id: str,
        active_only: bool = True,
        limit: int = 50,
    ) -> list[dict]:
        """Return user's conversations, most recent first."""
        sql = """
            SELECT session_id, title, is_active, created_at, last_active,
                   (SELECT COUNT(*) FROM conversation_turns t
                    WHERE t.session_id = s.session_id) AS turn_count
            FROM sessions s
            WHERE user_id = :uid
        """
        if active_only:
            sql += " AND is_active = TRUE"
        sql += " ORDER BY last_active DESC LIMIT :lim"

        with engine.connect() as conn:
            rows = conn.execute(text(sql), {"uid": user_id, "lim": limit}).fetchall()
        return [dict(r._mapping) for r in rows]

    # ── UPDATES ─────────────────────────────────────────────────────

    def touch(self, session_id: str) -> None:
        """Update last_active to NOW. Call on every turn."""
        with engine.begin() as conn:
            conn.execute(
                text("UPDATE sessions SET last_active = NOW() WHERE session_id = :sid"),
                {"sid": session_id},
            )

    def set_title(self, session_id: str, title: str) -> None:
        """Manually set or rename a conversation."""
        with engine.begin() as conn:
            conn.execute(
                text("UPDATE sessions SET title = :t WHERE session_id = :sid"),
                {"t": title, "sid": session_id},
            )

    def auto_title(self, session_id: str, first_message: str) -> str:
        """Generate a title from the first user message via LLM."""
        title = call_groq(
            TITLE_PROMPT.format(message=first_message),
            max_tokens=20,
        )
        # Strip quotes/punctuation the LLM sometimes adds
        title = title.strip().strip('"\'').strip()
        if not title:
            title = first_message[:50] + ("..." if len(first_message) > 50 else "")
        self.set_title(session_id, title)
        return title

    def archive(self, session_id: str) -> None:
        """Mark session inactive. It still exists but won't show by default."""
        with engine.begin() as conn:
            conn.execute(
                text("UPDATE sessions SET is_active = FALSE WHERE session_id = :sid"),
                {"sid": session_id},
            )
        print(f"[SESSION] Archived '{session_id}'")

    def delete(self, session_id: str) -> None:
        """Permanently delete session + all related turns + summary."""
        with engine.begin() as conn:
            conn.execute(
                text("DELETE FROM sessions WHERE session_id = :sid"),
                {"sid": session_id},
            )
        print(f"[SESSION] Deleted '{session_id}'")

    # ── INTROSPECTION ───────────────────────────────────────────────

    def turn_count(self, session_id: str) -> int:
        """How many turns are in this conversation."""
        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT COUNT(*) FROM conversation_turns WHERE session_id = :sid"),
                {"sid": session_id},
            ).fetchone()
        return row[0] if row else 0