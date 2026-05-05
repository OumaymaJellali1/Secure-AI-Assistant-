"""
memory/user_facts.py — Layer 4: persistent facts about a user.

Stored across all sessions. Updated when fact-worthy info is detected.
Capped at USER_FACTS_MAX entries; oldest are dropped when over the cap.
"""
import json
from typing import Optional
from sqlalchemy import text

from shared.db import engine
from memory.llm import call_groq, FACT_EXTRACTION_PROMPT


# ── CONFIG ────────────────────────────────────────────────────────
USER_FACTS_MAX = 15      # hard cap on stored facts per user


class UserMemory:
    """
    Persistent facts about a user, stored in user_facts table.
    One UserMemory instance per user.
    """

    def __init__(self, user_id: str, max_facts: int = USER_FACTS_MAX):
        self.user_id = user_id
        self.max_facts = max_facts

    # ── PUBLIC API ──────────────────────────────────────────────────

    def extract_and_store(self, message: str) -> list[str]:
        """
        Extract facts from a message, store new unique ones.
        Returns list of newly stored facts.
        """
        raw = call_groq(
            FACT_EXTRACTION_PROMPT.format(message=message),
            max_tokens=200,
        )
        new_facts = self._parse_facts(raw)

        if not new_facts:
            return []

        return self._store_facts(new_facts)

    def add_facts(self, facts: list[str]) -> list[str]:
        """Manually add facts (e.g., from summary promotion)."""
        return self._store_facts(facts)

    def get_facts(self) -> list[str]:
        """Return all current facts (most recent first)."""
        with engine.connect() as conn:
            rows = conn.execute(
                text("""
                    SELECT fact FROM user_facts
                    WHERE user_id = :uid
                    ORDER BY created_at DESC
                    LIMIT :lim
                """),
                {"uid": self.user_id, "lim": self.max_facts},
            ).fetchall()
        return [r[0] for r in rows]

    def get_prompt_context(self) -> str:
        """Format facts for prompt injection. Empty string if no facts."""
        facts = self.get_facts()
        if not facts:
            return ""
        bullet_list = "\n".join(f"  - {f}" for f in facts)
        return f"[KNOWN USER CONTEXT]\n{bullet_list}"

    def clear(self) -> None:
        """Delete all facts for this user."""
        with engine.begin() as conn:
            conn.execute(
                text("DELETE FROM user_facts WHERE user_id = :uid"),
                {"uid": self.user_id},
            )

    def count(self) -> int:
        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT COUNT(*) FROM user_facts WHERE user_id = :uid"),
                {"uid": self.user_id},
            ).fetchone()
        return row[0] if row else 0

    # ── INTERNAL ────────────────────────────────────────────────────

    def _store_facts(self, facts: list[str]) -> list[str]:
        """Insert new facts, dedupe, enforce cap. Returns inserted ones."""
        if not facts:
            return []

        inserted = []
        with engine.begin() as conn:
            for fact in facts:
                fact = str(fact).strip()
                if not fact or len(fact) > 200:
                    continue

                result = conn.execute(
                    text("""
                        INSERT INTO user_facts (user_id, fact)
                        VALUES (:uid, :f)
                        ON CONFLICT (user_id, fact) DO NOTHING
                        RETURNING fact
                    """),
                    {"uid": self.user_id, "f": fact},
                )
                row = result.fetchone()
                if row:
                    inserted.append(row[0])

            # Enforce cap: keep most recent max_facts, delete older
            conn.execute(
                text("""
                    DELETE FROM user_facts
                    WHERE user_id = :uid
                      AND id NOT IN (
                          SELECT id FROM user_facts
                          WHERE user_id = :uid
                          ORDER BY created_at DESC
                          LIMIT :lim
                      )
                """),
                {"uid": self.user_id, "lim": self.max_facts},
            )

        if inserted:
            print(f"[MEMORY] Stored {len(inserted)} new fact(s) for {self.user_id}")

        return inserted

    @staticmethod
    def _parse_facts(raw: str) -> list[str]:
        """Parse LLM output into list of fact strings. Tolerates markdown."""
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        try:
            facts = json.loads(raw)
            if isinstance(facts, list):
                return [
                    str(f).strip()
                    for f in facts
                    if str(f).strip() and len(str(f)) < 200
                ]
        except json.JSONDecodeError:
            pass
        return []