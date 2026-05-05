"""
memory/rewriter.py — Layer 2: query rewriting.

Expands vague follow-ups ("what about its limitations?") into
self-contained queries before retrieval. Skips the LLM call when
the query is obviously self-contained (no pronouns/references).
"""
import re
from typing import Optional

from memory.llm import call_groq, REWRITE_PROMPT


class QueryRewriter:
    """
    Rewrites vague queries using recent conversation context.
    Smart pre-check avoids LLM calls on obviously self-contained queries.
    """

    # Patterns that indicate a query MIGHT need rewriting
    _VAGUE_PATTERNS = [
        # Pronouns
        r"\b(it|its|they|them|their|theirs)\b",
        r"\b(this|that|these|those)\b",
        # Relative references
        r"\bthe (first|second|third|previous|latter|former|same|other)\b",
        r"\b(which one|the one)\b",
        r"\b(above|mentioned|just said)\b",
        # Short conversational follow-ups
        r"^(and|but|so|or|also)\s+\w+",
        r"^(what about|how about)\b",
    ]

    def rewrite(self, query: str, recent_turns: list) -> str:
        """
        Rewrite query if it has vague references AND there's history.
        Returns original query if no rewrite needed or LLM fails.
        """
        # No history → can't rewrite anyway
        if not recent_turns:
            return query

        # Quick check: does this query even need rewriting?
        if not self._needs_rewrite(query):
            return query

        # Build history text
        history_text = "\n".join(
            f"{t.role.upper()}: {t.content}" for t in recent_turns
        )

        # Call LLM
        rewritten = call_groq(
            REWRITE_PROMPT.format(history=history_text, query=query),
            max_tokens=60,
        )

        # Validate result — fall back to original on any issue
        if not rewritten or len(rewritten) < 3:
            return query

        # Strip common LLM preambles
        for prefix in ("rewritten question:", "question:", "answer:", "query:"):
            if rewritten.lower().startswith(prefix):
                rewritten = rewritten[len(prefix):].strip()

        # Sanity check: rewrite shouldn't be wildly longer than original
        if len(rewritten.split()) > max(len(query.split()) * 3, 25):
            return query

        if rewritten != query:
            print(f"[REWRITE] {query!r} → {rewritten!r}")

        return rewritten

    @classmethod
    def _needs_rewrite(cls, query: str) -> bool:
        """True if query has vague references that might need expansion."""
        q = query.lower()
        return any(re.search(p, q) for p in cls._VAGUE_PATTERNS)