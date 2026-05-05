"""
prompt_builder.py
═════════════════
Builds structured prompts for RAG answer generation with Qwen2.5.

Combines best of both worlds:
  - Detailed, comprehensive answers (no "be concise")
  - Clear inline citations: [Source: filename, page X]
  - Smart token budget management
  - Per-type chunk formatting (text / table / image)
  - Generous context window for richer answers
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


# ── CONFIG ────────────────────────────────────────────────────────────────────

# Rough character-to-token ratio for Qwen2.5
CHARS_PER_TOKEN: float = 3.8

# Token budgets (generous for detailed answers)
SYSTEM_PROMPT_TOKEN_BUDGET : int = 350
QUERY_TOKEN_BUDGET         : int = 100
ANSWER_HEADROOM            : int = 2000     # was 512 — allow long answers
CONTEXT_TOKEN_BUDGET       : int = 8000     # was 3500 — use more chunks

# Chunk type labels
CHUNK_TYPE_LABELS: dict[str, str] = {
    "text"  : "",
    "table" : "[TABLE] ",
    "image" : "[IMAGE CAPTION] ",
}


# ── SYSTEM PROMPT ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a precise, comprehensive question-answering assistant.
You are given numbered context excerpts retrieved from a document base.
Your task is to answer the user's question using ONLY the information in those excerpts.

STRICT RULES:
1. Ground every claim in the provided context. Do NOT use prior knowledge.
2. Cite sources inline using this exact format: [Source: filename, page X]
   - For multiple sources: [Source: file1.pdf, page 3] [Source: file2.pdf, page 5]
   - For chunks without page: [Source: filename]
3. Every factual claim must have a citation immediately after it.
4. If the context does not contain enough information, respond EXACTLY:
   "I don't have enough information in the provided context to answer this question."
5. Be thorough and detailed. Cover all relevant aspects from the context.
   Explain concepts fully, include examples, and provide depth.
6. Avoid filler phrases like "Based on the context..." — go straight to the answer.
7. Preserve exact numbers, names, and technical terms from the context.
8. If excerpts contradict each other, note the contradiction explicitly.
9. Preserve the original language of the question in your answer.
10. Structure long answers with paragraphs covering different aspects."""


# ── DATACLASS ─────────────────────────────────────────────────────────────────

@dataclass
class BuiltPrompt:
    """Output of PromptBuilder.build()."""
    messages             : list[dict]
    used_chunks          : list[dict]
    total_context_tokens : int


# ── PROMPT BUILDER ────────────────────────────────────────────────────────────

class PromptBuilder:
    """
    Constructs token-budget-aware ChatML prompts for Qwen2.5 RAG.

    Args:
        context_budget  : max tokens to spend on injected chunks
        answer_headroom : tokens reserved for model output
    """

    def __init__(
        self,
        context_budget  : int = CONTEXT_TOKEN_BUDGET,
        answer_headroom : int = ANSWER_HEADROOM,
    ) -> None:
        self.context_budget  = context_budget
        self.answer_headroom = answer_headroom

    # ── PUBLIC ────────────────────────────────────────────────────────────────

    def build(
        self,
        query  : str,
        chunks : list[dict],
    ) -> BuiltPrompt:
        """
        Build a ChatML message list with token budget management.

        Args:
            query  : raw user question
            chunks : list from Retriever.search()

        Returns:
            BuiltPrompt with .messages, .used_chunks, .total_context_tokens
        """
        if not query or not query.strip():
            raise ValueError("Query must be a non-empty string.")
        if not chunks:
            raise ValueError("Chunks list is empty.")

        # 1. Select chunks within budget
        selected, total_ctx_tokens = self._select_chunks(chunks)

        if not selected:
            # Always include at least the top chunk
            selected = [chunks[0]]
            total_ctx_tokens = self._estimate_tokens(chunks[0]["content"])

        # 2. Build context block
        context_block = self._build_context_block(selected)

        # 3. Build user message
        user_content = self._build_user_message(query, context_block)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ]

        return BuiltPrompt(
            messages             = messages,
            used_chunks          = selected,
            total_context_tokens = total_ctx_tokens,
        )

    # ── CHUNK SELECTION ───────────────────────────────────────────────────────

    def _select_chunks(self, chunks: list[dict]) -> tuple[list[dict], int]:
        """Greedy token-budget selection."""
        selected    : list[dict] = []
        tokens_used : int        = 0

        for chunk in chunks:
            chunk_tokens = self._get_chunk_tokens(chunk)

            if tokens_used + chunk_tokens > self.context_budget:
                continue   # try next chunk (might be smaller)

            selected.append(chunk)
            tokens_used += chunk_tokens

        return selected, tokens_used

    # ── CONTEXT BLOCK ─────────────────────────────────────────────────────────

    def _build_context_block(self, chunks: list[dict]) -> str:
        """
        Format chunks as numbered context excerpts.

        Format per chunk:
          [N] (filename: X | page: Y | section: Z | via: dense/sparse/both)
          <type_label><content>
        """
        lines: list[str] = ["--- CONTEXT EXCERPTS ---\n"]

        for idx, chunk in enumerate(chunks, start=1):
            header  = self._build_chunk_header(idx, chunk)
            label   = CHUNK_TYPE_LABELS.get(chunk.get("type", "text") or "text", "")
            content = chunk.get("content", "").strip()

            lines.append(header)
            lines.append(f"{label}{content}")
            lines.append("")

        lines.append("--- END OF CONTEXT ---")
        return "\n".join(lines)

    def _build_chunk_header(self, idx: int, chunk: dict) -> str:
        """Build metadata header for a chunk."""
        parts: list[str] = []

        if chunk.get("source"):
            parts.append(f"filename: {chunk['source']}")
        if chunk.get("page") is not None:
            parts.append(f"page: {chunk['page']}")
        if chunk.get("section"):
            parts.append(f"section: {chunk['section']}")
        if chunk.get("subject"):
            parts.append(f"email subject: {chunk['subject']}")
        if chunk.get("retrieval"):
            parts.append(f"via: {chunk['retrieval']}")

        meta = f" ({' | '.join(parts)})" if parts else ""
        return f"[{idx}]{meta}"

    # ── USER MESSAGE ──────────────────────────────────────────────────────────

    def _build_user_message(self, query: str, context_block: str) -> str:
        """Combine context + query into user turn."""
        return (
            f"{context_block}\n\n"
            f"Question: {query.strip()}\n\n"
            f"Provide a thorough, well-cited answer using the format "
            f"[Source: filename, page X] for citations:"
        )

    # ── TOKEN ESTIMATION ──────────────────────────────────────────────────────

    def _get_chunk_tokens(self, chunk: dict) -> int:
        """Get chunk token count (from metadata or estimate)."""
        if chunk.get("token_count") and isinstance(chunk["token_count"], int):
            return chunk["token_count"]
        return self._estimate_tokens(chunk.get("content", ""))

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough token estimate."""
        return max(1, int(len(text) / CHARS_PER_TOKEN))