"""
memory.py
─────────
4-layer memory system for RAG — plugs directly into your existing pipeline.

Layers:
  1. Conversation Buffer   — last N full turns kept verbatim in the prompt
  2. Query Rewriting       — rewrites vague follow-up queries before Qdrant search
  3. Conversation Summary  — compresses old turns into a rolling summary
  4. User Memory           — persists key facts about the user across sessions

Full pipeline with memory:
  query
    → Layer 2: rewrite query using conversation history
    → retriever.search(rewritten_query)
    → reranker.rerank(rewritten_query, candidates)
    → Layer 1+3: inject buffer + summary into prompt context
    → generator.generate_with_memory(query, chunks, memory_context)
    → Layer 4: extract and persist user facts
    → update conversation buffer (auto-triggers Layer 3 if needed)

Usage:
  from memory import MemoryManager

  memory = MemoryManager(user_id="user_001")

  # Every turn:
  result = memory.run(
      query     = "what about its limitations?",
      retriever = retriever,
      reranker  = reranker,
      generator = generator,
  )
  print(result["answer"])

  # Or use individual layers:
  rewritten = memory.rewrite_query("what about its limitations?")
  context   = memory.get_prompt_context()
  memory.add_turn("user", "what about its limitations?")
  memory.add_turn("assistant", result["answer"])
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import random
from dataclasses import dataclass, field
from typing import Optional

# ── PATH SETUP ────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from groq import Groq

# ── CONFIG ────────────────────────────────────────────────────────────────────

# Model used for rewriting + summarization (fast small model — not the judge)
MEMORY_MODEL       = "llama-3.1-8b-instant"

# Layer 1 — how many recent turns to keep verbatim in the prompt
BUFFER_MAX_TURNS   = 8       # 4 user + 4 assistant exchanges

# Layer 3 — compress when buffer exceeds this count
SUMMARY_THRESHOLD  = 6       # compress when we have 6+ turns
SUMMARY_KEEP_TURNS = 3       # keep 3 most recent turns after compression

# Layer 4 — max facts to inject per session (older ones dropped)
USER_MEMORY_MAX    = 20

# Where user memory JSON files are persisted
USER_MEMORY_DIR    = "user_memories"

# Groq retry config
MAX_RETRIES  = 3
RETRY_BASE_S = 1.0
RETRY_JITTER = 0.3

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


# ── PROMPTS ───────────────────────────────────────────────────────────────────

# FIX 1 — Added "Maximum 20 words. A single concise question only." constraint
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


# ── DATA CLASSES ──────────────────────────────────────────────────────────────

@dataclass
class Turn:
    role    : str   # "user" or "assistant"
    content : str

    def to_text(self) -> str:
        return f"{self.role.upper()}: {self.content}"


@dataclass
class MemoryState:
    """Full snapshot of memory at any point — useful for debugging."""
    summary      : str
    buffer       : list[Turn]
    user_facts   : list[str]
    turn_count   : int


# ── GROQ HELPER ───────────────────────────────────────────────────────────────

def _call_groq(prompt: str, max_tokens: int = 300) -> str:
    """
    Call Groq with retry + exponential backoff.
    Returns the raw text response.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = groq_client.chat.completions.create(
                model     = MEMORY_MODEL,
                messages  = [{"role": "user", "content": prompt}],
                max_tokens= max_tokens,
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


# ── LAYER 1+3: CONVERSATION MEMORY ───────────────────────────────────────────

class ConversationMemory:
    """
    Manages short-term (buffer) and medium-term (summary) conversation memory.

    Layer 1 — Conversation Buffer:
        Keeps the last BUFFER_MAX_TURNS turns verbatim. These are injected
        directly into the prompt so the LLM sees exact recent exchanges.

    Layer 3 — Rolling Summary:
        When the buffer exceeds SUMMARY_THRESHOLD turns, the oldest half is
        compressed into a summary paragraph. New turns are added to a fresh
        buffer. The summary grows incrementally — it is never discarded,
        only extended.

    This means the prompt always contains:
        [summary of everything before the recent buffer]
        [last N turns verbatim]
    """

    def __init__(
        self,
        max_turns        : int = BUFFER_MAX_TURNS,
        summary_threshold: int = SUMMARY_THRESHOLD,
        keep_turns       : int = SUMMARY_KEEP_TURNS,
    ) -> None:
        self.max_turns         = max_turns
        self.summary_threshold = summary_threshold
        self.keep_turns        = keep_turns

        self._buffer  : list[Turn] = []
        self._summary : str        = ""
        self._total_turns : int    = 0

    # ── PUBLIC ────────────────────────────────────────────────────────────────

    def add_turn(self, role: str, content: str) -> None:
        """
        Add one turn to the buffer. Automatically compresses if threshold hit.

        Args:
            role    : "user" or "assistant"
            content : turn text
        """
        self._buffer.append(Turn(role=role, content=content))
        self._total_turns += 1

        if len(self._buffer) > self.summary_threshold:
            self._compress()

    def get_prompt_context(self) -> str:
        """
        Build the memory section to inject into the LLM prompt.

        Returns a formatted string with:
          - Summary of older turns (if any)
          - Recent turns verbatim (buffer)

        Returns empty string if no memory yet.
        """
        parts = []

        if self._summary:
            parts.append(
                f"[CONVERSATION SUMMARY — earlier exchanges]\n{self._summary}"
            )

        if self._buffer:
            recent_text = "\n".join(t.to_text() for t in self._buffer)
            parts.append(
                f"[RECENT CONVERSATION — last {len(self._buffer)} turns]\n{recent_text}"
            )

        return "\n\n".join(parts)

    def get_recent_turns(self, n: int = 4) -> list[Turn]:
        """Return the N most recent turns from the buffer."""
        return self._buffer[-n:]

    def clear(self) -> None:
        """Reset all memory (start of a new session)."""
        self._buffer      = []
        self._summary     = ""
        self._total_turns = 0

    @property
    def total_turns(self) -> int:
        return self._total_turns

    @property
    def summary(self) -> str:
        return self._summary

    @property
    def buffer(self) -> list[Turn]:
        return list(self._buffer)

    # ── PRIVATE ───────────────────────────────────────────────────────────────

    def _compress(self) -> None:
        """
        Layer 3 — Compress oldest turns into the rolling summary.

        Takes everything except the last `keep_turns` turns, summarizes them,
        and appends the summary to the existing one. Keeps only the most
        recent turns in the buffer.
        """
        to_compress   = self._buffer[: -self.keep_turns]
        self._buffer  = self._buffer[-self.keep_turns :]

        if not to_compress:
            return

        conv_text = "\n".join(t.to_text() for t in to_compress)

        print(f"[MEMORY] Compressing {len(to_compress)} turns into summary...")
        new_summary = _call_groq(
            SUMMARY_PROMPT.format(conversation=conv_text),
            max_tokens=400,
        )

        if new_summary:
            if self._summary:
                # Append to existing summary
                self._summary = f"{self._summary}\n\n{new_summary}"
            else:
                self._summary = new_summary
            print(f"[MEMORY] Summary updated ({len(self._summary)} chars).")
        else:
            print("[MEMORY] Summarization failed — keeping turns in buffer.")
            # Fallback: keep the turns we tried to compress
            self._buffer = to_compress + self._buffer


# ── LAYER 2: QUERY REWRITER ───────────────────────────────────────────────────

class QueryRewriter:
    """
    Layer 2 — Rewrites follow-up queries into self-contained questions.

    This is the highest-leverage memory component. Without it, vague
    follow-ups like "what about its limitations?" hit Qdrant and retrieve
    completely wrong chunks because the vector store has no conversation context.

    The rewriter uses a fast small model (llama-3.1-8b-instant) to expand
    pronouns and references using the recent conversation history.

    Examples:
        History: user asked about "Naive RAG"
        Query:   "what are its limitations?"
        Rewrite: "what are the limitations of Naive RAG?"

        History: user asked about "BGE-M3 and ColBERT"
        Query:   "which one is faster?"
        Rewrite: "which is faster, BGE-M3 or ColBERT?"
    """

    def rewrite(self, query: str, recent_turns: list[Turn]) -> str:
        """
        Rewrite query to be self-contained using recent conversation turns.

        Args:
            query        : the raw user query
            recent_turns : recent Turn objects for context (4-6 is enough)

        Returns:
            Rewritten query string. Falls back to original if rewriting fails.
        """
        # No history → nothing to rewrite
        if not recent_turns:
            return query

        # Check if the query likely needs rewriting
        # (contains pronouns or vague references)
        if not self._needs_rewrite(query):
            return query

        history_text = "\n".join(t.to_text() for t in recent_turns)

        print(f"[MEMORY] Rewriting query: {query!r}")

        # FIX 3 — Reduced max_tokens from 150 to 40 (a question never needs more)
        rewritten = _call_groq(
            REWRITE_PROMPT.format(history=history_text, query=query),
            max_tokens=40,
        )

        if not rewritten or len(rewritten) < 3:
            print("[MEMORY] Rewrite failed — using original query.")
            return query

        # FIX 4 — Strip common LLM preamble patterns
        for prefix in ("rewritten question:", "question:", "answer:", "query:"):
            if rewritten.lower().startswith(prefix):
                rewritten = rewritten[len(prefix):].strip()

        # FIX 2 — Word-count based guard (replaces the old character * 4 check)
        # Allows up to 25 words max, or 3x the original word count — whichever is larger
        if len(rewritten.split()) > max(len(query.split()) * 3, 25):
            print(f"[MEMORY] Rewrite too long ({len(rewritten.split())} words) — using original query.")
            return query

        print(f"[MEMORY] Rewritten: {rewritten!r}")
        return rewritten

    @staticmethod
    def _needs_rewrite(query: str) -> bool:
        """
        Heuristic check: does the query contain references that need expansion?
        Avoids unnecessary Groq calls for self-contained questions.
        """
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

    After each user turn, the LLM extracts memorable facts (preferences,
    goals, background, constraints) and stores them in a JSON file keyed
    by user_id. On the next session, these facts are injected into the
    prompt so the model can personalize responses without the user needing
    to re-explain themselves.

    Storage: {USER_MEMORY_DIR}/{user_id}.json

    Example stored facts:
      ["uses Qdrant as vector DB",
       "RAG pipeline uses BGE-M3 for embeddings",
       "prefers detailed technical explanations",
       "working on a document QA system in Python"]
    """

    def __init__(
        self,
        user_id    : str,
        memory_dir : str = USER_MEMORY_DIR,
        max_facts  : int = USER_MEMORY_MAX,
    ) -> None:
        self.user_id    = user_id
        self.max_facts  = max_facts
        self.filepath   = os.path.join(memory_dir, f"{user_id}.json")

        os.makedirs(memory_dir, exist_ok=True)
        self._facts : list[str] = self._load()

        print(f"[MEMORY] User memory loaded: {len(self._facts)} facts for '{user_id}'")

    # ── PUBLIC ────────────────────────────────────────────────────────────────

    def extract_and_store(self, user_message: str) -> list[str]:
        """
        Extract memorable facts from a user message and persist them.

        Args:
            user_message : the raw user query or statement

        Returns:
            List of newly extracted facts (empty if none found).
        """
        raw = _call_groq(
            FACT_EXTRACTION_PROMPT.format(message=user_message),
            max_tokens=200,
        )

        new_facts = self._parse_facts(raw)

        if new_facts:
            # Deduplicate: skip facts we already know
            existing_lower = {f.lower() for f in self._facts}
            unique = [f for f in new_facts if f.lower() not in existing_lower]

            if unique:
                self._facts.extend(unique)
                # Keep only the most recent max_facts
                self._facts = self._facts[-self.max_facts :]
                self._save()
                print(f"[MEMORY] Stored {len(unique)} new fact(s): {unique}")

        return new_facts

    def get_prompt_context(self) -> str:
        """
        Return user facts formatted for prompt injection.
        Returns empty string if no facts stored.
        """
        if not self._facts:
            return ""
        facts_text = "\n".join(f"  - {f}" for f in self._facts[-self.max_facts :])
        return f"[KNOWN USER CONTEXT]\n{facts_text}"

    def all_facts(self) -> list[str]:
        """Return a copy of all stored facts."""
        return list(self._facts)

    def clear(self) -> None:
        """Delete all stored facts for this user."""
        self._facts = []
        if os.path.exists(self.filepath):
            os.remove(self.filepath)
        print(f"[MEMORY] Cleared all facts for user '{self.user_id}'")

    # ── PRIVATE ───────────────────────────────────────────────────────────────

    def _load(self) -> list[str]:
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, encoding="utf-8") as f:
                    data = json.load(f)
                    return data if isinstance(data, list) else []
            except (json.JSONDecodeError, IOError):
                return []
        return []

    def _save(self) -> None:
        try:
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump(self._facts, f, indent=2, ensure_ascii=False)
        except IOError as e:
            print(f"[MEMORY] Failed to save user memory: {e}")

    @staticmethod
    def _parse_facts(raw: str) -> list[str]:
        """Parse LLM response into a list of fact strings."""
        raw = raw.strip()

        # Strip markdown fences
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        try:
            facts = json.loads(raw)
            if isinstance(facts, list):
                # Filter: only strings, max 100 chars each
                return [str(f).strip() for f in facts if str(f).strip() and len(str(f)) < 100]
        except json.JSONDecodeError:
            pass

        return []


# ── MEMORY-AWARE PROMPT INJECTION ─────────────────────────────────────────────

def build_memory_context(
    conv_memory : ConversationMemory,
    user_memory : Optional[UserMemory] = None,
) -> str:
    """
    Combine all memory layers into a single context block for the prompt.

    Order:
      1. User facts (persistent, session-independent)
      2. Conversation summary (compressed older turns)
      3. Recent buffer (verbatim recent turns)

    Args:
        conv_memory : ConversationMemory instance
        user_memory : UserMemory instance (optional)

    Returns:
        Formatted string ready to prepend to the RAG context block.
        Returns empty string if all memory is empty.
    """
    parts = []

    # Layer 4: user facts
    if user_memory:
        user_ctx = user_memory.get_prompt_context()
        if user_ctx:
            parts.append(user_ctx)

    # Layers 1+3: conversation buffer + summary
    conv_ctx = conv_memory.get_prompt_context()
    if conv_ctx:
        parts.append(conv_ctx)

    return "\n\n".join(parts)


# ── MEMORY-AWARE GENERATOR WRAPPER ────────────────────────────────────────────

def generate_with_memory(
    query       : str,
    chunks      : list[dict],
    generator,                          # your Generator instance
    conv_memory : ConversationMemory,
    user_memory : Optional[UserMemory] = None,
) -> dict:
    """
    Wrap your existing Generator.generate() to inject memory context.

    Strategy: prepend the memory context block as an additional "chunk"
    at the front of the chunks list with a special source tag.
    This works without modifying your Generator or PromptBuilder at all —
    the memory context rides in as a high-priority chunk.

    Args:
        query       : original (not rewritten) user query
        chunks      : reranked chunks from your pipeline
        generator   : your Generator instance
        conv_memory : ConversationMemory instance
        user_memory : UserMemory instance (optional)

    Returns:
        Same dict as generator.generate() — fully compatible.
    """
    memory_ctx = build_memory_context(conv_memory, user_memory)

    if memory_ctx:
        # Inject memory as a synthetic leading chunk
        # It gets the highest priority because it's first in the list
        memory_chunk = {
            "content"     : memory_ctx,
            "score"       : 1.0,          # highest possible score
            "retrieval"   : "memory",
            "source"      : "conversation_memory",
            "source_type" : "memory",
            "type"        : "text",
            "section"     : "memory_context",
            "page"        : None,
            "chunk_index" : 0,
            "chunk_total" : None,
            "token_count" : len(memory_ctx) // 4,  # rough estimate
        }
        augmented_chunks = [memory_chunk] + chunks
    else:
        augmented_chunks = chunks

    return generator.generate(query, augmented_chunks)


# ── MAIN MEMORY MANAGER ───────────────────────────────────────────────────────

class MemoryManager:
    """
    Unified interface that wires all 4 memory layers together.

    This is the single object you instantiate and call in your pipeline.
    It owns:
      - ConversationMemory (Layers 1 + 3)
      - QueryRewriter      (Layer 2)
      - UserMemory         (Layer 4, optional)

    Usage:
        memory = MemoryManager(user_id="user_001")

        result = memory.run(
            query     = "what about its limitations?",
            retriever = retriever,
            reranker  = reranker,
            generator = generator,
        )
        print(result["answer"])
    """

    def __init__(
        self,
        user_id            : Optional[str]  = None,
        retrieval_pool     : int            = 20,
        top_n              : int            = 5,
        search_mode        : str            = "hybrid",
        buffer_max_turns   : int            = BUFFER_MAX_TURNS,
        summary_threshold  : int            = SUMMARY_THRESHOLD,
        summary_keep_turns : int            = SUMMARY_KEEP_TURNS,
        user_memory_dir    : str            = USER_MEMORY_DIR,
    ) -> None:
        self.retrieval_pool = retrieval_pool
        self.top_n          = top_n
        self.search_mode    = search_mode

        # Layer 1 + 3
        self.conv_memory = ConversationMemory(
            max_turns         = buffer_max_turns,
            summary_threshold = summary_threshold,
            keep_turns        = summary_keep_turns,
        )

        # Layer 2
        self.rewriter = QueryRewriter()

        # Layer 4 (only if user_id given)
        self.user_memory : Optional[UserMemory] = None
        if user_id:
            self.user_memory = UserMemory(
                user_id    = user_id,
                memory_dir = user_memory_dir,
            )

        print(f"\n[MEMORY] MemoryManager ready.")
        print(f"  buffer_max_turns   = {buffer_max_turns}")
        print(f"  summary_threshold  = {summary_threshold}")
        print(f"  user_memory        = {'enabled' if user_id else 'disabled'}")
        print(f"  retrieval_pool     = {retrieval_pool}")
        print(f"  top_n              = {top_n}")
        print(f"  search_mode        = {search_mode}\n")

    # ── MAIN ENTRY POINT ──────────────────────────────────────────────────────

    def run(
        self,
        query     : str,
        retriever,
        reranker,
        generator,
        filters   : Optional[dict] = None,
    ) -> dict:
        """
        Full RAG pipeline turn with all memory layers active.

        Steps:
          1. Rewrite query using conversation history    (Layer 2)
          2. Retrieve candidates using rewritten query
          3. Rerank candidates
          4. Generate answer with memory context         (Layers 1, 3, 4)
          5. Update conversation buffer                  (Layer 1 → triggers 3)
          6. Extract and persist user facts              (Layer 4)

        Args:
            query     : raw user question (as typed)
            retriever : your Retriever instance
            reranker  : your Reranker instance
            generator : your Generator instance
            filters   : optional Qdrant filters dict

        Returns:
            Same dict as generator.generate() with one extra key:
              "rewritten_query" : str — the query actually sent to Qdrant
        """
        print(f"\n{'─'*60}")
        print(f"  [MEMORY] Processing: {query!r}")
        print(f"{'─'*60}")

        # ── STEP 1: REWRITE QUERY ─────────────────────────────────────────────
        recent_turns  = self.conv_memory.get_recent_turns(n=4)
        rewritten     = self.rewriter.rewrite(query, recent_turns)

        # ── STEP 2: RETRIEVE ──────────────────────────────────────────────────
        print(f"\n[1/3] Retrieving (pool={self.retrieval_pool}, mode={self.search_mode})...")
        print(f"      Query sent to Qdrant: {rewritten!r}")

        candidates = retriever.search(
            rewritten,
            mode    = self.search_mode,
            top_n   = self.retrieval_pool,
            filters = filters,
        )

        if not candidates:
            print("  [No chunks retrieved]\n")
            # Still update memory even on empty retrieval
            self.conv_memory.add_turn("user", query)
            self.conv_memory.add_turn("assistant", "I could not find relevant information.")
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
            }

        # ── STEP 3: RERANK ────────────────────────────────────────────────────
        print(f"\n[2/3] Reranking → top {self.top_n}...")
        chunks = reranker.rerank(rewritten, candidates, top_n=self.top_n)

        # ── STEP 4: GENERATE WITH MEMORY ─────────────────────────────────────
        print(f"\n[3/3] Generating with memory context...")
        result = generate_with_memory(
            query       = query,          # original query for the answer
            chunks      = chunks,
            generator   = generator,
            conv_memory = self.conv_memory,
            user_memory = self.user_memory,
        )

        # ── STEP 5: UPDATE CONVERSATION BUFFER ───────────────────────────────
        self.conv_memory.add_turn("user",      query)
        self.conv_memory.add_turn("assistant", result["answer"])

        # ── STEP 6: EXTRACT USER FACTS ────────────────────────────────────────
        if self.user_memory:
            self.user_memory.extract_and_store(query)

        result["rewritten_query"] = rewritten
        return result

    # ── CONVENIENCE METHODS ───────────────────────────────────────────────────

    def rewrite_query(self, query: str) -> str:
        """Rewrite a query using current conversation history (Layer 2 only)."""
        return self.rewriter.rewrite(query, self.conv_memory.get_recent_turns(n=4))

    def get_prompt_context(self) -> str:
        """Get the full memory context string (Layers 1, 3, 4)."""
        return build_memory_context(self.conv_memory, self.user_memory)

    def add_turn(self, role: str, content: str) -> None:
        """Manually add a turn to the conversation buffer."""
        self.conv_memory.add_turn(role, content)

    def get_state(self) -> MemoryState:
        """Return a full snapshot of current memory state (for debugging)."""
        return MemoryState(
            summary    = self.conv_memory.summary,
            buffer     = self.conv_memory.buffer,
            user_facts = self.user_memory.all_facts() if self.user_memory else [],
            turn_count = self.conv_memory.total_turns,
        )

    def print_state(self) -> None:
        """Pretty-print current memory state."""
        state = self.get_state()
        print(f"\n{'='*60}")
        print(f"  MEMORY STATE  ({state.turn_count} total turns)")
        print(f"{'='*60}")

        if state.summary:
            print(f"\n  [Layer 3 — Summary]\n  {state.summary[:300]}{'...' if len(state.summary) > 300 else ''}")
        else:
            print(f"\n  [Layer 3 — Summary] empty")

        print(f"\n  [Layer 1 — Buffer] {len(state.buffer)} turns")
        for t in state.buffer:
            preview = t.content[:80].replace("\n", " ")
            print(f"    {t.role.upper()}: {preview}{'...' if len(t.content) > 80 else ''}")

        if state.user_facts:
            print(f"\n  [Layer 4 — User Facts] {len(state.user_facts)} facts")
            for f in state.user_facts:
                print(f"    - {f}")
        else:
            print(f"\n  [Layer 4 — User Facts] none yet")

        print(f"{'='*60}\n")

    def clear_session(self) -> None:
        """Clear conversation memory (start fresh session, keep user facts)."""
        self.conv_memory.clear()
        print("[MEMORY] Session cleared. User facts preserved.")

    def clear_all(self) -> None:
        """Clear everything including user facts."""
        self.conv_memory.clear()
        if self.user_memory:
            self.user_memory.clear()
        print("[MEMORY] All memory cleared.")


# ── INTERACTIVE REPL WITH MEMORY ─────────────────────────────────────────────

def main():
    """
    Drop-in replacement for the generator.py REPL — same interface,
    now with full 4-layer memory.

    Usage:
      python memory.py
      python memory.py --user_id john --top_n 5 --pool 20 --mode hybrid

    Commands during REPL:
      memory   → print current memory state
      clear    → clear session memory (keep user facts)
      clearall → clear everything
      exit     → quit
    """
    import sys

    args        = sys.argv[1:]
    user_id     = None
    top_n       = 5
    pool        = 20
    mode        = "hybrid"

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
        else:
            i += 1

    print(f"\n{'='*60}")
    print(f"  RAG + MEMORY REPL")
    print(f"{'='*60}")
    print(f"  Loading pipeline (one time)...")

    # Load your pipeline components
    from retrieve.retriever       import Retriever
    from reranking.reranker       import Reranker
    from answer_generation.generator import Generator

    retriever = Retriever()
    reranker  = Reranker()
    generator = Generator()

    # Initialise memory manager
    memory = MemoryManager(
        user_id        = user_id,
        retrieval_pool = pool,
        top_n          = top_n,
        search_mode    = mode,
    )

    print(f"\n{'='*60}")
    print(f"  READY. Type your question and press Enter.")
    print(f"  Commands: memory | clear | clearall | exit")
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

            try:
                result = memory.run(
                    query     = question,
                    retriever = retriever,
                    reranker  = reranker,
                    generator = generator,
                )

                print(f"\n{'─'*60}")
                print(f"  ANSWER")
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
                      f"no_answer={result['no_answer']}]\n")

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
        print("\n[MEMORY REPL] Goodbye.\n")


if __name__ == "__main__":
    main()