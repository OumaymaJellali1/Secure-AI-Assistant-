"""
memory/manager.py — Main orchestrator. Single entry point for the pipeline.

Wires together:
  - SessionManager     (manage conversations)
  - ConversationMemory (Layers 1+3: buffer + summary)
  - UserMemory         (Layer 4: persistent facts)
  - QueryRewriter      (Layer 2: pronoun expansion)
  - PermissionLoader   (Phase 3: doc access control)

Smart memory selection: LLM decides per-query which layers to inject.
"""
import json
from typing import Optional

from memory.llm           import call_groq, SELECTOR_PROMPT, SMART_MODEL
from memory.sessions      import SessionManager
from memory.conversation  import ConversationMemory
from memory.user_facts    import UserMemory
from memory.rewriter      import QueryRewriter
from memory.permissions   import PermissionLoader


class MemoryManager:
    """
    One instance per active conversation.

    Use:
      mm = MemoryManager(session_id="sess_abc", user_id="alice")
      rewritten = mm.rewrite_query(user_query)
      ctx       = mm.build_prompt_context(user_query)
      ...generate answer...
      mm.save_turn("user", user_query)
      mm.save_turn("assistant", answer)
    """

    def __init__(self, session_id: str, user_id: str):
        self.session_id = session_id
        self.user_id    = user_id

        self.sessions    = SessionManager()
        self.conv        = ConversationMemory(session_id, user_id)
        self.facts       = UserMemory(user_id)
        self.rewriter    = QueryRewriter()
        self.permissions = PermissionLoader()

    # ── PUBLIC API ──────────────────────────────────────────────────

    def rewrite_query(self, query: str) -> str:
        """Layer 2: rewrite vague queries using recent history."""
        recent = self.conv.get_buffer(n=4)
        return self.rewriter.rewrite(query, recent)

    def get_allowed_doc_ids(self) -> Optional[list[str]]:
        """Phase 3: documents the user can access. None = no filter."""
        return self.permissions.get_allowed_doc_ids(self.user_id)

    def build_prompt_context(self, query: str) -> str:
        """
        Build the memory section to inject into the LLM prompt.
        Smart selection: LLM decides which layers are needed.
        """
        # Load everything ONCE (avoid double DB hits)
        conv_ctx = self.conv.get_context()
        facts    = self.facts.get_facts()

        # Decide which layers to include
        layers = self._select_layers(
            query=query,
            has_summary=bool(conv_ctx["summary"]),
            has_facts=bool(facts),
        )

        # Build the context string
        parts = []

        if "facts" in layers and facts:
            facts_text = "\n".join(f"  - {f}" for f in facts)
            parts.append(f"[KNOWN USER CONTEXT]\n{facts_text}")

        if "summary" in layers and conv_ctx["summary"]:
            parts.append(f"[CONVERSATION SUMMARY]\n{conv_ctx['summary']}")

        # Buffer is ALWAYS included (cheapest, almost always relevant)
        if conv_ctx["buffer"]:
            recent = "\n".join(t.to_text() for t in conv_ctx["buffer"])
            parts.append(f"[RECENT CONVERSATION — last {len(conv_ctx['buffer'])} turns]\n{recent}")

        layer_str = "buffer"
        if "summary" in layers: layer_str += "+summary"
        if "facts"   in layers: layer_str += "+facts"
        print(f"[MEMORY] Layers: {layer_str}")

        return "\n\n".join(parts)

    def save_turn(self, role: str, content: str, extract_facts: bool = True) -> None:
        """
        Save a turn. For user turns, optionally extract memorable facts.
        Side effects: updates buffer, may trigger summary compression,
        may add facts to user memory, updates session last_active.
        """
        self.conv.add_turn(role, content)

        # Auto-extract facts only from user messages
        if extract_facts and role == "user":
            self.facts.extract_and_store(content)

        # Auto-title the session after first user turn (if no title yet)
        if role == "user":
            self._maybe_auto_title(content)

    def auto_title_if_needed(self, first_message: str) -> None:
        """Generate a title for the session if it doesn't have one."""
        self._maybe_auto_title(first_message)

    # ── SESSION MANAGEMENT ──────────────────────────────────────────

    def list_my_conversations(self, limit: int = 50) -> list[dict]:
        """List all of this user's conversations."""
        return self.sessions.list_for_user(self.user_id, limit=limit)

    def archive_current(self) -> None:
        """Archive this conversation (mark inactive)."""
        self.sessions.archive(self.session_id)

    def delete_current(self) -> None:
        """Permanently delete this conversation."""
        self.sessions.delete(self.session_id)

    def clear_buffer(self) -> None:
        """Clear conversation memory but keep user facts."""
        self.conv.clear()

    def clear_all(self) -> None:
        """Clear conversation + user facts."""
        self.conv.clear()
        self.facts.clear()

    # ── INTROSPECTION ───────────────────────────────────────────────

    def state(self) -> dict:
        """Return full memory state for debugging."""
        ctx = self.conv.get_context()
        return {
            "session_id":  self.session_id,
            "user_id":     self.user_id,
            "buffer_size": len(ctx["buffer"]),
            "buffer":      [t.to_dict() for t in ctx["buffer"]],
            "summary":     ctx["summary"],
            "facts":       self.facts.get_facts(),
        }

    def print_state(self) -> None:
        """Pretty-print memory state."""
        s = self.state()
        print(f"\n{'=' * 60}")
        print(f"  MEMORY STATE — session: {s['session_id']}, user: {s['user_id']}")
        print(f"{'=' * 60}")

        print(f"\n  [Layer 1 — Buffer] {s['buffer_size']} turn(s)")
        for t in s["buffer"]:
            preview = t["content"][:80].replace("\n", " ")
            suffix = "..." if len(t["content"]) > 80 else ""
            print(f"    {t['role'].upper():9} {preview}{suffix}")

        if s["summary"]:
            preview = s["summary"][:300]
            suffix = "..." if len(s["summary"]) > 300 else ""
            print(f"\n  [Layer 3 — Summary]\n    {preview}{suffix}")
        else:
            print(f"\n  [Layer 3 — Summary] (empty)")

        if s["facts"]:
            print(f"\n  [Layer 4 — User Facts] {len(s['facts'])} fact(s)")
            for f in s["facts"]:
                print(f"    - {f}")
        else:
            print(f"\n  [Layer 4 — User Facts] (empty)")

        print(f"{'=' * 60}\n")

    # ── INTERNAL ────────────────────────────────────────────────────

    def _select_layers(
        self,
        query: str,
        has_summary: bool,
        has_facts: bool,
    ) -> list[str]:
        """
        LLM-based layer selector. Buffer is always included separately.
        Returns subset of ["summary", "facts"] needed for this query.
        """
        # Skip LLM if both layers are empty anyway
        if not has_summary and not has_facts:
            return []

        # Skip LLM for very short queries (likely simple)
        if len(query.split()) < 4:
            return []

        raw = call_groq(
            SELECTOR_PROMPT.format(query=query),
            max_tokens=40,
            model=SMART_MODEL,
        )

        # Strip markdown fences
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        try:
            layers = json.loads(raw)
            if not isinstance(layers, list):
                return ["summary", "facts"] if has_summary or has_facts else []
        except json.JSONDecodeError:
            # Safe default on parse failure: include everything available
            return [
                l for l in ["summary", "facts"]
                if (l == "summary" and has_summary) or (l == "facts" and has_facts)
            ]

        # Filter out layers that are empty
        if not has_summary and "summary" in layers:
            layers.remove("summary")
        if not has_facts and "facts" in layers:
            layers.remove("facts")

        return layers

    def _maybe_auto_title(self, first_message: str) -> None:
        """If session has no title yet, generate one from this message."""
        sess = self.sessions.get(self.session_id)
        if sess and not sess.get("title"):
            self.sessions.auto_title(self.session_id, first_message)