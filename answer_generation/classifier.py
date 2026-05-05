"""
generation/classifier.py — LLM-based query classifier with heuristic shortcuts.

Improvements over v1:
  • Hard-coded shortcuts for obvious cases (greetings, thanks, etc.)
  • Improved prompt with MANY more examples
  • Decisive instruction (no defaulting to NEEDS_RAG)
  • Debug output shows the LLM's actual response

Returns: "NEEDS_RAG" or "DIRECT_ANSWER"
"""
from __future__ import annotations

import sys
import os
import re

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from memory.llm import call_groq


# ── HEURISTIC SHORTCUTS ───────────────────────────────────────────
# Catch obvious cases without an LLM call (faster + more reliable)

# Greetings / chitchat patterns (case-insensitive, full word match)
CHITCHAT_PATTERNS = [
    r"^\s*hi\s*[!.?]?\s*$",
    r"^\s*hello\s*[!.?]?\s*$",
    r"^\s*hey\s*[!.?]?\s*$",
    r"^\s*hola\s*[!.?]?\s*$",
    r"^\s*bonjour\s*[!.?]?\s*$",
    r"^\s*salut\s*[!.?]?\s*$",
    r"^\s*good\s+(morning|afternoon|evening|night)\s*[!.?]?\s*$",
    r"^\s*thanks?\s*[!.?]?\s*$",
    r"^\s*thank\s+you\s*[!.?]?\s*$",
    r"^\s*ok\s*[!.?]?\s*$",
    r"^\s*okay\s*[!.?]?\s*$",
    r"^\s*cool\s*[!.?]?\s*$",
    r"^\s*nice\s*[!.?]?\s*$",
    r"^\s*great\s*[!.?]?\s*$",
    r"^\s*bye\s*[!.?]?\s*$",
    r"^\s*goodbye\s*[!.?]?\s*$",
    r"^\s*see\s+you\s*[!.?]?\s*$",
]

# Follow-up patterns (operate on previous answer)
FOLLOWUP_PATTERNS = [
    r"\b(summarize|summarise|summary)\b.*\b(that|it|this|previous)\b",
    r"\b(make|rewrite)\b.*(shorter|longer|simpler|formal|informal)",
    r"\btranslate\b.*\b(that|it|this|to|in|into)\b",
    r"\bin\s+(french|english|arabic|spanish|german|chinese|japanese|italian)\b",
    r"\b(explain|elaborate)\b.*\b(more|better|simpler|further)\b",
    r"^\s*(what\s+do\s+you\s+mean|what\s+did\s+you\s+mean)\b",
    r"\brephrase\b",
    r"\bbullet\s+points?\b",
    r"^\s*(continue|go on|keep going)\s*[!.?]?\s*$",
]

# Meta questions about the assistant
META_PATTERNS = [
    r"^\s*(what|who)\s+(can|are|do)\s+you\b",
    r"\bwhat\s+(can|do)\s+you\s+(do|help)\b",
    r"\bhow\s+do\s+you\s+work\b",
    r"\bwhat\s+is\s+your\s+(name|purpose)\b",
]


def _matches_any(text: str, patterns: list[str]) -> bool:
    """Check if text matches any of the regex patterns (case-insensitive)."""
    text = text.strip()
    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def _check_heuristics(question: str) -> str | None:
    """
    Try to classify without an LLM call.
    Returns 'DIRECT_ANSWER' if obvious, None otherwise.
    """
    text = question.strip()
    
    # Very short messages (1-3 words) are usually chitchat
    word_count = len(text.split())
    
    if _matches_any(text, CHITCHAT_PATTERNS):
        return "DIRECT_ANSWER"
    
    if _matches_any(text, FOLLOWUP_PATTERNS):
        return "DIRECT_ANSWER"
    
    if _matches_any(text, META_PATTERNS):
        return "DIRECT_ANSWER"
    
    return None


# ── LLM-BASED CLASSIFIER PROMPT ──────────────────────────────────

CLASSIFIER_PROMPT = """You classify user questions for a RAG chatbot. Decide the route.

CONVERSATION HISTORY (last few turns):
{history}

USER'S LATEST MESSAGE:
"{question}"

YOU MUST CLASSIFY AS ONE OF:

DIRECT_ANSWER — Use this when the user is doing ANY of:
  - Greeting, thanking, casual chat (hi, hello, thanks, ok, cool, bye)
  - Asking what you can do, who you are, how you work
  - Acting on YOUR PREVIOUS ANSWER: summarize that, make it shorter, 
    translate it, rephrase, explain more, in French/Arabic/etc.
  - Asking a meta question (what does X mean — when X is in your previous answer)
  - Continuing the conversation casually (yes, no, why, really?)

NEEDS_RAG — Use this when the user is asking for NEW INFORMATION that 
would require searching documents:
  - "What is X?" (when X has not been discussed yet)
  - "Tell me about X" / "Explain X"
  - "How does X work?"
  - "What does the document say about X?"
  - "Find me information on X"

RULES:
- Single-word greetings ("hi", "hello", "thanks") are ALWAYS DIRECT_ANSWER.
- Single-word follow-ups ("yes", "no", "why", "ok") are ALWAYS DIRECT_ANSWER.
- "Summarize/translate/rephrase that" is ALWAYS DIRECT_ANSWER.
- Be DECISIVE. If the user is being casual, choose DIRECT_ANSWER.
- Only choose NEEDS_RAG if there is a clear request for NEW factual information.

Respond with EXACTLY one word: DIRECT_ANSWER or NEEDS_RAG"""


# ── MAIN CLASSIFIER FUNCTION ──────────────────────────────────────

def classify_query(question: str, history: list[dict]) -> str:
    """
    Classify the user's question.
    
    Strategy:
      1. Try heuristic patterns first (fast, deterministic)
      2. Fall back to LLM for ambiguous cases
    
    Returns "NEEDS_RAG" or "DIRECT_ANSWER".
    """
    # ── Heuristic shortcut (avoids LLM call for obvious cases) ─────
    heuristic = _check_heuristics(question)
    if heuristic:
        print(f"  [HEURISTIC] '{question[:50]}' → {heuristic}")
        return heuristic
    
    # ── LLM classification for ambiguous cases ────────────────────
    if history:
        history_str = "\n".join([
            f"{t['role'].upper()}: {t['content'][:300]}"
            + ("..." if len(t['content']) > 300 else "")
            for t in history[-4:]
        ])
    else:
        history_str = "(no previous turns — this is the first message)"

    prompt = CLASSIFIER_PROMPT.format(
        history=history_str,
        question=question,
    )

    try:
        response = call_groq(prompt, max_tokens=15)
        response_clean = response.strip().upper()
        
        # Show what the LLM said (for debugging)
        print(f"  [LLM] Raw response: {response_clean!r}")

        # Robust parsing
        if "DIRECT_ANSWER" in response_clean or "DIRECT ANSWER" in response_clean:
            return "DIRECT_ANSWER"
        if "NEEDS_RAG" in response_clean or "NEEDS RAG" in response_clean:
            return "NEEDS_RAG"

        # Unclear → check word count as last resort
        # Very short messages are almost always chitchat
        if len(question.strip().split()) <= 3:
            print(f"  [FALLBACK] Short message → DIRECT_ANSWER")
            return "DIRECT_ANSWER"
        
        # Otherwise default to NEEDS_RAG (safer for unclear longer queries)
        print(f"  [FALLBACK] Unclear → defaulting to NEEDS_RAG")
        return "NEEDS_RAG"

    except Exception as e:
        print(f"  [ERROR] Classifier failed: {e}")
        # On error, use word count heuristic
        if len(question.strip().split()) <= 3:
            return "DIRECT_ANSWER"
        return "NEEDS_RAG"
