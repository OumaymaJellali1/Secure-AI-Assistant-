"""
memory/llm.py — Centralized Groq client with retries.

All memory components use call_groq() — never instantiate Groq directly.
This ensures one client instance, one retry policy, one place to change models.
"""
import os
import time
import random
from groq import Groq
from dotenv import load_dotenv

load_dotenv()


# ── CONFIG ────────────────────────────────────────────────────────────
FAST_MODEL    = "llama-3.1-8b-instant"      # rewriter, summary, fact extraction
SMART_MODEL   = "llama-3.3-70b-versatile"   # selector, complex reasoning

MAX_RETRIES   = 3
RETRY_BASE_S  = 1.0
RETRY_JITTER  = 0.3


# ── SINGLE GROQ CLIENT ────────────────────────────────────────────────
_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


def call_groq(
    prompt: str,
    max_tokens: int = 300,
    model: str = FAST_MODEL,
) -> str:
    """
    Call Groq with retry + exponential backoff. Returns "" on failure.
    All callers must handle the empty-string fallback case.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = _client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt == MAX_RETRIES:
                print(f"[LLM] Groq failed after {MAX_RETRIES} attempts: {e}")
                return ""
            wait = RETRY_BASE_S * (2 ** (attempt - 1)) + random.uniform(0, RETRY_JITTER)
            print(f"[LLM] Retry {attempt}/{MAX_RETRIES} in {wait:.1f}s — {e}")
            time.sleep(wait)
    return ""


# ── SHARED PROMPTS ────────────────────────────────────────────────────

REWRITE_PROMPT = """You are a query rewriter for a RAG system.

Given the conversation history below and the user's latest question, rewrite the question so it is fully self-contained.

Rules:
- Replace pronouns ("it", "they", "this", "that") with the specific entity they refer to
- Expand vague references ("the second one", "the previous method") into explicit terms
- If the question is already self-contained, return it unchanged
- Return ONLY the rewritten question — no explanation, no preamble
- Maximum 25 words

Conversation history:
{history}

Latest question: {query}

Rewritten question:"""


SUMMARY_PROMPT = """Summarize this conversation segment into ONE concise paragraph.
Keep all important facts, decisions, names, numbers, and technical terms.

EXISTING SUMMARY (if any):
{existing_summary}

NEW TURNS TO ADD:
{conversation}

Updated summary (one dense paragraph, max 500 words):"""


CONDENSE_PROMPT = """Condense this summary to under {max_words} words.
Keep all key facts, names, decisions. Drop redundant details.

Summary:
{summary}

Condensed summary:"""


FACT_EXTRACTION_PROMPT = """Extract memorable, reusable facts about the USER from this message.
Focus on: preferences, goals, constraints, background, expertise, tech stack.
Ignore generic statements.

Message: {message}

Return ONLY a JSON array of short strings (max 15 words each).
Example: ["prefers Python over R", "uses Qdrant", "works on RAG systems"]
Return [] if nothing memorable."""


TITLE_PROMPT = """Generate a short, descriptive title (3-6 words) for a conversation that starts with this message.
Return ONLY the title — no quotes, no preamble.

First message: {message}

Title:"""


SELECTOR_PROMPT = """You decide which memory layers a chatbot needs to answer a question.

AVAILABLE LAYERS:
- summary: compressed history of older conversation turns
- facts: permanent facts about the user across all sessions

QUERY: "{query}"

Rules:
- The recent buffer is ALWAYS included (you don't choose it)
- Return "summary" if the query references earlier discussion
- Return "facts" if the answer should be personalized
- Return [] if recent buffer alone is enough

Return ONLY a JSON array. Examples:
"what is 2+2"           → []
"how to optimize that?" → ["summary"]
"what do I prefer?"     → ["facts"]
"based on what we said about my project" → ["summary", "facts"]"""