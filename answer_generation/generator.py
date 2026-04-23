"""
generator.py
════════════
Production-grade RAG answer generator using Groq API (llama-3.3-70b).

Combines best of both worlds:
  - Detailed, comprehensive answers
  - Inline source citations: [Source: filename, page X]
  - Smart token budget management (via prompt_builder.py)
  - Streaming with full answer capture (no bug)
  - Citation parsing — only cited sources returned
  - Observability — latency, tokens, no-answer detection

Prerequisites:
  pip install groq
  Set GROQ_API_KEY environment variable
  Get free key at: https://console.groq.com

Usage:
  from generation.generator import Generator
  from retrieval.retriever import Retriever

  retriever = Retriever()
  generator = Generator()

  chunks = retriever.search("What is RAG?", top_n=10)
  result = generator.generate("What is RAG?", chunks)

  print(result["answer"])
  print(result["sources"])  # only chunks actually cited
"""

from __future__ import annotations

import re
import sys
import os
import time
from typing import Iterator, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dotenv import load_dotenv
load_dotenv()
from groq import Groq
from answer_generation.prompt_builder import PromptBuilder, BuiltPrompt


# ── CONFIG ────────────────────────────────────────────────────────────────────

MODEL_NAME         : str   = "llama-3.3-70b-versatile"  # free 70b — beats qwen2.5:7b
TEMPERATURE        : float = 0.15    # low = factual (good for RAG)
TOP_P              : float = 0.90
MAX_TOKENS         : int   = 1024

# No-answer detection
NO_ANSWER_SENTINEL = "i don't have enough information in the provided context"


# ── GENERATOR ─────────────────────────────────────────────────────────────────

class Generator:
    """
    RAG answer generator using Groq API (llama-3.3-70b-versatile).

    Args:
        model       : Groq model name
        temperature : sampling temperature (lower = more factual)
        max_tokens  : max tokens to generate
    """

    def __init__(
        self,
        model       : str   = MODEL_NAME,
        temperature : float = TEMPERATURE,
        max_tokens  : int   = MAX_TOKENS,
    ) -> None:
        self.model           = model
        self.temperature     = temperature
        self.max_tokens      = max_tokens

        self._client         = Groq(api_key=os.environ["GROQ_API_KEY"])
        self._prompt_builder = PromptBuilder()

    # ── PUBLIC: STANDARD GENERATION ───────────────────────────────────────────

    def generate(
        self,
        query       : str,
        chunks      : list[dict],
        *,
        top_n       : Optional[int]   = None,
        temperature : Optional[float] = None,
    ) -> dict:
        """
        Generate a grounded answer from retrieved chunks.

        Args:
            query       : user's question
            chunks      : list of dicts from Retriever.search()
            top_n       : optionally limit to top-N chunks
            temperature : override instance temperature for this call

        Returns:
            {
              "answer"         : str,
              "sources"        : list  — only chunks actually cited
              "used_chunks"    : list  — all chunks injected into prompt
              "model"          : str,
              "tokens"         : {"prompt": int, "completion": int, "total": int},
              "context_tokens" : int,
              "no_answer"      : bool,
              "latency_s"      : float,
            }
        """
        if top_n:
            chunks = chunks[:top_n]

        built      = self._prompt_builder.build(query, chunks)
        system_msg = next((m["content"] for m in built.messages if m["role"] == "system"), "")
        user_msg   = next((m["content"] for m in built.messages if m["role"] == "user"), query)

        print(f"[GENERATOR] Calling {self.model} | "
              f"context_tokens≈{built.total_context_tokens} | "
              f"chunks_used={len(built.used_chunks)}")

        t0 = time.perf_counter()
        response = self._client.chat.completions.create(
            model       = self.model,
            temperature = temperature if temperature is not None else self.temperature,
            max_tokens  = self.max_tokens,
            messages    = [
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": user_msg},
            ],
        )
        latency = round(time.perf_counter() - t0, 3)

        answer    = response.choices[0].message.content.strip()
        no_answer = NO_ANSWER_SENTINEL in answer.lower()
        cited     = _extract_cited_sources(answer, built.used_chunks)

        prompt_tokens     = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens

        print(f"[GENERATOR] Done | latency={latency}s | "
              f"tokens={prompt_tokens}+{completion_tokens} | "
              f"sources cited={len(cited)} | no_answer={no_answer}")

        return {
            "answer"         : answer,
            "sources"        : cited,
            "used_chunks"    : built.used_chunks,
            "model"          : self.model,
            "tokens"         : {
                "prompt"     : prompt_tokens,
                "completion" : completion_tokens,
                "total"      : prompt_tokens + completion_tokens,
            },
            "context_tokens" : built.total_context_tokens,
            "no_answer"      : no_answer,
            "latency_s"      : latency,
        }

    # ── PUBLIC: STREAMING ─────────────────────────────────────────────────────

    def stream(
        self,
        query       : str,
        chunks      : list[dict],
        *,
        top_n       : Optional[int]   = None,
        temperature : Optional[float] = None,
    ) -> Iterator[str]:
        """
        Stream the answer token by token.
        Use stream_with_metadata() if you need sources + tokens.
        """
        result = self.stream_with_metadata(
            query, chunks, top_n=top_n, temperature=temperature
        )
        yield from result["answer"]

    def stream_with_metadata(
        self,
        query       : str,
        chunks      : list[dict],
        *,
        top_n       : Optional[int]   = None,
        temperature : Optional[float] = None,
        on_token                      = None,  # callback(token: str)
    ) -> dict:
        """
        Stream tokens AND capture full answer + metadata.

        Args:
            on_token : optional callback called with each token
                       (e.g., to print as they arrive)

        Returns:
            Same dict as generate()
        """
        if top_n:
            chunks = chunks[:top_n]

        built      = self._prompt_builder.build(query, chunks)
        system_msg = next((m["content"] for m in built.messages if m["role"] == "system"), "")
        user_msg   = next((m["content"] for m in built.messages if m["role"] == "user"), query)

        print(f"[GENERATOR] Streaming {self.model} | "
              f"context_tokens≈{built.total_context_tokens} | "
              f"chunks_used={len(built.used_chunks)}")

        full_answer       : list[str] = []
        prompt_tokens     : int       = 0
        completion_tokens : int       = 0

        t0 = time.perf_counter()

        stream = self._client.chat.completions.create(
            model       = self.model,
            temperature = temperature if temperature is not None else self.temperature,
            max_tokens  = self.max_tokens,
            stream      = True,
            messages    = [
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": user_msg},
            ],
        )

        for chunk in stream:
            token = chunk.choices[0].delta.content or ""
            if token:
                full_answer.append(token)
                if on_token:
                    on_token(token)
            # Capture token counts from last chunk (Groq-specific)
            if hasattr(chunk, "x_groq") and chunk.x_groq:
                usage = chunk.x_groq.usage
                if usage is not None:
                    prompt_tokens     = usage.prompt_tokens     or 0
                    completion_tokens = usage.completion_tokens or 0
        latency   = round(time.perf_counter() - t0, 3)
        answer    = "".join(full_answer).strip()
        no_answer = NO_ANSWER_SENTINEL in answer.lower()
        cited     = _extract_cited_sources(answer, built.used_chunks)

        print(f"\n[GENERATOR] Done | latency={latency}s | "
              f"tokens={prompt_tokens}+{completion_tokens} | "
              f"sources cited={len(cited)} | no_answer={no_answer}")

        return {
            "answer"         : answer,
            "sources"        : cited,
            "used_chunks"    : built.used_chunks,
            "model"          : self.model,
            "tokens"         : {
                "prompt"     : prompt_tokens,
                "completion" : completion_tokens,
                "total"      : prompt_tokens + completion_tokens,
            },
            "context_tokens" : built.total_context_tokens,
            "no_answer"      : no_answer,
            "latency_s"      : latency,
        }


# ── CITATION PARSER ───────────────────────────────────────────────────────────

def _extract_cited_sources(answer: str, used_chunks: list[dict]) -> list[dict]:
    """
    Parse [Source: filename, page X] citations and return matching chunks.

    Handles formats:
      [Source: report.pdf, page 3]
      [Source: report.pdf]
      [Source: report.pdf page 3]   (no comma)
      [Source: Report.PDF, Page 3]  (case insensitive)
    """
    pattern = re.compile(
        r"\[Source:\s*([^,\]\n]+?)(?:[,\s]+page\s+(\d+))?\s*\]",
        re.IGNORECASE,
    )

    cited_keys: set[tuple] = set()
    for match in pattern.finditer(answer):
        source = match.group(1).strip()
        page   = int(match.group(2)) if match.group(2) else None
        cited_keys.add((source, page))

    # Debug — remove once citations are confirmed working
    print(f"[CITATIONS] Found in answer : {cited_keys}")
    print(f"[CITATIONS] Chunks available: {[(c.get('source'), c.get('page')) for c in used_chunks]}")

    matched_chunks = []
    seen_chunk_ids : set = set()

    for chunk in used_chunks:
        chunk_source = chunk.get("source")
        chunk_page   = chunk.get("page")

        for cited_source, cited_page in cited_keys:
            if chunk_source and chunk_source.lower() == cited_source.lower():
                # No page in citation → match any chunk from that source
                # Page in citation    → must match exactly
                if cited_page is None or cited_page == chunk_page:
                    chunk_id = id(chunk)
                    if chunk_id not in seen_chunk_ids:
                        matched_chunks.append(chunk)
                        seen_chunk_ids.add(chunk_id)
                    break

    return matched_chunks


# ── PRETTY PRINT ──────────────────────────────────────────────────────────────

def print_answer(result: dict, show_sources: bool = True) -> None:
    """Pretty-print RAG result."""
    print(f"\n{'='*70}")
    print(f"  ANSWER")
    print(f"{'='*70}")
    print(f"\n{result['answer']}\n")

    if show_sources and result.get("sources"):
        print(f"\n{'='*70}")
        print(f"  SOURCES CITED ({len(result['sources'])})")
        print(f"{'='*70}")

        for i, src in enumerate(result["sources"], 1):
            source    = src.get("source", "unknown")
            page      = src.get("page")
            section   = src.get("section")
            retrieval = src.get("retrieval", "?")
            score     = src.get("score", 0)

            location = f"{source}"
            if page:
                location += f", page {page}"
            if section:
                location += f", section {section}"

            preview = src.get("content", "").replace("\n", " ")[:200]

            print(f"\n  [{i}] {location}")
            print(f"      score={score:.4f} | retrieval={retrieval}")
            print(f"      {preview}...")

    # Metadata footer
    print(f"\n{'─'*70}")
    print(f"  Model      : {result['model']}")
    print(f"  Latency    : {result.get('latency_s', '?')}s")
    print(f"  Tokens     : {result['tokens']['prompt']} prompt + "
          f"{result['tokens']['completion']} completion")
    print(f"  Ctx tokens : {result['context_tokens']}")
    print(f"  No-answer  : {result['no_answer']}")
    print(f"{'='*70}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Usage:
      python generator.py "What is RAG?"
      python generator.py "Explain RAG" --stream
      python generator.py "summary" --filter source_type=pdf --top_n 10
    """
    from retrieve.retriever import Retriever

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python generator.py <question>")
        print("  python generator.py <question> --stream")
        print("  python generator.py <question> --top_n 10 --filter source_type=pdf")
        sys.exit(1)

    # Parse args
    question = sys.argv[1]
    stream   = False
    filters  = None
    top_n    = 10

    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == "--stream":
            stream = True
            i += 1
        elif sys.argv[i] == "--filter" and i + 1 < len(sys.argv):
            key, val = sys.argv[i + 1].split("=", 1)
            filters = {key: val}
            i += 2
        elif sys.argv[i] == "--top_n" and i + 1 < len(sys.argv):
            top_n = int(sys.argv[i + 1])
            i += 2
        else:
            i += 1

    try:
        # Step 1: Retrieve
        retriever = Retriever()
        chunks = retriever.search(question, top_n=top_n, filters=filters)

        if not chunks:
            print("\n[GENERATOR] No chunks retrieved.\n")
            sys.exit(0)

        # Step 2: Generate
        gen = Generator()

        if stream:
            print(f"\n{'='*70}")
            print(f"  ANSWER (streaming)")
            print(f"{'='*70}\n")

            result = gen.stream_with_metadata(
                question, chunks,
                on_token=lambda t: print(t, end="", flush=True),
            )
            print()  # newline after stream

            if result.get("sources"):
                print(f"\n{'='*70}")
                print(f"  SOURCES CITED ({len(result['sources'])})")
                print(f"{'='*70}")
                for i, src in enumerate(result["sources"], 1):
                    src_name = src.get("source", "?")
                    page     = src.get("page")
                    section  = src.get("section")
                    location = src_name + (f", page {page}" if page else "")
                    if section:
                        location += f" | {section}"
                    print(f"  [{i}] {location}")
                print(f"{'='*70}\n")
        else:
            result = gen.generate(question, chunks)
            print_answer(result)

    except Exception as e:
        print(f"\n[GENERATOR] ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)