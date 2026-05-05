"""
generator.py
════════════
Production-grade RAG answer generator using llama-3.3-70b-versatile via Groq.

Pipeline: retriever → reranker → generator
  1. Retriever returns top-10 candidates (hybrid dense + sparse + RRF)
  2. Reranker scores all 10 with cross-encoder, returns top-N (default 5)
  3. Generator produces grounded answer with inline citations

Features:
  - Detailed, comprehensive answers
  - Inline source citations: [Source: filename, page X]
  - Smart token budget management (via prompt_builder.py)
  - Streaming with full answer capture
  - Citation parsing — only cited sources returned
  - Observability — latency, tokens, no-answer detection
  - Health check on first call
  - REPL mode — load models once, query many times

Prerequisites:
  pip install groq
  export GROQ_API_KEY=your_key_here

Usage:
  # REPL (recommended for development)
  python generator.py --repl

  # One-shot
  python generator.py "What is RAG?"
  python generator.py "Explain RAG" --stream
  python generator.py "summary" --filter source_type=pdf --top_n 5
"""

from __future__ import annotations

import re
import sys
import os
import time
from typing import Iterator, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from groq import Groq

from answer_generation.prompt_builder import PromptBuilder, BuiltPrompt
from reranking.reranker        import Reranker


# ── CONFIG ────────────────────────────────────────────────────────────────────

GROQ_API_KEY    : str   = os.environ.get("GROQ_API_KEY", "")
MODEL_NAME      : str   = "llama-3.3-70b-versatile"

# Generation params
TEMPERATURE     : float = 0.15          # low = factual, not creative
TOP_P           : float = 0.90
NUM_PREDICT     : int   = 1500          # max tokens the model generates

# No-answer detection — the model is prompted to say this when unsure
NO_ANSWER_SENTINEL = "i don't have enough information in the provided context"


# ── GENERATOR ─────────────────────────────────────────────────────────────────

class Generator:
    """
    RAG answer generator using llama-3.3-70b-versatile via Groq API.

    Args:
        model       : Groq model name
        temperature : sampling temperature (lower = more factual)
        top_p       : nucleus sampling
        num_predict : max tokens to generate
    """

    def __init__(
        self,
        model          : str   = MODEL_NAME,
        temperature    : float = TEMPERATURE,
        top_p          : float = TOP_P,
        num_predict    : int   = NUM_PREDICT,
    ) -> None:
        self.model          = model
        self.temperature    = temperature
        self.top_p          = top_p
        self.num_predict    = num_predict

        self._client         = Groq(api_key=GROQ_API_KEY)
        self._prompt_builder = PromptBuilder()
        self._healthy        : Optional[bool] = None

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

        Returns:
            {
              "answer"         : str,
              "sources"        : list — only chunks actually cited,
              "used_chunks"    : list — all chunks injected into prompt,
              "model"          : str,
              "tokens"         : {"prompt", "completion", "total"},
              "context_tokens" : int,
              "no_answer"      : bool,
              "latency_s"      : float,
            }
        """
        self._ensure_healthy()

        if top_n:
            chunks = chunks[:top_n]

        built = self._prompt_builder.build(query, chunks)

        print(f"[GENERATOR] Calling {self.model} | "
              f"context_tokens≈{built.total_context_tokens} | "
              f"chunks_used={len(built.used_chunks)}")

        t0 = time.perf_counter()
        response = self._client.chat.completions.create(
            model       = self.model,
            messages    = built.messages,
            temperature = temperature if temperature is not None else self.temperature,
            max_tokens  = self.num_predict,
            top_p       = self.top_p,
        )
        latency = round(time.perf_counter() - t0, 3)

        return self._parse_response(response, built, latency)

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
        Use stream_with_metadata() if you need sources + token counts.
        """
        self._ensure_healthy()

        if top_n:
            chunks = chunks[:top_n]

        built = self._prompt_builder.build(query, chunks)

        print(f"[GENERATOR] Streaming {self.model} | "
              f"context_tokens≈{built.total_context_tokens} | "
              f"chunks_used={len(built.used_chunks)}")

        stream = self._client.chat.completions.create(
            model       = self.model,
            messages    = built.messages,
            temperature = temperature if temperature is not None else self.temperature,
            max_tokens  = self.num_predict,
            top_p       = self.top_p,
            stream      = True,
        )

        for chunk in stream:
            token = chunk.choices[0].delta.content or ""
            if token:
                yield token

    def stream_with_metadata(
        self,
        query       : str,
        chunks      : list[dict],
        *,
        top_n       : Optional[int]   = None,
        temperature : Optional[float] = None,
        on_token              = None,   # callback(token: str)
    ) -> dict:
        """
        Stream tokens AND capture full answer + metadata.

        Args:
            on_token : optional callback called with each token
                       (e.g., to print as they arrive)

        Returns:
            Same dict shape as generate().
        """
        self._ensure_healthy()

        if top_n:
            chunks = chunks[:top_n]

        built = self._prompt_builder.build(query, chunks)

        print(f"[GENERATOR] Streaming {self.model} | "
              f"context_tokens≈{built.total_context_tokens} | "
              f"chunks_used={len(built.used_chunks)}")

        full_answer: list[str] = []
        prompt_tokens     = 0
        completion_tokens = 0

        t0 = time.perf_counter()

        stream = self._client.chat.completions.create(
            model       = self.model,
            messages    = built.messages,
            temperature = temperature if temperature is not None else self.temperature,
            max_tokens  = self.num_predict,
            top_p       = self.top_p,
            stream      = True,
        )

        for chunk in stream:
            token = chunk.choices[0].delta.content or ""
            if token:
                full_answer.append(token)
                if on_token:
                    on_token(token)
            # Capture token counts from the final chunk
            if chunk.x_groq and chunk.x_groq.usage:
                prompt_tokens     = chunk.x_groq.usage.prompt_tokens
                completion_tokens = chunk.x_groq.usage.completion_tokens

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

    # ── PRIVATE: RESPONSE PARSING ─────────────────────────────────────────────

    def _parse_response(
        self,
        response  ,
        built     : BuiltPrompt,
        latency_s : float,
    ) -> dict:
        """Parse non-streaming Groq response."""
        answer    = response.choices[0].message.content.strip()
        no_answer = NO_ANSWER_SENTINEL in answer.lower()

        prompt_tokens     = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens
        cited             = _extract_cited_sources(answer, built.used_chunks)

        print(f"[GENERATOR] Done | latency={latency_s}s | "
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
            "latency_s"      : latency_s,
        }

    # ── PRIVATE: HEALTH CHECK ─────────────────────────────────────────────────

    def _ensure_healthy(self) -> None:
        """Verify Groq API key is set. Cached after first call."""
        if self._healthy:
            return

        print(f"[GENERATOR] Health check — Groq API ...")

        if not GROQ_API_KEY:
            raise RuntimeError(
                "GROQ_API_KEY environment variable is not set.\n"
                "  Get a free key at: https://console.groq.com"
            )

        print(f"[GENERATOR] Model '{self.model}' is ready.")
        self._healthy = True


# ── CITATION PARSER ───────────────────────────────────────────────────────────

def _extract_cited_sources(answer: str, used_chunks: list[dict]) -> list[dict]:
    """
    Parse [Source: filename, page X] citations and return matching chunks.

    Handles formats:
      [Source: report.pdf, page 3]
      [Source: report.pdf]
      [Source: report.pdf, page 3] [Source: doc.docx]
    """
    pattern = re.compile(
        r"\[Source:\s*([^,\]]+)(?:,\s*page\s+(\d+))?\]",
        re.IGNORECASE,
    )

    cited_keys: set[tuple] = set()   # (source, page) pairs

    for match in pattern.finditer(answer):
        source = match.group(1).strip()
        page   = int(match.group(2)) if match.group(2) else None
        cited_keys.add((source, page))

    matched_chunks = []
    seen_chunk_ids = set()

    for chunk in used_chunks:
        chunk_source = chunk.get("source")
        chunk_page   = chunk.get("page")

        for cited_source, cited_page in cited_keys:
            if chunk_source and chunk_source.lower() == cited_source.lower():
                if cited_page is None or cited_page == chunk_page:
                    chunk_id = id(chunk)
                    if chunk_id not in seen_chunk_ids:
                        matched_chunks.append(chunk)
                        seen_chunk_ids.add(chunk_id)
                    break

    return matched_chunks


# ── PRETTY PRINT ──────────────────────────────────────────────────────────────

def print_answer(result: dict, show_sources: bool = True) -> None:
    """Pretty-print a RAG result."""
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
            score     = src.get("score", 0) or 0

            location = f"{source}"
            if page:
                location += f", page {page}"
            if section:
                location += f", section {section}"

            preview = src.get("content", "").replace("\n", " ")[:200]

            print(f"\n  [{i}] {location}")
            print(f"      score={score:.4f} | retrieval={retrieval}")
            print(f"      {preview}...")

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
      python generator.py --repl                            # interactive (recommended)
      python generator.py --repl --top_n 3                  # REPL with custom top_n
      python generator.py "What is RAG?"                    # one-shot
      python generator.py "Explain RAG" --stream            # one-shot streaming
      python generator.py "summary" --filter source_type=pdf --top_n 5
    """
    from retrieval.retriever import Retriever

    # ── REPL MODE ─────────────────────────────────────────────────────────────
    if "--repl" in sys.argv:
        repl_top_n = 5
        if "--top_n" in sys.argv:
            idx = sys.argv.index("--top_n")
            if idx + 1 < len(sys.argv):
                repl_top_n = int(sys.argv[idx + 1])

        retrieval_pool = 20

        print(f"\n{'='*70}")
        print(f"  REPL MODE — loading models (one time)...")
        print(f"{'='*70}\n")

        retriever = Retriever()
        reranker  = Reranker()
        gen       = Generator()

        print(f"\n{'='*70}")
        print(f"  READY — retrieval_pool={retrieval_pool}, top_n={repl_top_n}")
        print(f"  Type your question, or 'exit'/'quit' to leave.")
        print(f"{'='*70}\n")

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

                try:
                    t0 = time.perf_counter()

                    candidates = retriever.search(question, top_n=retrieval_pool)
                    if not candidates:
                        print("[No chunks retrieved]\n")
                        continue

                    chunks = reranker.rerank(question, candidates, top_n=repl_top_n)

                    print()
                    result = gen.stream_with_metadata(
                        question, chunks,
                        on_token=lambda t: print(t, end="", flush=True),
                    )

                    total = time.perf_counter() - t0

                    if result.get("sources"):
                        print(f"\n\n{'─'*70}")
                        print(f"  SOURCES CITED ({len(result['sources'])})")
                        print(f"{'─'*70}")
                        for i, src in enumerate(result["sources"], 1):
                            src_name = src.get("source", "?")
                            page     = src.get("page")
                            section  = src.get("section")
                            location = src_name + (f", page {page}" if page else "")
                            if section:
                                location += f" | {section}"
                            print(f"  [{i}] {location}")

                    print(f"\n[total={total:.2f}s | "
                          f"gen={result['latency_s']}s | "
                          f"tokens={result['tokens']['total']}]\n")

                except KeyboardInterrupt:
                    print("\n[Interrupted — type 'exit' to quit]\n")
                    continue
                except Exception as e:
                    print(f"\n[ERROR] {e}\n")
                    continue

        except KeyboardInterrupt:
            pass
        finally:
            print("\n[REPL] Goodbye.\n")
            retriever.close()

        sys.exit(0)

    # ── ONE-SHOT MODE ─────────────────────────────────────────────────────────
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python generator.py --repl")
        print("  python generator.py <question>")
        print("  python generator.py <question> --stream")
        print("  python generator.py <question> --top_n 5 --filter source_type=pdf")
        sys.exit(1)

    question = sys.argv[1]
    stream   = False
    filters  = None
    top_n    = 5

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
        retriever  = Retriever()
        candidates = retriever.search(question, top_n=10, filters=filters)

        if not candidates:
            print("\n[GENERATOR] No chunks retrieved.\n")
            sys.exit(0)

        reranker = Reranker()
        chunks   = reranker.rerank(question, candidates, top_n=top_n)

        gen = Generator()

        if stream:
            print(f"\n{'='*70}")
            print(f"  ANSWER (streaming)")
            print(f"{'='*70}\n")

            result = gen.stream_with_metadata(
                question, chunks,
                on_token=lambda t: print(t, end="", flush=True),
            )
            print()

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