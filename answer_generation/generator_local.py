from __future__ import annotations

import re
import sys
import os
import time
from typing import Iterator, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import ollama

from answer_generation.prompt_builder_local import PromptBuilder, BuiltPrompt
from reranking.reranker        import Reranker


# ── CONFIG ────────────────────────────────────────────────────────────────────

OLLAMA_HOST     : str   = "http://localhost:11434"
MODEL_NAME      : str   = "qwen2.5:7b"

# Context window (Qwen2.5 supports up to 128k; 4096 is plenty for 5 chunks).
# Smaller NUM_CTX = less KV-cache memory, faster prefill on CPU.
NUM_CTX         : int   = 4096

# Generation params
TEMPERATURE     : float = 0.15          # low = factual, not creative
TOP_P           : float = 0.90
TOP_K           : int   = 40
REPEAT_PENALTY  : float = 1.1
NUM_PREDICT     : int   = 1500          # max tokens the model generates

# Keep model loaded in Ollama for 30 minutes after last query.
# Use "-1" to keep forever, "5m" for aggressive memory saving.
KEEP_ALIVE      : str   = "30m"

# No-answer detection — the model is prompted to say this when unsure
NO_ANSWER_SENTINEL = "i don't have enough information in the provided context"


# ── GENERATOR ─────────────────────────────────────────────────────────────────

class Generator:
    """
    RAG answer generator using Qwen2.5-7B via Ollama.

    Args:
        model          : Ollama model tag
        host           : Ollama server URL
        num_ctx        : context window tokens
        temperature    : sampling temperature (lower = more factual)
        top_p          : nucleus sampling
        top_k          : top-k sampling
        repeat_penalty : penalize repetition
        num_predict    : max tokens to generate
        keep_alive     : how long Ollama keeps the model in memory
    """

    def __init__(
        self,
        model          : str   = MODEL_NAME,
        host           : str   = OLLAMA_HOST,
        num_ctx        : int   = NUM_CTX,
        temperature    : float = TEMPERATURE,
        top_p          : float = TOP_P,
        top_k          : int   = TOP_K,
        repeat_penalty : float = REPEAT_PENALTY,
        num_predict    : int   = NUM_PREDICT,
        keep_alive     : str   = KEEP_ALIVE,
    ) -> None:
        self.model          = model
        self.num_ctx        = num_ctx
        self.temperature    = temperature
        self.top_p          = top_p
        self.top_k          = top_k
        self.repeat_penalty = repeat_penalty
        self.num_predict    = num_predict
        self.keep_alive     = keep_alive

        self._client         = ollama.Client(host=host)
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

        built   = self._prompt_builder.build(query, chunks)
        options = self._build_options(temperature)

        print(f"[GENERATOR] Calling {self.model} | "
              f"context_tokens≈{built.total_context_tokens} | "
              f"chunks_used={len(built.used_chunks)}")

        t0 = time.perf_counter()
        response = self._client.chat(
            model      = self.model,
            messages   = built.messages,
            options    = options,
            keep_alive = self.keep_alive,
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

        built   = self._prompt_builder.build(query, chunks)
        options = self._build_options(temperature)

        print(f"[GENERATOR] Streaming {self.model} | "
              f"context_tokens≈{built.total_context_tokens} | "
              f"chunks_used={len(built.used_chunks)}")

        stream = self._client.chat(
            model      = self.model,
            messages   = built.messages,
            options    = options,
            stream     = True,
            keep_alive = self.keep_alive,
        )

        for chunk in stream:
            token = chunk["message"]["content"]
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

        built   = self._prompt_builder.build(query, chunks)
        options = self._build_options(temperature)

        print(f"[GENERATOR] Streaming {self.model} | "
              f"context_tokens≈{built.total_context_tokens} | "
              f"chunks_used={len(built.used_chunks)}")

        full_answer: list[str] = []
        prompt_tokens     = 0
        completion_tokens = 0

        t0 = time.perf_counter()

        stream = self._client.chat(
            model      = self.model,
            messages   = built.messages,
            options    = options,
            stream     = True,
            keep_alive = self.keep_alive,
        )

        for chunk in stream:
            token = chunk["message"]["content"]
            if token:
                full_answer.append(token)
                if on_token:
                    on_token(token)
            # Capture token counts from the final chunk
            if chunk.get("done"):
                prompt_tokens     = chunk.get("prompt_eval_count", 0)
                completion_tokens = chunk.get("eval_count", 0)

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
        response  : dict,
        built     : BuiltPrompt,
        latency_s : float,
    ) -> dict:
        """Parse non-streaming Ollama response."""
        answer    = response["message"]["content"].strip()
        no_answer = NO_ANSWER_SENTINEL in answer.lower()

        prompt_tokens     = response.get("prompt_eval_count", 0)
        completion_tokens = response.get("eval_count", 0)
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

    # ── PRIVATE: OPTIONS ──────────────────────────────────────────────────────

    def _build_options(self, temperature_override: Optional[float] = None) -> dict:
        return {
            "num_ctx"        : self.num_ctx,
            "temperature"    : temperature_override if temperature_override is not None else self.temperature,
            "top_p"          : self.top_p,
            "top_k"          : self.top_k,
            "repeat_penalty" : self.repeat_penalty,
            "num_predict"    : self.num_predict,
        }

    # ── PRIVATE: HEALTH CHECK ─────────────────────────────────────────────────

    def _ensure_healthy(self) -> None:
        """Verify Ollama is reachable and model is available. Cached."""
        if self._healthy:
            return

        print(f"[GENERATOR] Health check — Ollama @ {OLLAMA_HOST} ...")

        try:
            available = [m["model"] for m in self._client.list()["models"]]
        except Exception as exc:
            raise ConnectionError(
                f"Cannot reach Ollama. Is ollama serve running?\n  → {exc}"
            ) from exc

        normalized = [m.split(":")[0] for m in available]
        target     = self.model.split(":")[0]

        if target not in normalized:
            raise RuntimeError(
                f"Model '{self.model}' not in Ollama.\n"
                f"  Pull it: ollama pull {self.model}\n"
                f"  Available: {available}"
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
                # No page in citation → match any page from that source
                # Page in citation → must match exactly
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
    from retrieve.retriever import Retriever

    # ── REPL MODE ─────────────────────────────────────────────────────────────
    # Loads BGE-M3, reranker, and Ollama model once. Every subsequent query
    # is fast because nothing reloads.
    if "--repl" in sys.argv:
        repl_top_n = 5
        if "--top_n" in sys.argv:
            idx = sys.argv.index("--top_n")
            if idx + 1 < len(sys.argv):
                repl_top_n = int(sys.argv[idx + 1])

        retrieval_pool = 20   # candidates fed to reranker

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

                    # 1. Retrieve candidate pool
                    candidates = retriever.search(question, top_n=retrieval_pool)
                    if not candidates:
                        print("[No chunks retrieved]\n")
                        continue

                    # 2. Rerank — pick best chunks for the LLM
                    chunks = reranker.rerank(question, candidates, top_n=repl_top_n)

                    # 3. Generate answer (streamed)
                    print()
                    result = gen.stream_with_metadata(
                        question, chunks,
                        on_token=lambda t: print(t, end="", flush=True),
                    )

                    total = time.perf_counter() - t0

                    # Show sources cited
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

    # Parse args
    question = sys.argv[1]
    stream   = False
    filters  = None
    top_n    = 5   # final chunks going to the LLM (post-rerank)

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
        # Step 1: Retrieve candidate pool
        retriever  = Retriever()
        candidates = retriever.search(question, top_n=10, filters=filters)

        if not candidates:
            print("\n[GENERATOR] No chunks retrieved.\n")
            sys.exit(0)

        # Step 2: Rerank
        reranker = Reranker()
        chunks   = reranker.rerank(question, candidates, top_n=top_n)

        # Step 3: Generate
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