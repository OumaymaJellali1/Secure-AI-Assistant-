"""
reranker.py
───────────
Cross-encoder reranker for RAG — refines retrieved candidates using
Jina Reranker API (jina-reranker-v2-base-multilingual).

API-based (no local model):
  - zero RAM usage         : no model loaded locally
  - multilingual           : 100+ languages supported
  - free tier              : 1M tokens free on jina.ai
  - simple REST call       : just requests, no torch/FlagEmbedding needed

Pipeline position:
    query → retriever.search(top_n=10)  →  reranker.rerank(top_n=5)  →  LLM

Dependencies:
    pip install requests python-dotenv

Environment:
    JINA_API_KEY=jina_xxxxxxxxxxxxxxxxxxxxxxxx  (set in your .env)

Usage:
    from reranking.reranker import Reranker
    r = Reranker()
    top = r.rerank(query, candidates, top_n=5)
"""

from __future__ import annotations

import os
import time
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv()


API_URL       = "https://api.jina.ai/v1/rerank"
MODEL_NAME    = "jina-reranker-v2-base-multilingual"
DEFAULT_TOP_N = 5
MAX_CHUNK_CHARS = 2000


class Reranker:
    """
    API-based reranker using Jina AI (jina-reranker-v2-base-multilingual).

    Multilingual (100+ languages), no local model, no RAM overhead.
    Requires JINA_API_KEY environment variable.

    Args:
        model_name      : Jina model ID
        max_chunk_chars : truncate chunk content to this many chars before sending
    """

    def __init__(
        self,
        model_name      : str = MODEL_NAME,
        max_chunk_chars : int = MAX_CHUNK_CHARS,
    ) -> None:
        self.api_key = os.getenv("JINA_API_KEY")
        if not self.api_key:
            raise ValueError(
                "[RERANKER] JINA_API_KEY is not set. "
                "Add it to your .env file: JINA_API_KEY=jina_xxxxxxxx"
            )

        self.model_name     = model_name
        self.max_chunk_chars = max_chunk_chars
        self.api_url        = API_URL

        print(f"[RERANKER] Jina API ready. Model: {model_name}")


    def rerank(
        self,
        query  : str,
        chunks : list[dict],
        top_n  : Optional[int] = None,
    ) -> list[dict]:
        """
        Rerank candidate chunks by cross-encoder relevance via Jina API.

        Args:
            query  : user's question
            chunks : list of chunk dicts (must have "content" field)
            top_n  : how many top chunks to return (default 5)

        Returns:
            List of top-N chunks, sorted by rerank_score descending.
            Each chunk is augmented with:
              - rerank_score    : float in [0, 1] (relevance score)
              - retrieval_score : float (original hybrid score, preserved)
        """
        if not chunks:
            return []

        n = top_n if top_n is not None else DEFAULT_TOP_N

        lengths    = [len(c.get("content", "")) for c in chunks]
        avg_length = sum(lengths) / len(lengths) if lengths else 0
        max_length = max(lengths) if lengths else 0

        print(f"[RERANKER] Scoring {len(chunks)} candidates via Jina API "
              f"(avg={avg_length:.0f} chars, max={max_length} chars, "
              f"trunc={self.max_chunk_chars})...")

        t0 = time.perf_counter()

        # Prepare documents list — truncate to stay within token limits
        documents = [
            c.get("content", "")[: self.max_chunk_chars]
            for c in chunks
        ]

        try:
            response = requests.post(
                self.api_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type":  "application/json",
                },
                json={
                    "model":     self.model_name,
                    "query":     query,
                    "documents": documents,
                    "top_n":     len(chunks),   # get all scores, we sort ourselves
                },
                timeout=30,
            )
            response.raise_for_status()

        except requests.exceptions.Timeout:
            print("[RERANKER] WARNING: Jina API timed out. Returning original order.")
            return chunks[:n]

        except requests.exceptions.RequestException as e:
            print(f"[RERANKER] WARNING: Jina API error: {e}. Returning original order.")
            return chunks[:n]

        results = response.json().get("results", [])

        # Map scores back to original chunks by index
        for result in results:
            idx   = result["index"]
            score = result["relevance_score"]
            chunks[idx]["retrieval_score"] = chunks[idx].get("score")
            chunks[idx]["rerank_score"]    = float(score)

        # Fill any chunks that didn't get a score (safety net)
        for chunk in chunks:
            if "rerank_score" not in chunk:
                chunk["rerank_score"]    = 0.0
                chunk["retrieval_score"] = chunk.get("score")

        reranked = sorted(
            chunks,
            key     = lambda c: c["rerank_score"],
            reverse = True,
        )

        latency = time.perf_counter() - t0
        top_sc  = reranked[0]["rerank_score"] if reranked else 0.0
        bot_sc  = reranked[-1]["rerank_score"] if reranked else 0.0

        print(f"[RERANKER] Done. ({latency:.2f}s, "
              f"top={top_sc:.4f}, bottom={bot_sc:.4f})")

        return reranked[:n]


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from retrieve.retriever import Retriever

    if len(sys.argv) < 2:
        print("Usage: python reranker.py <query> [--pool N] [--top_n N]")
        sys.exit(1)

    query = sys.argv[1]
    pool  = 20
    top   = 5

    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == "--pool" and i + 1 < len(sys.argv):
            pool = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--top_n" and i + 1 < len(sys.argv):
            top = int(sys.argv[i + 1])
            i += 2
        else:
            i += 1

    retriever = Retriever()
    reranker  = Reranker()

    try:
        candidates = retriever.search(query, top_n=pool)
        if not candidates:
            print("No candidates retrieved.")
            sys.exit(0)

        retrieval_order = [
            (c.get("source"), c.get("page"), c.get("score"))
            for c in candidates
        ]

        top_chunks = reranker.rerank(query, candidates, top_n=top)

        print(f"\n{'='*80}")
        print(f"  RETRIEVAL TOP {min(top, len(retrieval_order))}  vs  "
              f"RERANKER TOP {len(top_chunks)}")
        print(f"{'='*80}")

        print(f"\n  BEFORE (hybrid retrieval):")
        for i, (src, page, score) in enumerate(retrieval_order[:top], 1):
            print(f"    [{i}] {src}, p{page}  (score={score:.4f})")

        print(f"\n  AFTER (reranker):")
        for i, chunk in enumerate(top_chunks, 1):
            src     = chunk.get("source", "?")
            page    = chunk.get("page", "?")
            rerank  = chunk.get("rerank_score", 0)
            retriev = chunk.get("retrieval_score", 0)
            preview = chunk.get("content", "")[:150].replace("\n", " ")
            print(f"    [{i}] {src}, p{page}  "
                  f"(rerank={rerank:.4f}, retrieval={retriev:.4f})")
            print(f"        {preview}...")

        print(f"\n{'='*80}\n")

    finally:
        retriever.close()