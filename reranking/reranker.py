"""
reranker.py
───────────
Cross-encoder reranker for RAG — refines retrieved candidates using
BAAI/bge-reranker-v2-m3.

CPU-optimized:
  - multi-threading       : forces PyTorch to use all available cores
  - device="cpu"          : explicit device selection
  - content truncation    : caps chunk length before reranker pass
  - fp16                  : halves memory use and speeds up scoring on CPU

Pipeline position:
    query → retriever.search(top_n=10)  →  reranker.rerank(top_n=5)  →  LLM

Dependencies:
    pip install FlagEmbedding torch

Usage:
    from retrieval.reranker import Reranker
    r = Reranker()
    top = r.rerank(query, candidates, top_n=5)
"""

from __future__ import annotations

import os
import time
from typing import Optional

import torch

# Force PyTorch to use all CPU cores BEFORE loading the model.
_CPU_COUNT = os.cpu_count() or 4
torch.set_num_threads(_CPU_COUNT)

from FlagEmbedding import FlagReranker


# ── CONFIG ────────────────────────────────────────────────────────────────────

MODEL_NAME        = "BAAI/bge-reranker-v2-m3"
DEVICE            = "cpu"
DEFAULT_TOP_N     = 5

# Max chunk content length passed to the reranker (in chars, not tokens).
# Quadratic cost in sequence length — keeping this tight saves a lot on CPU.
MAX_CHUNK_CHARS   = 2000

# bge-reranker-v2-m3's internal max sequence length (tokens).
# 512 is plenty for RAG and keeps CPU inference fast.
MAX_SEQ_LENGTH    = 512


# ── RERANKER CLASS ────────────────────────────────────────────────────────────

class Reranker:
    """
    Cross-encoder reranker using BAAI/bge-reranker-v2-m3.

    Multilingual (100+ languages), open-source, and significantly faster
    than jina-reranker-v2-base-multilingual on CPU.

    Args:
        model_name       : HuggingFace model ID
        device           : "cpu" or "cuda"
        max_chunk_chars  : truncate chunk content to this many chars for scoring
        max_seq_length   : transformer sequence length cap (tokens)
    """

    def __init__(
        self,
        model_name      : str = MODEL_NAME,
        device          : str = DEVICE,
        max_chunk_chars : int = MAX_CHUNK_CHARS,
        max_seq_length  : int = MAX_SEQ_LENGTH,
    ) -> None:
        print(f"[RERANKER] Loading {model_name} ...")
        print(f"[RERANKER] Config: device={device}, "
              f"cpu_threads={torch.get_num_threads()}, "
              f"max_seq_length={max_seq_length}")

        t0 = time.perf_counter()

        # use_fp16=True halves memory and speeds up CPU scoring with
        # negligible quality loss. Set to False if you see NaN scores.
        self.model = FlagReranker(
            model_name,
            use_fp16         = True,
            device           = device,
        )

        # Cap the sequence length to avoid slow long-context inference.
        self.model.model.config.max_position_embeddings = max_seq_length

        self.model_name      = model_name
        self.max_chunk_chars = max_chunk_chars

        print(f"[RERANKER] Ready. ({time.perf_counter() - t0:.1f}s)")

    # ── PUBLIC ────────────────────────────────────────────────────────────────

    def rerank(
        self,
        query   : str,
        chunks  : list[dict],
        top_n   : Optional[int] = None,
    ) -> list[dict]:
        """
        Rerank candidate chunks by cross-encoder relevance.

        Args:
            query  : user's question
            chunks : list of chunk dicts (must have "content" field)
            top_n  : how many top chunks to return (default 5)

        Returns:
            List of top-N chunks, sorted by rerank_score descending.
            Each chunk is augmented with:
              - rerank_score    : float in [0, 1] (sigmoid-normalized relevance)
              - retrieval_score : float (original hybrid score, preserved)
        """
        if not chunks:
            return []

        n = top_n if top_n is not None else DEFAULT_TOP_N

        lengths    = [len(c.get("content", "")) for c in chunks]
        avg_length = sum(lengths) / len(lengths) if lengths else 0
        max_length = max(lengths) if lengths else 0

        print(f"[RERANKER] Scoring {len(chunks)} candidates "
              f"(avg={avg_length:.0f} chars, max={max_length} chars, "
              f"trunc={self.max_chunk_chars})...")

        t0 = time.perf_counter()

        # Build (query, chunk_content) pairs, truncated for speed.
        pairs = [
            [query, c.get("content", "")[: self.max_chunk_chars]]
            for c in chunks
        ]

        # FlagReranker.compute_score returns a list of raw relevance scores.
        # normalize=True applies sigmoid → scores in [0, 1].
        scores = self.model.compute_score(pairs, normalize=True)

        # compute_score returns a float (not a list) when given a single pair.
        if not isinstance(scores, list):
            scores = [scores]

        # Attach scores, preserve original retrieval score.
        for chunk, score in zip(chunks, scores):
            chunk["retrieval_score"] = chunk.get("score")
            chunk["rerank_score"]    = float(score)

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


# ── CLI (for testing in isolation) ────────────────────────────────────────────

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