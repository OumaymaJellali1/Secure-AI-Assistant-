# pipeline.py

from __future__ import annotations

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from retrieve.retriever       import Retriever
from reranking.reranker       import Reranker
from answer_generation.generator import Generator

# ── SINGLETONS — load once, reuse across calls ────────────────────────────────

_retriever = None
_reranker  = None
_generator = None

def _get_components():
    global _retriever, _reranker, _generator

    if _retriever is None:
        _retriever = Retriever()
    if _reranker is None:
        _reranker = Reranker()
    if _generator is None:
        _generator = Generator()

    return _retriever, _reranker, _generator


# ── MAIN PIPELINE ─────────────────────────────────────────────────────────────

def run(
    question        : str,
    mode            : str        = "hybrid",
    filters         : dict|None  = None,
    retrieval_top_n : int        = 10,   # how many chunks RRF returns
    rerank_top_n    : int        = 5,    # how many chunks reranker keeps
    stream          : bool       = False,
) -> dict:
    """
    Full RAG pipeline: retrieve → rerank → generate.

    Args:
        question        : user query
        mode            : retriever mode — "hybrid" | "dense" | "sparse"
        filters         : optional metadata filters e.g. {"source_type": "pdf"}
        retrieval_top_n : chunks passed from RRF to reranker (default 25)
        rerank_top_n    : chunks passed from reranker to generator (default 5)
        stream          : stream tokens to stdout while capturing result

    Returns:
        Same dict as Generator.generate() / stream_with_metadata()
        {
          "answer"         : str,
          "sources"        : list,   # only cited chunks
          "used_chunks"    : list,   # all chunks sent to generator
          "model"          : str,
          "tokens"         : dict,
          "context_tokens" : int,
          "no_answer"      : bool,
          "latency_s"      : float,  # ← full pipeline latency (retrieve+rerank+generate)
        }
    """
    # ── START TOTAL PIPELINE TIMER ────────────────────────────────────────────
    pipeline_start = time.perf_counter()

    retriever, reranker, generator = _get_components()

    # ── STEP 1: RETRIEVE (dense + sparse → RRF) ───────────────────────────────
    chunks = retriever.search(
        query   = question,
        mode    = mode,
        filters = filters,
        top_n   = retrieval_top_n,   # 25 candidates for reranker
    )

    if not chunks:
        return {
            "answer"         : "I don't have enough information in the provided context to answer this question.",
            "sources"        : [],
            "used_chunks"    : [],
            "model"          : "",
            "tokens"         : {"prompt": 0, "completion": 0, "total": 0},
            "context_tokens" : 0,
            "no_answer"      : True,
            "latency_s"      : round(time.perf_counter() - pipeline_start, 3),
        }

    # ── STEP 2: RERANK (cross-encoder → top 5) ────────────────────────────────
    chunks = reranker.rerank(
        query  = question,
        chunks = chunks,
        top_n  = rerank_top_n,       # 5 best chunks for generator
    )

    # ── STEP 3: GENERATE ──────────────────────────────────────────────────────
    if stream:
        result = generator.stream_with_metadata(
            question, chunks,
            on_token=lambda t: print(t, end="", flush=True),
        )
        print()  # newline after stream ends
    else:
        result = generator.generate(question, chunks)

    # ── OVERRIDE latency_s with full pipeline duration ────────────────────────
    result["latency_s"] = round(time.perf_counter() - pipeline_start, 3)

    return result


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Usage:
      python pipeline.py "What is RAG?"
      python pipeline.py "What is RAG?" --stream
      python pipeline.py "What is RAG?" --mode dense --filter source_type=pdf
    """
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python pipeline.py <question>")
        print("  python pipeline.py <question> --stream")
        print("  python pipeline.py <question> --mode dense --filter source_type=pdf")
        sys.exit(1)

    question = sys.argv[1]
    stream   = "--stream" in sys.argv
    mode     = "hybrid"
    filters  = None

    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == "--mode" and i + 1 < len(sys.argv):
            mode = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--filter" and i + 1 < len(sys.argv):
            key, val = sys.argv[i + 1].split("=", 1)
            filters  = {key: val}
            i += 2
        else:
            i += 1

    result = run(question, mode=mode, filters=filters, stream=stream)

    # Print answer if not streaming (streaming already printed tokens)
    if not stream:
        from answer_generation.generator import print_answer
        print_answer(result)
    else:
        # Just print sources + metadata for stream mode
        if result.get("sources"):
            print(f"\n{'='*60}")
            print(f"  SOURCES CITED ({len(result['sources'])})")
            print(f"{'='*60}")
            for i, src in enumerate(result["sources"], 1):
                source  = src.get("source", "?")
                page    = src.get("page")
                section = src.get("section")
                location = source + (f", page {page}" if page else "")
                if section:
                    location += f" | {section}"
                print(f"  [{i}] {location}")

        print(f"\n{'─'*60}")
        print(f"  Model   : {result['model']}")
        print(f"  Latency : {result.get('latency_s')}s")
        print(f"  Tokens  : {result['tokens']['prompt']} + {result['tokens']['completion']}")
        print(f"{'='*60}\n")