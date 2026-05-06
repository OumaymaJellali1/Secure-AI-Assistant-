"""
retriever.py
────────────
Hybrid retriever for RAG — dense + sparse search over Qdrant with RRF fusion.

Three search modes:
  - hybrid : dense + sparse → RRF fusion (default, best overall)
  - dense  : semantic only  (better for vague / conceptual queries)
  - sparse : keyword only   (better for exact terms, IDs, model names)

Steps (hybrid mode):
  1. Embed query → dense vector + sparse vector  (via embedder.py)
  2. Search dense  vector space → top-K by cosine similarity
  3. Search sparse vector space → top-K by dot product
  4. RRF fusion on the two ranked lists
  5. Optional metadata filter (applied inside Qdrant before vector search)
  6. Return top-N chunks (content + metadata + retrieval source tag)

Dependencies:
  pip install qdrant-client
  (embedder.py and qdrant_store.py must be in the same directory)

Usage:
  from retriever import Retriever

  r = Retriever()
  results = r.search("what is RAG?")
  results = r.search("BGE-M3", mode="sparse")
  results = r.search("summarise the PDF", filters={"source_type": "pdf"})
"""

from __future__ import annotations

from typing import Literal, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Filter,
    FieldCondition,
    MatchValue,
    SparseVector,
    Query,
    FusionQuery,
    SparseIndexParams,
)

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from embedding.embedder import embed_query

# ── CONFIG ────────────────────────────────────────────────────────────────────

QDRANT_URL      = "http://localhost:6333"
COLLECTION_NAME = "rag_chunks"

DENSE_K  = 20
SPARSE_K = 20

TOP_N = 20


RRF_K = 20

SearchMode = Literal["hybrid", "dense", "sparse"]



_client: Optional[QdrantClient] = None


def _get_client() -> QdrantClient:
    global _client
    if _client is None:
        print(f"[RETRIEVER] Connecting to {QDRANT_URL}...")
        _client = QdrantClient(url=QDRANT_URL)
        print(f"[RETRIEVER] Connected.")
    return _client



class Retriever:
    """
    Hybrid retriever over a Qdrant collection.

    Args:
        collection_name : Qdrant collection to search (default: "rag_chunks")
        dense_k         : candidates fetched from dense search
        sparse_k        : candidates fetched from sparse search
        top_n           : results returned after fusion
        rrf_k           : RRF constant (controls rank-score steepness)
    """

    def __init__(
        self,
        collection_name : str = COLLECTION_NAME,
        dense_k         : int = DENSE_K,
        sparse_k        : int = SPARSE_K,
        top_n           : int = TOP_N,
        rrf_k           : int = RRF_K,
    ) -> None:
        self.collection_name = collection_name
        self.dense_k         = dense_k
        self.sparse_k        = sparse_k
        self.top_n           = top_n
        self.rrf_k           = rrf_k
        self.client          = _get_client()


    def search(
        self,
        query   : str,
        mode    : SearchMode        = "hybrid",
        filters : Optional[dict]    = None,
        top_n   : Optional[int]     = None,
    ) -> list[dict]:
        """
        Search the collection and return ranked chunks.

        Args:
            query   : natural language query string
            mode    : "hybrid" | "dense" | "sparse"
            filters : optional dict of payload field → value to filter by.
                      Supported fields: source, source_type, type, section, page.
                      Examples:
                        {"source_type": "pdf"}
                        {"type": "text"}
                        {"source": "report.pdf"}
            top_n   : override default result count for this call

        Returns:
            List of dicts, each containing:
              - content        : str
              - score          : float  (RRF score, higher = better)
              - retrieval      : "dense" | "sparse" | "both"
              - source         : str | None
              - source_type    : str | None
              - type           : str | None  ("text" | "table" | "image")
              - section        : str | None
              - page           : int | None
              - chunk_index    : int | None
              - chunk_total    : int | None
              - token_count    : int | None
        """
        n       = top_n or self.top_n
        qdrant_filter = _build_filter(filters) if filters else None

        print(f"\n[RETRIEVER] Query  : {query!r}")
        print(f"[RETRIEVER] Mode   : {mode}")
        if filters:
            print(f"[RETRIEVER] Filter : {filters}")

        print(f"[RETRIEVER] Embedding query...")
        dense_vec, sparse_vec = embed_query(query)

        # ── SEARCH
        if mode == "hybrid":
            results = self._hybrid_search(dense_vec, sparse_vec, qdrant_filter, n)
        elif mode == "dense":
            results = self._dense_search(dense_vec, qdrant_filter, n)
        elif mode == "sparse":
            results = self._sparse_search(sparse_vec, qdrant_filter, n)
        else:
            raise ValueError(f"Unknown mode: {mode!r}. Choose 'hybrid', 'dense', or 'sparse'.")

        print(f"[RETRIEVER] Returning {len(results)} chunks.\n")
        return results


    def _hybrid_search(
        self,
        dense_vec     : list[float],
        sparse_vec    : dict,
        qdrant_filter,
        n             : int,
    ) -> list[dict]:
        """Dense + sparse search → RRF fusion → top-N."""

        print(f"[RETRIEVER] Dense search  (K={self.dense_k})...")
        dense_hits = self._run_dense_search(dense_vec, qdrant_filter, self.dense_k)

        print(f"[RETRIEVER] Sparse search (K={self.sparse_k})...")
        sparse_hits = self._run_sparse_search(sparse_vec, qdrant_filter, self.sparse_k)

        print(f"[RETRIEVER] Dense  hits : {len(dense_hits)}")
        print(f"[RETRIEVER] Sparse hits : {len(sparse_hits)}")

        # RRF fusion
        fused = _rrf_fusion(dense_hits, sparse_hits, k=self.rrf_k)

        return _format_results(fused[:n])

    def _dense_search(
        self,
        dense_vec     : list[float],
        qdrant_filter,
        n             : int,
    ) -> list[dict]:
        """Dense-only search → top-N."""
        print(f"[RETRIEVER] Dense search (K={n})...")
        hits = self._run_dense_search(dense_vec, qdrant_filter, n)
        # Wrap in RRF structure for uniform formatting
        wrapped = [{"id": h.id, "payload": h.payload, "rrf_score": 1/(self.rrf_k + i + 1), "retrieval": "dense"}
                   for i, h in enumerate(hits)]
        return _format_results(wrapped)

    def _sparse_search(
        self,
        sparse_vec    : dict,
        qdrant_filter,
        n             : int,
    ) -> list[dict]:
        """Sparse-only search → top-N."""
        print(f"[RETRIEVER] Sparse search (K={n})...")
        hits = self._run_sparse_search(sparse_vec, qdrant_filter, n)
        wrapped = [{"id": h.id, "payload": h.payload, "rrf_score": 1/(self.rrf_k + i + 1), "retrieval": "sparse"}
                   for i, h in enumerate(hits)]
        return _format_results(wrapped)


    def _run_dense_search(self, dense_vec, qdrant_filter, k: int):
        """Execute dense vector search against Qdrant."""
        response = self.client.query_points(
            collection_name = self.collection_name,
            query           = dense_vec,
            using           = "dense",
            query_filter    = qdrant_filter,
            limit           = k,
            with_payload    = True,
        )
        return response.points

    def _run_sparse_search(self, sparse_vec: dict, qdrant_filter, k: int):
        """Execute sparse vector search against Qdrant."""
        response = self.client.query_points(
            collection_name = self.collection_name,
            query           = SparseVector(
                indices = sparse_vec["indices"],
                values  = sparse_vec["values"],
            ),
            using        = "sparse",
            query_filter = qdrant_filter,
            limit        = k,
            with_payload = True,
        )
        return response.points



def _rrf_fusion(
    dense_hits  : list,
    sparse_hits : list,
    k           : int = RRF_K,
) -> list[dict]:
    """
    Reciprocal Rank Fusion over two ranked hit lists.

    Formula: score(d) = Σ 1 / (k + rank(d))
    Rank is 1-indexed. Points appearing in both lists get scores summed.
    Tracks whether each point came from dense-only, sparse-only, or both.

    Args:
        dense_hits  : Qdrant ScoredPoint list from dense search
        sparse_hits : Qdrant ScoredPoint list from sparse search
        k           : RRF constant

    Returns:
        List of dicts sorted by rrf_score descending.
    """
    scores:   dict[str, float] = {}
    payloads: dict[str, dict]  = {}
    sources:  dict[str, set]   = {}

    def _accumulate(hits: list, label: str) -> None:
        for rank, hit in enumerate(hits, start=1):
            pid = str(hit.id)
            scores[pid]   = scores.get(pid, 0.0) + 1.0 / (k + rank)
            payloads[pid] = hit.payload or {}
            if pid not in sources:
                sources[pid] = set()
            sources[pid].add(label)

    _accumulate(dense_hits,  "dense")
    _accumulate(sparse_hits, "sparse")

    fused = [
        {
            "id"        : pid,
            "payload"   : payloads[pid],
            "rrf_score" : score,
            "retrieval" : "both" if len(sources[pid]) == 2 else next(iter(sources[pid])),
        }
        for pid, score in scores.items()
    ]

    fused.sort(key=lambda x: x["rrf_score"], reverse=True)
    return fused


# ── FILTER BUILDER ────────────────────────────────────────────────────────────

def _build_filter(filters: dict) -> Filter:
    """
    Convert a plain dict of field → value into a Qdrant Filter.

    Supported fields: source, source_type, type, section, page
    All conditions are ANDed together.

    Example:
      {"source_type": "pdf", "type": "text"}
      → Filter(must=[FieldCondition(key="source_type", match=MatchValue(value="pdf")),
                     FieldCondition(key="type",        match=MatchValue(value="text"))])
    """
    SUPPORTED = {"source", "source_type", "type", "section", "page"}
    conditions = []

    for field, value in filters.items():
        if field not in SUPPORTED:
            print(f"[RETRIEVER] Warning: filter field '{field}' is not indexed — may be slow.")
        conditions.append(
            FieldCondition(key=field, match=MatchValue(value=value))
        )

    return Filter(must=conditions)



def _format_results(fused: list[dict]) -> list[dict]:
    """
    Flatten fused RRF results into clean dicts ready for LLM injection.

    Each result dict contains:
      content, score, retrieval, and all metadata fields.
    """
    out = []
    for item in fused:
        p = item["payload"]
        out.append({
            # Core
            "content"     : p.get("content", ""),
            "score"       : round(item["rrf_score"], 6),
            "retrieval"   : item["retrieval"],
            # Metadata
            "source"      : p.get("source"),
            "source_type" : p.get("source_type"),
            "type"        : p.get("type"),
            "section"     : p.get("section"),
            "page"        : p.get("page"),
            "chunk_index" : p.get("chunk_index"),
            "chunk_total" : p.get("chunk_total"),
            "token_count" : p.get("token_count"),
        })
    return out


# ── CLI 

if __name__ == "__main__":
    """
    Quick test from CLI:
      python retriever.py "what is retrieval augmented generation?"
      python retriever.py "BGE-M3" --mode sparse
      python retriever.py "summarise findings" --mode dense --filter source_type=pdf
    """
    import sys

    raw     = sys.argv[1:]
    args    = []
    flags   = {}
    i       = 0
    while i < len(raw):
        token = raw[i]
        if token.startswith("--"):
            key = token.lstrip("--")
            if "=" in key:
                # --key=value style
                k, v = key.split("=", 1)
                flags[k] = v
            elif i + 1 < len(raw) and not raw[i + 1].startswith("--"):
                # --key value style
                flags[key] = raw[i + 1]
                i += 1
            else:
                # boolean flag
                flags[key] = True
        else:
            args.append(token)
        i += 1

    if not args:
        print("Usage: python retriever.py <query> [--mode hybrid|dense|sparse] [--filter field=value]")
        sys.exit(1)

    query   = args[0]
    mode    = flags.get("mode", "hybrid")
    filters = None

    if "filter" in flags:
        field, value = flags["filter"].split("=")
        filters = {field: value}

    retriever = Retriever()
    results   = retriever.search(query, mode=mode, filters=filters)

    print(f"\n{'='*60}")
    print(f"  TOP {len(results)} RESULTS  |  mode={mode}")
    print(f"{'='*60}")

    for i, r in enumerate(results, 1):
        print(f"\n── Result {i}")
        print(f"   Score      : {r['score']}  [{r['retrieval']}]")
        print(f"   Source     : {r['source']}  (type={r['source_type']})")
        print(f"   Chunk type : {r['type']}  |  page={r['page']}  |  section={r['section']}")
        print(f"   Tokens     : {r['token_count']}")
        print(f"   Content    :")
        preview = r["content"][:400].replace("\n", "\n             ")
        print(f"             {preview}")
        if len(r["content"]) > 400:
            print(f"             ... [{len(r['content'])} chars total]")

    print(f"\n{'='*60}\n")