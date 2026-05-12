"""
retriever.py
────────────
Hybrid retriever for RAG — dense + sparse search over Qdrant with RRF fusion.

UPDATED: Added id access-control filter.
  - search() now accepts caller_id + is_admin
  - When caller_id is provided and is_admin=False, a hard server-side filter
    on id is ANDed into every query — users only see their own Gmail data
  - Admin (is_admin=True) gets no id filter — sees all users' data
  - Non-Gmail chunks (pdf, docx, eml…) have id=None and are always visible

Three search modes:
  - hybrid : dense + sparse → RRF fusion (default, best overall)
  - dense  : semantic only  (better for vague / conceptual queries)
  - sparse : keyword only   (better for exact terms, IDs, model names)
"""

from __future__ import annotations

from typing import Literal, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Filter,
    FieldCondition,
    MatchValue,
    SparseVector,
    IsNullCondition,
    PayloadField,
)

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from embedding.embedder import embed_query

# ── CONFIG ────────────────────────────────────────────────────────

QDRANT_URL      = "http://localhost:6333"
COLLECTION_NAME = "rag_chunks_voyage"

DENSE_K  = 20
SPARSE_K = 20
TOP_N    = 20
RRF_K    = 20

SearchMode = Literal["hybrid", "dense", "sparse"]

_client: Optional[QdrantClient] = None


def _get_client() -> QdrantClient:
    global _client
    if _client is None:
        print(f"[RETRIEVER] Connecting to {QDRANT_URL}...")
        _client = QdrantClient(url=QDRANT_URL)
        print(f"[RETRIEVER] Connected.")
    return _client


# ── FILTER BUILDER ────────────────────────────────────────────────

def _build_filter(filters: dict) -> Filter:
    """
    Convert a plain dict of field → value into a Qdrant Filter.
    All conditions are ANDed together.
    """
    SUPPORTED = {"source", "source_type", "type", "section", "page",
                 "document_id", "uploaded_by"}
    conditions = []
    for field, value in filters.items():
        if field not in SUPPORTED:
            print(f"[RETRIEVER] Warning: filter field '{field}' may not be indexed.")
        conditions.append(FieldCondition(key=field, match=MatchValue(value=value)))
    return Filter(must=conditions)


def _build_user_acl_filter(caller_id: str) -> Filter:
    """
    Build the access-control filter for a regular (non-admin) user.

    Logic: return chunks where EITHER:
      (a) id == caller_id  (this user's Gmail data)
      OR
      (b) id IS NULL       (non-Gmail chunks: pdf, docx, eml… always public)

    This is expressed in Qdrant as a SHOULD (OR) condition.
    """
    return Filter(
        should=[
            # (a) chunk belongs to this user
            FieldCondition(
                key   = "id",
                match = MatchValue(value=caller_id),
            ),
            # (b) chunk has no id at all (shared / non-Gmail data)
            IsNullCondition(
                is_null=PayloadField(key="id")
            ),
        ]
    )


def _merge_filters(
    legacy_filters   : dict | None,
    retrieval_filter : dict | None,
    acl_filter       : Filter | None = None,
) -> Filter | None:
    """
    Combine:
      - legacy `filters` dict  (field→value)
      - scope `retrieval_filter` from API layer  ({"must": [...]})
      - ACL filter  (built from caller_id)

    All non-None filters are ANDed together at the top level.
    Returns None if there is nothing to filter on.
    """
    must_conditions = []

    if legacy_filters:
        for field, value in legacy_filters.items():
            must_conditions.append(
                FieldCondition(key=field, match=MatchValue(value=value))
            )

    if retrieval_filter:
        for c in retrieval_filter.get("must", []):
            must_conditions.append(
                FieldCondition(
                    key   = c["key"],
                    match = MatchValue(value=c["match"]["value"]),
                )
            )

    if acl_filter is not None:
        # Wrap the SHOULD (OR) ACL filter inside a must condition
        must_conditions.append(acl_filter)

    return Filter(must=must_conditions) if must_conditions else None


# ── MAIN CLASS ────────────────────────────────────────────────────

class Retriever:
    """
    Hybrid retriever over a Qdrant collection.

    Args:
        collection_name : Qdrant collection to search
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
        query            : str,
        mode             : SearchMode       = "hybrid",
        filters          : dict | None      = None,
        top_n            : int | None       = None,
        retrieval_filter : dict | None      = None,
        caller_id        : str | None       = None,   # ← NEW: id of the caller
        is_admin         : bool             = False,  # ← NEW: skip ACL if True
    ) -> list[dict]:
        """
        Search the collection and return ranked chunks.

        Args:
            query            : natural language query string
            mode             : "hybrid" | "dense" | "sparse"
            filters          : simple field→value dict (legacy, ANDed with others)
            top_n            : override default result count for this call
            retrieval_filter : scope filter from the API layer
                               {"must": [{"key": "...", "match": {"value": "..."}}]}
            caller_id        : id of whoever is querying.
                               When provided and is_admin=False, a server-side ACL
                               filter is applied: user sees only their own Gmail data
                               + all non-Gmail (id=None) chunks.
            is_admin         : if True, no ACL filter is applied — admin sees all data.

        Returns:
            List of chunk dicts with content, score, source, page, etc.
        """
        n = top_n or self.top_n

        # Build ACL filter
        acl_filter = None
        if caller_id and not is_admin:
            acl_filter = _build_user_acl_filter(caller_id)
            print(f"[RETRIEVER] ACL filter: id={caller_id} (own Gmail + shared docs)")
        elif is_admin:
            print(f"[RETRIEVER] ACL filter: ADMIN — no filter applied")

        qdrant_filter = _merge_filters(filters, retrieval_filter, acl_filter)

        print(f"\n[RETRIEVER] Query  : {query!r}")
        print(f"[RETRIEVER] Mode   : {mode}")
        if filters:
            print(f"[RETRIEVER] Legacy filter : {filters}")
        if retrieval_filter:
            print(f"[RETRIEVER] Scope  filter : {retrieval_filter}")

        print(f"[RETRIEVER] Embedding query...")
        dense_vec, sparse_vec = embed_query(query)

        if mode == "hybrid":
            results = self._hybrid_search(dense_vec, sparse_vec, qdrant_filter, n)
        elif mode == "dense":
            results = self._dense_search(dense_vec, qdrant_filter, n)
        elif mode == "sparse":
            results = self._sparse_search(sparse_vec, qdrant_filter, n)
        else:
            raise ValueError(
                f"Unknown mode: {mode!r}. Choose 'hybrid', 'dense', or 'sparse'."
            )

        print(f"[RETRIEVER] Returning {len(results)} chunks.\n")
        return results

    # ── private search helpers ────────────────────────────────────

    def _hybrid_search(self, dense_vec, sparse_vec, qdrant_filter, n):
        print(f"[RETRIEVER] Dense search  (K={self.dense_k})...")
        dense_hits = self._run_dense_search(dense_vec, qdrant_filter, self.dense_k)

        print(f"[RETRIEVER] Sparse search (K={self.sparse_k})...")
        sparse_hits = self._run_sparse_search(sparse_vec, qdrant_filter, self.sparse_k)

        print(f"[RETRIEVER] Dense  hits : {len(dense_hits)}")
        print(f"[RETRIEVER] Sparse hits : {len(sparse_hits)}")

        fused = _rrf_fusion(dense_hits, sparse_hits, k=self.rrf_k)
        return _format_results(fused[:n])

    def _dense_search(self, dense_vec, qdrant_filter, n):
        print(f"[RETRIEVER] Dense search (K={n})...")
        hits = self._run_dense_search(dense_vec, qdrant_filter, n)
        wrapped = [
            {"id": h.id, "payload": h.payload,
             "rrf_score": 1 / (self.rrf_k + i + 1), "retrieval": "dense"}
            for i, h in enumerate(hits)
        ]
        return _format_results(wrapped)

    def _sparse_search(self, sparse_vec, qdrant_filter, n):
        print(f"[RETRIEVER] Sparse search (K={n})...")
        hits = self._run_sparse_search(sparse_vec, qdrant_filter, n)
        wrapped = [
            {"id": h.id, "payload": h.payload,
             "rrf_score": 1 / (self.rrf_k + i + 1), "retrieval": "sparse"}
            for i, h in enumerate(hits)
        ]
        return _format_results(wrapped)

    def _run_dense_search(self, dense_vec, qdrant_filter, k):
        response = self.client.query_points(
            collection_name = self.collection_name,
            query           = dense_vec,
            using           = "dense",
            query_filter    = qdrant_filter,
            limit           = k,
            with_payload    = True,
        )
        return response.points

    def _run_sparse_search(self, sparse_vec, qdrant_filter, k):
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


# ── RRF FUSION ────────────────────────────────────────────────────

def _rrf_fusion(dense_hits, sparse_hits, k=RRF_K) -> list[dict]:
    scores:   dict[str, float] = {}
    payloads: dict[str, dict]  = {}
    sources:  dict[str, set]   = {}

    def _accumulate(hits, label):
        for rank, hit in enumerate(hits, start=1):
            pid = str(hit.id)
            scores[pid]   = scores.get(pid, 0.0) + 1.0 / (k + rank)
            payloads[pid] = hit.payload or {}
            sources.setdefault(pid, set()).add(label)

    _accumulate(dense_hits,  "dense")
    _accumulate(sparse_hits, "sparse")

    fused = [
        {
            "id":        pid,
            "payload":   payloads[pid],
            "rrf_score": score,
            "retrieval": "both" if len(sources[pid]) == 2 else next(iter(sources[pid])),
        }
        for pid, score in scores.items()
    ]
    fused.sort(key=lambda x: x["rrf_score"], reverse=True)
    return fused


# ── RESULT FORMATTER ──────────────────────────────────────────────

def _format_results(fused: list[dict]) -> list[dict]:
    out = []
    for item in fused:
        p = item["payload"]
        out.append({
            "content":      p.get("content", ""),
            "score":        round(item["rrf_score"], 6),
            "retrieval":    item["retrieval"],
            "source":       p.get("source"),
            "source_type":  p.get("source_type"),
            "type":         p.get("type"),
            "section":      p.get("section"),
            "page":         p.get("page"),
            "chunk_index":  p.get("chunk_index"),
            "chunk_total":  p.get("chunk_total"),
            "token_count":  p.get("token_count"),
            "document_id":  p.get("document_id"),
            "uploaded_by":  p.get("uploaded_by"),
            # Access-control fields (present for Gmail chunks, None for others)
            "id":      p.get("id"),
            "owner_email":  p.get("owner_email"),
        })
    return out


# ── CLI ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    raw, args, flags, i = sys.argv[1:], [], {}, 0
    while i < len(raw):
        token = raw[i]
        if token.startswith("--"):
            key = token.lstrip("--")
            if "=" in key:
                k, v = key.split("=", 1); flags[k] = v
            elif i + 1 < len(raw) and not raw[i + 1].startswith("--"):
                flags[key] = raw[i + 1]; i += 1
            else:
                flags[key] = True
        else:
            args.append(token)
        i += 1

    if not args:
        print("Usage: python retriever.py <query> [--mode hybrid|dense|sparse]"
              " [--filter field=value] [--caller_id id] [--admin]")
        sys.exit(1)

    query     = args[0]
    mode      = flags.get("mode", "hybrid")
    caller_id = flags.get("caller_id", None)
    is_admin  = "admin" in flags
    filters   = None
    if "filter" in flags:
        field, value = flags["filter"].split("=")
        filters = {field: value}

    retriever = Retriever()
    results   = retriever.search(
        query,
        mode      = mode,
        filters   = filters,
        caller_id = caller_id,
        is_admin  = is_admin,
    )

    print(f"\n{'='*60}")
    print(f"  TOP {len(results)} RESULTS  |  mode={mode}"
          f"  caller={caller_id or 'anonymous'}  admin={is_admin}")
    print(f"{'='*60}")
    for i, r in enumerate(results, 1):
        print(f"\n── Result {i}")
        print(f"   Score      : {r['score']}  [{r['retrieval']}]")
        print(f"   Source     : {r['source']}  (type={r['source_type']})")
        print(f"   Document ID: {r.get('document_id')}")
        print(f"   User ID    : {r.get('id') or '(shared)'}")
        print(f"   Content    :")
        preview = r["content"][:400].replace("\n", "\n             ")
        print(f"             {preview}")
    print(f"\n{'='*60}\n")