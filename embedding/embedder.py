"""
embedder.py
───────────
Embeds normalized chunks using Voyage AI (voyage-multilingual-2).
Dense vectors via Voyage API + sparse vectors via BM25 (local, lightweight).
Produces dense + sparse vectors for hybrid retrieval in Qdrant.

Usage:
    from embedder import embed_chunks, embed_query, embedding_health_check
"""

from __future__ import annotations
import os
import time
import numpy as np
from typing import Any


# ── CONFIG ────────────────────────────────────────────────────────────────────

EMBED_BATCH_SIZE  = 32           # Voyage supports up to 128 per batch
EMBED_MAX_LENGTH  = 512          # tokens (used by BM25 truncation hint)
VOYAGE_MODEL      = "voyage-multilingual-2"
VOYAGE_API_KEY    = os.environ.get("VOYAGE_API_KEY", "")

# Free tier: 3 RPM → sleep 21s between batches to stay safe.
# Set to 0 once you add a payment method to your Voyage account.
FREE_TIER_SLEEP_S = 21


# ── CLIENTS (lazy-loaded singletons) ─────────────────────────────────────────

_voyage_client = None


def _get_voyage():
    global _voyage_client
    if _voyage_client is None:
        if not VOYAGE_API_KEY:
            raise EnvironmentError(
                "[EMBEDDER] VOYAGE_API_KEY is not set. "
                "Export it or add it to your .env file."
            )
        import voyageai
        print("[EMBEDDER] Initializing Voyage AI client...")
        _voyage_client = voyageai.Client(api_key=VOYAGE_API_KEY)
        print("[EMBEDDER] Voyage AI client ready.")
    return _voyage_client


# ── SPARSE via TF-IDF-style token weights ────────────────────────────────────

def _to_sparse_bm25(text: str) -> dict:
    """
    Produce a lightweight sparse vector from raw token frequencies.
    This is a stateless BM25-lite: term frequency with basic normalization.
    Compatible with Qdrant sparse vector format.

    For production, replace with a fitted BM25 or SPLADE model.
    """
    import re
    from collections import Counter
    from math import log

    # Basic tokenization — lowercase, split on non-alphanumeric
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())

    if not tokens:
        return {"indices": [], "values": []}

    # Simple stopword filter
    STOPWORDS = {
        "the", "a", "an", "is", "it", "in", "on", "at", "to",
        "of", "and", "or", "for", "with", "this", "that", "be",
        "are", "was", "were", "have", "has", "had", "not", "by",
    }
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]

    if not tokens:
        return {"indices": [], "values": []}

    tf      = Counter(tokens)
    total   = len(tokens)
    vocab   = sorted(set(tokens))

    # Assign stable integer IDs via hash (keeps it stateless)
    indices, values = [], []
    for term in vocab:
        idx    = abs(hash(term)) % (2 ** 20)   # 1M token space
        freq   = tf[term] / total
        weight = freq * (1 + log(1 + tf[term]))  # TF-lite
        indices.append(idx)
        values.append(round(weight, 6))

    return {"indices": indices, "values": values}


# ── CORE EMBED FUNCTION ───────────────────────────────────────────────────────

def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed a list of normalized chunks.
    Dense vectors from Voyage AI, sparse from local BM25-lite.
    Handles free-tier rate limits (3 RPM) with sleep + exponential backoff retry.

    Args:
        chunks : normalized chunks from ingest_file()

    Returns:
        Same chunks + dense_vector (list[float], dim=1024)
                     + sparse_vector ({"indices": [...], "values": [...]})
    """
    if not chunks:
        print("[EMBEDDER] No chunks to embed.")
        return []

    # ── PRE-FLIGHT CHECK
    missing = [i for i, c in enumerate(chunks) if not c.get("embed_text", "").strip()]
    if missing:
        raise ValueError(
            f"[EMBEDDER] {len(missing)} chunk(s) missing 'embed_text' "
            f"at indices {missing[:10]}."
        )

    texts  = [c["embed_text"] for c in chunks]
    client = _get_voyage()

    print(f"[EMBEDDER] Embedding {len(texts)} chunks via Voyage AI "
          f"(model={VOYAGE_MODEL}, batch_size={EMBED_BATCH_SIZE})")

    if FREE_TIER_SLEEP_S > 0:
        print(f"[EMBEDDER] Free-tier mode: sleeping {FREE_TIER_SLEEP_S}s between batches. "
              f"Add a payment method to voyageai.com to remove this limit.")

    all_dense  = []
    all_sparse = []
    skipped    = []

    total_batches = (len(texts) + EMBED_BATCH_SIZE - 1) // EMBED_BATCH_SIZE

    for batch_idx in range(total_batches):
        start        = batch_idx * EMBED_BATCH_SIZE
        end          = min(start + EMBED_BATCH_SIZE, len(texts))
        batch_texts  = texts[start:end]
        batch_chunks = chunks[start:end]

        max_retries = 5
        success     = False

        for attempt in range(max_retries):
            try:
                # ── DENSE via Voyage API
                result     = client.embed(batch_texts, model=VOYAGE_MODEL, input_type="document")
                dense_vecs = result.embeddings   # list[list[float]]

                # ── SPARSE via local BM25-lite
                sparse_vecs = [_to_sparse_bm25(t) for t in batch_texts]

                # ── SANITY CHECK for zero-norm vectors
                for local_i, (dvec, chunk) in enumerate(zip(dense_vecs, batch_chunks)):
                    global_i = start + local_i
                    norm     = float(np.linalg.norm(dvec))
                    if norm == 0.0:
                        skipped.append({
                            "index"    : global_i,
                            "chunk_id" : chunk.get("chunk_id", "?"),
                            "reason"   : "zero-norm dense vector",
                            "content"  : chunk.get("content", "")[:80],
                        })

                all_dense.extend(dense_vecs)
                all_sparse.extend(sparse_vecs)

                print(f"[EMBEDDER]   Batch {batch_idx + 1}/{total_batches} done "
                      f"(chunks {start + 1}-{end})")

                success = True
                break  # exit retry loop on success

            except Exception as e:
                err_str       = str(e)
                is_rate_limit = (
                    "rate limit"  in err_str.lower() or
                    "payment"     in err_str.lower() or
                    "rpm"         in err_str.lower() or
                    "429"         in err_str
                )

                if is_rate_limit and attempt < max_retries - 1:
                    wait = 25 * (attempt + 1)   # 25s → 50s → 75s → 100s
                    print(f"[EMBEDDER]   Rate limited on batch {batch_idx + 1}, "
                          f"waiting {wait}s (attempt {attempt + 1}/{max_retries})...")
                    time.sleep(wait)
                    continue

                # Non-rate-limit error OR final retry exhausted
                print(f"[EMBEDDER] FAILED Batch {batch_idx + 1}: {e}")
                for local_i in range(len(batch_texts)):
                    global_i = start + local_i
                    skipped.append({
                        "index"    : global_i,
                        "chunk_id" : chunks[global_i].get("chunk_id", "?"),
                        "reason"   : f"batch exception: {e}",
                        "content"  : chunks[global_i].get("content", "")[:80],
                    })
                all_dense.extend([None] * len(batch_texts))
                all_sparse.extend([None] * len(batch_texts))
                break

        # ── Sleep between batches to respect free-tier rate limit
        if success and FREE_TIER_SLEEP_S > 0 and batch_idx < total_batches - 1:
            print(f"[EMBEDDER]   Sleeping {FREE_TIER_SLEEP_S}s (free-tier rate limit)...")
            time.sleep(FREE_TIER_SLEEP_S)

    # ── ATTACH VECTORS TO CHUNKS
    embedded      = []
    truly_skipped = []

    for chunk, dense_vec, sparse_vec in zip(chunks, all_dense, all_sparse):
        if dense_vec is None or sparse_vec is None:
            truly_skipped.append(chunk.get("chunk_id", "?"))
            continue

        ec = dict(chunk)
        ec["dense_vector"]  = dense_vec
        ec["sparse_vector"] = sparse_vec
        embedded.append(ec)

    _embedding_report(chunks, embedded, skipped, truly_skipped)
    return embedded


# ── QUERY EMBEDDER ────────────────────────────────────────────────────────────

def embed_query(query: str) -> tuple[list[float], dict]:
    """
    Embed a search query for hybrid retrieval.
    Dense via Voyage (input_type='query'), sparse via BM25-lite.

    Args:
        query : raw user question string

    Returns:
        (dense_vector, sparse_vector) ready for Qdrant hybrid search
    """
    if not query or not query.strip():
        raise ValueError("[EMBEDDER] Query string is empty.")

    client        = _get_voyage()
    result        = client.embed([query.strip()], model=VOYAGE_MODEL, input_type="query")
    dense_vector  = result.embeddings[0]
    sparse_vector = _to_sparse_bm25(query.strip())

    return dense_vector, sparse_vector


# ── HEALTH CHECK ──────────────────────────────────────────────────────────────

def embedding_health_check() -> bool:
    """Sanity check the Voyage client and BM25 sparse pipeline."""
    print("\n[EMBEDDER] ── Health Check ──────────────────────────────────")

    all_passed = True

    # CHECK 1: API key set
    ok = bool(VOYAGE_API_KEY)
    print(f"[EMBEDDER] {'PASS' if ok else 'FAIL'}  VOYAGE_API_KEY set  : {ok}")
    all_passed = all_passed and ok
    if not ok:
        print("[EMBEDDER] Set VOYAGE_API_KEY env var and retry.\n")
        return False

    # CHECK 2: Client init
    try:
        client = _get_voyage()
        print(f"[EMBEDDER] PASS  Client init")
    except Exception as e:
        print(f"[EMBEDDER] FAIL  Client init         : {e}")
        return False

    # CHECK 3: Dense embedding shape
    try:
        test_sentences = [
            "Retrieval-Augmented Generation improves LLM accuracy.",
            "RAG combines a retriever with a language model.",
            "The Eiffel Tower is located in Paris, France.",
        ]
        result = client.embed(test_sentences, model=VOYAGE_MODEL, input_type="document")
        dense  = result.embeddings
        dim    = len(dense[0])
        ok     = dim == 1024
        print(f"[EMBEDDER] {'PASS' if ok else 'FAIL'}  Dense dimension     : {dim}")
        all_passed = all_passed and ok
    except Exception as e:
        print(f"[EMBEDDER] FAIL  Dense embedding     : {e}")
        return False

    # CHECK 4: Non-zero vectors
    norms = [float(np.linalg.norm(d)) for d in dense]
    ok    = all(n > 0 for n in norms)
    print(f"[EMBEDDER] {'PASS' if ok else 'FAIL'}  Dense non-zero      : norms={[round(n, 3) for n in norms]}")
    all_passed = all_passed and ok

    # CHECK 5: Semantic ordering
    def _cosine(a, b):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        return float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else 0.0

    sim_related   = _cosine(dense[0], dense[1])
    sim_unrelated = _cosine(dense[0], dense[2])
    ok            = sim_related > sim_unrelated
    print(f"[EMBEDDER] {'PASS' if ok else 'FAIL'}  Semantic ordering   : "
          f"related={sim_related:.3f}  unrelated={sim_unrelated:.3f}")
    all_passed = all_passed and ok

    # CHECK 6: Sparse output
    sv = _to_sparse_bm25("What is retrieval augmented generation?")
    ok = len(sv["indices"]) > 0
    print(f"[EMBEDDER] {'PASS' if ok else 'FAIL'}  Sparse (BM25-lite)  : tokens={len(sv['indices'])}")
    all_passed = all_passed and ok

    # CHECK 7: embed_query end-to-end
    try:
        dv, sv = embed_query("What is RAG?")
        ok     = len(dv) == 1024 and len(sv["indices"]) > 0
        print(f"[EMBEDDER] {'PASS' if ok else 'FAIL'}  embed_query()       : "
              f"dense_dim={len(dv)}  sparse_tokens={len(sv['indices'])}")
    except Exception as e:
        print(f"[EMBEDDER] FAIL  embed_query()       : {e}")
        ok = False
    all_passed = all_passed and ok

    print(f"\n[EMBEDDER] {'All checks passed — embedder is ready.' if all_passed else 'Some checks FAILED.'}\n")
    return all_passed


# ── EMBEDDING REPORT (internal) ───────────────────────────────────────────────

def _embedding_report(input_chunks, embedded, zero_norm, batch_failed):
    total_in  = len(input_chunks)
    total_out = len(embedded)
    dropped   = total_in - total_out

    by_type: dict[str, int] = {}
    for c in embedded:
        t = c.get("type", "?")
        by_type[t] = by_type.get(t, 0) + 1

    sparse_lens  = [len(c["sparse_vector"]["indices"]) for c in embedded]
    dense_sample = embedded[0]["dense_vector"][:3] if embedded else []

    print(f"\n[EMBEDDER] ── Embedding Report ──────────────────────────────────")
    print(f"[EMBEDDER] Model              : {VOYAGE_MODEL} (Voyage AI)")
    print(f"[EMBEDDER] Input chunks       : {total_in}")
    print(f"[EMBEDDER] Embedded OK        : {total_out}" +
          (" (all good)" if dropped == 0 else f" ({dropped} dropped)"))

    for t, count in sorted(by_type.items()):
        print(f"[EMBEDDER]   {t:<10}          : {count}")

    if embedded:
        print(f"[EMBEDDER] Dense dim          : {len(embedded[0]['dense_vector'])}")
        print(f"[EMBEDDER] Dense sample [0:3] : {[round(x, 4) for x in dense_sample]}")
        print(f"[EMBEDDER] Sparse tok/chunk   : "
              f"min={min(sparse_lens)}  avg={int(sum(sparse_lens)/len(sparse_lens))}  max={max(sparse_lens)}")

    if zero_norm:
        print(f"\n[EMBEDDER] WARNING: Zero-norm vectors ({len(zero_norm)})")
        for item in zero_norm[:5]:
            print(f"           index={item['index']}  chunk_id={item['chunk_id'][:12]}  "
                  f"content='{item['content'][:60]}'")

    if batch_failed:
        print(f"\n[EMBEDDER] ERROR: Chunks NOT embedded ({len(batch_failed)})")
        for cid in batch_failed[:5]:
            print(f"           chunk_id={cid[:20]}")

    print(f"[EMBEDDER] ─────────────────────────────────────────────────────\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1 or sys.argv[1] == "--check":
        embedding_health_check()
        sys.exit(0)

    file_path   = sys.argv[1]
    source_type = sys.argv[2] if len(sys.argv) > 2 else None

    from ingest import ingest_file

    print(f"\n── Step 1: ingest + normalize")
    chunks = ingest_file(file_path, source_type=source_type, validate=True)

    print(f"\n── Step 2: health check")
    if not embedding_health_check():
        print("[EMBEDDER] Health check failed — aborting.")
        sys.exit(1)

    print(f"\n── Step 3: embed")
    embedded = embed_chunks(chunks)

    if embedded:
        c = embedded[0]
        print(f"\n── Step 4: spot check")
        print(f"  chunk_id      : {c['chunk_id']}")
        print(f"  dense_vector  : dim={len(c['dense_vector'])}  "
              f"first3={[round(x, 4) for x in c['dense_vector'][:3]]}")
        print(f"  sparse_vector : {len(c['sparse_vector']['indices'])} tokens")

    print(f"\n── Done. {len(embedded)}/{len(chunks)} chunks embedded successfully.")