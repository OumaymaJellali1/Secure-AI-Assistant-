"""
embedder.py
───────────
Embeds normalized chunks using BGE-M3.
Produces dense + sparse vectors for hybrid retrieval in Qdrant.

Input  : list of normalized chunks from ingest.py (already have embed_text)
Output : list of EmbeddedChunk dicts — same chunk + both vectors attached

Usage:
    from embedder import embed_chunks, embed_query, embedding_health_check

    embedded = embed_chunks(chunks)          # document embedding
    dense, sparse = embed_query("your question")  # query embedding
"""

from __future__ import annotations
import numpy as np


# ── CONFIG ────────────────────────────────────────────────────────────────────

EMBED_BATCH_SIZE = 32     
EMBED_MAX_LENGTH = 512   
USE_FP16         = True   # matches pdf_chunker / txt_chunker



_model = None

def _get_model():
    global _model
    if _model is None:
        print("[EMBEDDER] Loading BGE-M3 model...")
        from FlagEmbedding import BGEM3FlagModel
        _model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=USE_FP16)
        print("[EMBEDDER] BGE-M3 ready.")
    return _model


# ── SPARSE CONVERSION 

def _to_sparse_vector(sparse_weights: dict) -> dict:
    """
    Convert BGE-M3 sparse output → Qdrant-ready format.

    BGE-M3 returns : {token_id (int): weight (float), ...}
    Qdrant expects : {"indices": [...], "values": [...]}

    Zero-weight entries are dropped — they waste memory and
    corrupt dot-product scoring in Qdrant sparse search.
    """
    indices, values = [], []
    for token_id, weight in sparse_weights.items():
        w = float(weight)
        if w > 0.0:
            indices.append(int(token_id))
            values.append(w)
    return {"indices": indices, "values": values}


# ── CORE EMBED FUNCTION 

def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed a list of normalized chunks (output of ingest.py).

    Each chunk must have an "embed_text" field — set by the normalizer.
    Returns the same chunks with "dense_vector" and "sparse_vector" added.
    Input order is always preserved.

    Args:
        chunks : normalized chunks from ingest_file()

    Returns:
        Same chunks + dense_vector (list[float], dim=1024)
                     + sparse_vector ({"indices": [...], "values": [...]})

    Raises:
        ValueError : if any chunk is missing "embed_text"
                     (means normalizer was skipped)
    """
    if not chunks:
        print("[EMBEDDER] No chunks to embed.")
        return []

    # ── PRE-FLIGHT CHECK 
    missing_embed_text = [
        i for i, c in enumerate(chunks)
        if not c.get("embed_text", "").strip()
    ]
    if missing_embed_text:
        raise ValueError(
            f"[EMBEDDER] {len(missing_embed_text)} chunk(s) missing 'embed_text' "
            f"at indices {missing_embed_text[:10]}{'...' if len(missing_embed_text) > 10 else ''}.\n"
            f"Make sure you ran ingest_file() before embed_chunks()."
        )

    texts = [c["embed_text"] for c in chunks]
    model = _get_model()

    print(f"[EMBEDDER] Embedding {len(texts)} chunks "
          f"(batch_size={EMBED_BATCH_SIZE}, max_length={EMBED_MAX_LENGTH}, fp16={USE_FP16})")

    all_dense  = []
    all_sparse = []
    skipped    = []   

    # ── BATCH LOOP 
    total_batches = (len(texts) + EMBED_BATCH_SIZE - 1) // EMBED_BATCH_SIZE

    for batch_idx in range(total_batches):
        start = batch_idx * EMBED_BATCH_SIZE
        end   = min(start + EMBED_BATCH_SIZE, len(texts))
        batch = texts[start:end]

        try:
            result = model.encode(
                batch,
                batch_size          = EMBED_BATCH_SIZE,
                max_length          = EMBED_MAX_LENGTH,
                return_dense        = True,
                return_sparse       = True,
                return_colbert_vecs = False,
            )

            batch_dense  = result["dense_vecs"]       
            batch_sparse = result["lexical_weights"]  

            # ── SANITY CHECK: dense vectors must be non-zero 
            for local_i, (dvec, chunk) in enumerate(
                zip(batch_dense, chunks[start:end])
            ):
                global_i = start + local_i
                norm     = float(np.linalg.norm(dvec))

                if norm == 0.0:
                    skipped.append({
                        "index"    : global_i,
                        "chunk_id" : chunk.get("chunk_id", "?"),
                        "reason"   : "zero-norm dense vector",
                        "content"  : chunk.get("content", "")[:80],
                    })
                    print(f"[EMBEDDER] WARNING: Zero-norm vector at index {global_i} "
                          f"(chunk_id={chunk.get('chunk_id','?')[:12]})")

            all_dense.extend(batch_dense)
            all_sparse.extend(batch_sparse)

            print(f"[EMBEDDER]   Batch {batch_idx + 1}/{total_batches} done "
                  f"(chunks {start + 1}-{end})")

        except Exception as e:
            print(f"[EMBEDDER] FAILED Batch {batch_idx + 1}: {e}")
            for local_i in range(len(batch)):
                global_i = start + local_i
                skipped.append({
                    "index"    : global_i,
                    "chunk_id" : chunks[global_i].get("chunk_id", "?"),
                    "reason"   : f"batch exception: {e}",
                    "content"  : chunks[global_i].get("content", "")[:80],
                })
            all_dense.extend([None] * len(batch))
            all_sparse.extend([None] * len(batch))

    # ── ATTACH VECTORS 
    embedded      = []
    truly_skipped = []

    for chunk, dense_vec, sparse_dict in zip(chunks, all_dense, all_sparse):
        if dense_vec is None or sparse_dict is None:
            truly_skipped.append(chunk.get("chunk_id", "?"))
            continue   # drop failed chunks — don't insert broken points to Qdrant

        ec = dict(chunk)
        ec["dense_vector"]  = dense_vec.tolist()
        ec["sparse_vector"] = _to_sparse_vector(sparse_dict)
        embedded.append(ec)

    # ── FINAL REPORT 
    _embedding_report(chunks, embedded, skipped, truly_skipped)

    return embedded


# ── QUERY EMBEDDER 

def embed_query(query: str) -> tuple[list[float], dict]:
    """
    Embed a search query for retrieval.

    IMPORTANT — uses "query: " prefix.
    BGE-M3 is asymmetric: documents and queries live in different spaces.
    Documents are embedded via embed_text (with type prefix from normalizer).
    Queries MUST use "query: " prefix to align with the document space.
    If you skip this prefix, retrieval scores will be wrong.

    Args:
        query : raw user question string

    Returns:
        (dense_vector, sparse_vector) ready for Qdrant hybrid search
    """
    if not query or not query.strip():
        raise ValueError("[EMBEDDER] Query string is empty.")

    query_text = f"query: {query.strip()}"
    model      = _get_model()

    result = model.encode(
        [query_text],
        batch_size          = 1,
        max_length          = 512,
        return_dense        = True,
        return_sparse       = True,
        return_colbert_vecs = False,
    )

    dense_vector  = result["dense_vecs"][0].tolist()
    sparse_vector = _to_sparse_vector(result["lexical_weights"][0])

    return dense_vector, sparse_vector


# ── HEALTH CHECK 

def embedding_health_check() -> bool:
    """
    Run a quick sanity check on the BGE-M3 model before embedding a real file.

    Checks:
      1. Model loads without error
      2. Dense vector has correct dimension (1024)
      3. Dense vectors are non-zero (model is not broken)
      4. Sparse vectors have at least 1 token
      5. Similar sentences score higher than dissimilar ones
         (semantic ordering is correct)
      6. Query prefix produces a different vector than document prefix
         (asymmetric encoding is working)
      7. embed_query() wrapper works end-to-end

    Prints PASS / FAIL for each check.
    Returns True if all checks pass, False otherwise.
    """
    print("\n[EMBEDDER] ── Health Check ──────────────────────────────────")

    try:
        model = _get_model()
    except Exception as e:
        print(f"[EMBEDDER] FAIL  Model load: {e}")
        return False

    all_passed = True

    # ── TEST SENTENCES 
    sent_similar_a = "Retrieval-Augmented Generation improves LLM accuracy."
    sent_similar_b = "RAG combines a retriever with a language model."
    sent_different = "The Eiffel Tower is located in Paris, France."
    query_text     = "query: What is RAG?"
    doc_text       = sent_similar_a  

    try:
        result = model.encode(
            [sent_similar_a, sent_similar_b, sent_different,
             query_text, doc_text],
            batch_size          = 8,
            max_length          = 512,
            return_dense        = True,
            return_sparse       = True,
            return_colbert_vecs = False,
        )
        dense  = result["dense_vecs"]      
        sparse = result["lexical_weights"] 

    except Exception as e:
        print(f"[EMBEDDER] FAIL  Encode call: {e}")
        return False

    def _cosine(a, b):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        return float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else 0.0

    # CHECK 1: Dense dimension
    dim = dense.shape[1] if hasattr(dense, "shape") else len(dense[0])
    ok  = dim == 1024
    print(f"[EMBEDDER] {'PASS' if ok else 'FAIL'}  Dense dimension     : {dim} "
          f"{'(expected 1024)' if not ok else ''}")
    all_passed = all_passed and ok

    # CHECK 2: Dense non-zero
    norms = [float(np.linalg.norm(dense[i])) for i in range(3)]
    ok    = all(n > 0 for n in norms)
    print(f"[EMBEDDER] {'PASS' if ok else 'FAIL'}  Dense non-zero      : "
          f"norms={[round(n, 3) for n in norms]}")
    all_passed = all_passed and ok

    # CHECK 3: Sparse has tokens
    sparse_lens = [len(sparse[i]) for i in range(3)]
    ok          = all(l > 0 for l in sparse_lens)
    print(f"[EMBEDDER] {'PASS' if ok else 'FAIL'}  Sparse tokens       : "
          f"counts={sparse_lens}")
    all_passed = all_passed and ok

    # CHECK 4: Semantic ordering
    sim_related   = _cosine(dense[0], dense[1])  # similar_a vs similar_b
    sim_unrelated = _cosine(dense[0], dense[2])  # similar_a vs different
    ok            = sim_related > sim_unrelated
    print(f"[EMBEDDER] {'PASS' if ok else 'FAIL'}  Semantic ordering   : "
          f"related={sim_related:.3f}  unrelated={sim_unrelated:.3f}")
    all_passed = all_passed and ok

    # CHECK 5: Query vs document asymmetry
    diff = float(np.linalg.norm(np.array(dense[3]) - np.array(dense[4])))
    ok   = diff > 0.01
    print(f"[EMBEDDER] {'PASS' if ok else 'FAIL'}  Query/doc asymmetry : "
          f"vector distance={diff:.4f} "
          f"{'(prefix working)' if ok else '(prefix has no effect!)'}")
    all_passed = all_passed and ok

    # CHECK 6: embed_query() end-to-end
    try:
        dv, sv = embed_query("What is RAG?")
        ok     = len(dv) == 1024 and len(sv["indices"]) > 0
        print(f"[EMBEDDER] {'PASS' if ok else 'FAIL'}  embed_query()       : "
              f"dense_dim={len(dv)}  sparse_tokens={len(sv['indices'])}")
    except Exception as e:
        print(f"[EMBEDDER] FAIL  embed_query()       : {e}")
        ok = False
    all_passed = all_passed and ok

    # SUMMARY
    print(f"[EMBEDDER] ")
    if all_passed:
        print(f"[EMBEDDER] All checks passed — embedder is ready.\n")
    else:
        print(f"[EMBEDDER] Some checks FAILED — review output above.\n")

    return all_passed


# ── EMBEDDING REPORT (internal) 

def _embedding_report(
    input_chunks : list[dict],
    embedded     : list[dict],
    zero_norm    : list[dict],
    batch_failed : list[str],
) -> None:
    """Print a full summary after embed_chunks() completes."""

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
    print(f"[EMBEDDER] Input chunks       : {total_in}")
    print(f"[EMBEDDER] Embedded OK        : {total_out}"
          + (" (all good)" if dropped == 0 else f" ({dropped} dropped)"))

    for t, count in sorted(by_type.items()):
        print(f"[EMBEDDER]   {t:<10}          : {count}")

    if embedded:
        print(f"[EMBEDDER] Dense dim          : {len(embedded[0]['dense_vector'])}")
        print(f"[EMBEDDER] Dense sample [0:3] : {[round(x, 4) for x in dense_sample]}")
        print(f"[EMBEDDER] Sparse tok/chunk   : "
              f"min={min(sparse_lens)}  "
              f"avg={int(sum(sparse_lens) / len(sparse_lens))}  "
              f"max={max(sparse_lens)}")

    if zero_norm:
        print(f"\n[EMBEDDER] WARNING: Zero-norm vectors ({len(zero_norm)}) — "
              f"these were embedded but may hurt retrieval:")
        for item in zero_norm[:5]:
            print(f"           index={item['index']}  "
                  f"chunk_id={item['chunk_id'][:12]}  "
                  f"content='{item['content'][:60]}'")
        if len(zero_norm) > 5:
            print(f"           ... and {len(zero_norm) - 5} more")

    if batch_failed:
        print(f"\n[EMBEDDER] ERROR: Chunks NOT embedded — will NOT be in Qdrant "
              f"({len(batch_failed)}):")
        for cid in batch_failed[:5]:
            print(f"           chunk_id={cid[:20]}")
        if len(batch_failed) > 5:
            print(f"           ... and {len(batch_failed) - 5} more")

    print(f"[EMBEDDER] ─────────────────────────────────────────────────────\n")


# ── CLI 

if __name__ == "__main__":
    import sys

    # MODE 1: health check only
    #   python embedder.py --check
    if len(sys.argv) == 1 or sys.argv[1] == "--check":
        embedding_health_check()
        sys.exit(0)

    # MODE 2: embed a real file end-to-end
    #   python embedder.py <file_path> [source_type]
    file_path   = sys.argv[1]
    source_type = sys.argv[2] if len(sys.argv) > 2 else None

    from ingest import ingest_file

    print(f"\n── Step 1: ingest + normalize")
    chunks = ingest_file(file_path, source_type=source_type, validate=True)

    print(f"\n── Step 2: health check")
    ok = embedding_health_check()
    if not ok:
        print("[EMBEDDER] Health check failed — aborting.")
        sys.exit(1)

    print(f"\n── Step 3: embed")
    embedded = embed_chunks(chunks)

    print(f"\n── Step 4: spot check first embedded chunk")
    if embedded:
        c = embedded[0]
        print(f"  chunk_id      : {c['chunk_id']}")
        print(f"  type          : {c['type']}")
        print(f"  source_type   : {c['metadata']['source_type']}")
        print(f"  dense_vector  : dim={len(c['dense_vector'])}  "
              f"first3={[round(x, 4) for x in c['dense_vector'][:3]]}")
        print(f"  sparse_vector : {len(c['sparse_vector']['indices'])} tokens  "
              f"top3_idx={c['sparse_vector']['indices'][:3]}  "
              f"top3_val={[round(v, 4) for v in c['sparse_vector']['values'][:3]]}")

    print(f"\n── Done. {len(embedded)}/{len(chunks)} chunks embedded successfully.")