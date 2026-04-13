import uuid
from txt_chunker import (
    _detect_doc_type,
    _estimate_tokens,
    _late_chunk_pipeline,
    _hierarchical_sliding_pipeline,
    LATE_CHUNK_MAX_TOK,
)
from File_processing.scanned_pdf import extract_scanned_pdf



def chunk_scanned_pdf(parsed: dict) -> dict:
    source   = parsed.get("source", "unknown.pdf")
    ocr_text = parsed.get("text",   "").strip()
    tables   = parsed.get("tables", [])
    images   = parsed.get("images", [])

    print(f"\n[PDF CHUNKER] File     : {source}")
    print(f"[PDF CHUNKER] Tables   : {len(tables)}")
    print(f"[PDF CHUNKER] Images   : {len(images)}")

    base_metadata = {"file_name": source}
    all_chunks    = []

    # ── TEXT CHUNKS ───────────────────────────────────
    if ocr_text:
        doc_type     = _detect_doc_type(ocr_text)
        total_tokens = _estimate_tokens(ocr_text)

        print(f"[PDF CHUNKER] Doc type : {doc_type}")
        print(f"[PDF CHUNKER] Tokens   : {total_tokens}")

        if doc_type == "narrative" and total_tokens <= LATE_CHUNK_MAX_TOK:
            print("[PDF CHUNKER] Strategy : LATE CHUNKING")
            text_chunks = _late_chunk_pipeline(ocr_text, base_metadata)
        else:
            if doc_type == "narrative":
                print("[PDF CHUNKER] Strategy : SLIDING WINDOW (doc too long for late chunking)")
            else:
                print(f"[PDF CHUNKER] Strategy : HIERARCHICAL + SLIDING WINDOW ({doc_type})")
            text_chunks = _hierarchical_sliding_pipeline(ocr_text, base_metadata, doc_type)

        all_chunks.extend(text_chunks)
        print(f"[PDF CHUNKER] Text chunks produced: {len(text_chunks)}")
    else:
        print("[PDF CHUNKER] No OCR text found — skipping text chunking.")

    # ── TABLE CHUNKS ──────────────────────────────────
    for table in tables:
        chunk = _make_table_chunk(table, base_metadata)
        all_chunks.append(chunk)

    if tables:
        print(f"[PDF CHUNKER] Table chunks produced: {len(tables)}")

    # ── IMAGE CHUNKS ──────────────────────────────────
    for image in images:
        chunk = _make_image_chunk(image, base_metadata)
        all_chunks.append(chunk)

    if images:
        print(f"[PDF CHUNKER] Image chunks produced: {len(images)}")

    print(f"[PDF CHUNKER] Total chunks produced: {len(all_chunks)}")
    return {"filename": source, "chunks": all_chunks}


#  CHUNK HELPERS

def _make_table_chunk(table: dict, base_metadata: dict) -> dict:
    """One chunk per table extracted by Groq vision."""
    return {
        "type"    : "table",
        "content" : table.get("content", ""),
        "metadata": {
            "file_name"  : base_metadata.get("file_name"),
            "chunk_id"   : str(uuid.uuid4()),
            "table_index": table.get("table_index"),
            "page"       : table.get("page"),
        }
    }


def _make_image_chunk(image: dict, base_metadata: dict) -> dict:
    """One chunk per chart/figure described by Groq vision."""
    return {
        "type"    : "image",
        "content" : image.get("content", ""),
        "metadata": {
            "file_name"  : base_metadata.get("file_name"),
            "chunk_id"   : str(uuid.uuid4()),
            "image_index": image.get("image_index"),
            "page"       : image.get("page"),
        }
    }


#  TEST

if __name__ == "__main__":
    import json

    parsed  = extract_scanned_pdf("222.pdf")
    chunked = chunk_scanned_pdf(parsed)

    print(f"\n[TEST] Total chunks: {len(chunked['chunks'])}")
    for i, c in enumerate(chunked["chunks"]):
        preview  = c["content"][:100].replace("\n", " ")
        chunk_id = c["metadata"]["chunk_id"]
        ctype    = c["type"]
        page     = c["metadata"].get("page", "-")
        print(f"  [{i}] type={ctype} | page={page} | id={chunk_id}")
        print(f"       {preview}...")

    out = chunked["filename"].replace(".pdf", "_chunked.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(chunked, f, ensure_ascii=False, indent=2)
    print(f"\n Saved to: {out}")