import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import uuid
from txt_chunker import (
    _detect_doc_type,
    _estimate_tokens,
    _late_chunk_pipeline,
    _hierarchical_sliding_pipeline,
    LATE_CHUNK_MAX_TOK,
)
from File_processing.scanned_pdf import extract_scanned_pdf
from File_processing.cleaner import clean


def chunk_scanned_pdf(parsed: dict) -> list[dict]:
    parsed = clean(parsed)
    source   = parsed.get("source", "unknown.pdf")
    ocr_text = parsed.get("text",   "").strip()
    tables   = parsed.get("tables", [])
    images   = parsed.get("images", [])
    file_id  = str(uuid.uuid4())  # one GUID for the whole file

    print(f"\n[PDF CHUNKER] File     : {source}")
    print(f"[PDF CHUNKER] File ID  : {file_id}")
    print(f"[PDF CHUNKER] Tables   : {len(tables)}")
    print(f"[PDF CHUNKER] Images   : {len(images)}")

    base_metadata = {"file_name": source}
    all_chunks    = []

    # TEXT CHUNKS
    if ocr_text:
        doc_type     = _detect_doc_type(ocr_text)
        total_tokens = _estimate_tokens(ocr_text)

        print(f"[PDF CHUNKER] Doc type : {doc_type}")
        print(f"[PDF CHUNKER] Tokens   : {total_tokens}")

        if doc_type == "narrative" and total_tokens <= LATE_CHUNK_MAX_TOK:
            print("[PDF CHUNKER] Strategy : LATE CHUNKING")
            text_chunks = _late_chunk_pipeline(ocr_text, base_metadata, file_id)
        else:
            if doc_type == "narrative":
                print("[PDF CHUNKER] Strategy : SLIDING WINDOW (doc too long for late chunking)")
            else:
                print(f"[PDF CHUNKER] Strategy : HIERARCHICAL + SLIDING WINDOW ({doc_type})")
            text_chunks = _hierarchical_sliding_pipeline(ocr_text, base_metadata, doc_type, file_id)

        text_chunks = _merge_parasitic_chunks(text_chunks)
        print(f"[PDF CHUNKER] Text chunks produced: {len(text_chunks)}")
        all_chunks.extend(text_chunks)

    else:
        print("[PDF CHUNKER] No OCR text found — skipping text chunking.")

    # TABLE CHUNKS
    for table in tables:
        chunk = _make_table_chunk(table, base_metadata, file_id)
        all_chunks.append(chunk)

    if tables:
        print(f"[PDF CHUNKER] Table chunks produced: {len(tables)}")

    # IMAGE CHUNKS
    for image in images:
        chunk = _make_image_chunk(image, base_metadata, file_id)
        all_chunks.append(chunk)

    if images:
        print(f"[PDF CHUNKER] Image chunks produced: {len(images)}")

    print(f"[PDF CHUNKER] Total chunks produced: {len(all_chunks)}")
    return {"filename": source, "chunks": all_chunks}


# CHUNK HELPERS

def _make_table_chunk(table: dict, base_metadata: dict, file_id: str) -> dict:
    """One chunk per table extracted by Groq vision."""
    return {
        "chunk_id": str(uuid.uuid4()),
        "type"    : "table",
        "content" : table.get("content", ""),
        "metadata": {
            "file_name"  : base_metadata.get("file_name"),
            "file_id"    : file_id,
            "table_index": table.get("table_index"),
            "page"       : table.get("page"),
        }
    }


def _make_image_chunk(image: dict, base_metadata: dict, file_id: str) -> dict:
    """One chunk per chart/figure described by Groq vision."""
    return {
        "chunk_id": str(uuid.uuid4()),
        "type"    : "image",
        "content" : image.get("content", ""),
        "metadata": {
            "file_name"  : base_metadata.get("file_name"),
            "file_id"    : file_id,
            "image_index": image.get("image_index"),
            "page"       : image.get("page"),
        }
    }


def _merge_parasitic_chunks(chunks: list[dict]) -> list[dict]:
    """
    Fusionne les chunks qui sont de purs artefacts de pagination
    avec leur voisin le plus proche, sans perdre de contenu.
    """
    import re

    def _is_page_artifact(content: str) -> bool:
        """Détecte si un chunk n'est QUE un séparateur de page."""
        cleaned = content.strip()
        cleaned = re.sub(r"[-_=]{2,}", "", cleaned)
        cleaned = re.sub(r"\bPage\s+\d+\b", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\bP\.\s*\d+\b", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\d+", "", cleaned)
        cleaned = cleaned.strip()
        return len(cleaned) == 0

    result = []

    for chunk in chunks:
        content = chunk.get("content", "")

        if _is_page_artifact(content):
            if result:
                prev = result[-1]
                prev["content"] = prev["content"].rstrip() + "\n" + content.strip()
                print(f"[MERGE] Artifact merged into previous chunk: {repr(content[:60])}")
            else:
                result.append(chunk)
        else:
            if result and _is_page_artifact(result[-1]["content"]):
                pending = result.pop()
                chunk["content"] = pending["content"].strip() + "\n" + chunk["content"]
                print(f"[MERGE] Pending artifact merged into current chunk: {repr(pending['content'][:60])}")
            result.append(chunk)

    return result


# TEST

if __name__ == "__main__":
    import json

    parsed  = extract_scanned_pdf("222.pdf")
    chunked = chunk_scanned_pdf(parsed)

    print(f"\n[TEST] Total chunks: {len(chunked['chunks'])}")
    for i, c in enumerate(chunked["chunks"]):
        preview  = c["content"][:100].replace("\n", " ")
        chunk_id = c.get("chunk_id", "n/a")
        file_id  = c["metadata"].get("file_id", "n/a")
        ctype    = c["type"]
        page     = c["metadata"].get("page", "-")
        print(f"  [{i}] type={ctype} | page={page} | chunk_id={chunk_id} | file_id={file_id}")
        print(f"       {preview}...")

    out = chunked["filename"].replace(".pdf", "_chunked.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(chunked, f, ensure_ascii=False, indent=2)
    print(f"\n Saved to: {out}")