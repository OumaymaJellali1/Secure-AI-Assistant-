import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uuid
import json
from pathlib import Path
from config import DEBUG

from File_processing.eml_handler import extract_eml
from File_processing.cleaner import clean


# ── HELPERS 

def _count_tokens(text: str) -> int:
    return max(1, int(len(text) / 3.5))


def _build_metadata(eml_metadata: dict, attachment_names: list,
                    source_type: str) -> dict:
    """Standard metadata block for every chunk."""
    return {
        "source"      : eml_metadata.get("source",  "unknown"),
        "source_type" : source_type,
        "eml_source"  : eml_metadata.get("source",  ""),
        "subject"     : eml_metadata.get("subject", ""),
        "from"        : eml_metadata.get("from",    ""),
        "to"          : eml_metadata.get("to",      ""),
        "date"        : eml_metadata.get("date",    ""),
        "attachments" : attachment_names,
    }


def _make_text_chunk(content: str, metadata: dict) -> dict:
    return {
        "chunk_id" : str(uuid.uuid4()),
        "type"     : "text",
        "content"  : content.strip(),
        "metadata" : metadata,
    }


def _table_to_text(headers: list, rows: list) -> str:
    """Convert a structured table into markdown-style text."""
    lines = []
    if headers:
        lines.append(" | ".join(str(h) for h in headers))
        lines.append(" | ".join(["---"] * len(headers)))
    for row in rows:
        lines.append(" | ".join(str(c) for c in row))
    return "\n".join(lines)


def _make_table_chunk(headers: list, rows: list, metadata: dict) -> dict:
    return {
        "chunk_id" : str(uuid.uuid4()),
        "type"     : "table",
        "content"  : _table_to_text(headers, rows),
        "metadata" : metadata,
    }


# ── BODY CHUNKER (text only) 

def _chunk_body(body: str, metadata: dict) -> list[dict]:
    """Split plain text body into ~400-token chunks."""
    if not body or not body.strip():
        return []

    MAX_TOKENS = 450
    MIN_TOKENS = 80

    paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]
    if not paragraphs:
        return []

    chunks = []
    buffer = ""

    for para in paragraphs:
        candidate = (buffer + " " + para).strip() if buffer else para

        if _count_tokens(candidate) <= MAX_TOKENS:
            buffer = candidate
        else:
            if buffer:
                chunks.append(buffer)
            buffer = para

    if buffer:
        if chunks and _count_tokens(buffer) < MIN_TOKENS:
            chunks[-1] = chunks[-1] + " " + buffer
        else:
            chunks.append(buffer)

    return [_make_text_chunk(c, metadata) for c in chunks if c.strip()]


# ── CONTENT ROUTER 

def _chunk_content(content, eml_metadata: dict, attachment_names: list) -> list[dict]:
    """
    Route content based on shape:
      - list of {type, ...} chunks → dispatch per type
      - plain string               → paragraph-chunker (legacy)
    """
    all_chunks = []

    # ── LEGACY: plain string ──────────────────────────
    if isinstance(content, str):
        meta = _build_metadata(eml_metadata, attachment_names, "eml_body")
        return _chunk_body(content, meta)

    if not isinstance(content, list):
        return []

    # ── NEW: list of typed chunks ─────────────────────
    for item in content:
        if not isinstance(item, dict):
            continue

        item_type = item.get("type", "")

        if item_type == "text":
            text = item.get("content", "")
            if text and text.strip():
                meta = _build_metadata(eml_metadata, attachment_names, "eml_body")
                all_chunks.extend(_chunk_body(text, meta))

        elif item_type == "table":
            headers = item.get("headers", []) or []
            rows    = item.get("rows",    []) or []
            if headers or rows:
                meta = _build_metadata(eml_metadata, attachment_names, "eml_body_table")
                all_chunks.append(_make_table_chunk(headers, rows, meta))
                if DEBUG:
                    print(f"[EML CHUNKER] Body table chunk: "
                          f"{len(headers)} headers, {len(rows)} rows")

        else:
            text = item.get("content", "")
            if text and isinstance(text, str) and text.strip():
                meta = _build_metadata(eml_metadata, attachment_names, "eml_body")
                all_chunks.extend(_chunk_body(text, meta))

    return all_chunks


# ── ATTACHMENT CHUNK COLLECTOR ────────────────────────

def _normalize_attachment_chunk(chunk: dict, attach_name: str,
                                 eml_source: str, eml_metadata: dict) -> dict:
    """Force any attachment chunk into the minimal schema."""
    return {
        "chunk_id" : chunk.get("chunk_id", str(uuid.uuid4())),
        "type"     : chunk.get("type", "text"),
        "content"  : chunk.get("content", ""),
        "metadata" : {
            "source"      : attach_name,
            "source_type" : chunk.get("metadata", {}).get(
                                "source_type",
                                Path(attach_name).suffix.lstrip(".") or "attachment"),
            "eml_source"  : eml_source,
            "subject"     : eml_metadata.get("subject", ""),
            "from"        : eml_metadata.get("from",    ""),
            "to"          : eml_metadata.get("to",      ""),
            "date"        : eml_metadata.get("date",    ""),
            "attachments" : [],
        }
    }


def _collect_attachment_chunks(attachments: list[dict], eml_source: str,
                                eml_metadata: dict) -> list[dict]:
    """
    Collect chunks from already-processed attachments.
    Normalizes all attachment chunks to the minimal schema.
    """
    all_chunks = []

    for attachment in attachments:
        name   = attachment.get("name",   "unknown")
        result = attachment.get("result")
        ext    = Path(name).suffix.lower()

        if DEBUG:
            print(f"[EML CHUNKER] Processing attachment: {name}")

        if result is None:
            continue

        # CSV → Postgres only, skip
        if ext == ".csv":
            if DEBUG:
                print(f"[EML CHUNKER] CSV '{name}' → stored in Postgres, skipping")
            continue

        # CASE 1: flat list[dict] (PDF/DOCX)
        if isinstance(result, list):
            for chunk in result:
                all_chunks.append(
                    _normalize_attachment_chunk(chunk, name, eml_source, eml_metadata)
                )

        # CASE 2: dict with 'chunks' key (PPTX/TXT/scanned PDF)
        elif isinstance(result, dict) and "chunks" in result:
            for chunk in result["chunks"]:
                all_chunks.append(
                    _normalize_attachment_chunk(chunk, name, eml_source, eml_metadata)
                )

        # CASE 3: single dict (image handler)
        elif isinstance(result, dict) and "content" in result:
            all_chunks.append(
                _normalize_attachment_chunk(
                    {"type": result.get("type", "image"), "content": result.get("content", "")},
                    name, eml_source, eml_metadata
                )
            )

    return all_chunks


# ── MAIN CHUNKER ──────────────────────────────────────

def chunk_eml(file_path: str) -> list[dict]:
    """
    Full pipeline: parse → clean → chunk.

    Every chunk has exactly these fields:
      - chunk_id  : UUID string
      - type      : "text" | "table" | "image" | ...
      - content   : string
      - metadata  : {source, source_type, eml_source,
                     subject, from, to, date, attachments}
    """
    if DEBUG:
        print(f"\n[EML CHUNKER] File: {file_path}")

    parsed = extract_eml(file_path)
    parsed = clean(parsed)

    eml_source  = parsed["metadata"].get("source", Path(file_path).name)
    content     = parsed.get("content", "")
    attachments = parsed.get("attachments", [])

    attachment_names = [
        a.get("name", "")
        for a in parsed["metadata"].get("attachments", [])
    ]

    if DEBUG:
        if isinstance(content, list):
            print(f"[EML CHUNKER] Content      : list ({len(content)} items, "
                  f"types={[c.get('type') for c in content if isinstance(c, dict)]})")
        else:
            print(f"[EML CHUNKER] Content      : string ({len(content)} chars)")
        print(f"[EML CHUNKER] Attachments  : {len(attachments)}")
        print(f"[EML CHUNKER] Attach names : {attachment_names}")

    all_chunks = []

    # Body chunks
    body_chunks = _chunk_content(content, parsed["metadata"], attachment_names)
    all_chunks.extend(body_chunks)

    if DEBUG:
        body_types = [c.get("type") for c in body_chunks]
        print(f"[EML CHUNKER] Body chunks  : {len(body_chunks)} ({body_types})")

    # Attachment chunks
    attachment_chunks = _collect_attachment_chunks(
        attachments, eml_source, parsed["metadata"]
    )
    all_chunks.extend(attachment_chunks)

    if DEBUG:
        print(f"[EML CHUNKER] Attach chunks: {len(attachment_chunks)}")
        print(f"[EML CHUNKER] Total chunks : {len(all_chunks)}")

    return all_chunks


# ── TEST ──────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python chunking/eml_chunking.py <file.eml>")
        sys.exit(1)

    eml_path = sys.argv[1]
    chunks   = chunk_eml(eml_path)

    print("\n===== CHUNKS PREVIEW =====")
    for i, chunk in enumerate(chunks[:5]):
        print(f"\nChunk {i+1}:")
        print(f"  chunk_id   : {chunk['chunk_id']}")
        print(f"  type       : {chunk['type']}")
        print(f"  content    : {chunk['content'][:120]}...")
        print(f"  metadata   :")
        for k, v in chunk['metadata'].items():
            print(f"      {k:12s}: {v}")

    print(f"\n Total chunks: {len(chunks)}")

    output_path = Path(eml_path).stem + "_chunks.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f" Saved to   : {output_path}")