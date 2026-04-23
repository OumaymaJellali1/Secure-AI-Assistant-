import uuid


TYPE_PREFIX = {
    "text"  : "",              
    "table" : "Table: ",
    "image" : "Image description: ",
}


def _count_tokens(text: str) -> int:
    """Fast character-based token estimate (same heuristic used across chunkers)."""
    return max(1, int(len(text) / 3.5))


def _make_embed_text(content: str, chunk_type: str) -> str:
    """Prepend type prefix to content for embedding."""
    prefix = TYPE_PREFIX.get(chunk_type, "")
    return (prefix + content).strip()


def _safe_str(value, fallback: str = "") -> str:
    return str(value).strip() if value is not None else fallback


def _safe_int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None



def _meta_from_pdf(raw: dict, source_type: str = "pdf_digital") -> dict:
    m = raw.get("metadata", {})
    return {
        "source"      : _safe_str(m.get("source")),
        "source_type" : _safe_str(m.get("source_type", source_type)),
        "section"     : _safe_str(m.get("section")) or None,
        "page"        : _safe_int(m.get("page_start")),
        "chunk_index" : _safe_int(m.get("chunk_index")),
        "chunk_total" : _safe_int(m.get("chunk_total")),
        "token_count" : _safe_int(m.get("token_count")) or _count_tokens(raw.get("content", "")),
        "subject"     : None,
        "from"        : None,
        "date"        : None,
    }


def _meta_from_eml(raw: dict) -> dict:
    m = raw.get("metadata", {})
    return {
        "source"      : _safe_str(m.get("source") or m.get("eml_source")),
        "source_type" : _safe_str(m.get("source_type", "eml")),
        "section"     : None,
        "page"        : None,
        "chunk_index" : None,
        "chunk_total" : None,
        "token_count" : _count_tokens(raw.get("content", "")),
        "subject"     : _safe_str(m.get("subject")) or None,
        "from"        : _safe_str(m.get("from")) or None,
        "date"        : _safe_str(m.get("date")) or None,
    }


def _meta_from_pptx(raw: dict) -> dict:
    m = raw.get("metadata", {})
    return {
        "source"      : _safe_str(m.get("source")),
        "source_type" : "pptx",
        "section"     : _safe_str(m.get("section")) or None,
        "page"        : _safe_int(m.get("slide")),        # slide number → page
        "chunk_index" : None,
        "chunk_total" : None,
        "token_count" : _count_tokens(raw.get("content", "")),
        "subject"     : None,
        "from"        : None,
        "date"        : None,
    }


def _meta_from_scanned_pdf(raw: dict) -> dict:
    m = raw.get("metadata", {})
    return {
        "source"      : _safe_str(m.get("source")),
        "source_type" : "pdf_scanned",
        "section"     : None,
        "page"        : _safe_int(m.get("page")),
        "chunk_index" : None,
        "chunk_total" : None,
        "token_count" : _count_tokens(raw.get("content", "")),
        "subject"     : None,
        "from"        : None,
        "date"        : None,
    }


def _meta_from_txt(raw: dict) -> dict:
    m = raw.get("metadata", {})
    return {
        "source"      : _safe_str(m.get("file_name")),
        "source_type" : "txt",
        "section"     : None,
        "page"        : None,
        "chunk_index" : None,
        "chunk_total" : None,
        "token_count" : _count_tokens(raw.get("content", "")),
        "subject"     : None,
        "from"        : None,
        "date"        : None,
    }


def _meta_from_image(raw: dict) -> dict:
    m = raw.get("metadata", {})
    return {
        "source"      : _safe_str(m.get("source")),
        "source_type" : "image",
        "section"     : None,
        "page"        : None,
        "chunk_index" : None,
        "chunk_total" : None,
        "token_count" : _count_tokens(raw.get("content", "")),
        "subject"     : None,
        "from"        : None,
        "date"        : None,
    }



_META_EXTRACTORS = {
    "pdf_digital" : _meta_from_pdf,
    "pdf_scanned" : _meta_from_scanned_pdf,
    "docx"        : lambda r: _meta_from_pdf(r, source_type="docx"),
    "pptx"        : _meta_from_pptx,
    "eml"         : _meta_from_eml,
    "eml_body"         : _meta_from_eml,
    "eml_body_table"   : _meta_from_eml,
    "txt"         : _meta_from_txt,
    "image"       : _meta_from_image,
}



def normalize_chunk(raw: dict, source_type: str) -> dict | None:
    """
    Normalize a single raw chunk from any chunker into the canonical schema.

    Args:
        raw         : one chunk dict as produced by any chunker
        source_type : one of "pdf_digital" | "pdf_scanned" | "docx" |
                              "pptx" | "eml" | "txt" | "image"

    Returns:
        Normalized chunk dict, or None if content is empty.
    """
    content = _safe_str(raw.get("content", ""))
    if not content:
        return None

    chunk_type = _safe_str(raw.get("type", "text"))
    if chunk_type not in ("text", "table", "image"):
        chunk_type = "text"

    chunk_id = (
    _safe_str(raw.get("chunk_id"))
    or _safe_str(raw.get("metadata", {}).get("chunk_id"))
    or str(uuid.uuid5(
        uuid.NAMESPACE_DNS,
        f"{raw.get('metadata', {}).get('source', '')}"
        f":{raw.get('metadata', {}).get('file_name', '')}"
        f":{content[:200]}"
    ))
)

    extractor = _META_EXTRACTORS.get(source_type, _meta_from_pdf)
    metadata  = extractor(raw)

    return {
        "chunk_id"   : chunk_id,
        "type"       : chunk_type,
        "content"    : content,
        "embed_text" : _make_embed_text(content, chunk_type),
        "metadata"   : metadata,
    }



def normalize_chunks(chunker_output, source_type: str) -> list[dict]:
    """
    Normalize the output of ANY chunker into a flat list of canonical chunks.

    Handles all output shapes:
      - list[dict]           → pdf, docx, eml
      - dict with "chunks"   → pptx, scanned_pdf, txt
      - single dict          → image_handler

    Args:
        chunker_output : raw return value from any chunker
        source_type    : one of "pdf_digital" | "pdf_scanned" | "docx" |
                                 "pptx" | "eml" | "txt" | "image"

    Returns:
        Flat list of normalized chunk dicts ready for embedding.
    """

    # Shape 1: dict with "chunks" key  (pptx, scanned_pdf, txt)
    if isinstance(chunker_output, dict) and "chunks" in chunker_output:
        raw_list = chunker_output["chunks"]

    # Shape 2: single chunk dict with "content"  (image_handler)
    elif isinstance(chunker_output, dict) and "content" in chunker_output:
        raw_list = [chunker_output]

    # Shape 3: flat list  (pdf, docx, eml)
    elif isinstance(chunker_output, list):
        raw_list = chunker_output

    else:
        return []

    normalized = []
    for raw in raw_list:
        if not isinstance(raw, dict):
            continue
        chunk = normalize_chunk(raw, source_type)
        if chunk is not None:
            normalized.append(chunk)

    return normalized


# ── VALIDATION HELPER (optional, for debugging) 

REQUIRED_FIELDS    = {"chunk_id", "type", "content", "embed_text", "metadata"}
REQUIRED_META      = {"source", "source_type", "section", "page",
                      "chunk_index", "chunk_total", "token_count",
                      "subject", "from", "date"}
VALID_TYPES        = {"text", "table", "image"}
VALID_SOURCE_TYPES = {"pdf_digital", "pdf_scanned", "docx", "pptx",
                      "eml", "txt", "image"}


def validate_chunks(chunks: list[dict]) -> list[str]:
    """
    Validate a list of normalized chunks.
    Returns a list of error strings (empty = all good).
    Useful during development — not needed in production hot path.
    """
    errors = []

    for i, chunk in enumerate(chunks):
        prefix = f"Chunk[{i}] ({chunk.get('chunk_id', '?')[:8]})"

        missing_top = REQUIRED_FIELDS - set(chunk.keys())
        if missing_top:
            errors.append(f"{prefix} missing top-level fields: {missing_top}")

        if chunk.get("type") not in VALID_TYPES:
            errors.append(f"{prefix} invalid type: {chunk.get('type')!r}")

        if not chunk.get("content", "").strip():
            errors.append(f"{prefix} empty content")

        if not chunk.get("embed_text", "").strip():
            errors.append(f"{prefix} empty embed_text")

        meta = chunk.get("metadata", {})
        missing_meta = REQUIRED_META - set(meta.keys())
        if missing_meta:
            errors.append(f"{prefix} missing metadata fields: {missing_meta}")

        if meta.get("source_type") not in VALID_SOURCE_TYPES:
            errors.append(f"{prefix} invalid source_type: {meta.get('source_type')!r}")

    return errors


#  QUICK TEST 

if __name__ == "__main__":


    fake_pdf_chunk = {
        "chunk_id": "abc-123",
        "type": "text",
        "content": "This is a paragraph from a PDF.",
        "metadata": {
            "source": "report.pdf",
            "source_type": "pdf_digital",
            "section": "Introduction",
            "page_start": 1,
            "page_end": 1,
            "chunk_index": 1,
            "chunk_total": 10,
            "token_count": 9,
        }
    }

    fake_txt_chunk = {
        "type": "text",
        "content": "This is a paragraph from a TXT file.",
        "metadata": {
            "file_name": "notes.txt",
            "chunk_id": "txt-chunk-uuid",
        }
    }

    fake_pptx_output = {
        "chunks": [
            {
                "chunk_id": "pptx-1",
                "type": "table",
                "content": "Col1 | Col2\n--- | ---\nA | B",
                "metadata": {
                    "source": "deck.pptx",
                    "slide": 3,
                    "section": "Results",
                }
            }
        ],
        "metadata": {"source": "deck.pptx", "slide_count": 10}
    }

    fake_image_output = {
        "chunk_id": "img-1",
        "type": "image",
        "content": "A bar chart showing quarterly revenue growth.",
        "metadata": {"source": "chart.png", "image_index": 1}
    }

    tests = [
        (fake_pdf_chunk,    "pdf_digital", "PDF chunk"),
        (fake_txt_chunk,    "txt",         "TXT chunk"),
        (fake_pptx_output,  "pptx",        "PPTX output"),
        (fake_image_output, "image",       "Image output"),
    ]

    for raw, stype, label in tests:
        result = normalize_chunks(raw, stype)
        errors = validate_chunks(result)

        print(f"\n── {label} ({'✓' if not errors else '✗'})")
        for chunk in result:
            print(f"  chunk_id    : {chunk['chunk_id'][:20]}")
            print(f"  type        : {chunk['type']}")
            print(f"  source_type : {chunk['metadata']['source_type']}")
            print(f"  embed_text  : {chunk['embed_text'][:60]}...")
            print(f"  token_count : {chunk['metadata']['token_count']}")
        if errors:
            for e in errors:
                print(f"  ERROR: {e}")