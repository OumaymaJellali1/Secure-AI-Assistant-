import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HUGGINGFACE_HUB_VERBOSITY"] = "error"

import re
import uuid
import numpy as np
from itertools import groupby
from FlagEmbedding import BGEM3FlagModel
from config import DEBUG


# ── MODEL ─────────────────────────────────────────────
_model = None

def _get_model() -> BGEM3FlagModel:
    global _model
    if _model is None:
        if DEBUG:
            print("[PDF CHUNKER] Loading BGE-M3 model...")
        _model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
        if DEBUG:
            print("[PDF CHUNKER] BGE-M3 ready")
    return _model


# ── SETTINGS ──────────────────────────────────────────
MAX_TOKENS        = 520
TABLE_MAX_TOKENS  = 600
OVERLAP_TOKENS    = 0
SIMILARITY_CUTOFF = 0.65
MIN_TOKENS        = 100

SKIP_SECTIONS = {
    "references",
    "bibliography",
    "acknowledgments",
    "acknowledgements",
}

# ── CAPTION PATTERNS ──────────────────────────────────
CAPTION_RE = re.compile(
    r'^(Fig\.|Figure|Table)\s+\d+[.:]\s+\S',
    re.IGNORECASE
)
LABEL_RE = re.compile(
    r'^(Fig\.|Figure|Table)\s+\d+[.?\s*$',
    re.IGNORECASE
)
ANY_CAPTION_RE = re.compile(
    r'^(Fig\.|Figure|Table|Tableau|FIGURE|TABLE)\s*[\dIVXLC]+[.:\s]',
    re.IGNORECASE
)


#  MAIN ENTRY POINT

def chunk_pdf(handler_output: dict) -> list[dict]:
    """
    Takes the output of extract_digital_pdf() and returns
    a flat list of normalized chunks ready for embedding + storage.

    Works for ANY PDF type: academic papers, books, reports, manuals.

    Strategy:
      - Process in ORIGINAL ORDER to keep captions adjacent to images/tables
      - text   → separate caption lines from body → merge body per section
                 → semantic split if too large
      - table  → split by rows, repeat header in every chunk
      - image  → always atomic
      - after  → attach captions (checks BOTH before and after)
                 propagate caption to ALL row chunks of same table
                 merge tiny text chunks (forward AND backward)
      - skip   → references, bibliography, acknowledgments
    """

    filename   = handler_output.get("filename", "unknown")
    raw_chunks = handler_output.get("chunks",   [])

    if DEBUG:
        print(f"\n[PDF CHUNKER] File      : {filename}")
        print(f"[PDF CHUNKER] Raw chunks: {len(raw_chunks)}")

    processed   = []
    text_buffer = []

    def flush_text_buffer(buf: list) -> list:
        """
        Process accumulated text chunks:
        - Separate caption lines from body text
        - Merge body text per section
        - Semantic split if too large
        - Keep caption lines standalone for _attach_captions()
        """
        if not buf:
            return []

        result = []

        for section_key, group in groupby(
            buf,
            key=lambda c: c.get("metadata", {}).get("section", "")
        ):
            group         = list(group)
            body_chunks   = []
            caption_lines = []

            for c in group:
                content = c["content"].strip()
                # separate short caption-like lines from body text
                if ANY_CAPTION_RE.match(content) and len(content) < 200:
                    caption_lines.append(c)
                else:
                    body_chunks.append(c)

            meta = group[0].get("metadata", {})

            # process body text
            if body_chunks:
                merged = " ".join(
                    c["content"].strip()
                    for c in body_chunks
                    if c["content"].strip()
                )
                if merged:
                    tokens = _count_tokens(merged)
                    if tokens <= MAX_TOKENS:
                        result.append(_make_chunk(merged, "text", meta))
                    else:
                        if DEBUG:
                            print(f"[PDF CHUNKER] Splitting section "
                                  f"'{section_key[:40]}' ({tokens} tokens)")
                        for piece in _semantic_split(merged):
                            if piece.strip():
                                result.append(_make_chunk(piece, "text", meta))

            # keep caption lines separate so _attach_captions() finds them
            for c in caption_lines:
                result.append(_make_chunk(c["content"].strip(), "text", meta))

        return result

    # ── process in original order ──────────────────────
    for raw in raw_chunks:
        chunk_type = raw.get("type",    "text")
        content    = raw.get("content", "").strip()
        metadata   = raw.get("metadata", {})

        if not content:
            continue

        section = metadata.get("section", "").strip().lower()
        if any(skip in section for skip in SKIP_SECTIONS):
            if DEBUG:
                print(f"[PDF CHUNKER] Skipping section: {section[:50]}")
            continue

        if chunk_type in ("table", "image"):
            processed.extend(flush_text_buffer(text_buffer))
            text_buffer = []

            if chunk_type == "table":
                processed.extend(_split_table(content, metadata))
            else:
                processed.append(_make_chunk(content, "image", metadata))

        else:
            text_buffer.append(raw)

    processed.extend(flush_text_buffer(text_buffer))

    # ── attach captions to images and tables ──────────
    normalized = _attach_captions(processed)

    # ── propagate caption to ALL chunks of same table ─
    normalized = _propagate_table_captions(normalized)

    # ── merge tiny text chunks ─────────────────────────
    normalized = _merge_tiny_chunks(normalized)

    # ── fill chunk_index / chunk_total ────────────────
    total = len(normalized)
    for i, chunk in enumerate(normalized):
        chunk["metadata"]["chunk_index"] = i + 1
        chunk["metadata"]["chunk_total"] = total

    if DEBUG:
        _print_summary(normalized)

    return normalized


#  TABLE SPLIT BY ROWS

def _split_table(content: str, metadata: dict) -> list[dict]:
    """
    Split a large markdown table into smaller chunks by rows.
    Header is repeated in every chunk — meaning always preserved.

    Every chunk:
      | Col1 | Col2 | Col3 |   ← header always present
      |------|------|------|
      | row1 | ...  | ...  |
      | row2 | ...  | ...  |
    """
    if _count_tokens(content) <= TABLE_MAX_TOKENS:
        return [_make_chunk(content, "table", metadata)]

    lines = content.strip().split("\n")

    header_lines = []
    data_lines   = []
    header_done  = False

    for line in lines:
        if not header_done:
            header_lines.append(line)
            stripped = (
                line.replace("|", "")
                    .replace("-", "")
                    .replace(":", "")
                    .replace(" ", "")
            )
            if stripped == "":
                header_done = True
        else:
            if line.strip():
                data_lines.append(line)

    if not data_lines:
        return [_make_chunk(content, "table", metadata)]

    header    = "\n".join(header_lines)
    threshold = TABLE_MAX_TOKENS - 50

    if DEBUG:
        print(f"[PDF CHUNKER] Splitting table: "
              f"{len(data_lines)} rows, {_count_tokens(content)} tokens")

    chunks = []
    batch  = []

    for row in data_lines:
        batch.append(row)
        current = header + "\n" + "\n".join(batch)

        if _count_tokens(current) > threshold:
            batch.pop()
            if batch:
                table_content = header + "\n" + "\n".join(batch)
                chunks.append(_make_chunk(table_content, "table", metadata))
            batch = [row]

    if batch:
        table_content = header + "\n" + "\n".join(batch)
        chunks.append(_make_chunk(table_content, "table", metadata))

    if DEBUG:
        print(f"[PDF CHUNKER] Table → {len(chunks)} chunks")

    return chunks


#  CAPTION ATTACHMENT

def _attach_captions(chunks: list[dict]) -> list[dict]:
    """
    Attaches captions to image/table chunks.
    Works for ANY PDF layout — checks BOTH before AND after.

    Priority:
      1. Check AFTER  (caption below — most PDFs, books, reports)
      2. Check BEFORE (caption above — academic papers)

    Handles:
      Pattern A — full caption after:
        image/table → text: "Fig. 2. A representative instance..."

      Pattern B — split label after:
        image/table → text: "Fig. 2."
                    → text: "Full caption sentence..."

      Pattern C — caption before:
        text: "TABLE I SUMMARY OF RAG METHODS"
        table → rows...

      Pattern D — any caption-like text adjacent:
        image/table → text: "TABLE III SUMMARY OF METRICS..."
    """
    result = []
    i = 0

    while i < len(chunks):
        chunk = chunks[i]

        if chunk["type"] not in ("image", "table"):
            result.append(chunk)
            i += 1
            continue

        next1 = chunks[i + 1] if i + 1 < len(chunks) else None
        next2 = chunks[i + 2] if i + 2 < len(chunks) else None

        caption_text = ""
        remainder    = ""
        skip         = 0
        found_before = False

        # ── check AFTER (caption below) ───────────────
        if next1 and next1["type"] == "text":
            n1 = next1["content"].strip()

            # Pattern B: "Fig. 2." → "Full caption... body"
            if LABEL_RE.match(n1) and next2 and next2["type"] == "text":
                n2 = next2["content"].strip()
                caption_text, remainder = _extract_caption_and_remainder(
                    n1 + " " + n2
                )
                skip = 2

            # Pattern A or D
            elif ANY_CAPTION_RE.match(n1):
                caption_text, remainder = _extract_caption_and_remainder(n1)
                skip = 1

        # ── check BEFORE (caption above) ──────────────
        if not caption_text and result and result[-1]["type"] == "text":
            prev_content = result[-1]["content"].strip()
            if ANY_CAPTION_RE.match(prev_content):
                caption_text = result.pop()["content"].strip()
                found_before = True

        # ── attach caption ─────────────────────────────
        if caption_text:
            chunk = dict(chunk)
            if found_before:
                chunk["content"] = (
                    f"Caption: {caption_text}\n\n" + chunk["content"]
                )
            else:
                chunk["content"] = (
                    chunk["content"].rstrip() + f"\n\nCaption: {caption_text}"
                )

        result.append(chunk)

        if remainder and len(remainder) > 20:
            leftover            = dict(chunks[i + skip])
            leftover["content"] = remainder
            result.append(leftover)

        i += skip + 1

    return result


def _extract_caption_and_remainder(text: str):
    """
    Extract caption sentence from text like:
      "Fig. 2. A representative instance of RAG. widespread adoption..."
      "TABLE I SUMMARY OF RAG METHODS"

    Returns (caption_text, remainder_body)
    """
    text = text.strip()

    label_match = re.match(
        r'^((?:Fig\.|Figure|Table|Tableau|FIGURE|TABLE)\s*[\dIVXLC]+[.:\s]*)',
        text,
        re.IGNORECASE
    )

    if not label_match:
        parts = re.split(r'(?<=[.!?])\s+', text, maxsplit=1)
        return parts[0].strip(), parts[1].strip() if len(parts) > 1 else ""

    label_end = label_match.end()
    rest      = text[label_end:]

    if not rest.strip():
        return text, ""

    sentence_match = re.search(r'(?<=[.!?])\s+', rest)

    if sentence_match:
        caption_text = text[:label_end + sentence_match.start()].strip()
        remainder    = rest[sentence_match.end():].strip()
    else:
        caption_text = text
        remainder    = ""

    return caption_text, remainder


#  PROPAGATE TABLE CAPTION TO ALL ROW CHUNKS

def _propagate_table_captions(chunks: list[dict]) -> list[dict]:
    """
    After _attach_captions(), the caption lands on only ONE chunk
    of a split table (whichever was adjacent to the caption text).

    This function copies the caption to ALL consecutive table chunks
    that belong to the same table (same section).

    Why every chunk needs the caption:
      - Each row chunk is retrieved independently by similarity search
      - Without the caption, the LLM doesn't know what table it came from
      - With caption on every chunk, any retrieved row is self-contained

    Result:
      chunk 24: "| Method | Source |...\n\nCaption: TABLE I SUMMARY..." ✅
      chunk 25: "| Method | Source |...\n\nCaption: TABLE I SUMMARY..." ✅
      ...
      chunk 34: "| Method | Source |...\n\nCaption: TABLE I SUMMARY..." ✅
    """
    result = list(chunks)

    # pass 1 — find all table chunks that have a caption
    # collect: caption text + index range of sibling chunks
    i = 0
    while i < len(result):
        chunk = result[i]

        if chunk["type"] == "table" and "Caption:" in chunk.get("content", ""):

            # extract caption part
            content    = chunk["content"]
            cap_marker = content.find("\n\nCaption:")
            if cap_marker == -1:
                cap_marker = content.find("Caption:")
                if cap_marker == -1:
                    i += 1
                    continue

            # "\n\nCaption: TABLE I SUMMARY..." or "Caption: TABLE I..."
            if content[cap_marker:cap_marker+2] == "\n\n":
                caption_part = content[cap_marker:]       # includes \n\n
            else:
                caption_part = "\n\n" + content[cap_marker:]

            table_only = content[:cap_marker].strip()

            # find ALL sibling chunks (consecutive tables, same section)
            section = chunk["metadata"]["section"]

            # walk backward
            start = i
            while (start > 0
                   and result[start - 1]["type"] == "table"
                   and result[start - 1]["metadata"]["section"] == section):
                start -= 1

            # walk forward
            end = i
            while (end + 1 < len(result)
                   and result[end + 1]["type"] == "table"
                   and result[end + 1]["metadata"]["section"] == section):
                end += 1

            # apply caption to ALL siblings
            for idx in range(start, end + 1):
                sibling         = dict(result[idx])
                sibling_content = sibling["content"]

                # remove existing caption if present (avoid duplicates)
                cap_pos = sibling_content.find("\n\nCaption:")
                if cap_pos == -1:
                    cap_pos = sibling_content.find("Caption:")
                if cap_pos != -1:
                    sibling_content = sibling_content[:cap_pos].strip()

                sibling["content"] = sibling_content + caption_part
                sibling["metadata"]["token_count"] = _count_tokens(
                    sibling["content"]
                )
                result[idx] = sibling

            if DEBUG:
                print(f"[PDF CHUNKER] Caption propagated to "
                      f"{end - start + 1} table chunks "
                      f"(chunks {start+1}–{end+1})")

        i += 1

    return result


#  MERGE TINY CHUNKS

def _merge_tiny_chunks(chunks: list[dict]) -> list[dict]:
    """
    Merge any text chunk under MIN_TOKENS with its neighbor.
    Tries forward first, then backward.
    """
    result = []
    i = 0

    while i < len(chunks):
        chunk = chunks[i]

        if (chunk["type"] == "text"
                and _count_tokens(chunk["content"]) < MIN_TOKENS):

            # try FORWARD
            if (i + 1 < len(chunks)
                    and chunks[i + 1]["type"] == "text"):
                next_chunk = dict(chunks[i + 1])
                next_chunk["content"] = (
                    chunk["content"].strip()
                    + " "
                    + next_chunk["content"].strip()
                )
                next_chunk["metadata"]["token_count"] = (
                    _count_tokens(next_chunk["content"])
                )
                chunks[i + 1] = next_chunk
                i += 1
                continue

            # try BACKWARD
            elif result and result[-1]["type"] == "text":
                result[-1]["content"] = (
                    result[-1]["content"].strip()
                    + " "
                    + chunk["content"].strip()
                )
                result[-1]["metadata"]["token_count"] = (
                    _count_tokens(result[-1]["content"])
                )
                i += 1
                continue

        result.append(chunk)
        i += 1

    return result


#  SEMANTIC SPLIT

def _semantic_split(text: str) -> list[str]:
    """
    Split an oversized text block into semantically coherent pieces.

    Steps:
      1. Split into sentences
      2. Embed with BGE-M3
      3. Cosine similarity between adjacent sentences
      4. Cut where similarity < SIMILARITY_CUTOFF (0.65)
      5. Merge pieces under MIN_TOKENS
    """
    sentences = _split_sentences(text)

    if len(sentences) <= 1:
        return [text]

    model      = _get_model()
    embeddings = model.encode(
        sentences,
        batch_size          = 32,
        max_length          = 512,
        return_dense        = True,
        return_sparse       = False,
        return_colbert_vecs = False,
    )["dense_vecs"]

    similarities = _adjacent_cosine(embeddings)

    if DEBUG:
        print(f"[PDF CHUNKER]   Sentences  : {len(sentences)}")
        print(f"[PDF CHUNKER]   Sim (min/avg/max): "
              f"{min(similarities):.2f} / "
              f"{sum(similarities)/len(similarities):.2f} / "
              f"{max(similarities):.2f}")

    cut_indices = [
        i + 1
        for i, sim in enumerate(similarities)
        if sim < SIMILARITY_CUTOFF
    ]

    if DEBUG:
        print(f"[PDF CHUNKER]   Cut points : {cut_indices}")

    segments = _build_segments(sentences, cut_indices)
    segments = _merge_small_segments(segments, min_tokens=MIN_TOKENS)

    return segments


#  HELPERS

def _split_sentences(text: str) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    sentences  = []

    for para in paragraphs:
        parts = re.split(r"(?<=[.!?؟])\s+", para)
        for part in parts:
            part = part.strip()
            if part:
                sentences.append(part)

    if not sentences:
        return paragraphs if paragraphs else [text]

    return sentences


def _adjacent_cosine(embeddings: np.ndarray) -> list[float]:
    similarities = []
    for i in range(len(embeddings) - 1):
        a      = embeddings[i]
        b      = embeddings[i + 1]
        dot    = float(np.dot(a, b))
        norm_a = float(np.linalg.norm(a))
        norm_b = float(np.linalg.norm(b))

        if norm_a == 0 or norm_b == 0:
            similarities.append(0.0)
        else:
            similarities.append(dot / (norm_a * norm_b))

    return similarities


def _build_segments(sentences: list[str], cut_indices: list[int]) -> list[str]:
    if not cut_indices:
        return [" ".join(sentences)]

    segments   = []
    prev_index = 0

    for cut in cut_indices:
        segment = " ".join(sentences[prev_index:cut]).strip()
        if segment:
            segments.append(segment)
        prev_index = cut

    last = " ".join(sentences[prev_index:]).strip()
    if last:
        segments.append(last)

    return segments


def _merge_small_segments(
    segments: list[str],
    min_tokens: int = MIN_TOKENS
) -> list[str]:
    if len(segments) <= 1:
        return segments

    merged = [segments[0]]

    for seg in segments[1:]:
        prev       = merged[-1]
        combined   = prev + " " + seg
        prev_tok   = _count_tokens(prev)
        seg_tok    = _count_tokens(seg)
        combined_t = _count_tokens(combined)

        if (prev_tok < min_tokens or seg_tok < min_tokens) \
                and combined_t <= MAX_TOKENS:
            merged[-1] = combined.strip()
        else:
            merged.append(seg)

    return merged


def _count_tokens(text: str) -> int:
    return max(1, int(len(text) / 3.5))


def _make_chunk(content: str, chunk_type: str, metadata: dict) -> dict:
    return {
        "chunk_id" : str(uuid.uuid4()),
        "type"     : chunk_type,
        "content"  : content,
        "metadata" : {
            "source"      : metadata.get("source",     "unknown"),
            "source_type" : "pdf_digital",
            "section"     : metadata.get("section",    ""),
            "page_start"  : metadata.get("page_start", None),
            "page_end"    : metadata.get("page_end",   None),
            "token_count" : _count_tokens(content),
            "chunk_index" : 0,
            "chunk_total" : 0,
        }
    }


def _print_summary(chunks: list[dict]):
    text_chunks  = [c for c in chunks if c["type"] == "text"]
    table_chunks = [c for c in chunks if c["type"] == "table"]
    image_chunks = [c for c in chunks if c["type"] == "image"]

    text_tokens  = [c["metadata"]["token_count"] for c in text_chunks]
    table_tokens = [c["metadata"]["token_count"] for c in table_chunks]
    image_tokens = [c["metadata"]["token_count"] for c in image_chunks]
    captions     = [c for c in chunks if "Caption:" in c.get("content", "")]

    tbl_with_cap = [c for c in table_chunks if "Caption:" in c.get("content","")]

    print(f"\n[PDF CHUNKER] ── Summary ─────────────────────")
    print(f"[PDF CHUNKER] Total chunks      : {len(chunks)}")
    print(f"[PDF CHUNKER]   text            : {len(text_chunks)}")
    print(f"[PDF CHUNKER]   table           : {len(table_chunks)}")
    print(f"[PDF CHUNKER]   image           : {len(image_chunks)}")
    print(f"[PDF CHUNKER]   captions total  : {len(captions)}")
    print(f"[PDF CHUNKER]   table w/caption : {len(tbl_with_cap)}/{len(table_chunks)}")

    if text_tokens:
        over  = [t for t in text_tokens if t > MAX_TOKENS]
        small = [t for t in text_tokens if t < MIN_TOKENS]
        print(f"[PDF CHUNKER] Text tokens       : "
              f"min={min(text_tokens)}  "
              f"avg={int(sum(text_tokens)/len(text_tokens))}  "
              f"max={max(text_tokens)}")
        print(f"[PDF CHUNKER] Text over  {MAX_TOKENS}  : {len(over)}")
        print(f"[PDF CHUNKER] Text under {MIN_TOKENS}  : {len(small)}")

    if table_tokens:
        over_t = [t for t in table_tokens if t > TABLE_MAX_TOKENS]
        print(f"[PDF CHUNKER] Table tokens      : "
              f"min={min(table_tokens)}  "
              f"avg={int(sum(table_tokens)/len(table_tokens))}  "
              f"max={max(table_tokens)}")
        print(f"[PDF CHUNKER] Table over {TABLE_MAX_TOKENS} : {len(over_t)}")

    if image_tokens:
        print(f"[PDF CHUNKER] Image tokens      : "
              f"min={min(image_tokens)}  "
              f"avg={int(sum(image_tokens)/len(image_tokens))}  "
              f"max={max(image_tokens)}")

    print(f"[PDF CHUNKER] ─────────────────────────────────")


#  TEST

if __name__ == "__main__":
    import json
    from processing.digital_pdf import extract_digital_pdf

    if len(sys.argv) < 2:
        print("Usage: python chunking/pdf_chunker.py <file.pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]

    print(f"\n── Step 1: ingestion ─────────────────────────")
    handler_output = extract_digital_pdf(pdf_path)

    print(f"\n── Step 2: chunking ──────────────────────────")
    chunks = chunk_pdf(handler_output)

    print(f"\n── Step 3: preview (first 5 chunks) ─────────")
    for chunk in chunks[:5]:
        print(f"\n  chunk_id   : {chunk['chunk_id']}")
        print(f"  type       : {chunk['type']}")
        print(f"  section    : {chunk['metadata']['section'][:50]}")
        print(f"  pages      : {chunk['metadata']['page_start']} → "
              f"{chunk['metadata']['page_end']}")
        print(f"  tokens     : {chunk['metadata']['token_count']}")
        print(f"  index      : {chunk['metadata']['chunk_index']} / "
              f"{chunk['metadata']['chunk_total']}")
        print(f"  content    : {chunk['content'][:150]}...")

    output_path = pdf_path.replace(".pdf", "_chunks.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"\n Saved to: {output_path}")