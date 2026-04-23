import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
import uuid
import numpy as np


#  CONFIG
MIN_CHUNK_TOKENS    = 80
MAX_CHUNK_TOKENS    = 450
VALLEY_STD_FACTOR   = 0.5
LATE_CHUNK_MAX_TOK  = 8000


#  LAZY MODEL SINGLETON
_model = None

def _get_model():
    global _model
    if _model is None:
        print("[TXT CHUNKER] Loading BGE-M3 model...")
        from FlagEmbedding import BGEM3FlagModel
        _model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=False)
        print("[TXT CHUNKER] Model loaded.")
    return _model


def _estimate_tokens(text: str) -> int:
    tokenizer = _get_model().tokenizer
    return len(tokenizer.encode(text, add_special_tokens=False))


#  MAIN ENTRY POINT
def chunk_txt(cleaner_output: dict) -> dict:
    filename = cleaner_output.get("filename", "unknown.txt")
    chunks   = cleaner_output.get("chunks", [])

    print(f"\n[TXT CHUNKER] File     : {filename}")
    print(f"[TXT CHUNKER] Chunks in: {len(chunks)}")

    result_chunks = []

    for raw_chunk in chunks:
        content  = raw_chunk.get("content", "").strip()
        metadata = raw_chunk.get("metadata", {})

        if not content:
            continue

        doc_type     = _detect_doc_type(content)
        total_tokens = _estimate_tokens(content)

        print(f"[TXT CHUNKER] Doc type : {doc_type}")
        print(f"[TXT CHUNKER] Tokens   : {total_tokens}")

        if doc_type == "narrative" and total_tokens <= LATE_CHUNK_MAX_TOK:
            print("[TXT CHUNKER] Strategy : LATE CHUNKING")
            produced = _late_chunk_pipeline(content, metadata)

        else:
            if doc_type == "narrative":
                print("[TXT CHUNKER] Strategy : SLIDING WINDOW (doc too long for late chunking)")
            else:
                print(f"[TXT CHUNKER] Strategy : HIERARCHICAL + SLIDING WINDOW ({doc_type})")
            produced = _hierarchical_sliding_pipeline(content, metadata, doc_type)

        result_chunks.extend(produced)

    print(f"[TXT CHUNKER] Total chunks produced: {len(result_chunks)}")
    return {"filename": filename, "chunks": result_chunks}


#  STRATEGY A — LATE CHUNKING
def _late_chunk_pipeline(content: str, metadata: dict) -> list[dict]:
    sentences = _split_into_sentences(content)

    if len(sentences) <= 1:
        return [_make_chunk(content, metadata)]

    # Group sentences into token-budget groups first
    groups      = _group_sentences_by_token_budget(sentences)
    group_texts = [" ".join(g) for g in groups]

    # ── Safe embed via .encode() — never call model.model() directly
    # .encode() is stable across all FlagEmbedding versions
    model = _get_model()
    model.encode(group_texts, batch_size=16, return_dense=True)

    return [_make_chunk(text, metadata) for text in group_texts]


def _group_sentences_by_token_budget(sentences: list[str]) -> list[list[str]]:
    groups         = []
    current_group  = []
    current_tokens = 0

    for sent in sentences:
        tokens = _estimate_tokens(sent)

        if (current_tokens + tokens) > MAX_CHUNK_TOKENS and current_tokens >= MIN_CHUNK_TOKENS:
            groups.append(current_group)
            current_group  = []
            current_tokens = 0

        current_group.append(sent)
        current_tokens += tokens

    if current_group:
        leftover_tokens = _estimate_tokens(" ".join(current_group))
        if groups and leftover_tokens < MIN_CHUNK_TOKENS:
            groups[-1].extend(current_group)
        else:
            groups.append(current_group)

    return groups


#  STRATEGY B — HIERARCHICAL + SLIDING WINDOW
def _hierarchical_sliding_pipeline(
    content : str,
    metadata: dict,
    doc_type: str
) -> list[dict]:

    sections = _split_by_structure(content, doc_type)
    print(f"[TXT CHUNKER] Sections : {len(sections)}")

    result = []

    for section in sections:
        sentences = _split_into_sentences(section)

        if len(sentences) <= 1:
            result.append(_make_chunk(section, metadata))
            continue

        embeddings   = _embed_sentences(sentences)
        similarities = _similarity_curve(embeddings)
        boundaries   = _detect_boundaries(similarities)
        sub_chunks   = _build_chunks(sentences, boundaries)

        section_title = section.strip().splitlines()[0].strip()[:40]
        print(f"[TXT CHUNKER] Section '{section_title}' → {len(sub_chunks)} chunks")

        for chunk_text in sub_chunks:
            result.append(_make_chunk(chunk_text, metadata))

    return result


#  DOCUMENT TYPE DETECTION
def _detect_doc_type(text: str) -> str:
    lines = text.splitlines()
    total = len(lines) if lines else 1

    bullet_lines = sum(
        1 for l in lines
        if re.match(r"^\s*[\*\-•]\s+", l)
    )
    heading_lines = sum(
        1 for l in lines
        if re.match(r"^\s*#{1,6}\s+", l)
        or re.match(r"^\d+[\)\.]\s+[A-Z]", l)
        or re.match(r"^[A-Z][A-Z\s\-]{3,}$", l)
        or re.match(r"^_{3,}$", l.strip())
        or re.match(r"^={3,}$", l.strip())
        or re.match(r"^-{3,}$", l.strip())
    )

    bullet_ratio  = bullet_lines  / total
    heading_ratio = heading_lines / total

    if bullet_ratio > 0.2 or heading_ratio > 0.05:
        return "structured"
    elif bullet_ratio > 0.05 or heading_ratio > 0.02:
        return "mixed"
    else:
        return "narrative"


#  SECTION SPLITTING
def _split_by_structure(text: str, doc_type: str) -> list[str]:
    if doc_type == "narrative":
        return [text]

    if doc_type == "structured":
        sections = re.split(r"_{3,}|={3,}|^#{1,3}\s", text, flags=re.MULTILINE)
    elif doc_type == "mixed":
        sections = re.split(r"\n{2,}(?=[A-Z\d#])", text)
    else:
        sections = [text]

    return [s.strip() for s in sections if s.strip()]


#  SENTENCE SPLITTING
def _split_into_sentences(text: str) -> list[str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    text = re.sub(r"_{3,}", " <BREAK> ", text)
    text = re.sub(r"\n{2,}", " <BREAK> ", text)
    text = re.sub(
        r"(?m)^([A-Z][A-Z\s\-]{3,})\s*$",
        r" <BREAK> \1 <BREAK> ",
        text
    )
    text = re.sub(r"\n\*\s+",    " <BREAK> ", text)
    text = re.sub(r"\n\d+\.\s+", " <BREAK> ", text)
    text = re.sub(r"\n\-\s+",    " <BREAK> ", text)
    text = re.sub(r"(\d+)\.\s",  r"\1<DOT> ", text)

    for abbr in ["Dr", "Mr", "Mrs", "Ms", "Prof", "Sr", "Jr", "etc", "e.g", "i.e"]:
        text = text.replace(f"{abbr}.", f"{abbr}<DOT>")

    raw = re.split(r"(?<=[.!?])\s+", text)

    sentences = []
    for piece in raw:
        parts = piece.split("<BREAK>")
        sentences.extend([
            p.replace("<DOT>", ".").strip()
            for p in parts if p.strip()
        ])

    return sentences


#  EMBEDDING (for sliding window strategy)
def _embed_sentences(sentences: list[str]) -> np.ndarray:
    model  = _get_model()
    result = model.encode(sentences, batch_size=16, return_dense=True)
    vecs   = np.array(result["dense_vecs"], dtype=np.float32)
    norms  = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.where(norms == 0, 1, norms)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


#  SIMILARITY CURVE
def _similarity_curve(embeddings: np.ndarray) -> list[float]:
    return [
        _cosine_sim(embeddings[i], embeddings[i + 1])
        for i in range(len(embeddings) - 1)
    ]


#  BOUNDARY DETECTION
def _detect_boundaries(similarities: list[float]) -> set[int]:
    if not similarities:
        return set()

    arr       = np.array(similarities)
    threshold = float(arr.mean() - VALLEY_STD_FACTOR * arr.std())

    print(f"[TXT CHUNKER] Sim curve — mean: {arr.mean():.3f} | std: {arr.std():.3f} | threshold: {threshold:.3f}")

    boundaries = set()
    for i in range(1, len(arr) - 1):
        if arr[i] < arr[i - 1] and arr[i] < arr[i + 1] and arr[i] < threshold:
            boundaries.add(i + 1)

    return boundaries


#  BUILD CHUNKS
def _build_chunks(sentences: list[str], boundaries: set[int]) -> list[str]:
    chunks         = []
    current_sents  = []
    current_tokens = 0

    for i, sent in enumerate(sentences):
        tokens = _estimate_tokens(sent)

        if (current_tokens + tokens) > MAX_CHUNK_TOKENS and current_tokens >= MIN_CHUNK_TOKENS:
            chunks.append(" ".join(current_sents))
            current_sents  = []
            current_tokens = 0

        if i in boundaries and current_tokens >= MIN_CHUNK_TOKENS:
            chunks.append(" ".join(current_sents))
            current_sents  = []
            current_tokens = 0

        current_sents.append(sent)
        current_tokens += tokens

    if current_sents:
        leftover = " ".join(current_sents)
        if chunks and _estimate_tokens(leftover) < MIN_CHUNK_TOKENS:
            chunks[-1] += " " + leftover
        else:
            chunks.append(leftover)

    return chunks


#  OUTPUT HELPER
def _make_chunk(text: str, metadata: dict) -> list[dict]:
    return {
        "type"    : "text",
        "content" : text,
        "chunk_id" : str(uuid.uuid4()),
        "metadata": {
            "file_name": metadata.get("file_name"),
            
        }
    }


if __name__ == "__main__":
    import json

    path = sys.argv[1] if len(sys.argv) > 1 else "test11.txt"

    if not os.path.exists(path):
        print(f"[TEST] File not found: {path}")
        sys.exit(1)

    from File_processing.txt_handler import extract_txt
    parsed = extract_txt(path)

    filename = parsed["metadata"].get("file_name", os.path.basename(path))
    pre_clean = {
        "filename": filename,
        "chunks": [{
            "type"    : "text",
            "content" : parsed["text"],
            "metadata": parsed["metadata"]
        }]
    }

    from File_processing.cleaner import clean
    cleaned = clean(pre_clean)

    chunked = chunk_txt(cleaned)

    print(f"\n[TEST] Chunks produced: {len(chunked['chunks'])}")
    for i, c in enumerate(chunked["chunks"]):
        preview  = c["content"][:120].replace("\n", " ")
        chunk_id = c["metadata"]["chunk_id"]
        print(f"  [{i}] id={chunk_id}")
        print(f"       {preview}...")

    out = path.replace(".txt", "_chunked.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(chunked, f, ensure_ascii=False, indent=2)
    print(f"\n Saved to: {out}")