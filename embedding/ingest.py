import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from pathlib import Path

from normalizer import normalize_chunks, validate_chunks

from chunking.pdf_chunker        import chunk_pdf
from chunking.docx_chunker       import chunk_docx
from chunking.pptx_chunker       import extract_pptx
from chunking.eml_chunker        import chunk_eml
from chunking.txt_chunker        import chunk_txt
from chunking.scanned_pdf_chunker import chunk_scanned_pdf
from File_processing.image_handler import extract_image

from File_processing.digital_pdf  import extract_digital_pdf
from File_processing.txt_handler  import extract_txt
from File_processing.cleaner      import clean


EXT_TO_SOURCE_TYPE = {
    ".pdf"  : "pdf_digital",   
    ".docx" : "docx",
    ".pptx" : "pptx",
    ".eml"  : "eml",
    ".txt"  : "txt",
    ".jpg"  : "image",
    ".jpeg" : "image",
    ".png"  : "image",
    ".bmp"  : "image",
    ".tiff" : "image",
    ".tif"  : "image",
}



def _run_pdf_digital(file_path: str):
    handler_output = extract_digital_pdf(file_path)
    handler_output = clean(handler_output)
    return chunk_pdf(handler_output)


def _run_pdf_scanned(file_path: str):
    from File_processing.scanned_pdf import extract_scanned_pdf
    parsed = extract_scanned_pdf(file_path)
    return chunk_scanned_pdf(parsed)


def _run_docx(file_path: str):
    return chunk_docx(file_path)


def _run_pptx(file_path: str):
    return extract_pptx(file_path)


def _run_eml(file_path: str):
    return chunk_eml(file_path)


def _run_txt(file_path: str):
    parsed   = extract_txt(file_path)
    filename = parsed["metadata"].get("file_name", Path(file_path).name)
    pre_clean = {
        "filename": filename,
        "chunks": [{
            "type"    : "text",
            "content" : parsed["text"],
            "metadata": parsed["metadata"],
        }]
    }
    cleaned = clean(pre_clean)
    return chunk_txt(cleaned)


def _run_image(file_path: str):
    return extract_image(file_path)


CHUNKER_MAP = {
    "pdf_digital" : _run_pdf_digital,
    "pdf_scanned" : _run_pdf_scanned,
    "docx"        : _run_docx,
    "pptx"        : _run_pptx,
    "eml"         : _run_eml,
    "txt"         : _run_txt,
    "image"       : _run_image,
}



def ingest_file(
    file_path   : str,
    source_type : str | None = None,   
    validate    : bool       = False,  
    save_json   : bool       = False,  
) -> list[dict]:
    """
    Full pipeline for any supported file type:
      1. Auto-detect source_type from extension (or use override)
      2. Run the appropriate chunker
      3. Normalize all chunks to canonical schema
      4. Optionally validate + save JSON preview
      5. Return list of normalized chunks → ready for embedding

    Args:
        file_path   : path to the file
        source_type : override auto-detection ("pdf_digital" | "pdf_scanned" |
                      "docx" | "pptx" | "eml" | "txt" | "image")
        validate    : run validate_chunks() and print any errors
        save_json   : save normalized chunks to <stem>_normalized.json

    Returns:
        list[dict] — normalized chunks in canonical schema
    """
    file_path = str(file_path)
    ext       = Path(file_path).suffix.lower()

    if source_type is None:
        source_type = EXT_TO_SOURCE_TYPE.get(ext)
        if source_type is None:
            raise ValueError(
                f"Cannot auto-detect source_type for extension '{ext}'. "
                f"Pass source_type explicitly."
            )

    print(f"\n{'='*60}")
    print(f"  FILE        : {Path(file_path).name}")
    print(f"  SOURCE TYPE : {source_type}")
    print(f"{'='*60}")

    chunker = CHUNKER_MAP.get(source_type)
    if chunker is None:
        raise ValueError(f"No chunker registered for source_type='{source_type}'")

    print(f"\n[INGEST] Step 1 — Chunking...")
    raw_output = chunker(file_path)

    # ── NORMALIZE 
    print(f"[INGEST] Step 2 — Normalizing...")
    chunks = normalize_chunks(raw_output, source_type)
    print(f"[INGEST] Normalized chunks : {len(chunks)}")

    # ── VALIDATE (optional) 
    if validate:
        print(f"[INGEST] Step 3 — Validating...")
        errors = validate_chunks(chunks)
        if errors:
            print(f"[INGEST] ⚠  {len(errors)} validation error(s):")
            for e in errors:
                print(f"          {e}")
        else:
            print(f"[INGEST]   All chunks valid")

    # ── PREVIEW 
    _print_preview(chunks, source_type)

    # ── SAVE JSON (optional) 
    if save_json:
        out = Path(file_path).stem + "_normalized.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        print(f"\n[INGEST] Saved to : {out}")

    return chunks


# ── PREVIEW PRINTER 

def _print_preview(chunks: list[dict], source_type: str, n: int = 3):
    """Print a clean summary + first N chunks."""

    type_counts = {}
    for c in chunks:
        t = c.get("type", "?")
        type_counts[t] = type_counts.get(t, 0) + 1

    print(f"\n{'─'*60}")
    print(f"  SUMMARY")
    print(f"{'─'*60}")
    print(f"  Total chunks  : {len(chunks)}")
    for t, count in sorted(type_counts.items()):
        print(f"  {t:<12}  : {count}")

    print(f"\n{'─'*60}")
    print(f"  FIRST {n} CHUNKS")
    print(f"{'─'*60}")

    for i, chunk in enumerate(chunks[:n]):
        m = chunk["metadata"]
        print(f"\n  Chunk [{i+1}]")
        print(f"    chunk_id    : {chunk['chunk_id']}")
        print(f"    type        : {chunk['type']}")
        print(f"    source      : {m.get('source', '')}")
        print(f"    source_type : {m.get('source_type', '')}")
        print(f"    section     : {m.get('section')}")
        print(f"    page        : {m.get('page')}")
        print(f"    token_count : {m.get('token_count')}")
        print(f"    chunk_index : {m.get('chunk_index')} / {m.get('chunk_total')}")

        # eml-specific fields
        if source_type in ("eml", "eml_body", "eml_body_table"):
            print(f"    subject     : {m.get('subject')}")
            print(f"    from        : {m.get('from')}")
            print(f"    date        : {m.get('date')}")

        print(f"    content     : {chunk['content'][:120].replace(chr(10), ' ')}...")
        print(f"    embed_text  : {chunk['embed_text'][:120].replace(chr(10), ' ')}...")

    print(f"\n{'='*60}\n")


# ── CLI 

if __name__ == "__main__":
    """
    Usage:
        python ingest.py <file_path> [source_type]

    Examples:
        python ingest.py report.pdf
        python ingest.py scan.pdf pdf_scanned
        python ingest.py deck.pptx
        python ingest.py email.eml
        python ingest.py notes.txt
        python ingest.py chart.png
        python ingest.py doc.docx
    """
    if len(sys.argv) < 2:
        print("Usage: python ingest.py <file_path> [source_type]")
        sys.exit(1)

    fp          = sys.argv[1]
    stype       = sys.argv[2] if len(sys.argv) > 2 else None

    chunks = ingest_file(
        file_path   = fp,
        source_type = stype,
        validate    = True,   
        save_json   = True,    
    )

    print(f"[INGEST] Done. {len(chunks)} chunks ready for embedding.")