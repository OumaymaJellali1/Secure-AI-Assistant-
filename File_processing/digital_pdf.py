import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DEBUG, GROQ_API_KEY, GROQ_VISION_MODEL, TEMP_DIR
from groq import Groq
from pathlib import Path
import json
import nest_asyncio
nest_asyncio.apply()

# ── CONFIGURE GROQ ────────────────────────────────────
groq_client = Groq(api_key=GROQ_API_KEY)

# ── CONFIGURE LLAMAPARSE ──────────────────────────────
LLAMA_API_KEY = os.environ.get("LLAMA_CLOUD_API_KEY", "")


def extract_digital_pdf(file_path: str) -> dict:
    """
    Extract content from a digital PDF using LlamaParse.
    - Text   → grouped by section
    - Tables → one chunk each with metadata
    - Images → Groq description + metadata
    """
    if DEBUG:
        print(f"\n[DIGITAL PDF] Processing: {file_path}")

    filename = Path(file_path).name

    # ── PARSE WITH LLAMAPARSE ─────────────────────────
    from llama_parse import LlamaParse
    from llama_index.core.schema import TextNode

    parser = LlamaParse(
        api_key=LLAMA_API_KEY,
        result_type="markdown",        # clean markdown output
        verbose=DEBUG,
        language="en",
    )

    documents = parser.load_data(file_path)

    # ── STATE ─────────────────────────────────────────
    chunks          = []
    tbl_counter     = 0
    img_counter     = 0

    # ── PROCESS EACH PAGE ─────────────────────────────
    for page_idx, doc in enumerate(documents):
        page_no  = page_idx + 1
        content  = doc.text.strip()

        if not content:
            continue

        lines            = content.split("\n")
        current_section  = "Unknown"
        current_text     = ""

        def flush_text():
            nonlocal current_text
            if current_text.strip():
                chunks.append({
                    "type"    : "text",
                    "content" : current_text.strip(),
                    "metadata": {
                        "source"    : filename,
                        "page_start": page_no,
                        "page_end"  : page_no,
                        "section"   : current_section,
                    }
                })
                if DEBUG:
                    print(f"[DIGITAL PDF] Text chunk saved "
                          f"(page {page_no}, section: {current_section[:30]})")
            current_text = ""

        for line in lines:
            stripped = line.strip()

            # ── DETECT SECTION HEADERS (markdown headings)
            if stripped.startswith("#"):
                flush_text()
                current_section = stripped.lstrip("#").strip()
                if DEBUG:
                    print(f"[DIGITAL PDF] New section: {current_section}")

            # ── DETECT TABLES (markdown table syntax)
            elif stripped.startswith("|") and "---" in stripped or (
                stripped.startswith("|") and current_text == ""
            ):
                flush_text()
                # collect full table
                table_lines = [stripped]
                continue  # handled below in table accumulation

            else:
                current_text += stripped + " "

        flush_text()

    # ── BETTER TABLE EXTRACTION ───────────────────────
    # Re-process to properly capture markdown tables
    chunks = []  # reset and do it properly

    for page_idx, doc in enumerate(documents):
        page_no = page_idx + 1
        content = doc.text.strip()
        if not content:
            continue

        current_section = "Unknown"
        current_text    = ""
        lines           = content.split("\n")
        i               = 0

        def flush_text():
            nonlocal current_text
            if current_text.strip():
                chunks.append({
                    "type"    : "text",
                    "content" : current_text.strip(),
                    "metadata": {
                        "source"    : filename,
                        "page_start": page_no,
                        "page_end"  : page_no,
                        "section"   : current_section,
                    }
                })
            current_text = ""

        while i < len(lines):
            line     = lines[i]
            stripped = line.strip()

            # ── SECTION HEADER
            if stripped.startswith("#"):
                flush_text()
                current_section = stripped.lstrip("#").strip()
                i += 1

            # ── TABLE (starts with | and next line has ---)
            elif stripped.startswith("|") and i + 1 < len(lines) and "---" in lines[i + 1]:
                flush_text()
                table_lines = []
                while i < len(lines) and lines[i].strip().startswith("|"):
                    table_lines.append(lines[i].strip())
                    i += 1
                tbl_counter += 1
                chunks.append({
                    "type"    : "table",
                    "content" : "\n".join(table_lines),
                    "metadata": {
                        "source"     : filename,
                        "page_start" : page_no,
                        "page_end"   : page_no,
                        "section"    : current_section,
                        "table_index": tbl_counter,
                    }
                })
                if DEBUG:
                    print(f"[DIGITAL PDF] Table {tbl_counter} (page {page_no})")

            # ── NORMAL TEXT
            else:
                if stripped:
                    current_text += stripped + " "
                i += 1

        flush_text()

    # ── SAVE AS JSON ──────────────────────────────────
    output_path = Path(file_path).stem + "_parsed.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"filename": filename, "chunks": chunks}, f,
                  ensure_ascii=False, indent=2)

    if DEBUG:
        text_count  = len([c for c in chunks if c["type"] == "text"])
        table_count = len([c for c in chunks if c["type"] == "table"])
        print(f"\n[DIGITAL PDF] Text chunks : {text_count}")
        print(f"[DIGITAL PDF] Tables      : {table_count}")
        print(f"[DIGITAL PDF] Saved to    : {output_path}")

    return {"filename": filename, "chunks": chunks}


# ── TEST ──────────────────────────────────────────────
if __name__ == "__main__":
    result = extract_digital_pdf("tree.pdf")
    print("\n===== CHUNKS PREVIEW =====")
    for i, chunk in enumerate(result["chunks"][:5]):
        print(f"\nChunk {i+1}:")
        print(f"  Type    : {chunk['type']}")
        print(f"  Page    : {chunk['metadata']['page_start']}")
        print(f"  Section : {chunk['metadata']['section']}")
        print(f"  Content : {chunk['content'][:100]}...")
    print(f"\nTotal chunks: {len(result['chunks'])}")