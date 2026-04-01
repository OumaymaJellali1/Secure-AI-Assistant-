import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
from config import DEBUG


def clean_parsed_result(result: dict) -> dict:
    """
    Clean parsed document content.
    Works with the chunks-based format from all extractors.
    """

    if DEBUG:
        print(f"\n[CLEANER] Cleaning parsed content...")

    chunks = result.get("chunks", [])

    # ── CLEAN EACH CHUNK IN PLACE ─────────────────────
    cleaned_chunks = []
    for chunk in chunks:
        cleaned = dict(chunk)  # shallow copy
        if chunk["type"] == "text":
            cleaned["content"] = _clean_text(chunk["content"])
        elif chunk["type"] == "image":
            cleaned["content"] = _clean_description(chunk["content"])
        elif chunk["type"] == "table":
            cleaned["content"] = chunk["content"]  # tables stay as markdown
        cleaned_chunks.append(cleaned)

    # ── STATS FOR DEBUG ───────────────────────────────
    text_chunks  = [c for c in cleaned_chunks if c["type"] == "text"]
    table_chunks = [c for c in cleaned_chunks if c["type"] == "table"]
    image_chunks = [c for c in cleaned_chunks if c["type"] == "image"]

    text_chars = sum(len(c["content"]) for c in text_chunks)

    if DEBUG:
        print(f"[CLEANER] Text chunks : {len(text_chunks)} ({text_chars} chars)")
        print(f"[CLEANER] Tables      : {len(table_chunks)}")
        print(f"[CLEANER] Images      : {len(image_chunks)}")

    return {
        "filename": result.get("filename", ""),
        "chunks"  : cleaned_chunks
    }


# ── CLEAN TEXT ────────────────────────────────────────
def _clean_text(text: str) -> str:
    """
     Clean text while keeping:
     tables in their place (markdown)
     image descriptions in their place
     natural punctuation
    Remove:
     markdown headers
     **label:** markers
     bullet points
     page separators
     extra whitespace
    """

    # ── 1. REMOVE MARKDOWN HEADERS ────────────────────
    text = re.sub(r'#{1,6}\s+.*', '', text)

    # ── 2. CLEAN LABELS (keep content after them) ─────
    text = re.sub(r'\*\*Table\s+\d+:\*\*',        'Table:', text)
    text = re.sub(r'\*\*Image\s+\d+.*?:\*\*',     'Image:', text)
    text = re.sub(r'\*\*Drawing.*?:\*\*',          'Drawing:', text)
    text = re.sub(r'\*\*Visual Description.*?\*\*','Visual content:', text)
    text = re.sub(r'\*\*Calendar Description:\*\*','Calendar:', text)
    text = re.sub(r'\*\*Footer:\*\*',              'Footer:', text)
    text = re.sub(r'\*\*.*?:\*\*',                '', text)

    # ── 3. REMOVE BULLET POINTS ───────────────────────
    text = re.sub(
        r'^\s*[\*\-•]\s+', '',
        text, flags=re.MULTILINE
    )

    # ── 4. REMOVE REMAINING ** MARKERS ───────────────
    text = re.sub(r'\*{1,3}(.+?)\*{1,3}', r'\1', text)
    text = re.sub(r'\*+', '', text)

    # ── 5. REMOVE PAGE MARKERS ────────────────────────
    text = re.sub(r'^---+$',            '', text, flags=re.MULTILINE)
    text = re.sub(r'^##\s+Page\s+\d+',  '', text, flags=re.MULTILINE)
    text = re.sub(r'^Page\s+\d+$',      '', text, flags=re.MULTILINE)

    # ── 6. REMOVE SPECIAL CHARS ───────────────────────
    text = re.sub(r'["\u201c\u201d\u2018\u2019]', '', text)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    text = re.sub(r'\u00a0', ' ', text)
    text = re.sub(r'_{2,}', '', text)
    text = re.sub(r'`+', '', text)

    # ── 7. NORMALIZE WHITESPACE ───────────────────────
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r'^\s+$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # ── 8. FINAL STRIP ────────────────────────────────
    text = text.strip()

    return text


# ── CLEAN IMAGE DESCRIPTION ───────────────────────────
def _clean_description(description: str) -> str:
    """
    Clean Groq image description:
    - remove bullet points
    - remove markdown bold/italic
    - convert to plain paragraph
    """

    # remove bullet points
    description = re.sub(
        r'^\s*[\*\-•]\s+', '',
        description, flags=re.MULTILINE
    )

    # remove nested bullets
    description = re.sub(
        r'^\s+[\*\-•]\s+', '',
        description, flags=re.MULTILINE
    )

    # remove markdown bold/italic
    description = re.sub(
        r'\*{1,3}(.+?)\*{1,3}', r'\1',
        description
    )
    description = re.sub(r'\*+', '', description)

    # remove special chars
    description = re.sub(
        r'["\u201c\u201d\u2018\u2019]', '',
        description
    )
    description = re.sub(r'\u00a0', ' ', description)

    # normalize whitespace
    description = re.sub(r'[ \t]{2,}', ' ', description)
    description = re.sub(
        r'^\s+$', '',
        description, flags=re.MULTILINE
    )
    description = re.sub(r'\n{3,}', '\n\n', description)

    # final strip
    description = description.strip()

    return description