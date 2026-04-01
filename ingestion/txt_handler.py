import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import chardet
from pathlib import Path
from config import DEBUG


def extract_txt(file_path: str) -> dict:

    if DEBUG:
        print(f"\n[TXT HANDLER] Reading file: {file_path}")

    # ── READ FILE ─────────────────────────────────────
    with open(file_path, "rb") as f:
        raw = f.read()

    encoding = chardet.detect(raw[:50_000]).get("encoding") or "utf-8"
    text     = raw.decode(encoding, errors="replace")

    if DEBUG:
        print(f"[TXT HANDLER] Encoding : {encoding}")
        print(f"[TXT HANDLER] Chars    : {len(text)}")

    # ── METADATA ──────────────────────────────────────
    metadata = {
        "file_name" : Path(file_path).name,
        "encoding"  : encoding,
        "word_count": len(text.split()),
        "line_count": len(text.splitlines()),
    }

    result = {
        "text"    : text,
        "metadata": metadata,
    }

    # ── SAVE JSON ─────────────────────────────────────
    output_path = Path(file_path).stem + "_parsed.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    if DEBUG:
        print(f"[TXT HANDLER] Saved to : {output_path}")

    return result


# ── TEST ──────────────────────────────────────────────
if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "../test2.txt"
    result = extract_txt(path)

    print(f"\n===== METADATA =====")
    for k, v in result["metadata"].items():
        print(f"  {k}: {v}")

    print(f"\n===== TEXT PREVIEW =====")
    print(result["text"][:500])