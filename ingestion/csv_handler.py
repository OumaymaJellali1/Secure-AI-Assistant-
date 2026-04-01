import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import chardet
import pandas as pd
from pathlib import Path
from config import DEBUG


def extract_csv(file_path: str) -> dict:

    if DEBUG:
        print(f"\n[CSV HANDLER] Reading file: {file_path}")

    # ── DETECT ENCODING ───────────────────────────────
    with open(file_path, "rb") as f:
        encoding = chardet.detect(f.read(50_000)).get("encoding") or "utf-8"

    # ── READ CSV ──────────────────────────────────────
    df = pd.read_csv(file_path, encoding=encoding)

    if DEBUG:
        print(f"[CSV HANDLER] Rows    : {df.shape[0]}")
        print(f"[CSV HANDLER] Columns : {df.shape[1]}")
        print(f"[CSV HANDLER] Cols    : {list(df.columns)}")

    filename = Path(file_path).name

    result = {
        "content" : df.to_markdown(index=False),
        "metadata": {
            "filename"    : filename,
            "row_count"   : df.shape[0],
            "column_count": df.shape[1],
            "columns"     : df.columns.tolist(),
        }
    }

    # ── SAVE JSON ─────────────────────────────────────
    output_path = Path(file_path).stem + "_parsed.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    if DEBUG:
        print(f"[CSV HANDLER] Saved to: {output_path}")

    return result


# ── TEST ──────────────────────────────────────────────
if __name__ == "__main__":
    result = extract_csv("../test.csv")

    print("\n===== METADATA =====")
    print(f"  Filename : {result['metadata']['filename']}")
    print(f"  Rows     : {result['metadata']['row_count']}")
    print(f"  Columns  : {result['metadata']['column_count']}")
    print(f"  Col names: {result['metadata']['columns']}")

    print("\n===== CONTENT =====")
    print(result['content'][:200], "...")

    print(f"\n Saved to: {Path('../test.csv').stem}_parsed.json")