import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
from config import DEBUG


MIN_TEXT_LENGTH  = 20
MIN_IMAGE_LENGTH = 20


#  MAIN ENTRY POINT
def clean(handler_output: dict) -> dict:
    """
    Walk any handler output structure and clean every content field by its type.
    Works on ANY JSON structure — no structure detection needed.
    Just looks for 'type' + 'content' pairs anywhere in the JSON.
    """
    return _walk(handler_output)


def _walk(obj):
    """
    Recursively walk dict/list and clean any content field found.
    Rules:
      - dict with "type" == "table"             → clean headers/rows strings
      - dict with "type" + "content"            → clean content by type
      - dict with "content" (list or dict)      → walk into it
      - dict with "content" (string) only       → clean as plain text
      - dict with "text"                        → clean as plain text
    Metadata is never touched.
    """
    if isinstance(obj, list):
        result = []
        for item in obj:
            walked = _walk(item)
            # drop empty text/image/email chunks
            if isinstance(walked, dict) and "type" in walked:
                if walked.get("type") in ("text", "image", "email") \
                        and walked.get("content", "x") == "":
                    if DEBUG:
                        print(f"[CLEANER] Dropped empty chunk after cleaning "
                              f"(type={walked.get('type')})")
                    continue
                # drop empty tables
                if walked.get("type") == "table" \
                        and not walked.get("rows") and not walked.get("headers"):
                    if DEBUG:
                        print(f"[CLEANER] Dropped empty table chunk")
                    continue
            result.append(walked)
        return result

    if isinstance(obj, dict):
        # ── TABLE CHUNK: clean headers + each row cell ─
        if obj.get("type") == "table":
            obj["headers"] = [clean_text(str(h)) for h in obj.get("headers", [])]
            obj["rows"]    = [
                [clean_text(str(cell)) for cell in row]
                for row in obj.get("rows", [])
            ]
            return obj

        # ── has "type" + "content" → clean by type ────
        if "type" in obj and "content" in obj:
            chunk_type = obj["type"]
            content    = obj["content"]
            cleaned    = _clean_by_type(chunk_type, content)

            if _should_drop(chunk_type, cleaned):
                obj["content"] = ""
            else:
                obj["content"] = cleaned

        # ── has "content" (no type): walk if container, clean if string ─
        elif "content" in obj and "type" not in obj:
            content = obj["content"]
            if isinstance(content, (list, dict)):
                obj["content"] = _walk(content)
            elif isinstance(content, str):
                obj["content"] = clean_text(content)

        # ── has "text" key → clean as text ────────────
        if "text" in obj and isinstance(obj.get("text"), str):
            obj["text"] = clean_text(obj["text"])

        # ── recurse into all values except protected keys ──
        for key in list(obj.keys()):
            if key not in ("content", "text", "metadata", "type",
                           "headers", "rows"):
                obj[key] = _walk(obj[key])

    return obj


#  ROUTE CLEANING BY TYPE
def _clean_by_type(chunk_type: str, content) -> str:
    # defensive: non-string content (table with legacy format, etc.)
    if not isinstance(content, str):
        return ""

    if chunk_type == "text":
        return clean_text(content)
    elif chunk_type == "table":
        return clean_table(content)
    elif chunk_type == "image":
        return clean_image_description(content)
    elif chunk_type == "email":
        return clean_email(content)
    else:
        return clean_text(content)  # fallback


def _should_drop(chunk_type: str, content: str) -> bool:
    if not content.strip():
        if DEBUG:
            print(f"[CLEANER] Dropped empty chunk (type={chunk_type})")
        return True
    min_len = MIN_IMAGE_LENGTH if chunk_type == "image" else MIN_TEXT_LENGTH
    if len(content.strip()) < min_len:
        if DEBUG:
            print(f"[CLEANER]  Dropped short chunk (type={chunk_type}, len={len(content.strip())})")
        return True
    return False


#  CLEAN TEXT
def clean_text(text: str) -> str:
    if not text:
        return ""

    # remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # normalize unicode whitespace
    text = text.replace("\xa0", " ")
    text = text.replace("\u200b", "")
    text = text.replace("\ufffd", "")
    # remove horizontal lines (--- or more)
    text = re.sub(r"-{3,}", "", text)
    # remove control characters (keep \n and \t)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # collapse multiple spaces
    text = re.sub(r" {2,}", " ", text)

    # collapse more than 2 consecutive newlines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # strip leading/trailing whitespace per line
    lines = [line.strip() for line in text.splitlines()]
    text  = "\n".join(lines)

    return text.strip()


#  CLEAN TABLE (string form — kept for backward compat)
def clean_table(text: str) -> str:
    if not text:
        return ""

    lines = text.splitlines()
    cleaned_lines = []

    for line in lines:
        if not line.strip():
            continue
        # keep markdown separator lines
        if re.match(r"^[\|\s\-:]+$", line):
            cleaned_lines.append(line)
            continue
        line = re.sub(r" {2,}", " ", line)
        line = line.strip()
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()


#  CLEAN IMAGE DESCRIPTION
def clean_image_description(text: str) -> str:
    if not text:
        return ""

    # drop useless descriptions
    useless_phrases = [
        "the image contains no charts",
        "the image contains no figures",
        "the image contains no diagrams",
        "no charts, figures, diagrams, or images",
    ]
    for phrase in useless_phrases:
        if phrase.lower() in text.lower():
            if DEBUG:
                print(f"[CLEANER]  Dropped useless image description")
            return ""

    # remove redundant AI opening phrases
    redundant_phrases = [
        r"^the image (shows?|presents?|depicts?|displays?|contains?|illustrates?)[,.]?\s*",
        r"^this image (shows?|presents?|depicts?|displays?|contains?|illustrates?)[,.]?\s*",
        r"^the (picture|photo|photograph|diagram|chart|figure|screenshot) (shows?|presents?|depicts?)[,.]?\s*",
        r"^in this image[,.]?\s*",
        r"^as (seen|shown) in the image[,.]?\s*",
    ]
    for pattern in redundant_phrases:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    # remove markdown bold/italic (* and **)
    text = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", text)
    text = re.sub(r"\*+", "", text)
    text = text.replace("\\", "")
    text = text.replace('\"', '"')
    text = text.replace('"', '')
    # remove markdown bullet lines (lines starting with * or -)
    text = re.sub(r"^\s*[\*\-]\s+", "", text, flags=re.MULTILINE)

    # remove lines that are only whitespace or symbols
    lines = [line for line in text.splitlines() if line.strip()]
    text  = "\n".join(lines)

    return clean_text(text)


#  CLEAN EMAIL
def clean_email(text: str) -> str:
    if not text:
        return ""

    # remove tracking URLs (long urls with tokens)
    text = re.sub(r"https?://\S{60,}", "", text)

    # remove inline image references like [image: Mailsuite]
    text = re.sub(r"\[image:[^\]]*\]", "", text)

    # remove email tracking footers
    tracking_patterns = [
        r"email tracked with \w+.*",
        r"\·\s*opt out.*",
        r"opt out\s*·?.*",
        r"unsubscribe.*",
        r"this email was sent to.*",
        r"you (are|were) receiving this (email|message) because.*",
        r"©\s*\d{4}.*",
        r"all rights reserved.*",
        r"\d{2}/\d{2}/\d{2,4}\s+\d{2}:\d{2}:\d{2}",
    ]
    for pattern in tracking_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL)

    # remove forwarded/original message headers
    text = re.sub(
        r"-{3,}\s*(forwarded message|original message)\s*-{3,}[^\n]*",
        "", text, flags=re.IGNORECASE
    )

    # remove quoted reply lines (lines starting with >)
    lines = [line for line in text.splitlines() if not line.strip().startswith(">")]
    text  = "\n".join(lines)

    return clean_text(text)


#  TEST — supports any file type
if __name__ == "__main__":
    import json

    if len(sys.argv) < 2:
        print("Usage: python processing/cleaner.py <file_path>")
        print("Example: python processing/cleaner.py test.pdf")
        sys.exit(1)

    file_path = sys.argv[1]
    ext       = os.path.splitext(file_path)[1].lower()

    print(f"[TEST] File  : {file_path}")
    print(f"[TEST] Exists: {os.path.exists(file_path)}")
    print(f"[TEST] Type  : {ext}")

    if ext == ".eml":
        from File_processing.eml_handler import extract_eml
        result = extract_eml(file_path)

    elif ext == ".pdf":
        import fitz
        doc        = fitz.open(file_path)
        is_digital = any(page.get_text().strip() for page in doc)
        doc.close()
        if is_digital:
            from File_processing.digital_pdf import extract_digital_pdf
            result = extract_digital_pdf(file_path)
        else:
            from File_processing.scanned_pdf import extract_scanned_pdf
            result = extract_scanned_pdf(file_path)

    elif ext == ".docx":
        from File_processing.docx_handler import extract_docx
        result = extract_docx(file_path)

    elif ext == ".txt":
        from File_processing.txt_handler import extract_txt
        result = extract_txt(file_path)

    elif ext == ".csv":
        from File_processing.csv_handler import extract_csv
        result = extract_csv(file_path)

    elif ext == ".pptx":
        from File_processing.pptx_handler import extract_pptx
        result = extract_pptx(file_path)

    elif ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]:
        from File_processing.image_handler import extract_image
        result = extract_image(file_path)

    else:
        print(f"[TEST] Unsupported extension: {ext}")
        sys.exit(1)

    cleaned = clean(result)

    output_path = os.path.splitext(file_path)[0] + "_cleaned.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, indent=2, ensure_ascii=False)
    print(f"\n Saved to: {output_path}")

    print(f"\n Cleaned output keys: {list(cleaned.keys())}")
    print("\n── Preview (first 1500 chars) ──")
    print(json.dumps(cleaned, indent=2, ensure_ascii=False)[:1500])