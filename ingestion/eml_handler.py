import sys
import os
import gc
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from email import policy
from email.parser import BytesParser
from pathlib import Path
from config import DEBUG, TEMP_DIR


# ── IMPORT HANDLERS ───────────────────────────────────
from ingestion.digital_pdf        import extract_digital_pdf
from ingestion.image_handler       import extract_image
from ingestion.scanned_pdf import extract_scanned_pdf
from ingestion.csv_handler         import extract_csv
from ingestion.txt_handler         import extract_txt
from ingestion.pptx_handler        import extract_pptx

# ── EXTENSION → HANDLER MAP ───────────────────────────
HANDLER_MAP = {
    ".pdf" : extract_digital_pdf,
    ".jpg" : extract_image,
    ".jpeg": extract_image,
    ".png" : extract_image,
    ".bmp" : extract_image,
    ".tiff": extract_image,
    ".tif" : extract_image,
    ".csv" : extract_csv,
    ".txt" : extract_txt,
    ".pptx": extract_pptx,
}

# ── MULTIPART CONTAINER TYPES (never attachments) ─────
_CONTAINER_TYPES = {
    "text/plain",
    "text/html",
    "multipart/alternative",
    "multipart/mixed",
    "multipart/related",
    "multipart/signed",
    "message/rfc822",
}


def _is_attachment(part) -> bool:
    """
    Return True if this MIME part should be treated as an attachment.

    Catches all three common cases:
      1. Content-Disposition: attachment  (classic)
      2. Content-Disposition: inline  with a filename  (inline attachment)
      3. No Content-Disposition at all  but has a filename and is not a
         known body/container content-type  (many mail clients omit the header)
    """
    disposition  = str(part.get("Content-Disposition", "")).lower()
    content_type = part.get_content_type()
    filename     = part.get_filename()

    if "attachment" in disposition:
        return True

    if filename and content_type not in _CONTAINER_TYPES:
        return True

    return False


def extract_eml(file_path: str) -> dict:

    if DEBUG:
        print(f"\n[EML HANDLER] Reading file: {file_path}")

    # ── READ & PARSE ──────────────────────────────────
    with open(file_path, "rb") as f:
        msg = BytesParser(policy=policy.default).parse(f)

    filename = Path(file_path).name

    # ── METADATA ──────────────────────────────────────
    metadata = {
        "source"          : filename,
        "subject"         : str(msg.get("subject", "")),
        "from"            : str(msg.get("from", "")),
        "to"              : str(msg.get("to", "")),
        "date"            : str(msg.get("date", "")),
        "has_attachments" : False,
        "attachments"     : [],
    }

    body_content        = ""
    found_plain         = False
    attachments_results = []
    attachment_counter  = 0

    # ── WALK PARTS ────────────────────────────────────
    for part in msg.walk():
        content_type = part.get_content_type()
        disposition  = str(part.get("Content-Disposition", "")).lower()

        # ── PLAIN TEXT BODY ───────────────────────────
        if content_type == "text/plain" and not _is_attachment(part):
            try:
                body = part.get_content()
                if body and body.strip():
                    body_content = body.strip()
                    found_plain  = True
                    if DEBUG:
                        print(f"[EML HANDLER] Body extracted ✅")
            except Exception as e:
                if DEBUG:
                    print(f"[EML HANDLER] Body error: {e}")

        # ── HTML FALLBACK ─────────────────────────────
        elif content_type == "text/html" and not _is_attachment(part) and not found_plain:
            try:
                from html.parser import HTMLParser

                class _StripHTML(HTMLParser):
                    def __init__(self):
                        super().__init__()
                        self.text = []
                    def handle_data(self, data):
                        self.text.append(data)
                    def get_text(self):
                        return " ".join(self.text).strip()

                html   = part.get_content()
                parser = _StripHTML()
                parser.feed(html)
                plain  = parser.get_text()

                if plain:
                    body_content = plain
                    if DEBUG:
                        print(f"[EML HANDLER] HTML body extracted ✅")
            except Exception as e:
                if DEBUG:
                    print(f"[EML HANDLER] HTML error: {e}")

        # ── ATTACHMENTS → ROUTE TO HANDLERS ──────────
        elif _is_attachment(part):
            attach_name = part.get_filename() or "unnamed"
            ext         = Path(attach_name).suffix.lower()
            handler     = HANDLER_MAP.get(ext)

            metadata["has_attachments"] = True
            metadata["attachments"].append({
                "name"        : attach_name,
                "content_type": content_type,
            })

            if DEBUG:
                print(f"\n[EML HANDLER] Attachment : {attach_name}")
                print(f"[EML HANDLER] Extension  : {ext}")
                print(f"[EML HANDLER] Handler    : "
                      f"{handler.__name__ if handler else 'none'}")

            if handler:
                attachment_counter += 1

                # ── SAFE TEMP FILENAME ────────────────
                # avoid pdfium/Windows issues with
                # special chars, spaces, parentheses
                safe_name = f"attachment_{attachment_counter}{ext}"
                temp_path = TEMP_DIR / safe_name

                try:
                    # ── SAVE TO TEMP FILE ─────────────
                    payload = part.get_payload(decode=True)
                    with open(temp_path, "wb") as tmp:
                        tmp.write(payload)

                    if DEBUG:
                        print(f"[EML HANDLER] Temp file  : {temp_path}")

                    # ── CALL HANDLER ──────────────────
                    result = handler(str(temp_path))
                    attachments_results.append({
                        "name"  : attach_name,
                        "result": result,
                    })

                    if DEBUG:
                        print(f"[EML HANDLER] {attach_name} ✅")

                except Exception as e:
                    if DEBUG:
                        print(f"[EML HANDLER] Error processing "
                              f"{attach_name}: {e}")

                finally:
                    # ── CLEANUP ───────────────────────
                    # gc.collect + sleep needed because
                    # pdfium holds the file handle open
                    # briefly after handler returns
                    gc.collect()
                    time.sleep(0.5)
                    try:
                        if temp_path.exists():
                            os.remove(temp_path)
                            if DEBUG:
                                print(f"[EML HANDLER] Temp deleted ✅")
                    except PermissionError:
                        if DEBUG:
                            print(f"[EML HANDLER] Could not delete "
                                  f"{temp_path} — still locked")
            else:
                if DEBUG:
                    print(f"[EML HANDLER] No handler for {ext} → skipped")

    result = {
        "content"    : body_content,
        "metadata"   : metadata,
        "attachments": attachments_results,
    }

    # ── SAVE JSON ─────────────────────────────────────
    output_path = Path(file_path).stem + "_parsed.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    if DEBUG:
        print(f"\n[EML HANDLER] Attachments processed : "
              f"{len(attachments_results)}")
        print(f"[EML HANDLER] Saved to              : {output_path}")

    return result


# ── TEST ──────────────────────────────────────────────
if __name__ == "__main__":
    result = extract_eml("../test.eml")

    print("\n===== METADATA =====")
    for k, v in result["metadata"].items():
        print(f"  {k}: {v}")

    print("\n===== CONTENT =====")
    print(result["content"][:300], "...")

    print("\n===== ATTACHMENTS =====")
    for att in result["attachments"]:
        print(f"\n  📎 {att['name']}")
        print(f"     Keys: {list(att['result'].keys())}")

    print(f"\n📄 Saved to: {Path('../test.eml').stem}_parsed.json")