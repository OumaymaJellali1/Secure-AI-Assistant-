import sys
import os
import gc
import time
import re

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from email import policy
from email.parser import BytesParser
from pathlib import Path
from config import DEBUG, TEMP_DIR
import hashlib


# ── IMPORT HANDLERS ───────────────────────────────────
from File_processing.digital_pdf        import extract_digital_pdf
from File_processing.image_handler       import extract_image
from File_processing.scanned_pdf        import extract_scanned_pdf
from File_processing.txt_handler         import extract_txt
from chunking.pptx_chunker import extract_pptx
from chunking.scanned_pdf_chunker import chunk_scanned_pdf
from csv_processing.csv_executor_postgres import import_csv

# ── CLEANUP SIDE-EFFECT JSON ──────────────────────────
def _cleanup_json(file_path: str):
    json_path = Path(Path(file_path).stem + "_parsed.json")
    try:
        if json_path.exists():
            os.remove(json_path)
            if DEBUG:
                print(f"[EML HANDLER] Deleted side-effect JSON: {json_path}")
    except Exception as e:
        if DEBUG:
            print(f"[EML HANDLER] Could not delete {json_path}: {e}")


# ── SMART PDF ROUTER ──────────────────────────────────
def _smart_pdf_handler(file_path: str) -> dict:
    result = extract_digital_pdf(file_path)
    _cleanup_json(file_path)

    has_text = any(
        c.get("type") == "text" and c.get("content", "").strip()
        for c in result.get("chunks", [])
    )

    if not has_text:
        if DEBUG:
            print(f"[EML HANDLER] No text chunks → trying OCR + chunker")
        scanned = extract_scanned_pdf(file_path)
        _cleanup_json(file_path)
        result = chunk_scanned_pdf(scanned)
    else:
        result = {"chunks": result.get("chunks", result) if isinstance(result, dict) else result}

    return result

# ── EXTENSION → HANDLER MAP ───────────────────────────
HANDLER_MAP = {
    ".pdf" : _smart_pdf_handler,
    ".jpg" : extract_image,
    ".jpeg": extract_image,
    ".png" : extract_image,
    ".bmp" : extract_image,
    ".tiff": extract_image,
    ".tif" : extract_image,
    ".txt" : extract_txt,
    ".pptx": extract_pptx,
}

_CONTAINER_TYPES = {
    "text/plain", "text/html",
    "multipart/alternative", "multipart/mixed",
    "multipart/related", "multipart/signed",
    "message/rfc822",
}


def _is_attachment(part) -> bool:
    disposition  = str(part.get("Content-Disposition", "")).lower()
    content_type = part.get_content_type()
    filename     = part.get_filename()

    if "attachment" in disposition:
        return True
    if filename and content_type not in _CONTAINER_TYPES:
        return True
    return False


# ── INLINE TABLE HELPERS ──────────────────────────────
def _table_to_dict(table_tag) -> dict:
    headers = []
    rows    = []

    thead = table_tag.find("thead")
    if thead:
        headers = [th.get_text(strip=True) for th in thead.find_all(["th", "td"])]

    for tr in table_tag.find_all("tr"):
        cells_tags = tr.find_all(["td", "th"])
        cells      = [c.get_text(strip=True) for c in cells_tags]
        if not cells:
            continue

        if not headers and all(c.name == "th" for c in cells_tags):
            headers = cells
        else:
            if thead and tr.find_parent("thead"):
                continue
            rows.append(cells)

    # Fallback: if no headers were detected but we have rows,
    # promote the first row to headers (common for pasted markdown tables)
    if not headers and rows:
        headers = rows.pop(0)

    return {"headers": headers, "rows": rows}


def _is_data_table(table_tag) -> bool:
    rows = table_tag.find_all("tr")
    if len(rows) < 2:
        return False
    max_cols = max(len(tr.find_all(["td", "th"])) for tr in rows)
    return max_cols >= 2


# ── FORWARDED HEADER STRIPPER ─────────────────────────
# NOTE: for FORWARDED emails we strip only the metadata header
#       (From/Date/Subject/To block), keeping the actual content.
#       For REPLIES we cut everything after the quote marker.

_FORWARD_HEADER_MARKERS = [
    r"-+\s*Forwarded message\s*-+",       # Gmail
    r"Begin forwarded message:",          # Apple Mail
    r"^From:\s.+\r?\nSent:\s",            # Outlook (English)
    r"^From:\s.+\r?\nDate:\s",            # Outlook Web (English)
    r"^De:\s.+\r?\nEnvoy\u00e9:\s",       # Outlook (French)
]

_REPLY_MARKERS = [
    r"-+\s*Original Message\s*-+",       # Outlook reply (English)
    r"-+\s*Message d'origine\s*-+",      # Outlook reply (French)
    r"On\s.+\swrote:",                    # Gmail/Apple Mail reply (English)
    r"Le\s.+\sa\s\u00e9crit\s*:",        # Gmail/Apple Mail reply (French)
]


def _strip_forward_header(text: str) -> str:
    """
    Remove the 'Forwarded message' metadata block but keep the forwarded content.
    The block looks like:
        ---------- Forwarded message ---------
        From: ...
        Date: ...
        Subject: ...
        To: ...
    """
    if not text:
        return text

    for pattern in _FORWARD_HEADER_MARKERS:
        m = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if not m:
            continue

        before = text[:m.start()].rstrip()
        after  = text[m.end():]

        # Skip leading blank lines, then skip header-style lines (Key: value)
        lines = after.split("\n")
        i = 0
        while i < len(lines) and not lines[i].strip():
            i += 1
        header_keys = re.compile(
            r"^\s*(From|To|Cc|Bcc|Date|Sent|Subject|"   # English
            r"De|\u00c0|A|Objet|Envoy\u00e9)\s*:",       # French
            flags=re.IGNORECASE,
        )
        while i < len(lines) and header_keys.match(lines[i]):
            i += 1

        remaining = "\n".join(lines[i:]).strip()
        text = (before + "\n" + remaining).strip() if before else remaining
        break

    return text


def _strip_replies(text: str) -> str:
    """Cut everything from the first reply marker onward."""
    if not text:
        return text
    earliest = len(text)
    for pattern in _REPLY_MARKERS:
        m = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if m and m.start() < earliest:
            earliest = m.start()
    return text[:earliest].strip()


def _clean_body_text(text: str) -> str:
    """Strip forwarded-header metadata AND reply quotes."""
    text = _strip_forward_header(text)
    text = _strip_replies(text)
    return text


def extract_eml(file_path: str) -> dict:

    if DEBUG:
        print(f"\n[EML HANDLER] Reading file: {file_path}")

    with open(file_path, "rb") as f:
        msg = BytesParser(policy=policy.default).parse(f)

    filename = Path(file_path).name

    metadata = {
        "source"          : filename,
        "subject"         : str(msg.get("subject", "")),
        "from"            : str(msg.get("from", "")),
        "to"              : str(msg.get("to", "")),
        "date"            : str(msg.get("date", "")),
        "has_attachments" : False,
        "attachments"     : [],
    }

    body_chunks         = []
    plain_text_body     = ""   # buffered text/plain
    html_body           = ""   # buffered text/html
    attachments_results = []
    attachment_counter  = 0

    for part in msg.walk():
        content_type = part.get_content_type()
        disposition  = str(part.get("Content-Disposition", "")).lower()

        # ── BUFFER PLAIN TEXT BODY ────────────────────
        if content_type == "text/plain" and not _is_attachment(part):
            try:
                body = part.get_content()
                if body and body.strip() and not plain_text_body:
                    plain_text_body = body.strip()
            except Exception as e:
                if DEBUG:
                    print(f"[EML HANDLER] Body error: {e}")

        # ── BUFFER HTML BODY ──────────────────────────
        elif content_type == "text/html" and not _is_attachment(part):
            try:
                body = part.get_content()
                if body and body.strip() and not html_body:
                    html_body = body
            except Exception as e:
                if DEBUG:
                    print(f"[EML HANDLER] HTML read error: {e}")

        # ── ATTACHMENTS ───────────────────────────────
        elif _is_attachment(part):
            attach_name = part.get_filename() or "unnamed"
            ext         = Path(attach_name).suffix.lower()
            handler     = HANDLER_MAP.get(ext)

            metadata["has_attachments"] = True
            metadata["attachments"].append({
                "name"        : attach_name,
                "content_type": content_type,
            })

            if ext == ".csv":
                attachment_counter += 1
                safe_name = f"attachment_{attachment_counter}{ext}"
                temp_path = TEMP_DIR / safe_name
                try:
                    payload = part.get_payload(decode=True)
                    with open(temp_path, "wb") as tmp:
                        tmp.write(payload)
                    real_table = re.sub(r"[^a-z0-9]+", "", Path(attach_name).stem.lower()).strip("")
                    import_csv(str(temp_path), table_name=real_table)
                    if DEBUG:
                        print(f"[EML HANDLER] {attach_name} → imported to Postgres")
                except Exception as e:
                    if DEBUG:
                        print(f"[EML HANDLER] CSV import error for {attach_name}: {e}")
                finally:
                    try:
                        if temp_path.exists():
                            os.remove(temp_path)
                    except Exception:
                        pass
                continue

            if handler:
                attachment_counter += 1
                safe_name = f"attachment_{attachment_counter}{ext}"
                temp_path = TEMP_DIR / safe_name

                try:
                    payload = part.get_payload(decode=True)
                    with open(temp_path, "wb") as tmp:
                        tmp.write(payload)

                    if DEBUG:
                        print(f"[EML HANDLER] Temp file  : {temp_path}")

                    result = handler(str(temp_path))
                    _cleanup_json(str(temp_path))
                    attachments_results.append({
                        "name"  : attach_name,
                        "result": result,
                    })

                    if DEBUG:
                        print(f"[EML HANDLER] {attach_name} processed")

                except Exception as e:
                    if DEBUG:
                        print(f"[EML HANDLER] Error processing {attach_name}: {e}")

                finally:
                    gc.collect()
                    time.sleep(0.5)
                    try:
                        if temp_path.exists():
                            os.remove(temp_path)
                            if DEBUG:
                                print(f"[EML HANDLER] Temp deleted ")
                    except PermissionError:
                        if DEBUG:
                            print(f"[EML HANDLER] Could not delete {temp_path} — still locked")
            else:
                if DEBUG:
                    print(f"[EML HANDLER] No handler for {ext} → skipped")

    # ── BUILD BODY CHUNKS: PREFER HTML IF PRESENT ─────
    # (avoids duplication between text/plain and text/html parts)
    if html_body:
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(html_body, "html.parser")

            # Strip only forward metadata header (gmail_attr), keep gmail_quote content
            for attr in soup.find_all("div", class_=re.compile(r"gmail_attr")):
                attr.decompose()
            # Thunderbird reply citation header
            for cite in soup.find_all(class_=re.compile(r"moz-cite-prefix")):
                cite.decompose()
            # Outlook reply/forward separator + classic Outlook reply wrappers
            for ol in soup.find_all(id=re.compile(r"divRplyFwdMsg|OLK_SRC_BODY_SECTION", re.I)):
                ol.decompose()
            for ol in soup.find_all("div", class_=re.compile(r"OutlookMessageHeader", re.I)):
                ol.decompose()
            # <blockquote> = quoted reply → remove
            for bq in soup.find_all("blockquote"):
                bq.decompose()
            # Tracker pixels
            for img in soup.find_all("img"):
                w = img.get("width", "")
                h = img.get("height", "")
                if str(w).strip() in ("0", "1") and str(h).strip() in ("0", "1"):
                    img.decompose()
            # Mailsuite / Mailtrack tracker signatures
            for sig in soup.find_all(id=re.compile(r"mt-signature", re.I)):
                sig.decompose()

            # Extract tables
            html_tables = []
            for table_tag in soup.find_all("table"):
                if not _is_data_table(table_tag):
                    table_tag.decompose()
                    continue
                table_data = _table_to_dict(table_tag)
                if table_data["rows"] or table_data["headers"]:
                    html_tables.append({"type": "table", **table_data})
                    if DEBUG:
                        print(f"[EML HANDLER] Inline table found: "
                              f"{len(table_data['rows'])} rows, "
                              f"headers={table_data['headers']}")
                table_tag.decompose()   # remove so text extraction skips it

            # Clean text (tables already removed)
            plain = soup.get_text(separator=" ", strip=True)
            plain = _clean_body_text(plain)

            if plain:
                body_chunks.append({"type": "text", "content": plain})
            body_chunks.extend(html_tables)

            if DEBUG:
                print(f"[EML HANDLER] Used HTML body ({len(html_tables)} tables)")

        except Exception as e:
            if DEBUG:
                print(f"[EML HANDLER] HTML error: {e} — falling back to plain")
            clean = _clean_body_text(plain_text_body)
            if clean:
                body_chunks = [{"type": "text", "content": clean}]

    elif plain_text_body:
        clean = _clean_body_text(plain_text_body)
        if clean:
            body_chunks = [{"type": "text", "content": clean}]
            if DEBUG:
                print(f"[EML HANDLER] Used plain text body (no HTML present)")

    result = {
        "content"    : body_chunks,
        "metadata"   : metadata,
        "attachments": attachments_results,
    }

    output_path = Path(file_path).stem + "_parsed.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    if DEBUG:
        print(f"\n[EML HANDLER] Attachments processed : {len(attachments_results)}")
        print(f"[EML HANDLER] Body chunks            : {[c['type'] for c in body_chunks]}")
        print(f"[EML HANDLER] Saved to              : {output_path}")

    return result


# ── TEST ──────────────────────────────────────────────
if __name__ == "__main__":
    result = extract_eml("../teee.eml")

    print("\n===== METADATA =====")
    for k, v in result["metadata"].items():
        print(f"  {k}: {v}")

    print("\n===== CONTENT CHUNKS =====")
    for chunk in result["content"]:
        t = chunk["type"]
        if t == "text":
            print(f"  [TEXT] {chunk['content'][:200]}...")
        elif t == "table":
            print(f"  [TABLE] headers={chunk['headers']}")
            for row in chunk["rows"]:
                print(f"         {row}")

    print("\n===== ATTACHMENTS =====")
    for att in result["attachments"]:
        print(f"\n   {att['name']}")
        print(f"     Keys: {list(att['result'].keys())}")

    print(f"\n Saved to: {Path('../teee.eml').stem}_parsed.json")