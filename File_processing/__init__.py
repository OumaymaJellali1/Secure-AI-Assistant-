import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DEBUG
from File_processing.detector import detect_file_type


def parse_document(file_path: str) -> dict:
    """
    Main entry point for document parsing.
    Detects file type and routes to correct handler.

    Returns:
        {
            "text"  : full extracted text,
            "tables": [list of tables],
            "images": [list of image descriptions]
        }
    """

    if DEBUG:
        print(f"\n[PARSER] Received: {file_path}")

    # ── DETECT FILE TYPE ──────────────────────────────
    file_type = detect_file_type(file_path)

    if DEBUG:
        print(f"[PARSER] Detected type: {file_type}")

    # ── ROUTE TO CORRECT HANDLER ──────────────────────

    # ── DIGITAL PDF ───────────────────────────────────
    if file_type == "pdf_digital":
        from File_processing.digital_pdf import extract_digital_pdf
        return extract_digital_pdf(file_path)

    # ── SCANNED PDF ───────────────────────────────────
    elif file_type == "pdf_scanned":
        from File_processing.scanned_pdf import extract_scanned_pdf
        return extract_scanned_pdf(file_path)

    # ── WORD DOCUMENT ─────────────────────────────────
    elif file_type == "docx":
        from File_processing.docx_handler import extract_docx
        return extract_docx(file_path)

    # ── PLAIN TEXT ────────────────────────────────────
    elif file_type == "txt":
        from File_processing.txt_handler import extract_txt
        return extract_txt(file_path)

    # ── CSV ───────────────────────────────────────────
    elif file_type == "csv":
        from File_processing.csv_handler import extract_csv
        return extract_csv(file_path)

    # ── IMAGE ─────────────────────────────────────────
    elif file_type == "image":
        from File_processing.image_handler import extract_image
        return extract_image(file_path)
    #-------------------pptx-------------------------------
    elif file_type == "pptx":
     from File_processing.pptx_handler import extract_pptx
     return extract_pptx(file_path)

    # ── UNSUPPORTED ───────────────────────────────────
    else:
        raise ValueError(
            f"[PARSER] Unsupported file type: {file_type}"
        )