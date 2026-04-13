import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from config import DEBUG


def detect_file_type(file_path: str) -> str:
    ext = Path(file_path).suffix.lower()

    if DEBUG:
        print(f"[DETECTOR] File     : {file_path}")
        print(f"[DETECTOR] Extension: {ext}")

    if ext == ".docx":
        return "docx"

    elif ext == ".txt":
        return "txt"

    elif ext == ".csv":
        return "csv"

    elif ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]:
        return "image"

    elif ext == ".pdf":
        try:
            import fitz
            doc = fitz.open(file_path)
            for page in doc:
                if page.get_text().strip():
                    if DEBUG:
                        print(f"[DETECTOR] Type: pdf_digital ")
                    return "pdf_digital"
            if DEBUG:
                print(f"[DETECTOR] Type: pdf_scanned ")
            return "pdf_scanned"

        except Exception as e:
            if DEBUG:
                print(f"[DETECTOR] PDF error: {e}")
            return "pdf_digital"

    elif ext == ".pptx":
        if DEBUG:
            print(f"[DETECTOR] Type: pptx ")
        return "pptx"

    elif ext == ".eml":
        if DEBUG:
            print(f"[DETECTOR] Type: eml ")
        return "eml"

    else:
        if DEBUG:
            print(f"[DETECTOR] Type: unknown ")
        return "unknown"