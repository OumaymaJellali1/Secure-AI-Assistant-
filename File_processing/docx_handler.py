import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DEBUG
from pathlib import Path
import subprocess

from ingestion.digital_pdf import extract_digital_pdf


# ── LIBREOFFICE PATH (Windows default) ───────────────
LIBREOFFICE_PATH = r"C:\Program Files\LibreOffice\program\soffice.exe"


def convert_docx_to_pdf(docx_path: Path) -> Path:
    """
    Convert a .docx file to .pdf using LibreOffice headless.
    Output PDF is saved in the same folder as the .docx.
    Raises RuntimeError if conversion fails.
    """
    if not Path(LIBREOFFICE_PATH).exists():
        raise RuntimeError(
            f"LibreOffice not found at: {LIBREOFFICE_PATH}\n"
            f"Download from https://www.libreoffice.org/download/download/"
        )

    output_dir = docx_path.parent

    if DEBUG:
        print(f"[DOCX] Converting to PDF via LibreOffice: {docx_path.name}")

    result = subprocess.run(
        [
            LIBREOFFICE_PATH,
            "--headless",
            "--convert-to", "pdf",
            "--outdir", str(output_dir),
            str(docx_path)
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60  # seconds
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"LibreOffice conversion failed:\n"
            f"{result.stderr.decode('utf-8', errors='ignore')}"
        )

    pdf_path = output_dir / (docx_path.stem + ".pdf")

    if not pdf_path.exists():
        raise RuntimeError(
            f"PDF not found after conversion: {pdf_path}"
        )

    if DEBUG:
        print(f"[DOCX] PDF created: {pdf_path.name} ")

    return pdf_path


def extract_docx(file_path: str) -> dict:
    """
    Extract content from a .docx file by:
      1. Converting it to PDF via LibreOffice headless
      2. Running it through extract_digital_pdf()
      3. Cleaning up the temporary PDF
    """
    docx_path = Path(file_path)

    if not docx_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if docx_path.suffix.lower() != ".docx":
        raise ValueError(f"Expected a .docx file, got: {docx_path.suffix}")

    pdf_path = None

    try:
        # ── STEP 1: CONVERT ───────────────────────────
        pdf_path = convert_docx_to_pdf(docx_path)

        # ── STEP 2: EXTRACT (same pipeline as digital PDF) ──
        result = extract_digital_pdf(str(pdf_path))

        # ── STEP 3: FIX FILENAME IN RESULT ───────────
        # Report original .docx name, not the temp .pdf name
        result["filename"] = docx_path.name
        for chunk in result["chunks"]:
            chunk["metadata"]["source"] = docx_path.name

        return result

    finally:
        # ── STEP 4: CLEANUP TEMP PDF ──────────────────
        if pdf_path and pdf_path.exists():
            try:
                pdf_path.unlink()
                if DEBUG:
                    print(f"[DOCX] Temp PDF deleted: {pdf_path.name} ")
            except Exception as e:
                if DEBUG:
                    print(f"[DOCX] Could not delete temp PDF: {e}")


# ── TEST ──────────────────────────────────────────────
if __name__ == "__main__":
    result = extract_docx("../test10.docx")

    print("\n===== CHUNKS PREVIEW =====")
    for i, chunk in enumerate(result["chunks"][:5]):
        print(f"\nChunk {i+1}:")
        print(f"  Type      : {chunk['type']}")
        print(f"  Page start: {chunk['metadata']['page_start']}")
        print(f"  Page end  : {chunk['metadata']['page_end']}")
        print(f"  Section   : {chunk['metadata']['section']}")
        print(f"  Content   : {chunk['content'][:100]}...")

    print(f"\n Total chunks: {len(result['chunks'])}")