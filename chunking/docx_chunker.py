import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from File_processing.docx_handler import extract_docx
from chunking.pdf_chunker import chunk_pdf

def chunk_docx(file_path: str) -> list[dict]:
    handler_output = extract_docx(file_path)
    return chunk_pdf(handler_output)


# TEST
if __name__ == "__main__":
    chunks = chunk_docx("test2.docx")

    output_path = "test10_chunks.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f" Saved {len(chunks)} chunks to: {output_path}")