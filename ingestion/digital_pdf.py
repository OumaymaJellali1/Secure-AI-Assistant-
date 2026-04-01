import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DEBUG, GROQ_API_KEY, GROQ_VISION_MODEL, TEMP_DIR
from groq import Groq
from pathlib import Path
import pandas as pd
import base64
import json

# ── CONFIGURE GROQ ────────────────────────────────────
groq_client = Groq(api_key=GROQ_API_KEY)


def extract_digital_pdf(file_path: str) -> dict:
    """
    Extract content from a digital PDF using Docling.
    - Text   → grouped by SECTION (not by page)
    - Tables → one chunk each with metadata
    - Images → Groq description + metadata
    """

    if DEBUG:
        print(f"\n[DIGITAL PDF] Processing: {file_path}")

    filename = Path(file_path).name

    # ── LOAD PDF WITH DOCLING ─────────────────────────
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling_core.types.doc import (
        TextItem, TableItem, PictureItem, SectionHeaderItem
    )

    pipeline_options = PdfPipelineOptions()
    pipeline_options.generate_picture_images = True
    pipeline_options.images_scale= 2.0
    pipeline_options.do_ocr = False

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options
            )
        }
    )

    result = converter.convert(Path(file_path))
    doc    = result.document

    # ── STATE ─────────────────────────────────────────
    chunks             = []
    img_counter        = 0
    tbl_counter        = 0
    current_section    = "Unknown"
    current_text       = ""

 
    current_page_start = 1
    current_page_end   = 1

    # ── HELPER: FLUSH ACCUMULATED TEXT ────────────────
    def flush_text():
        nonlocal current_text
        if current_text.strip():
            chunks.append({
                "type"    : "text",
                "content" : current_text.strip(),
                "metadata": {
                    "source"    : filename,
                    "page_start": current_page_start,
                    "page_end"  : current_page_end,
                    "section"   : current_section
                }
            })
            if DEBUG:
                print(f"[DIGITAL PDF] Text chunk saved "
                      f"(pages {current_page_start}-{current_page_end}, "
                      f"section: {current_section[:30]}) ")
        current_text = ""

    for element, _ in doc.iterate_items():

        page_no = 1
        if element.prov:
            page_no = element.prov[0].page_no

        if isinstance(element, SectionHeaderItem):
            # flush whatever text we have before starting new section
            flush_text()
            current_section    = element.text.strip()
            # CHANGE 3: reset page range when new section starts
            current_page_start = page_no
            current_page_end   = page_no
            if DEBUG:
                print(f"[DIGITAL PDF] New section: {current_section} "
                      f"(page {page_no})")

        elif isinstance(element, TextItem):
            text = element.text.strip()
            if text:
                
                if current_text == "":
                    current_page_start = page_no  
                current_page_end = page_no         
                current_text += text + " "

        # ── TABLE ─────────────────────────────────────
        elif isinstance(element, TableItem):
            flush_text()

            try:
                table_data = element.data
                rows       = {}

                for cell in table_data.table_cells:
                    row_idx = cell.start_row_offset_idx
                    col_idx = cell.start_col_offset_idx
                    if row_idx not in rows:
                        rows[row_idx] = {}
                    rows[row_idx][col_idx] = cell.text

                if rows:
                    max_row = max(rows.keys())
                    max_col = max(
                        col
                        for row in rows.values()
                        for col in row.keys()
                    )

                    data = []
                    for r in range(max_row + 1):
                        row_data = []
                        for c in range(max_col + 1):
                            row_data.append(
                                rows.get(r, {}).get(c, "")
                            )
                        data.append(row_data)

                    if len(data) > 1:
                        df = pd.DataFrame(
                            data[1:], columns=data[0]
                        )
                    else:
                        df = pd.DataFrame(data)

                    tbl_counter += 1
                    md_table     = df.to_markdown(index=False)

                    chunks.append({
                        "type"    : "table",
                        "content" : md_table,
                        "metadata": {
                            "source"     : filename,
                            "page_start" : page_no,
                            "page_end"   : page_no,
                            "section"    : current_section,
                            "table_index": tbl_counter
                        }
                    })

                    if DEBUG:
                        print(f"[DIGITAL PDF] Table {tbl_counter} "
                              f"(page {page_no}) ")

            except Exception as e:
                if DEBUG:
                    print(f"[DIGITAL PDF] Table error: {e}")
                try:
                    tbl_counter += 1
                    md_table     = element.export_to_markdown()
                    chunks.append({
                        "type"    : "table",
                        "content" : md_table,
                        "metadata": {
                            "source"     : filename,
                            "page_start" : page_no,
                            "page_end"   : page_no,
                            "section"    : current_section,
                            "table_index": tbl_counter
                        }
                    })
                except:
                    continue

        # ── IMAGE ─────────────────────────────────────
        elif isinstance(element, PictureItem):
            flush_text()

            try:
                img = element.get_image(doc)

                if img:
                    temp_img_path = TEMP_DIR / f"img_{img_counter}.png"
                    img.save(str(temp_img_path))

                    if DEBUG:
                        print(f"[DIGITAL PDF] Describing image "
                              f"{img_counter+1} (page {page_no})...")

                    with open(str(temp_img_path), "rb") as img_file:
                        img_base64 = base64.b64encode(
                            img_file.read()
                        ).decode("utf-8")

                    response = groq_client.chat.completions.create(
                        model=GROQ_VISION_MODEL,
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/png;base64,{img_base64}"
                                        }
                                    },
                                    {
                                        "type": "text",
                                        "text": (
                                            "Describe this image or chart in detail. "
                                            "If it contains data or values mention them. "
                                            "If it is a chart describe type, values, and trends."
                                        )
                                    }
                                ]
                            }
                        ],
                        max_tokens=500
                    )

                    description = response.choices[0].message.content
                    img_counter += 1

                    chunks.append({
                        "type"    : "image",
                        "content" : description,
                        "metadata": {
                            "source"     : filename,
                            "page_start" : page_no,
                            "page_end"   : page_no,
                            "section"    : current_section,
                            "image_index": img_counter
                        }
                    })

                    if DEBUG:
                        print(f"[DIGITAL PDF] Image {img_counter} "
                              f"(page {page_no}) ")

                    os.remove(str(temp_img_path))

            except Exception as e:
                if DEBUG:
                    print(f"[DIGITAL PDF] Image error: {e}")
                continue

    # ── FLUSH REMAINING TEXT ──────────────────────────
    flush_text()

    # ── SAVE AS JSON ──────────────────────────────────
    output_path = Path(file_path).stem + "_parsed.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "filename": filename,
            "chunks"  : chunks
        }, f, ensure_ascii=False, indent=2)

    if DEBUG:
        text_count  = len([c for c in chunks if c["type"] == "text"])
        table_count = len([c for c in chunks if c["type"] == "table"])
        image_count = len([c for c in chunks if c["type"] == "image"])
        print(f"\n[DIGITAL PDF] Text chunks : {text_count}")
        print(f"[DIGITAL PDF] Tables      : {table_count}")
        print(f"[DIGITAL PDF] Images      : {image_count}")
        print(f"[DIGITAL PDF] Saved to    : {output_path}")

    return {
        "filename": filename,
        "chunks"  : chunks
    }


# ── TEST ──────────────────────────────────────────────
if __name__ == "__main__":
    result = extract_digital_pdf(r"C:\Users\Dell\Desktop\PFE\test.pdf")

    print("\n===== CHUNKS PREVIEW =====")
    for i, chunk in enumerate(result["chunks"][:5]):
        print(f"\nChunk {i+1}:")
        print(f"  Type      : {chunk['type']}")
        print(f"  Page start: {chunk['metadata']['page_start']}")
        print(f"  Page end  : {chunk['metadata']['page_end']}")
        print(f"  Section   : {chunk['metadata']['section']}")
        print(f"  Content   : {chunk['content'][:100]}...")

    print(f"\n Total chunks: {len(result['chunks'])}")
    print(f" Saved to: test_parsed.json")