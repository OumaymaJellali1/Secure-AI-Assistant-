import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DEBUG
from File_processing.pptx_handler import parse_pptx, clean_content
from File_processing.cleaner import clean
from pathlib import Path
import json
import uuid


MAX_CAPTION_CHARS = 150



def chunk_parsed_pptx(parsed: dict) -> list[dict]:
    """
    Takes the output of parse_pptx() and returns a list of chunks
    ready for embedding and storage in a vector DB.
    """
    filename  = parsed["filename"]
    chunks    = []
    img_counter = 0
    tbl_counter = 0

    for slide in parsed["slides"]:


        if slide["is_section"]:
            continue

        slide_num       = slide["slide_num"]
        current_section = slide["current_section"]
        text            = slide["text"]
        tables          = slide["tables"]
        images          = slide["images"]

        base_meta = {
            "source":  filename,
            "slide":   slide_num,
            "section": current_section,
        }

        text_len      = len(text)
        has_meaningful = text_len > 0
        is_short_text  = has_meaningful and text_len < MAX_CAPTION_CHARS

        # TABLE CHUNKS 
        is_table_caption   = bool(tables) and is_short_text
        table_caption_used = False

        for md_table in tables:
            tbl_counter += 1

            if is_table_caption and not table_caption_used:
                content            = clean_content(f"{text}\n\n{md_table}")
                table_caption_used = True
            else:
                content = clean_content(md_table)

            chunks.append({
                "chunk_id": str(uuid.uuid4()),
                "type":    "table",
                "content": content,
                "metadata": {**base_meta, "table_index": tbl_counter},
            })

        # IMAGE CHUNKS 
        is_image_caption   = bool(images) and is_short_text
        image_caption_used = False

        for description in images:
            img_counter += 1

            if is_image_caption and not image_caption_used:
                content            = clean_content(f"{text}\n\n{description}")
                image_caption_used = True
            else:
                content = clean_content(description)

            chunks.append({
                "chunk_id": str(uuid.uuid4()),
                "type":    "image",
                "content": content,
                "metadata": {**base_meta, "image_index": img_counter},
            })

        # TEXT CHUNK (only if not merged) 
        caption_was_used = table_caption_used or image_caption_used

        if has_meaningful and not caption_was_used:
            chunks.append({
                "chunk_id": str(uuid.uuid4()),
                "type":    "text",
                "content": text,
                "metadata": {**base_meta},
            })

    return {
        "chunks": chunks,
        "metadata": {
            "source":      filename,
            "slide_count": parsed["total_slides"],
            "text_count":  len([c for c in chunks if c["type"] == "text"]),
            "table_count": len([c for c in chunks if c["type"] == "table"]),
            "image_count": len([c for c in chunks if c["type"] == "image"]),
        },
    }


# SECTION GROUPING 
def group_chunks_by_section(chunks: list) -> dict:
    """
    Merges all chunks of the same section into a single entry.
    Useful for section-level summarization.
    """
    sections = {}

    for chunk in chunks:
        sec = chunk["metadata"].get("section") or "__no_section__"

        if sec not in sections:
            sections[sec] = {"text": "", "slides": [], "tables": [], "images": []}

        slide_num = chunk["metadata"]["slide"]
        if slide_num not in sections[sec]["slides"]:
            sections[sec]["slides"].append(slide_num)

        if chunk["type"] == "text":
            sections[sec]["text"] += chunk["content"] + "\n\n"
        elif chunk["type"] == "table":
            sections[sec]["tables"].append(chunk["content"])
        elif chunk["type"] == "image":
            sections[sec]["images"].append(chunk["content"])

    return sections


#  MAIN PIPELINE 
def extract_pptx(file_path: str) -> dict:
    """Full pipeline: parse → chunk → save JSON."""

    parsed = parse_pptx(file_path)
    result = chunk_parsed_pptx(parsed)
    result = clean(result)

    output_path = Path(file_path).stem + "_chunked.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    if DEBUG:
        print(f"\n[PPTX] Slides : {result['metadata']['slide_count']}")
        print(f"[PPTX] Text   : {result['metadata']['text_count']}")
        print(f"[PPTX] Tables : {result['metadata']['table_count']}")
        print(f"[PPTX] Images : {result['metadata']['image_count']}")

    return result


# TEST 
if __name__ == "__main__":
    result = extract_pptx("test4.pptx")

    print("\n===== CHUNKS PREVIEW =====")
    for i, chunk in enumerate(result["chunks"][:5]):
        print(f"\nChunk {i+1}:")
        print(f"  chunk_id: {chunk['chunk_id']}")
        print(f"  Type   : {chunk['type']}")
        print(f"  Slide  : {chunk['metadata']['slide']}")
        print(f"  Section: {chunk['metadata']['section']}")
        print(f"  Content: {chunk['content'][:100]}...")

    print(f"\nTotal chunks : {len(result['chunks'])}")

    sections = group_chunks_by_section(result["chunks"])
    print("\n===== SECTIONS =====")
    for sec, data in sections.items():
        print(f"\n  [{sec}]  slides={data['slides']}")
        print(f"  text preview: {data['text'][:80]}...")