import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DEBUG, GROQ_API_KEY, GROQ_VISION_MODEL, TEMP_DIR
from groq import Groq
from pathlib import Path
import base64
import json

# ── CONFIGURE GROQ ────────────────────────────────────
groq_client = Groq(api_key=GROQ_API_KEY)


def extract_image(file_path: str) -> dict:
    """
    Extract description from a standalone image file.
    - Image → Groq llama-4-scout → text description
    - Saves result as JSON file

    Supports: .jpg .jpeg .png .bmp .tiff

    Returns:
        {
            "type"    : "image",
            "content" : image description,
            "metadata": { "source": file_path, "image_index": 1 }
        }
    """

    if DEBUG:
        print(f"\n[IMAGE] Processing: {file_path}")

    # ── GET IMAGE EXTENSION ───────────────────────────
    ext      = Path(file_path).suffix.lower().replace(".", "")
    mime_map = {
        "jpg" : "jpeg",
        "jpeg": "jpeg",
        "png" : "png",
        "bmp" : "bmp",
        "tiff": "tiff",
        "tif" : "tiff",
    }
    mime_type = mime_map.get(ext, "jpeg")

    if DEBUG:
        print(f"[IMAGE] Extension : {ext}")
        print(f"[IMAGE] MIME type : {mime_type}")

    # ── CHECK IMAGE SIZE ──────────────────────────────
    from PIL import Image as PILImage
    img  = PILImage.open(file_path)
    w, h = img.size

    if DEBUG:
        print(f"[IMAGE] Size      : {w}x{h}")

    # ── ENCODE TO BASE64 ──────────────────────────────
    with open(file_path, "rb") as img_file:
        img_base64 = base64.b64encode(
            img_file.read()
        ).decode("utf-8")

    if DEBUG:
        print(f"[IMAGE] Calling Groq...")

    # ── SEND TO GROQ ──────────────────────────────────
    response = groq_client.chat.completions.create(
        model=GROQ_VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/{mime_type};base64,{img_base64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": (
                            "Describe this image in detail. "
                            "If it contains a chart or graph describe "
                            "the type, values, and trends. "
                            "If it contains a table extract the data. "
                            "If it contains text read and transcribe it. "
                            "Be precise and detailed."
                        )
                    }
                ]
            }
        ],
        max_tokens=500
    )

    description = response.choices[0].message.content

    if DEBUG:
        print(f"[IMAGE] Description: {description[:100]}...")

    # ── BUILD JSON ────────────────────────────────────
    result = {
        "type"    : "image",
        "content" : description,
        "metadata": {
            "source"     : file_path,
            "image_index": 1
        }
    }

    # ── SAVE AS JSON ──────────────────────────────────
    output_path = Path(file_path).stem + "_parsed.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    if DEBUG:
        print(f"[IMAGE] JSON saved to: {output_path}")

    return result


# ── TEST ──────────────────────────────────────────────
if __name__ == "__main__":
    result = extract_image("../1.png")

    print("\n===== JSON OUTPUT =====")
    print(json.dumps(result, indent=4, ensure_ascii=False))

    print(f"\n Description length: {len(result['content'])} chars")
    print(f"\n Output saved to  : 1111.json")