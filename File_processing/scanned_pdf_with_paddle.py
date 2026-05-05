import sys
import os
import os
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["PPOCR_LOG_LEVEL"]                        = "ERROR"
os.environ["FLAGS_use_mkldnn"]                       = "0"   # ← disables oneDNN
os.environ["FLAGS_mkldnn_cache_capacity"]            = "0"   # ← disables oneDNN cache

# ── SUPPRESS PADDLEOCR VERBOSE OUTPUT ────────────────
# must be set BEFORE any paddleocr import


import json
import re
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DEBUG, GROQ_API_KEY, GROQ_VISION_MODEL, TEMP_DIR
from groq import Groq
from pathlib import Path
import base64

# ── CONFIGURE GROQ ────────────────────────────────────
groq_client  = Groq(api_key=GROQ_API_KEY)
POPPLER_PATH = r"C:\Users\hp\Downloads\Release-25.12.0-0\poppler-25.12.0\Library\bin"

from paddleocr import PaddleOCR

# ── DEUX MOTEURS OCR : latin (fr+en) et arabe ────────
# Le modèle "fr" couvre français ET anglais (alphabet latin partagé).
# Le modèle "ar" est nécessaire pour l'arabe (script complètement différent).
# Les deux sont chargés une seule fois au démarrage pour éviter la latence.

_ocr_engine_latin = PaddleOCR(
    lang="fr",
    use_angle_cls=True,
    use_gpu=False,
    show_log=False
)

_ocr_engine_arabic = PaddleOCR(
    lang="ar",
    use_angle_cls=True,
    use_gpu=False,
    show_log=False
)


def _clean_model_output(text: str) -> str:
    """
    Remove model reasoning/thinking steps from output.
    Strips lines like:
      ## Step 1: ...
      ## Step 2: ...
      Step 1
      Let me analyze...
      Here is the table:
    Keeps only the actual content.
    """
    text = re.sub(r"##\s*Step\s*\d+[:\-].*\n?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\*\*Step\s*\d+.*?\*\*\n?", "", text, flags=re.IGNORECASE)

    preamble_patterns = [
        r"^(Let me|I will|I'll|First,|Here is|Here are|The following|Below is|Below are|Sure,|Certainly,|Of course,).*\n?",
        r"^There (is|are) \w+ table[s]? in the image.*\n?",
        r"^There (is|are) \w+ chart[s]? in the image.*\n?",
        r"^I (can see|found|identified|detected|notice).*\n?",
    ]
    for pattern in preamble_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.MULTILINE)

    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_scanned_pdf(file_path: str) -> dict:
    """
    Extract content from a scanned PDF (Arabic, French, English).
    - Language detection → Groq vision (per page)
    - Text             → PaddleOCR v3 (arabic engine OR latin engine)
    - Page type        → Groq vision (classify: text / chart / table / mixed)
    - Charts           → Groq vision → stored in images[] with metadata
    - Tables           → Groq vision → stored in tables[] with metadata
    - Output           → JSON file
    """

    if DEBUG:
        print(f"\n[SCANNED PDF] Processing: {file_path}")

    from pdf2image import convert_from_path

    pages = convert_from_path(
        file_path,
        dpi=300,
        poppler_path=POPPLER_PATH
    )

    if DEBUG:
        print(f"[SCANNED PDF] Total pages: {len(pages)}")

    source_name  = Path(file_path).name
    full_text    = ""
    all_images   = []
    all_tables   = []
    image_index  = 0
    table_index  = 0
    page_counter = 0

    for page_img in pages:
        page_counter += 1

        if DEBUG:
            print(f"\n[SCANNED PDF] Processing page "
                  f"{page_counter}/{len(pages)}...")

        # ── SAVE PAGE AS TEMP IMAGE ───────────────────
        temp_page_path = TEMP_DIR / f"page_{page_counter}.png"
        page_img.save(str(temp_page_path))

        # ── STEP 1: DÉTECTER LA LANGUE ────────────────
        page_language = _detect_language(str(temp_page_path))

        # ── STEP 2: OCR → TEXT ONLY ───────────────────
        text = _ocr_page(str(temp_page_path), language=page_language)
        if text:
            full_text += f"\n---\nPage {page_counter}\n\n{text}\n\n"
            if DEBUG:
                print(f"[SCANNED PDF] Page {page_counter} OCR ✓ "
                      f"(lang={page_language})")

        # ── STEP 3: CLASSIFY PAGE WITH GROQ ──────────
        visual_type = _detect_visual_type(str(temp_page_path))

        if DEBUG:
            print(f"[SCANNED PDF] Page {page_counter} "
                  f"→ type: {visual_type}")

        # ── STEP 4: ROUTE BASED ON CLASSIFICATION ────
        if visual_type == "text_only":
            if DEBUG:
                print(f"[SCANNED PDF] Text only → no extra call ✓")

        elif visual_type in ("has_chart", "mixed"):
            if DEBUG:
                print(f"[SCANNED PDF] Chart/mixed detected "
                      f"→ calling Groq to describe...")

            raw_description = _describe_visual(str(temp_page_path), page_language)
            description     = _clean_model_output(raw_description)

            if description:
                image_index += 1
                all_images.append({
                    "image_index": image_index,
                    "page"       : page_counter,
                    "source"     : source_name,
                    "language"   : page_language,
                    "content"    : description
                })
                if DEBUG:
                    print(f"[SCANNED PDF] Page {page_counter} "
                          f"described ✓")

        elif visual_type == "has_table":
            if DEBUG:
                print(f"[SCANNED PDF] Table detected "
                      f"→ calling Groq to extract...")

            raw_table  = _extract_table(str(temp_page_path), page_language)
            table_data = _clean_model_output(raw_table)

            if table_data:
                table_index += 1
                all_tables.append({
                    "table_index": table_index,
                    "page"       : page_counter,
                    "source"     : source_name,
                    "language"   : page_language,
                    "content"    : table_data
                })
                if DEBUG:
                    print(f"[SCANNED PDF] Page {page_counter} "
                          f"table extracted ✓")

        # ── CLEANUP ───────────────────────────────────
        os.remove(str(temp_page_path))

    # ── SAVE AS JSON ──────────────────────────────────
    output_path = Path(file_path).stem + "_parsed.json"

    output_data = {
        "source" : source_name,
        "pages"  : page_counter,
        "text"   : full_text.strip(),
        "tables" : all_tables,
        "images" : all_images
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    if DEBUG:
        print(f"\n[SCANNED PDF] Visuals described : {len(all_images)}")
        print(f"\n[SCANNED PDF] Tables extracted  : {len(all_tables)}")
        print(f"[SCANNED PDF] Saved to          : {output_path}")

    return output_data


# ── DÉTECTER LA LANGUE DE LA PAGE ────────────────────
def _detect_language(image_path: str) -> str:
    """
    Utilise Groq vision pour détecter la langue dominante de la page.

    Retourne :
        "arabic" → texte majoritairement en arabe
        "latin"  → texte majoritairement en français ou anglais
    """
    try:
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")

        response = groq_client.chat.completions.create(
            model=GROQ_VISION_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_b64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": (
                            "Look at the text on this page. "
                            "Reply with ONE word only:\n"
                            "'arabic' → if the dominant text is Arabic\n"
                            "'latin'  → if the dominant text is English or French\n"
                            "ONE word only. No explanation. No punctuation."
                        )
                    }
                ]
            }],
            max_tokens=5
        )

        label  = response.choices[0].message.content.strip().lower()
        result = label if label in ("arabic", "latin") else "latin"

        if DEBUG:
            print(f"[SCANNED PDF] Language detected → '{result}'")

        return result

    except Exception as e:
        if DEBUG:
            print(f"[SCANNED PDF] Language detection error: {e}")
        return "latin"   # fallback sûr


# ── OCR PAGE (PaddleOCR v3) ───────────────────────────
def _ocr_page(image_path: str, language: str = "latin") -> str:
    try:
        engine = _ocr_engine_arabic if language == "arabic" else _ocr_engine_latin
        result = engine.ocr(image_path, cls=True)

        lines = []
        if result and result[0]:
            for line in result[0]:
                text = line[1][0].strip()
                if text:
                    lines.append(text)

        return "\n".join(lines)

    except Exception as e:
        if DEBUG:
            print(f"[SCANNED PDF] OCR error: {e}")
        return ""


# ── CLASSIFY PAGE WITH GROQ ───────────────────────────
def _detect_visual_type(image_path: str) -> str:
    """
    Use Groq vision to classify the page content.

    Returns:
        "text_only" → plain text page  → skip extra calls
        "has_chart" → charts/diagrams  → send to _describe_visual()
        "has_table" → tables           → send to _extract_table()
        "mixed"     → text + visuals   → send to _describe_visual()
    """
    try:
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")

        response = groq_client.chat.completions.create(
            model=GROQ_VISION_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_b64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": (
                            "Look at this page carefully. "
                            "Reply with ONE word only:\n"
                            "'text_only' → plain text, no visuals\n"
                            "'has_chart' → contains charts, graphs, or diagrams\n"
                            "'has_table' → contains tables or structured data grids\n"
                            "'mixed'     → contains both text and charts/diagrams\n\n"
                            "ONE word only. No explanation. No punctuation."
                        )
                    }
                ]
            }],
            max_tokens=10
        )

        label = response.choices[0].message.content.strip().lower()

        valid  = ("text_only", "has_chart", "has_table", "mixed")
        result = label if label in valid else "text_only"

        if DEBUG:
            print(f"[SCANNED PDF] Groq classification → '{label}' "
                  f"→ using '{result}'")

        return result

    except Exception as e:
        if DEBUG:
            print(f"[SCANNED PDF] Classification error: {e}")
        return "text_only"


# ── DESCRIBE VISUAL ───────────────────────────────────
def _describe_visual(image_path: str, language: str = "latin") -> str:
    """
    Send page to Groq when chart or mixed content detected.
    Describes charts, figures, diagrams and their data.
    Adapts the prompt language hint based on detected page language.
    """
    lang_hint = (
        "The document is in Arabic — preserve any Arabic text exactly as written."
        if language == "arabic"
        else "The document is in English or French — preserve text as-is."
    )

    try:
        with open(image_path, "rb") as img_file:
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
                                f"This is a scanned page. {lang_hint} "
                                "Describe ONLY the charts, figures, "
                                "diagrams or images you see. "
                                "Mention any data values or trends. "
                                "Ignore the text content. "
                                "Do NOT include reasoning steps or explanations. "
                                "Output the description directly, nothing else."
                            )
                        }
                    ]
                }
            ],
            max_tokens=500
        )

        return response.choices[0].message.content

    except Exception as e:
        if DEBUG:
            print(f"[SCANNED PDF] Groq describe error: {e}")
        return ""


# ── EXTRACT TABLE ─────────────────────────────────────
def _extract_table(image_path: str, language: str = "latin") -> str:
    """
    Send page to Groq when a table is detected.
    Returns the table as clean markdown.
    Adapts the prompt language hint based on detected page language.
    """
    lang_hint = (
        "The table may contain Arabic text — preserve it exactly as written, right-to-left."
        if language == "arabic"
        else "The table is in English or French — preserve all text as-is."
    )

    try:
        with open(image_path, "rb") as img_file:
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
                                f"Extract the table from this scanned page. {lang_hint} "
                                "Return it as a clean markdown table. "
                                "Preserve all rows, columns and values exactly. "
                                "Use | separators for columns. "
                                "Add a header separator row (|---|---|). "
                                "Return ONLY the markdown table. "
                                "No reasoning, no steps, no explanation."
                            )
                        }
                    ]
                }
            ],
            max_tokens=800
        )

        return response.choices[0].message.content

    except Exception as e:
        if DEBUG:
            print(f"[SCANNED PDF] Groq table error: {e}")
        return ""


# ── TEST ──────────────────────────────────────────────
if __name__ == "__main__":
    result = extract_scanned_pdf(r"C:\Users\hp\Downloads\Secure-AI-Assistant--main\Secure-AI-Assistant--main\File_processing\tests\222.pdf")

    print("\n===== TEXT (OCR only) =====")
    print(result["text"][:2000])

    print(f"\n Visuals described : {len(result['images'])}")
    print(f" Tables extracted  : {len(result['tables'])}")
    print(f"\n Output saved to  : 222.json")