import os
from pathlib import Path
import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_CACHE"] = r"C:\Users\hp\AppData\Local\huggingface\hub"

# ── PROJECT PATHS ─────────────────────────────────────
BASE_DIR = Path(__file__).parent
TEMP_DIR = BASE_DIR / "temp"
TEMP_DIR.mkdir(exist_ok=True)



from dotenv import load_dotenv
load_dotenv()


# ── GROQ ─────────────────────────────────────────────
GROQ_API_KEY      = os.environ.get("GROQ_API_KEY")
GROQ_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# ── FILE TYPES ────────────────────────────────────────
SUPPORTED_EXTENSIONS = {
    "pdf"  : [".pdf"],
    "image": [".jpg", ".jpeg", ".png", ".bmp", ".tiff"],
    "txt"  : [".txt"],
    "docx" : [".docx"],
    "csv"  : [".csv"],
}

# ── DEBUG ─────────────────────────────────────────────
DEBUG = True

# ── CAMELOT SETTINGS ──────────────────────────────────
CAMELOT_FLAVOR = "stream"