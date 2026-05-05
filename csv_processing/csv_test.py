# csv_agent.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import io
import json
import re
import hashlib
import builtins
import chardet
import requests
import pandas as pd

from pathlib import Path
from groq import Groq
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct,
)
from RestrictedPython import compile_restricted
from RestrictedPython.Guards import guarded_iter_unpack_sequence, safer_getattr

from embedder import embed_query
from config import GROQ_API_KEY, GROQ_VISION_MODEL

import uuid

load_dotenv()


# ── CONFIG ────────────────────────────────────────────────────────────────────

QDRANT_URL     = "http://localhost:6333"
CSV_COLLECTION = "csv_index"
DENSE_DIM      = 1024

SHAREPOINT_USER     = os.environ.get("SHAREPOINT_USER")
SHAREPOINT_PASSWORD = os.environ.get("SHAREPOINT_PASSWORD")

_groq_client = Groq(api_key=GROQ_API_KEY)
_qdrant      = None
_session     = None


# ── .env ──────────────────────────────────────────────────────────────────────
# SHAREPOINT_USER     = your.email@company.com
# SHAREPOINT_PASSWORD = yourpassword
# GROQ_API_KEY        = your_groq_key


# ── SHAREPOINT SESSION ────────────────────────────────────────────────────────

def _get_session() -> requests.Session:
    """
    Create one authenticated session and reuse it for all requests.
    Tries NTLM first (most company SharePoints), falls back to Basic auth.
    """
    global _session
    if _session is not None:
        return _session

    if not SHAREPOINT_USER or not SHAREPOINT_PASSWORD:
        raise EnvironmentError(
            "SHAREPOINT_USER and SHAREPOINT_PASSWORD must be set in .env file."
        )

    # try NTLM first — used by most on-premise SharePoint
    try:
        from requests_ntlm import HttpNtlmAuth
        session      = requests.Session()
        session.auth = HttpNtlmAuth(SHAREPOINT_USER, SHAREPOINT_PASSWORD)
        session.headers.update({"User-Agent": "Mozilla/5.0"})
        print("[SHAREPOINT] Using NTLM authentication.")
        _session = session
        return _session

    except ImportError:
        # requests_ntlm not installed — fall back to Basic auth
        # works for SharePoint Online (Microsoft 365)
        session      = requests.Session()
        session.auth = (SHAREPOINT_USER, SHAREPOINT_PASSWORD)
        session.headers.update({"User-Agent": "Mozilla/5.0"})
        print("[SHAREPOINT] Using Basic authentication.")
        print("[SHAREPOINT] Tip: pip install requests_ntlm for NTLM support.")
        _session = session
        return _session


def _build_download_url(sharepoint_url: str) -> str:
    """
    Force direct file download by appending ?download=1.
    Without this SharePoint returns an HTML preview page
    instead of the raw CSV bytes.
    """
    if "download=1" in sharepoint_url:
        return sharepoint_url
    separator = "&" if "?" in sharepoint_url else "?"
    return sharepoint_url + separator + "download=1"


def _url_to_filename(sharepoint_url: str) -> str:
    """Extract filename from URL — removes query params first."""
    return Path(sharepoint_url.split("?")[0]).name


# ── DOWNLOAD CSV BYTES FROM SHAREPOINT ───────────────────────────────────────

def _download_bytes(sharepoint_url: str) -> bytes:
    """
    Download raw bytes from SharePoint URL.
    Nothing written to disk — bytes stay in RAM.
    Raises clear errors for auth/permission/not-found issues.
    """
    url      = _build_download_url(sharepoint_url)
    session  = _get_session()

    print(f"[SHAREPOINT] Downloading: {_url_to_filename(sharepoint_url)}")

    try:
        response = session.get(url, timeout=30)
    except requests.exceptions.ConnectionError:
        raise ConnectionError(
            f"Could not connect to SharePoint. "
            f"Check your network and the URL:\n{sharepoint_url}"
        )
    except requests.exceptions.Timeout:
        raise TimeoutError(
            f"SharePoint request timed out after 30s.\n{sharepoint_url}"
        )

    # clear error messages for common HTTP errors
    if response.status_code == 401:
        raise PermissionError(
            "Authentication failed (401). "
            "Check SHAREPOINT_USER and SHAREPOINT_PASSWORD in .env"
        )
    if response.status_code == 403:
        raise PermissionError(
            "Access denied (403). "
            "You don't have permission to access this file."
        )
    if response.status_code == 404:
        raise FileNotFoundError(
            f"File not found on SharePoint (404):\n{sharepoint_url}"
        )

    response.raise_for_status()

    # verify we got CSV bytes not an HTML page
    content_type = response.headers.get("Content-Type", "")
    if "text/html" in content_type:
        raise ValueError(
            "SharePoint returned an HTML page instead of CSV bytes.\n"
            "This usually means authentication failed silently.\n"
            "Check your credentials or try adding ?download=1 to the URL."
        )

    return response.content


# ── CSV LOADING ───────────────────────────────────────────────────────────────

def load_csv_from_url(sharepoint_url: str) -> pd.DataFrame:
    """
    Download CSV from SharePoint → DataFrame.
    Nothing saved to disk — everything in RAM.

    Args:
        sharepoint_url : SharePoint file URL

    Returns:
        pandas DataFrame with cleaned column names
    """
    content  = _download_bytes(sharepoint_url)
    buffer   = io.BytesIO(content)
    encoding = chardet.detect(content[:50_000]).get("encoding") or "utf-8"

    df = pd.read_csv(buffer, encoding=encoding)
    df.columns = [
        re.sub(r"[^a-z0-9]+", "", col.lower()).strip()
        for col in df.columns
    ]

    print(f"[SHAREPOINT] Loaded '{_url_to_filename(sharepoint_url)}' "
          f"({len(df)} rows, {len(df.columns)} cols) — RAM only, not saved.")

    return df


def get_df_info(df: pd.DataFrame, name: str) -> dict:
    """Schema + 3 sample rows for LLM prompt."""
    return {
        "table"  : name,
        "shape"  : df.shape,
        "columns": df.columns.tolist(),
        "dtypes" : {col: str(dt) for col, dt in df.dtypes.items()},
        "sample" : df.head(3).to_dict(orient="records"),
    }


# ── QDRANT ────────────────────────────────────────────────────────────────────

def _get_qdrant() -> QdrantClient:
    global _qdrant
    if _qdrant is None:
        print(f"[QDRANT] Connecting to {QDRANT_URL}...")
        _qdrant = QdrantClient(url=QDRANT_URL)
        print(f"[QDRANT] Connected.")
    return _qdrant


def _ensure_csv_collection():
    """Create csv_index collection in Qdrant if it doesn't exist."""
    client   = _get_qdrant()
    existing = [c.name for c in client.get_collections().collections]

    if CSV_COLLECTION in existing:
        return

    print(f"[QDRANT] Creating collection '{CSV_COLLECTION}'...")
    client.create_collection(
        collection_name = CSV_COLLECTION,
        vectors_config  = {
            "dense": VectorParams(
                size     = DENSE_DIM,
                distance = Distance.COSINE,
            )
        },
    )
    print(f"[QDRANT] Collection '{CSV_COLLECTION}' ready.")


# ── SUMMARY GENERATOR ─────────────────────────────────────────────────────────

def _summarize_csv(file_name: str, df_info: dict) -> str:
    """
    Ask LLM to write a natural language description of the CSV.
    This is what gets embedded — must be descriptive enough
    for similarity search to work well.
    """
    response = _groq_client.chat.completions.create(
        model    = GROQ_VISION_MODEL,
        messages = [{
            "role"   : "user",
            "content": (
                f"CSV file : '{file_name}'\n"
                f"Columns  : {json.dumps(df_info['dtypes'], indent=2)}\n"
                f"Sample   : {json.dumps(df_info['sample'], indent=2)}\n\n"
                "Write 2-3 sentences describing:\n"
                "1. What domain or topic this dataset covers\n"
                "2. What the key columns represent\n"
                "3. What kind of questions it can answer\n"
                "Be specific — mention actual column names."
            )
        }],
        max_tokens  = 200,
        temperature = 0.0,
    )
    return response.choices[0].message.content.strip()


# ── INDEXING ──────────────────────────────────────────────────────────────────

def index_csv_from_url(sharepoint_url: str, force: bool = False) -> str:
    """
    Index one CSV from SharePoint into Qdrant.

    What happens:
      1. Download bytes from SharePoint into RAM
      2. Compute MD5 hash from bytes
      3. Skip if already indexed and hash unchanged
      4. Load bytes into DataFrame
      5. LLM writes summary of the DataFrame
      6. Embed summary → dense vector
      7. Save {url, summary, hash, columns, vector} in Qdrant
      8. Bytes and DataFrame discarded — nothing saved to disk

    Args:
        sharepoint_url : direct URL to CSV on SharePoint
        force          : re-index even if file unchanged

    Returns:
        Qdrant point ID (deterministic — same file always same ID)
    """
    _ensure_csv_collection()

    file_name = _url_to_filename(sharepoint_url)
    point_id  = str(uuid.uuid5(uuid.NAMESPACE_DNS, file_name))
    client    = _get_qdrant()

    # download once — reuse bytes for both hash and DataFrame
    content   = _download_bytes(sharepoint_url)
    curr_hash = hashlib.md5(content).hexdigest()

    # skip if already indexed and unchanged
    if not force:
        existing = client.retrieve(
            collection_name = CSV_COLLECTION,
            ids             = [point_id],
            with_payload    = True,
        )
        if existing:
            stored_hash = existing[0].payload.get("file_hash", "")
            if stored_hash == curr_hash:
                print(f"[CSV INDEX] Skipped  '{file_name}' — no changes.")
                return point_id
            else:
                print(f"[CSV INDEX] Updated  '{file_name}' — file changed, re-indexing.")
        else:
            print(f"[CSV INDEX] New      '{file_name}' — indexing for first time.")

    # bytes → DataFrame (RAM only)
    buffer   = io.BytesIO(content)
    encoding = chardet.detect(content[:50_000]).get("encoding") or "utf-8"
    df       = pd.read_csv(buffer, encoding=encoding)
    df.columns = [
        re.sub(r"[^a-z0-9]+", "", col.lower()).strip()
        for col in df.columns
    ]

    df_info = get_df_info(df, Path(file_name).stem)
    summary = _summarize_csv(file_name, df_info)

    print(f"[CSV INDEX] Summary  : {summary}")

    # embed using your existing embedder
    dense_vector, _ = embed_query(summary)

    # save to Qdrant — URL is the pointer back to SharePoint
    client.upsert(
        collection_name = CSV_COLLECTION,
        points = [
            PointStruct(
                id      = point_id,
                vector  = {"dense": dense_vector},
                payload = {
                    "file_name"     : file_name,
                    "sharepoint_url": sharepoint_url,
                    "summary"       : summary,
                    "file_hash"     : curr_hash,
                    "columns"       : df_info["columns"],
                }
            )
        ],
        wait = True,
    )

    print(f"[CSV INDEX] Done     '{file_name}' → saved in Qdrant only.")
    return point_id


def index_multiple_urls(urls: list[str], force: bool = False) -> list[str]:
    """
    Index multiple SharePoint CSV URLs.
    Skips unchanged files automatically.

    Args:
        urls  : list of SharePoint file URLs
        force : re-index all even if unchanged
    """
    if not urls:
        print("[CSV INDEX] No URLs provided.")
        return []

    print(f"\n[CSV INDEX] Indexing {len(urls)} file(s)...")
    ids     = []
    errors  = []

    for url in urls:
        try:
            pid = index_csv_from_url(url, force=force)
            ids.append(pid)
        except Exception as e:
            print(f"[CSV INDEX] ERROR on '{_url_to_filename(url)}': {e}")
            errors.append({"url": url, "error": str(e)})

    print(f"\n[CSV INDEX] ── Index Report ─────────────────────────")
    print(f"[CSV INDEX] Total    : {len(urls)}")
    print(f"[CSV INDEX] Indexed  : {len(ids)}")
    print(f"[CSV INDEX] Errors   : {len(errors)}")
    if errors:
        for e in errors:
            print(f"[CSV INDEX]   {_url_to_filename(e['url'])}: {e['error']}")
    print(f"[CSV INDEX] ─────────────────────────────────────────\n")

    return ids


def list_indexed_csvs():
    """Print all CSVs currently indexed in Qdrant."""
    _ensure_csv_collection()
    client = _get_qdrant()

    results = client.scroll(
        collection_name = CSV_COLLECTION,
        with_payload    = True,
        limit           = 100,
    )

    points = results[0]
    if not points:
        print("[CSV INDEX] No CSVs indexed yet.")
        return

    print(f"\n[CSV INDEX] ── Indexed CSVs ─────────────────────────")
    for p in points:
        print(f"  file     : {p.payload['file_name']}")
        print(f"  columns  : {p.payload['columns']}")
        print(f"  summary  : {p.payload['summary'][:80]}...")
        print(f"  url      : {p.payload['sharepoint_url'][:60]}...")
        print()
    print(f"[CSV INDEX] Total: {len(points)} file(s)")
    print(f"[CSV INDEX] ─────────────────────────────────────────\n")


# ── ROUTING ───────────────────────────────────────────────────────────────────

def detect_csv(question: str) -> tuple[str, str] | tuple[None, None]:
    """
    Embed question → search Qdrant → return (file_name, sharepoint_url).
    The URL comes from Qdrant payload — no local paths involved.
    """
    _ensure_csv_collection()
    client = _get_qdrant()

    info = client.get_collection(CSV_COLLECTION)
    if info.points_count == 0:
        print("[ROUTER] No CSVs indexed. Run index_multiple_urls() first.")
        return None, None

    dense_vector, _ = embed_query(question)

    results = client.search(
        collection_name = CSV_COLLECTION,
        query_vector    = ("dense", dense_vector),
        limit           = 1,
        with_payload    = True,
    )

    if not results:
        print("[ROUTER] No matching CSV found.")
        return None, None

    best = results[0]
    print(f"[ROUTER] Best match : '{best.payload['file_name']}'")
    print(f"[ROUTER] Score      : {best.score:.4f}")
    print(f"[ROUTER] Summary    : {best.payload['summary'][:80]}...")

    return best.payload["file_name"], best.payload["sharepoint_url"]


# ── SANDBOX ───────────────────────────────────────────────────────────────────

def _safe_getitem(obj, key):
    return obj[key]


class _PrintCollector:
    def __init__(self):
        self._lines = []

    def __call__(self, *args, **kwargs):
        self._lines.append(" ".join(str(a) for a in args))

    def _call_print(self, *args, **kwargs):
        self._lines.append(" ".join(str(a) for a in args))

    def __getattr__(self, name):
        return lambda *a, **k: None

    def get_output(self):
        return "\n".join(self._lines)


def _build_safe_globals(df: pd.DataFrame) -> dict:
    import math, statistics

    ALLOWED_BUILTINS = [
        "len", "range", "enumerate", "zip", "map", "filter",
        "sorted", "reversed", "list", "dict", "set", "tuple",
        "str", "int", "float", "bool", "round", "abs", "sum",
        "min", "max", "type", "isinstance", "repr", "any", "all",
    ]
    safe_bi = {k: getattr(builtins, k) for k in ALLOWED_BUILTINS
               if hasattr(builtins, k)}
    _print  = _PrintCollector()

    return {
        "__builtins__"          : safe_bi,
        "pd"                    : pd,
        "math"                  : math,
        "statistics"            : statistics,
        "df"                    : df,
        "print"                 : _print,
        "_getattr_"             : safer_getattr,
        "_getitem_"             : _safe_getitem,
        "_write_"               : lambda x: x,
        "_inplacevar_"          : lambda op, x, y: x,
        "_getiter_"             : iter,
        "_iter_unpack_sequence_": guarded_iter_unpack_sequence,
        "_print_"               : _print,
    }


def execute_code(code: str, df: pd.DataFrame) -> dict:
    try:
        byte_code = compile_restricted(
            code, filename="<llm_code>", mode="exec"
        )
    except SyntaxError as e:
        return {
            "success": False, "output": "",
            "result": None, "error": f"Syntax error: {e}"
        }

    safe_env   = _build_safe_globals(df)
    local_vars = {}

    try:
        exec(byte_code, safe_env, local_vars)
    except Exception as e:
        return {
            "success": False, "output": "",
            "result": None, "error": f"{type(e).__name__}: {e}"
        }

    _print = safe_env.get("_print")
    output = _print.get_output() if _print is not None else ""

    return {
        "success": True,
        "output" : output.strip(),
        "result" : local_vars.get("result"),
        "error"  : None,
    }


# ── CODE + ANSWER GENERATORS ──────────────────────────────────────────────────

def generate_pandas_code(question: str, df_info: dict) -> str:
    system_prompt = """You are a pandas code generator.
You receive a question about a dataframe and its schema.
Write Python code to answer the question.

Rules:
- The dataframe is always called df
- Store the final answer in a variable called result
- Do NOT use print() — only store the answer in result
- Do NOT use imports — pd, math, statistics are already available
- Do NOT use os, sys, subprocess, open(), or any file operations
- Return ONLY raw Python code — no markdown, no explanation"""

    user_message = (
        f"Question: {question}\n\n"
        f"Dataframe schema:\n"
        f"  Table   : {df_info['table']}\n"
        f"  Shape   : {df_info['shape'][0]} rows x {df_info['shape'][1]} cols\n"
        f"  Columns : {json.dumps(df_info['dtypes'], indent=4)}\n"
        f"  Sample  : {json.dumps(df_info['sample'], indent=4)}\n\n"
        f"Write pandas code. Store the answer in result."
    )

    response = _groq_client.chat.completions.create(
        model      = GROQ_VISION_MODEL,
        messages   = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ],
        max_tokens  = 1024,
        temperature = 0.0,
    )

    code = response.choices[0].message.content.strip()
    if code.startswith("```"):
        lines = code.split("\n")
        code  = "\n".join(
            lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
        )
    return code


def generate_answer(question: str, code: str, execution: dict) -> str:
    result_text = (
        f"stdout : {execution['output']}\nresult : {execution['result']}"
        if execution["success"]
        else f"Error: {execution['error']}"
    )

    response = _groq_client.chat.completions.create(
        model      = GROQ_VISION_MODEL,
        messages   = [{
            "role"   : "user",
            "content": (
                f"Question: {question}\n\n"
                f"Code that ran:\n{code}\n\n"
                f"Execution result:\n{result_text}\n\n"
                "Give a clear, direct, concise answer. No preamble."
            ),
        }],
        max_tokens  = 512,
        temperature = 0.3,
    )
    return response.choices[0].message.content.strip()


# ── MAIN PIPELINE ─────────────────────────────────────────────────────────────

def ask(question: str, verbose: bool = True) -> dict:
    """
    Full pipeline — nothing touches disk at any point:
      1. Embed question → Qdrant finds best matching CSV URL
      2. Download CSV from SharePoint into RAM (io.BytesIO)
      3. LLM generates pandas code
      4. Sandbox executes code on DataFrame
      5. LLM formats the answer
      6. DataFrame discarded from RAM
    """
    if verbose:
        print(f"\n[ROUTER] Question: \"{question}\"")

    file_name, sharepoint_url = detect_csv(question)

    if file_name is None:
        return {
            "question": question,
            "file"    : None,
            "answer"  : (
                "No CSV files indexed. "
                "Run index_multiple_urls() first."
            ),
        }

    # download from SharePoint → RAM only, never disk
    df      = load_csv_from_url(sharepoint_url)
    df_info = get_df_info(df, Path(file_name).stem)

    if verbose:
        print(f"[EXECUTOR] File  : {file_name}")
        print(f"[EXECUTOR] Shape : "
              f"{df_info['shape'][0]} rows x {df_info['shape'][1]} cols")

    code = generate_pandas_code(question, df_info)

    if verbose:
        print(f"[EXECUTOR] Code:\n{'─'*50}\n{code}\n{'─'*50}")

    execution = execute_code(code, df)

    if verbose:
        if execution["success"]:
            print(f"[EXECUTOR] Result : {execution['result']}")
        else:
            print(f"[EXECUTOR] Error  : {execution['error']}")

    answer = generate_answer(question, code, execution)

    if verbose:
        print(f"[EXECUTOR] Answer : {answer}")

    return {
        "question" : question,
        "file"     : file_name,
        "code"     : code,
        "execution": execution,
        "answer"   : answer,
    }


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── Step 1: paste your SharePoint CSV URLs here ───────────────────────────
    SHAREPOINT_URLS = [
        "https://yourcompany.sharepoint.com/sites/mysite/Shared Documents/sales.csv",
        "https://yourcompany.sharepoint.com/sites/mysite/Shared Documents/hr_data.csv",
        "https://yourcompany.sharepoint.com/sites/mysite/Shared Documents/finance.csv",
    ]

    # ── Step 2: index them (skips unchanged files on next run) ────────────────
    index_multiple_urls(SHAREPOINT_URLS)

    # ── Step 3: see what is indexed ───────────────────────────────────────────
    list_indexed_csvs()

    # ── Step 4: ask questions — routing is automatic ──────────────────────────
    questions = [
        "What is the total sales by region?",
        "What is the average employee salary?",
        "What is the net profit for Q3?",
    ]

    for q in questions:
        result = ask(q)
        print(f"\n  Q    : {result['question']}")
        print(f"  File : {result['file']}")
        print(f"  A    : {result['answer']}")