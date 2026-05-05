# csv_agent.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import re
import hashlib
import builtins
import chardet
import pandas as pd

from pathlib import Path
from groq import Groq
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct,
    SparseVectorParams, SparseIndexParams, SparseVector
)
from RestrictedPython import compile_restricted
from RestrictedPython.Guards import guarded_iter_unpack_sequence, safer_getattr

from embedding.embedder import embed_query
from config import GROQ_API_KEY, GROQ_VISION_MODEL

import uuid


# ── CONFIG ────────────────────────────────────────────────────────────────────

QDRANT_URL        = "http://localhost:6333"
CSV_COLLECTION    = "csv_index"
DENSE_DIM         = 1024                  # BGE-M3
ROUTING_THRESHOLD = 0.45                  # below this → no confident match

_groq_client = Groq(api_key=GROQ_API_KEY)
_qdrant      = None


# ── QDRANT CLIENT ─────────────────────────────────────────────────────────────

def _get_qdrant() -> QdrantClient:
    global _qdrant
    if _qdrant is None:
        _qdrant = QdrantClient(url=QDRANT_URL)
    return _qdrant


# ── COLLECTION SETUP ──────────────────────────────────────────────────────────

def _ensure_csv_collection():
    client     = _get_qdrant()
    existing   = [c.name for c in client.get_collections().collections]

    if CSV_COLLECTION in existing:
        return

    print(f"[CSV INDEX] Creating collection '{CSV_COLLECTION}'...")
    client.create_collection(
        collection_name = CSV_COLLECTION,
        vectors_config  = {
            "dense": VectorParams(
                size     = DENSE_DIM,
                distance = Distance.COSINE,
            )
        },
    )
    print(f"[CSV INDEX] Collection ready.")


# ── CSV LOADING ───────────────────────────────────────────────────────────────

def _slugify(name: str) -> str:
    name = Path(name).stem
    name = re.sub(r"[^a-zA-Z0-9_]+", "", name).strip("_")
    if name and name[0].isdigit():
        name = "t_" + name
    return name


def load_csv(file_path: str) -> pd.DataFrame:
    """Load CSV → DataFrame with cleaned column names."""
    with open(file_path, "rb") as f:
        encoding = chardet.detect(f.read(50_000)).get("encoding") or "utf-8"

    df = pd.read_csv(file_path, encoding=encoding)
    df.columns = [
        re.sub(r"[^a-z0-9]+", "", col.lower()).strip()
        for col in df.columns
    ]
    return df


def get_df_info(df: pd.DataFrame, name: str) -> dict:
    """Return schema + 3 sample rows."""
    return {
        "table"  : name,
        "shape"  : df.shape,
        "columns": df.columns.tolist(),
        "dtypes" : {col: str(dt) for col, dt in df.dtypes.items()},
        "sample" : df.head(3).to_dict(orient="records"),
    }


def _compute_hash(file_path: str) -> str:
    md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            md5.update(chunk)
    return md5.hexdigest()


# ── FIX #1: IMPROVED SUMMARY GENERATOR ───────────────────────────────────────

def _summarize_csv(file_path: str, df_info: dict) -> str:
    """
    Generate a keyword-rich, routing-optimised summary.
    The summary is embedded into Qdrant — it must surface the right file
    even when user questions use different vocabulary.
    """
    response = _groq_client.chat.completions.create(
        model    = GROQ_VISION_MODEL,
        messages = [{
            "role"   : "user",
            "content": (
                f"CSV file: '{Path(file_path).name}'\n"
                f"Columns : {json.dumps(df_info['dtypes'], indent=2)}\n"
                f"Sample  : {json.dumps(df_info['sample'], indent=2)}\n\n"
                "Write a structured dataset description for semantic search routing.\n\n"
                "Include:\n"
                "- domain keywords (very important for search matching)\n"
                "- ALL column names listed explicitly\n"
                "- 3 concrete example questions this dataset can answer\n"
                "- what the dataset CANNOT answer (to prevent wrong routing)\n\n"
                "Be very explicit and keyword-rich. Do NOT be vague.\n"
                "Use plain sentences, not markdown headers or bullets."
            )
        }],
        max_tokens  = 300,
        temperature = 0.0,
    )
    summary = response.choices[0].message.content.strip()

    # FIX #2: Append raw column names to make embeddings strongly aware of schema
    summary += "\n\nCOLUMNS: " + ", ".join(df_info["columns"])

    return summary


# ── INDEXING ──────────────────────────────────────────────────────────────────

def index_csv(file_path: str, force: bool = False) -> str:
    """
    Index a single CSV file into Qdrant.
    Skips unchanged files, re-indexes if hash changed.
    """
    _ensure_csv_collection()

    file_path  = str(file_path)
    file_name  = Path(file_path).name
    point_id   = str(uuid.uuid5(uuid.NAMESPACE_DNS, file_name))
    curr_hash  = _compute_hash(file_path)
    client     = _get_qdrant()

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
                print(f"[CSV INDEX] Updated  '{file_name}' — file changed.")
        else:
            print(f"[CSV INDEX] New file '{file_name}' — indexing...")

    df       = load_csv(file_path)
    df_info  = get_df_info(df, _slugify(file_path))
    summary  = _summarize_csv(file_path, df_info)

    print(f"[CSV INDEX] Summary  : {summary[:120]}...")

    dense_vector, _ = embed_query(summary)

    client.upsert(
        collection_name = CSV_COLLECTION,
        points = [
            PointStruct(
                id      = point_id,
                vector  = {"dense": dense_vector},
                payload = {
                    "file_path" : file_path,
                    "file_name" : file_name,
                    "summary"   : summary,
                    "file_hash" : curr_hash,
                    "columns"   : df_info["columns"],
                }
            )
        ],
        wait = True,
    )

    print(f"[CSV INDEX] Indexed  '{file_name}' → point_id={point_id[:8]}...")
    return point_id


def index_csv_folder(folder: str, force: bool = False) -> list[str]:
    """Index all CSVs in a folder. Skips unchanged files automatically."""
    csv_files = list(Path(folder).glob("*.csv"))

    if not csv_files:
        print(f"[CSV INDEX] No CSV files found in '{folder}'")
        return []

    ids = []
    for f in csv_files:
        pid = index_csv(str(f), force=force)
        ids.append(pid)

    print(f"[CSV INDEX] Done — {len(ids)} file(s) processed.")
    return ids


# ── FIX: ROUTING WITH CONFIDENCE THRESHOLD ───────────────────────────────────

def detect_csv(question: str) -> str | None:
    """
    Embed question → Qdrant search → return best file path.
    Returns None if best score is below ROUTING_THRESHOLD.
    """
    _ensure_csv_collection()
    client = _get_qdrant()

    info = client.get_collection(CSV_COLLECTION)
    if info.points_count == 0:
        print("[ROUTER] No CSVs indexed. Run index_csv_folder() first.")
        return None

    dense_vector, _ = embed_query(question)

    results = client.query_points(
        collection_name = CSV_COLLECTION,
        query           = dense_vector,
        using           = "dense",
        limit           = 1,
        with_payload    = True
    ).points

    if not results:
        print("[ROUTER] No results found.")
        return None

    best      = results[0]
    file_path = best.payload["file_path"]
    score     = best.score
    columns   = best.payload.get("columns", [])

    print(f"[ROUTER] Best match : '{best.payload['file_name']}'")
    print(f"[ROUTER] Score      : {score:.4f}")
    print(f"[ROUTER] Columns    : {columns}")

    # Confidence threshold guard
    if score < ROUTING_THRESHOLD:
        print(f"[ROUTER] Score {score:.4f} below threshold {ROUTING_THRESHOLD} — no confident match.")
        return None

    return file_path


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
    safe_bi = {k: getattr(builtins, k) for k in ALLOWED_BUILTINS if hasattr(builtins, k)}
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
        byte_code = compile_restricted(code, filename="<llm_code>", mode="exec")
    except SyntaxError as e:
        return {"success": False, "output": "", "result": None, "error": f"Syntax error: {e}"}

    safe_env   = _build_safe_globals(df)
    local_vars = {}

    try:
        exec(byte_code, safe_env, local_vars)
    except Exception as e:
        return {"success": False, "output": "", "result": None, "error": f"{type(e).__name__}: {e}"}

    _print = safe_env.get("_print")
    output = _print.get_output() if _print is not None else ""

    return {
        "success": True,
        "output" : output.strip(),
        "result" : local_vars.get("result"),
        "error"  : None,
    }


# ── FIX #3: SCHEMA-ENFORCED CODE GENERATOR ───────────────────────────────────

def _extract_code(raw: str) -> str:
    """Strip markdown fences from LLM output reliably."""
    if "```" in raw:
        match = re.search(r"```(?:python)?\n(.*?)```", raw, re.DOTALL)
        if match:
            return match.group(1).strip()
    return raw.strip()


def generate_pandas_code(question: str, df_info: dict) -> str:
    system_prompt = f"""You are a pandas code generator. You must be strictly schema-aware.

DATAFRAME SCHEMA:
  Table   : {df_info['table']}
  Columns : {json.dumps(df_info['dtypes'], indent=2)}
  Sample  : {json.dumps(df_info['sample'], indent=2)}

ABSOLUTE RULES:
- The dataframe is always called df
- Store the final answer in a variable called result
- ONLY use columns that exist in the schema above — never invent column names
- If the question asks for a column that does not exist, set:
    result = "Column not available: <column_name> is not in this dataset"
- Do NOT use print() — only use result
- Do NOT use imports — pd, math, statistics are already available
- Do NOT use os, sys, subprocess, open(), or file operations
- Return ONLY raw Python code — no markdown, no explanation, no prose before or after"""

    user_message = f"Question: {question}\n\nWrite pandas code. Store the answer in result."

    response = _groq_client.chat.completions.create(
        model      = GROQ_VISION_MODEL,
        messages   = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ],
        max_tokens  = 1024,
        temperature = 0.0,
    )

    return _extract_code(response.choices[0].message.content)


# ── FIX #4: COLUMN VALIDATION BEFORE EXECUTION ───────────────────────────────

def validate_columns(code: str, df: pd.DataFrame) -> list[str]:
    """
    Extract all df['col'] references from generated code.
    Returns list of invalid column names found.
    """
    referenced = re.findall(r"""df\[['"]([^'"]+)['"]\]""", code)
    invalid    = [col for col in referenced if col not in df.columns]
    return invalid


# ── FIX: CODE RETRY LOOP ──────────────────────────────────────────────────────

def _generate_fix(question: str, code: str, error: str, df_info: dict) -> str:
    """Ask the LLM to fix a failing code attempt."""
    prompt = (
        f"This pandas code failed:\n{code}\n\n"
        f"Error: {error}\n\n"
        f"Available columns ONLY: {json.dumps(df_info['dtypes'], indent=2)}\n"
        f"Sample rows: {json.dumps(df_info['sample'], indent=2)}\n\n"
        "Fix the code. Use ONLY the columns listed above.\n"
        "Return ONLY raw Python code — no markdown, no explanation.\n"
        "Store the answer in result."
    )
    response = _groq_client.chat.completions.create(
        model    = GROQ_VISION_MODEL,
        messages = [{"role": "user", "content": prompt}],
        max_tokens  = 1024,
        temperature = 0.0,
    )
    return _extract_code(response.choices[0].message.content)


# ── ANSWER GENERATOR ──────────────────────────────────────────────────────────

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
                "Give a clear, direct, concise answer in plain English. No preamble. "
                "Do not include code snippets in your answer."
            ),
        }],
        max_tokens  = 256,
        temperature = 0.3,
    )
    return response.choices[0].message.content.strip()


# ── MAIN PIPELINE ─────────────────────────────────────────────────────────────

def query_csv(
    file_path : str,
    question  : str,
    verbose   : bool = True,
    max_retries: int = 2,
) -> dict:
    """Direct query on a specific CSV — no routing needed."""
    df      = load_csv(file_path)
    df_info = get_df_info(df, _slugify(file_path))

    if verbose:
        print(f"\n[EXECUTOR] File    : {file_path}")
        print(f"[EXECUTOR] Shape   : {df_info['shape'][0]} rows × {df_info['shape'][1]} cols")
        print(f"[EXECUTOR] Columns : {df_info['columns']}")
        print(f"[EXECUTOR] Question: {question}")

    code = generate_pandas_code(question, df_info)

    if verbose:
        print(f"[EXECUTOR] Code:\n{'─'*50}\n{code}\n{'─'*50}")

    # FIX #4: Validate columns before running
    invalid_cols = validate_columns(code, df)
    if invalid_cols:
        print(f"[EXECUTOR] ⚠ Hallucinated columns detected: {invalid_cols} — triggering fix...")
        code = _generate_fix(
            question,
            code,
            f"Columns do not exist in dataset: {invalid_cols}. "
            f"Available columns: {df_info['columns']}",
            df_info,
        )
        if verbose:
            print(f"[EXECUTOR] Fixed Code:\n{'─'*50}\n{code}\n{'─'*50}")

    execution = execute_code(code, df)

    # Retry loop on execution failure
    for attempt in range(max_retries):
        if execution["success"]:
            break

        print(f"[EXECUTOR] Retry {attempt + 1}/{max_retries} — Error: {execution['error']}")
        code      = _generate_fix(question, code, execution["error"], df_info)
        if verbose:
            print(f"[EXECUTOR] Retry Code:\n{'─'*50}\n{code}\n{'─'*50}")
        execution = execute_code(code, df)

    if verbose:
        if execution["success"]:
            print(f"[EXECUTOR] Result  : {execution['result']}")
        else:
            print(f"[EXECUTOR] Failed after {max_retries} retries: {execution['error']}")

    answer = generate_answer(question, code, execution)

    if verbose:
        print(f"[EXECUTOR] Answer  : {answer}")

    return {
        "question" : question,
        "file"     : file_path,
        "code"     : code,
        "execution": execution,
        "answer"   : answer,
    }


def ask(question: str, verbose: bool = True) -> dict:
    """
    Full pipeline:
      1. Embed question → Qdrant finds best CSV (with confidence threshold)
      2. Load CSV → pandas
      3. LLM generates schema-aware code → column validation → sandbox
      4. Auto-retry on failure (up to 2 times)
      5. LLM formats final answer
    """
    if verbose:
        print(f"\n[ROUTER] Question: \"{question}\"")

    file_path = detect_csv(question)

    if file_path is None:
        return {
            "question": question,
            "file"    : None,
            "answer"  : (
                "No CSV matched your question with sufficient confidence. "
                "Either no relevant CSV is indexed, or the question is outside "
                "the scope of available datasets."
            ),
        }

    return query_csv(file_path, question, verbose)


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Step 1 — re-index with force=True to regenerate improved summaries
    # Use force=True once after upgrading to get the new keyword-rich summaries
    index_csv_folder("data_csv", force=False)

    # Step 2 — ask questions (routing is automatic)
    questions = [
        "What percentage of patients have heart disease?",
        "What is the total revenue by region?",
        "What is the average salary per department?",
    ]

    for q in questions:
        result = ask(q)
        print(f"\n  Q    : {result['question']}")
        print(f"  File : {result['file']}")
        print(f"  A    : {result['answer']}")