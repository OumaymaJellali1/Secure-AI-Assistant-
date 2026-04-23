import sys, os, json, traceback, tempfile
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100 MB


def _flat(result) -> list:
    if isinstance(result, list):
        return result
    if isinstance(result, dict):
        return result.get("chunks", [])
    return []


def _run(file, chunker_fn):
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name
    try:
        result = chunker_fn(tmp_path)
        return _flat(result)
    finally:
        os.unlink(tmp_path)


# ── Index ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


# ── Chunker route ─────────────────────────────────────────────────────────────

@app.route("/chunk", methods=["POST"])
def chunk():
    file    = request.files.get("file")
    chunker = request.form.get("chunker", "")

    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    try:
        if chunker == "pdf_digital":
            from File_processing.digital_pdf import extract_digital_pdf
            from File_processing.cleaner     import clean
            from chunking.pdf_chunker        import chunk_pdf
            def fn(p):
                out = extract_digital_pdf(p)
                out = clean(out)
                return chunk_pdf(out)

        elif chunker == "pdf_scanned":
            from File_processing.scanned_pdf      import extract_scanned_pdf
            from chunking.scanned_pdf_chunker import chunk_scanned_pdf
            def fn(p):
                parsed = extract_scanned_pdf(p)
                return chunk_scanned_pdf(parsed)

        elif chunker == "docx":
            from chunking.docx_chunker import chunk_docx
            fn = chunk_docx

        elif chunker == "eml":
            from chunking.eml_chunker import chunk_eml
            fn = chunk_eml

        elif chunker == "pptx":
            from chunking.pptx_chunker import extract_pptx
            fn = extract_pptx

        elif chunker == "txt":
            from File_processing.txt_handler import extract_txt
            from File_processing.cleaner     import clean
            from chunking.txt_chunker        import chunk_txt
            def fn(p):
                parsed   = extract_txt(p)
                filename = parsed["metadata"].get("file_name", os.path.basename(p))
                pre = {
                    "filename": filename,
                    "chunks": [{"type": "text", "content": parsed["text"],
                                "metadata": parsed["metadata"]}]
                }
                cleaned = clean(pre)
                return chunk_txt(cleaned)

        else:
            return jsonify({"error": f"Unknown chunker: {chunker}"}), 400

        chunks = _run(file, fn)
        return jsonify({"chunks": chunks, "total": len(chunks)})

    except Exception:
        return jsonify({"error": traceback.format_exc()}), 500


# ── CSV routes ────────────────────────────────────────────────────────────────

@app.route("/csv/import", methods=["POST"])
def csv_import():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400
    try:
        from csv_processing.csv_executor_postgres import import_csv
        basename = os.path.splitext(file.filename)[0]
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv",
                                         prefix=basename + "_") as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name
        table_name = import_csv(tmp_path)
        os.unlink(tmp_path)
        return jsonify({"table": table_name, "message": f"Imported as '{table_name}'"})
    except Exception:
        return jsonify({"error": traceback.format_exc()}), 500


@app.route("/csv/tables", methods=["GET"])
def csv_tables():
    try:
        from csv_processing.csv_executor_postgres import list_tables, get_table_info
        tables = list_tables()
        info   = []
        for t in tables:
            try:
                ti = get_table_info(t)
                info.append({
                    "name"   : t,
                    "rows"   : ti["shape"][0],
                    "cols"   : ti["shape"][1],
                    "columns": ti["columns"],
                    "sample" : ti["sample"],
                })
            except Exception:
                info.append({"name": t, "rows": "?", "cols": "?", "columns": [], "sample": []})
        return jsonify({"tables": info})
    except Exception:
        return jsonify({"error": traceback.format_exc()}), 500


@app.route("/csv/query", methods=["POST"])
def csv_query():
    body     = request.get_json() or {}
    table    = body.get("table")
    question = body.get("question", "").strip()

    if not table or not question:
        return jsonify({"error": "Both 'table' and 'question' are required."}), 400

    try:
        from csv_processing.csv_executor_postgres import query_table
        result = query_table(table, question, verbose=False)
        return jsonify(result)
    except Exception:
        return jsonify({"error": traceback.format_exc()}), 500


@app.route("/csv/registry", methods=["GET"])
def csv_registry():
    try:
        from csv_processing.csv_executor_postgres import _ensure_registry, get_engine
        from sqlalchemy import text as sa_text
        _ensure_registry()
        engine = get_engine()
        with engine.connect() as conn:
            rows = conn.execute(
                sa_text("SELECT table_name, file_hash, imported_at FROM _csv_registry ORDER BY imported_at DESC")
            ).fetchall()
        data = [{"table": r[0], "hash": r[1][:8] + "...", "imported_at": str(r[2])} for r in rows]
        return jsonify({"registry": data})
    except Exception:
        return jsonify({"error": traceback.format_exc()}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)