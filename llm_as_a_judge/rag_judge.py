"""
eval_interactive.py
───────────────────
Interactive RAG evaluator — plugs into YOUR real pipeline.

Flow per question:
  1. You type a question (or load from --batch file)
  2. Retriever.search()   → candidate chunks  (Qdrant hybrid)
  3. Reranker.rerank()    → top-N chunks      (BGE-M3 cross-encoder)
  4. Generator.generate() → answer            (Qwen2.5 via Ollama)
  5. Groq judge scores    → Faithfulness / Context Relevance / Answer Relevance
                            + optional Correctness (if --reference provided)
                            + per-chunk relevance breakdown
  6. Results printed + appended to eval_results.csv

Usage:
  python eval_interactive.py
  python eval_interactive.py --retrieval_pool 20 --top_n 5 --mode hybrid
  python eval_interactive.py --batch questions.txt
  python eval_interactive.py --batch questions.csv   # columns: question[,reference]
"""

from __future__ import annotations

import os
import csv
import json
import re
import time
import sys
import random
from datetime import datetime
from typing import Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from groq import Groq

from retrieve.retriever import Retriever
from reranking.reranker import Reranker
from answer_generation.generator import Generator  # Ollama/local generator



JUDGE_MODEL = "qwen/qwen3-32b"
RESULTS_FILE   = "eval_results.csv"

RETRIEVAL_POOL = 20      
TOP_N          = 5       
SEARCH_MODE    = "hybrid" 

MAX_RETRIES    = 5
RETRY_BASE_S   = 1.5     
RETRY_JITTER   = 0.5     

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


# ── JUDGE PROMPTS 

FAITHFULNESS_PROMPT = """You are an expert evaluator for RAG (Retrieval Augmented Generation) systems.

Your task is to evaluate if the ANSWER is faithful to the CONTEXT — meaning the answer only uses information present in the context and does not hallucinate or add outside knowledge.

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
{answer}

Evaluate faithfulness using this scale:
1 - The answer contradicts or ignores the context entirely
2 - The answer has major hallucinations or unsupported claims
3 - The answer is mostly faithful with minor unsupported details
4 - The answer is faithful with very minor issues
5 - The answer is completely faithful to the context

First explain your reasoning in 2-3 sentences, then give the score.

Respond in this exact JSON format:
{{"reasoning": "your reasoning here", "score": <1-5>}}"""


CONTEXT_RELEVANCE_PROMPT = """You are an expert evaluator for RAG (Retrieval Augmented Generation) systems.

Your task is to evaluate if the RETRIEVED CHUNKS are relevant to the QUESTION — meaning the chunks actually contain information needed to answer the question.

QUESTION:
{query}

RETRIEVED CHUNKS:
{context}

Evaluate context relevance using this scale:
1 - The chunks are completely irrelevant to the question
2 - The chunks have little relevant information
3 - The chunks are partially relevant, some useful information
4 - The chunks are mostly relevant with minor irrelevant parts
5 - The chunks are highly relevant and contain exactly what is needed

First explain your reasoning in 2-3 sentences, then give the score.

Respond in this exact JSON format:
{{"reasoning": "your reasoning here", "score": <1-5>}}"""


ANSWER_RELEVANCE_PROMPT = """You are an expert evaluator for RAG (Retrieval Augmented Generation) systems.

Your task is to evaluate if the ANSWER actually addresses the QUESTION — meaning the answer is useful, complete, and directly responds to what was asked.

QUESTION:
{query}

ANSWER:
{answer}

Evaluate answer relevance using this scale:
1 - The answer does not address the question at all
2 - The answer barely addresses the question
3 - The answer partially addresses the question
4 - The answer mostly addresses the question well
5 - The answer completely and directly addresses the question

First explain your reasoning in 2-3 sentences, then give the score.

Respond in this exact JSON format:
{{"reasoning": "your reasoning here", "score": <1-5>}}"""


CORRECTNESS_PROMPT = """You are an expert evaluator for RAG (Retrieval Augmented Generation) systems.

Your task is to evaluate if the ANSWER is factually correct compared to the REFERENCE ANSWER.

QUESTION:
{query}

REFERENCE ANSWER (ground truth):
{reference}

ANSWER (to evaluate):
{answer}

Evaluate correctness using this scale:
1 - The answer is completely wrong or contradicts the reference
2 - The answer has major factual errors compared to the reference
3 - The answer is partially correct, missing key facts from the reference
4 - The answer is mostly correct with minor omissions or inaccuracies
5 - The answer is fully correct and consistent with the reference

First explain your reasoning in 2-3 sentences, then give the score.

Respond in this exact JSON format:
{{"reasoning": "your reasoning here", "score": <1-5>}}"""


CHUNK_RELEVANCE_PROMPT = """You are an expert evaluator for RAG systems.

Rate how relevant the following CHUNK is for answering the QUESTION.

QUESTION:
{query}

CHUNK [{chunk_idx}]:
{chunk}

Score (1-5):
1 - Completely irrelevant
2 - Slightly relevant, tangential
3 - Partially relevant, some useful info
4 - Mostly relevant, directly useful
5 - Perfectly relevant, exactly what is needed

Respond in this exact JSON format only:
{{"score": <1-5>, "reasoning": "one sentence"}}"""



def call_judge(prompt: str) -> dict:
    """
    Send a prompt to the Groq judge and parse the JSON response.
    Retries with exponential backoff + jitter on rate-limit or transient errors.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = groq_client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.choices[0].message.content.strip()

            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()

            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                match = re.search(r'"score"\s*:\s*(\d)', raw)
                score = int(match.group(1)) if match else 0
                return {"reasoning": raw, "score": score}

        except Exception as e:
            err_str = str(e).lower()
            is_rate_limit = "rate" in err_str or "429" in err_str or "limit" in err_str

            if attempt == MAX_RETRIES:
                print(f"\n  [Judge failed after {MAX_RETRIES} attempts: {e}]")
                return {"reasoning": f"Judge error: {e}", "score": 0}

            wait = RETRY_BASE_S * (2 ** (attempt - 1)) + random.uniform(0, RETRY_JITTER)
            reason = "rate limit" if is_rate_limit else "transient error"
            print(f"\n  [Attempt {attempt}/{MAX_RETRIES} — {reason}, retrying in {wait:.1f}s...]")
            time.sleep(wait)

    return {"reasoning": "Unreachable", "score": 0}


# ── PER-CHUNK RELEVANCE SCORING 

def score_chunks(query: str, chunks: list[dict]) -> list[dict]:
    """
      Score each chunk individually for relevance.
    Returns a list of dicts with chunk index, score, and reasoning.
    """
    results = []
    for i, chunk in enumerate(chunks, 1):
        result = call_judge(
            CHUNK_RELEVANCE_PROMPT.format(
                query=query,
                chunk_idx=i,
                chunk=chunk["content"],
            )
        )
        results.append({
            "chunk_idx" : i,
            "score"     : result.get("score", 0),
            "reasoning" : result.get("reasoning", ""),
            "preview"   : chunk["content"][:120].replace("\n", " ") + "...",
        })
    return results



def evaluate(
    query     : str,
    chunks    : list[dict],
    answer    : str,
    reference : Optional[str] = None,
    latency_s : Optional[float] = None,
    tokens    : Optional[dict]  = None,
) -> dict:
    """
    Run all judges on a real pipeline result.
    chunks must be the reranked chunks (list of dicts with 'content' key).
    """
    context = "\n\n".join(
        f"[Chunk {i}]\n{c['content']}" for i, c in enumerate(chunks, 1)
    )

    print(f"\n{'─'*60}")
    print(f"  JUDGING YOUR PIPELINE OUTPUT")
    print(f"{'─'*60}")

    # Faithfulness
    print("  → Faithfulness ...", end=" ", flush=True)
    faithfulness = call_judge(
        FAITHFULNESS_PROMPT.format(context=context, query=query, answer=answer)
    )
    print(f"score: {faithfulness['score']}/5")

    # Context Relevance
    print("  → Context Relevance ...", end=" ", flush=True)
    context_relevance = call_judge(
        CONTEXT_RELEVANCE_PROMPT.format(context=context, query=query)
    )
    print(f"score: {context_relevance['score']}/5")

    # Answer Relevance
    print("  → Answer Relevance ...", end=" ", flush=True)
    answer_relevance = call_judge(
        ANSWER_RELEVANCE_PROMPT.format(query=query, answer=answer)
    )
    print(f"score: {answer_relevance['score']}/5")

    correctness = None
    if reference:
        print("  → Correctness ...", end=" ", flush=True)
        correctness = call_judge(
            CORRECTNESS_PROMPT.format(query=query, reference=reference, answer=answer)
        )
        print(f"score: {correctness['score']}/5")

    print("  → Per-chunk relevance ...")
    chunk_scores = score_chunks(query, chunks)

    # Average score
    scores = [
        faithfulness["score"],
        context_relevance["score"],
        answer_relevance["score"],
    ]
    if correctness:
        scores.append(correctness["score"])
    avg = round(sum(scores) / len(scores), 2)

    result = {
        "query"                      : query,
        "answer"                     : answer,
        "reference"                  : reference or "",
        "faithfulness_score"         : faithfulness["score"],
        "faithfulness_reasoning"     : faithfulness["reasoning"],
        "context_relevance_score"    : context_relevance["score"],
        "context_relevance_reasoning": context_relevance["reasoning"],
        "answer_relevance_score"     : answer_relevance["score"],
        "answer_relevance_reasoning" : answer_relevance["reasoning"],
        "correctness_score"          : correctness["score"] if correctness else "",
        "correctness_reasoning"      : correctness["reasoning"] if correctness else "",
        "avg_score"                  : avg,
        "num_chunks_retrieved"       : len(chunks),
        "latency_s"                  : latency_s if latency_s is not None else "",
        "total_tokens"               : tokens.get("total", "") if tokens else "",
        "prompt_tokens"              : tokens.get("prompt", "") if tokens else "",
        "completion_tokens"          : tokens.get("completion", "") if tokens else "",
        "chunk_scores_json"          : json.dumps(chunk_scores),
        "timestamp"                  : datetime.now().isoformat(),
    }

    return result, chunk_scores


# ── PRINT RESULT 

def print_eval_result(result: dict, chunk_scores: list[dict]) -> None:
    """Pretty-print scores + reasoning for one evaluated question."""
    print(f"\n{'='*60}")
    print(f"  EVALUATION SCORES")
    print(f"{'='*60}")
    print(f"  Faithfulness      : {result['faithfulness_score']}/5")
    print(f"    → {result['faithfulness_reasoning']}")
    print(f"\n  Context Relevance : {result['context_relevance_score']}/5")
    print(f"    → {result['context_relevance_reasoning']}")
    print(f"\n  Answer Relevance  : {result['answer_relevance_score']}/5")
    print(f"    → {result['answer_relevance_reasoning']}")

    if result["correctness_score"] != "":
        print(f"\n  Correctness       : {result['correctness_score']}/5")
        print(f"    → {result['correctness_reasoning']}")

    print(f"\n  Average Score     : {result['avg_score']}/5")

    print(f"\n  Per-Chunk Relevance:")
    for c in chunk_scores:
        bar = "█" * c["score"] + "░" * (5 - c["score"])
        print(f"    Chunk {c['chunk_idx']} [{bar}] {c['score']}/5 — {c['reasoning']}")
        print(f"           \"{c['preview']}\"")

    if result["latency_s"] != "":
        print(f"\n  Performance       : {result['latency_s']}s | "
              f"{result['total_tokens']} tokens total")

    print()
    if result["faithfulness_score"] < 3:
        print("    LOW FAITHFULNESS — model may be hallucinating outside the context")
    if result["context_relevance_score"] < 3:
        print("    LOW CONTEXT RELEVANCE — retriever/reranker returning wrong chunks")
    if result["answer_relevance_score"] < 3:
        print("   LOW ANSWER RELEVANCE — answer doesn't address the question")
    if result.get("correctness_score") and result["correctness_score"] < 3:
        print("    LOW CORRECTNESS — answer deviates from ground truth")

    low_chunks = [c for c in chunk_scores if c["score"] < 3]
    if low_chunks:
        idxs = ", ".join(str(c["chunk_idx"]) for c in low_chunks)
        print(f"    WEAK CHUNKS [{idxs}] — reranker kept low-quality chunks")

    if result["avg_score"] >= 4.0:
        print("   Looks good overall")

    print(f"{'='*60}\n")


# ── SAVE TO CSV 

def append_to_csv(result: dict, filepath: str = RESULTS_FILE) -> None:
    """Append one evaluation result to the CSV log."""
    file_exists = os.path.isfile(filepath)
    fieldnames  = list(result.keys())

    with open(filepath, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)

    print(f"  Saved → {filepath}")



def print_session_summary(session_results: list[dict]) -> None:
    """Print averages across all questions asked in this session."""
    if not session_results:
        return

    n           = len(session_results)
    avg_faith   = round(sum(r["faithfulness_score"]      for r in session_results) / n, 2)
    avg_ctx     = round(sum(r["context_relevance_score"] for r in session_results) / n, 2)
    avg_ans     = round(sum(r["answer_relevance_score"]  for r in session_results) / n, 2)
    avg_overall = round(sum(r["avg_score"]               for r in session_results) / n, 2)

    correct_results = [r for r in session_results if r.get("correctness_score") != ""]
    avg_correct = (
        round(sum(r["correctness_score"] for r in correct_results) / len(correct_results), 2)
        if correct_results else None
    )

    print(f"\n{'='*60}")
    print(f"  SESSION SUMMARY  ({n} question{'s' if n > 1 else ''} evaluated)")
    print(f"{'='*60}")
    print(f"  Faithfulness        : {avg_faith}/5")
    print(f"  Context Relevance   : {avg_ctx}/5")
    print(f"  Answer Relevance    : {avg_ans}/5")
    if avg_correct is not None:
        print(f"  Correctness         : {avg_correct}/5  ({len(correct_results)} w/ reference)")
    print(f"  Overall Average     : {avg_overall}/5")
    print(f"{'='*60}")

    if avg_faith < 3.0:
        print("    LOW FAITHFULNESS — check generation prompt or chunk quality")
    if avg_ctx < 3.0:
        print("    LOW CONTEXT RELEVANCE — tune retrieval_pool / reranker top_n")
    if avg_ans < 3.0:
        print("    LOW ANSWER RELEVANCE — check generation prompt instructions")
    if avg_correct is not None and avg_correct < 3.0:
        print("    LOW CORRECTNESS — answers diverge from ground truth frequently")
    if avg_overall >= 4.0:
        print("    Pipeline looks healthy overall")

    print(f"\n  Full log saved to: {RESULTS_FILE}\n")



def load_batch(filepath: str) -> list[tuple[str, Optional[str]]]:
    """
    Load questions (and optional references) from a file.

    Supported formats:
      - .txt  → one question per line
      - .csv  → columns: question[,reference]
    Returns list of (question, reference_or_None) tuples.
    """
    items: list[tuple[str, Optional[str]]] = []
    ext = os.path.splitext(filepath)[1].lower()

    with open(filepath, newline="", encoding="utf-8") as f:
        if ext == ".csv":
            reader = csv.reader(f)
            header = next(reader, None)  # skip header row if present
            for row in reader:
                if not row or not row[0].strip():
                    continue
                question  = row[0].strip()
                reference = row[1].strip() if len(row) > 1 and row[1].strip() else None
                items.append((question, reference))
        else:
            for line in f:
                line = line.strip()
                if line:
                    items.append((line, None))

    return items



def run_question(
    question     : str,
    reference    : Optional[str],
    retriever    : Retriever,
    reranker     : Reranker,
    generator    : Generator,
    retrieval_pool: int,
    top_n        : int,
    search_mode  : str,
) -> Optional[dict]:
    """Retrieve → Rerank → Generate → Judge one question. Returns result dict or None."""
    try:
        # ── STEP 1: RETRIEVE 
        print(f"\n[1/3] Retrieving (pool={retrieval_pool}, mode={search_mode})...")
        candidates = retriever.search(question, mode=search_mode, top_n=retrieval_pool)

        if not candidates:
            print("  [No chunks retrieved — try a different question]\n")
            return None

        # ── STEP 2: RERANK 
        print(f"[2/3] Reranking → top {top_n}...")
        chunks = reranker.rerank(question, candidates, top_n=top_n)

        # ── STEP 3: GENERATE 
        print(f"[3/3] Generating answer...\n")
        gen_result = generator.generate(question, chunks)
        answer     = gen_result["answer"]

        print(f"\n{'─'*60}")
        print(f"  ANSWER")
        print(f"{'─'*60}")
        print(answer)

        if gen_result.get("sources"):
            print(f"\n  Sources cited: {len(gen_result['sources'])}")
            for idx, src in enumerate(gen_result["sources"], 1):
                loc = src.get("source", "?")
                if src.get("page"):
                    loc += f", page {src['page']}"
                print(f"    [{idx}] {loc}")

        print(f"\n  [latency={gen_result['latency_s']}s | "
              f"tokens={gen_result['tokens']['total']} | "
              f"no_answer={gen_result['no_answer']}]")

        # ── STEP 4: JUDGE 
        eval_result, chunk_scores = evaluate(
            query     = question,
            chunks    = chunks,
            answer    = answer,
            reference = reference,
            latency_s = gen_result.get("latency_s"),
            tokens    = gen_result.get("tokens"),
        )

        print_eval_result(eval_result, chunk_scores)
        append_to_csv(eval_result)
        return eval_result

    except KeyboardInterrupt:
        print("\n[Interrupted — type 'exit' to quit]\n")
        return None
    except Exception as e:
        print(f"\n[ERROR] {e}\n")
        import traceback
        traceback.print_exc()
        return None



def main():
    args           = sys.argv[1:]
    retrieval_pool = RETRIEVAL_POOL
    top_n          = TOP_N
    search_mode    = SEARCH_MODE
    batch_file     = None

    i = 0
    while i < len(args):
        if args[i] == "--retrieval_pool" and i + 1 < len(args):
            retrieval_pool = int(args[i + 1]); i += 2
        elif args[i] == "--top_n" and i + 1 < len(args):
            top_n = int(args[i + 1]); i += 2
        elif args[i] == "--mode" and i + 1 < len(args):
            search_mode = args[i + 1]; i += 2
        elif args[i] == "--batch" and i + 1 < len(args):
            batch_file = args[i + 1]; i += 2
        else:
            i += 1

    print(f"\n{'='*60}")
    print(f"  RAG INTERACTIVE EVALUATOR")
    print(f"{'='*60}")
    print(f"  Loading your pipeline (one time)...")
    print(f"  retrieval_pool={retrieval_pool}  top_n={top_n}  mode={search_mode}")
    print(f"  Judge model: {JUDGE_MODEL}")
    if batch_file:
        print(f"  Batch file: {batch_file}")
    print(f"{'='*60}\n")

    retriever = Retriever()
    reranker  = Reranker()
    generator = Generator()

    session_results: list[dict] = []

    if batch_file:
        questions = load_batch(batch_file)
        print(f"  Loaded {len(questions)} question(s) from {batch_file}\n")
        for idx, (question, reference) in enumerate(questions, 1):
            print(f"\n{'─'*60}")
            print(f"  [{idx}/{len(questions)}] {question}")
            if reference:
                print(f"  Reference: {reference[:100]}{'...' if len(reference) > 100 else ''}")
            print(f"{'─'*60}")
            result = run_question(
                question, reference, retriever, reranker, generator,
                retrieval_pool, top_n, search_mode,
            )
            if result:
                session_results.append(result)

        print_session_summary(session_results)
        print("[Batch complete]\n")
        return

    print(f"\n{'='*60}")
    print(f"  READY. Type your question and press Enter.")
    print(f"  Commands:")
    print(f"    summary          → session stats")
    print(f"    q: <question>    → question with no reference (default)")
    print(f"    q: <q> | r: <ref>→ question + reference answer for correctness scoring")
    print(f"    exit             → quit")
    print(f"{'='*60}\n")

    try:
        while True:
            try:
                raw_input = input("Q> ").strip()
            except EOFError:
                break

            if not raw_input:
                continue

            if raw_input.lower() in {"exit", "quit", ":q"}:
                break

            if raw_input.lower() == "summary":
                print_session_summary(session_results)
                continue

            reference = None
            if " | r: " in raw_input:
                parts     = raw_input.split(" | r: ", 1)
                question  = parts[0].replace("q: ", "").strip()
                reference = parts[1].strip()
            else:
                question = raw_input.replace("q: ", "").strip()

            result = run_question(
                question, reference, retriever, reranker, generator,
                retrieval_pool, top_n, search_mode,
            )
            if result:
                session_results.append(result)

    except KeyboardInterrupt:
        pass
    finally:
        print_session_summary(session_results)
        print("[Goodbye]\n")


if __name__ == "__main__":
    main()