"""
generation/rag_graph.py — LangGraph RAG with classifier + corrective retry loop.

UPDATED: direct_answer node now uses direct_chat (conversational LLM)
instead of going through the strict-RAG generator prompt.
"""
from __future__ import annotations

import sys
import os
import time
from typing import TypedDict, Literal

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langgraph.graph import StateGraph, END

from retrieve.retriever  import Retriever
from reranking.reranker   import Reranker
from answer_generation.generator import Generator
from memory               import MemoryManager

from llm_as_a_judge.rag_judge import score_chunks
from answer_generation.classifier       import classify_query
from answer_generation.direct_chat      import direct_chat


# ── CONFIG ────────────────────────────────────────────────────────
RETRIEVAL_POOL         = 20
TOP_N                  = 5
MAX_ATTEMPTS           = 2
WEAK_CHUNK_THRESHOLD   = 3
WEAK_CHUNK_RATIO       = 0.5

NEXT_MODE: dict[str, str] = {
    "hybrid": "dense",
    "dense":  "sparse",
    "sparse": "hybrid",
}


# ── STATE ─────────────────────────────────────────────────────────
class RAGState(TypedDict):
    question:        str
    user_id:         str
    session_id:      str
    query_type:      Literal["NEEDS_RAG", "DIRECT_ANSWER"]
    history:         list[dict]
    rewritten_query: str
    retrieval_mode:  Literal["hybrid", "dense", "sparse"]
    attempt:         int
    candidates:      list[dict]
    chunks:          list[dict]
    chunk_scores:    list[dict]
    should_retry:    bool
    memory_context:  str
    answer:          str
    sources:         list[dict]
    no_answer:       bool
    tokens:          dict
    latency_s:       float
    total_latency_s: float


# ── HELPERS ───────────────────────────────────────────────────────

def _build_memory_chunk(memory_context: str) -> dict:
    return {
        "content":     memory_context,
        "score":       1.0,
        "source":      "conversation_memory",
        "source_type": "memory",
        "type":        "text",
        "section":     "memory_context",
        "page":        None,
        "chunk_index": 0,
        "chunk_total": None,
        "retrieval":   "memory",
        "token_count": len(memory_context) // 4,
    }


def _get_document_id(chunk: dict) -> str | None:
    doc_id = chunk.get("document_id")
    if doc_id:
        return doc_id
    payload = chunk.get("payload", {})
    if isinstance(payload, dict):
        doc_id = payload.get("document_id")
        if doc_id:
            return doc_id
    metadata = chunk.get("metadata", {})
    if isinstance(metadata, dict):
        doc_id = metadata.get("document_id")
        if doc_id:
            return doc_id
    return None


def _load_recent_history_from_db(session_id: str, n: int = 4) -> list[dict]:
    from sqlalchemy import text
    from shared.db import engine
    
    try:
        with engine.connect() as conn:
            rows = conn.execute(
                text("""
                    SELECT role, content
                    FROM conversation_turns
                    WHERE session_id = :sid
                    ORDER BY created_at DESC
                    LIMIT :n
                """),
                {"sid": session_id, "n": n},
            ).fetchall()
        return [{"role": r[0], "content": r[1]} for r in reversed(rows)]
    except Exception as e:
        print(f"  ⚠ Could not load history: {e}")
        return []


# ── NODE FACTORIES ────────────────────────────────────────────────

def make_classify_node():
    def classify_node(state: RAGState) -> RAGState:
        print(f"\n[GRAPH 1/6] Classifying query...")
        history = _load_recent_history_from_db(state["session_id"], n=4)
        category = classify_query(state["question"], history)
        print(f"  Question: {state['question'][:80]}{'...' if len(state['question']) > 80 else ''}")
        print(f"  → Category: {category}")
        return {**state, "query_type": category, "history": history}
    return classify_node


def make_rewrite_node(memory: MemoryManager):
    def rewrite_node(state: RAGState) -> RAGState:
        print(f"\n[GRAPH 2/6] Rewriting query...")
        rewritten      = memory.rewrite_query(state["question"])
        memory_context = memory.build_prompt_context(state["question"])
        if rewritten != state["question"]:
            print(f"  Original : {state['question']}")
            print(f"  Rewritten: {rewritten}")
        else:
            print(f"  Unchanged: {rewritten}")
        return {**state, "rewritten_query": rewritten, "memory_context": memory_context}
    return rewrite_node


def make_retrieve_node(retriever: Retriever, memory: MemoryManager, pool: int):
    def retrieve_node(state: RAGState) -> RAGState:
        attempt = state["attempt"]
        mode    = state["retrieval_mode"]
        print(f"\n[GRAPH 3/6] Retrieving (attempt={attempt}, mode={mode}, pool={pool})...")

        allowed_docs = memory.get_allowed_doc_ids() or []
        allowed_set  = set(allowed_docs)
        print(f"  User has access to {len(allowed_set)} private docs")

        raw_candidates = retriever.search(
            state["rewritten_query"],
            mode=mode,
            top_n=pool * 3,
        )

        candidates = []
        n_dropped  = 0
        for c in raw_candidates:
            doc_id = _get_document_id(c)
            if not doc_id:
                candidates.append(c)
            elif doc_id in allowed_set:
                candidates.append(c)
            else:
                n_dropped += 1
            if len(candidates) >= pool:
                break

        n_public  = sum(1 for c in candidates if not _get_document_id(c))
        n_private = len(candidates) - n_public
        print(f"  Kept {len(candidates)} chunks ({n_public} public, {n_private} private), dropped {n_dropped} from other users")
        return {**state, "candidates": candidates}
    return retrieve_node


def make_rerank_node(reranker: Reranker, top_n: int):
    def rerank_node(state: RAGState) -> RAGState:
        print(f"\n[GRAPH 4/6] Reranking → top {top_n}...")
        if not state["candidates"]:
            print("  No candidates to rerank.")
            return {**state, "chunks": []}
        chunks = reranker.rerank(state["rewritten_query"], state["candidates"], top_n=top_n)
        print(f"  Kept {len(chunks)} chunks.")
        return {**state, "chunks": chunks}
    return rerank_node


def make_grade_chunks_node():
    def grade_chunks_node(state: RAGState) -> RAGState:
        print(f"\n[GRAPH 5/6] Grading chunks...")
        chunks = state["chunks"]
        if not chunks:
            print("  No chunks to grade — flagging for retry.")
            return {**state, "chunk_scores": [], "should_retry": state["attempt"] < MAX_ATTEMPTS}
        try:
            chunk_scores = score_chunks(state["rewritten_query"], chunks)
        except Exception as e:
            print(f"  ⚠ Chunk grading failed: {e}")
            return {**state, "chunk_scores": [], "should_retry": False}

        weak       = [c for c in chunk_scores if c["score"] < WEAK_CHUNK_THRESHOLD]
        weak_ratio = len(weak) / len(chunk_scores) if chunk_scores else 1.0
        print(f"  Weak chunks: {len(weak)}/{len(chunk_scores)} ({weak_ratio:.0%})")

        should_retry = weak_ratio > WEAK_CHUNK_RATIO and state["attempt"] < MAX_ATTEMPTS
        if should_retry:
            print(f"  → Retry triggered (attempt {state['attempt']} < max {MAX_ATTEMPTS})")
        else:
            print(f"  → {'Max attempts reached' if state['attempt'] >= MAX_ATTEMPTS else 'Chunks look good'}, proceeding to generation.")
        return {**state, "chunk_scores": chunk_scores, "should_retry": should_retry}
    return grade_chunks_node


def make_reformulate_node():
    def reformulate_node(state: RAGState) -> RAGState:
        next_mode = NEXT_MODE[state["retrieval_mode"]]
        print(f"\n[GRAPH] Reformulating: switching mode "
              f"{state['retrieval_mode']} → {next_mode}, "
              f"attempt {state['attempt']} → {state['attempt'] + 1}")
        return {**state, "retrieval_mode": next_mode, "attempt": state["attempt"] + 1}
    return reformulate_node


def make_generate_node(generator: Generator, memory: MemoryManager):
    """RAG-style generation (uses retrieved chunks)."""
    def generate_node(state: RAGState) -> RAGState:
        print(f"\n[GRAPH 6/6] Generating (with retrieved chunks)...")
        chunks = state["chunks"]
        if state["memory_context"]:
            memory_chunk = _build_memory_chunk(state["memory_context"])
            augmented    = [memory_chunk] + chunks
        else:
            augmented = chunks

        if not augmented:
            print("  No chunks available — returning no-answer.")
            memory.save_turn("user",      state["question"], extract_facts=False)
            memory.save_turn("assistant", "I couldn't find relevant information.", extract_facts=False)
            return {
                **state,
                "answer":    "I couldn't find relevant information in the knowledge base.",
                "sources":   [],
                "no_answer": True,
                "tokens":    {"prompt": 0, "completion": 0, "total": 0},
                "latency_s": 0.0,
            }

        result = generator.generate(state["rewritten_query"], augmented)
        memory.save_turn("user",      state["question"], extract_facts=True)
        memory.save_turn("assistant", result.get("answer", ""), extract_facts=False)

        return {
            **state,
            "answer":    result.get("answer", ""),
            "sources":   result.get("sources", []),
            "no_answer": result.get("no_answer", False),
            "tokens":    result.get("tokens", {"prompt": 0, "completion": 0, "total": 0}),
            "latency_s": result.get("latency_s", 0.0),
        }
    return generate_node


def make_direct_answer_node(memory: MemoryManager):
    """
    Direct conversational answer — NO retrieval, NO RAG prompt.
    Uses direct_chat which has a conversational system prompt.
    """
    def direct_answer_node(state: RAGState) -> RAGState:
        print(f"\n[GRAPH] Direct answer (conversational)...")
        
        history = state.get("history", [])
        result = direct_chat(state["question"], history)
        
        memory.save_turn("user",      state["question"], extract_facts=False)
        memory.save_turn("assistant", result.get("answer", ""), extract_facts=False)
        
        return {
            **state,
            "answer":          result.get("answer", ""),
            "sources":         [],
            "no_answer":       result.get("no_answer", False),
            "tokens":          result.get("tokens", {"prompt": 0, "completion": 0, "total": 0}),
            "latency_s":       result.get("latency_s", 0.0),
            "rewritten_query": state["question"],
        }
    return direct_answer_node


# ── ROUTING FUNCTIONS ─────────────────────────────────────────────

def route_after_classify(state: RAGState) -> str:
    if state["query_type"] == "DIRECT_ANSWER":
        return "direct_answer"
    return "rewrite"


def route_after_grading(state: RAGState) -> str:
    return "reformulate" if state["should_retry"] else "generate"


# ── GRAPH BUILDER (FULL — WITH GENERATION) ────────────────────────
def build_rag_graph(
    retriever : Retriever,
    reranker  : Reranker,
    generator : Generator,
    memory    : MemoryManager,
    pool      : int = RETRIEVAL_POOL,
    top_n     : int = TOP_N,
):
    builder = StateGraph(RAGState)

    builder.add_node("classify",      make_classify_node())
    builder.add_node("rewrite",       make_rewrite_node(memory))
    builder.add_node("retrieve",      make_retrieve_node(retriever, memory, pool))
    builder.add_node("rerank",        make_rerank_node(reranker, top_n))
    builder.add_node("grade_chunks",  make_grade_chunks_node())
    builder.add_node("reformulate",   make_reformulate_node())
    builder.add_node("generate",      make_generate_node(generator, memory))
    builder.add_node("direct_answer", make_direct_answer_node(memory))

    builder.set_entry_point("classify")

    builder.add_conditional_edges(
        "classify",
        route_after_classify,
        {"rewrite": "rewrite", "direct_answer": "direct_answer"},
    )

    builder.add_edge("rewrite",       "retrieve")
    builder.add_edge("retrieve",      "rerank")
    builder.add_edge("rerank",        "grade_chunks")
    builder.add_edge("reformulate",   "retrieve")
    builder.add_edge("generate",      END)
    builder.add_edge("direct_answer", END)

    builder.add_conditional_edges(
        "grade_chunks",
        route_after_grading,
        {"reformulate": "reformulate", "generate": "generate"},
    )

    return builder.compile()


# ── GRAPH BUILDER (STREAMING — WITHOUT GENERATION) ────────────────
def build_rag_graph_streaming(
    retriever : Retriever,
    reranker  : Reranker,
    memory    : MemoryManager,
    pool      : int = RETRIEVAL_POOL,
    top_n     : int = TOP_N,
):
    """
    Build streaming graph: classifier + retrieval pipeline.
    Generation happens in the streaming endpoint (so tokens stream live).
    """
    builder = StateGraph(RAGState)

    builder.add_node("classify",     make_classify_node())
    builder.add_node("rewrite",      make_rewrite_node(memory))
    builder.add_node("retrieve",     make_retrieve_node(retriever, memory, pool))
    builder.add_node("rerank",       make_rerank_node(reranker, top_n))
    builder.add_node("grade_chunks", make_grade_chunks_node())
    builder.add_node("reformulate",  make_reformulate_node())

    builder.set_entry_point("classify")

    def route_after_classify_streaming(state):
        if state["query_type"] == "DIRECT_ANSWER":
            return END
        return "rewrite"

    builder.add_conditional_edges(
        "classify",
        route_after_classify_streaming,
        {"rewrite": "rewrite", END: END},
    )

    builder.add_edge("rewrite",     "retrieve")
    builder.add_edge("retrieve",    "rerank")
    builder.add_edge("rerank",      "grade_chunks")
    builder.add_edge("reformulate", "retrieve")

    def route_streaming(state: RAGState) -> str:
        return "reformulate" if state["should_retry"] else END

    builder.add_conditional_edges(
        "grade_chunks",
        route_streaming,
        {"reformulate": "reformulate", END: END},
    )

    return builder.compile()


# ── RUN HELPERS ───────────────────────────────────────────────────
def run_graph(graph, question, user_id, session_id, mode="hybrid"):
    t_total = time.perf_counter()
    initial_state = _initial_state(question, user_id, session_id, mode)
    final_state = graph.invoke(initial_state)
    total_latency = round(time.perf_counter() - t_total, 3)

    return {
        "answer":           final_state["answer"],
        "sources":          final_state["sources"],
        "no_answer":        final_state["no_answer"],
        "original_query":   question,
        "rewritten_query":  final_state["rewritten_query"],
        "session_id":       session_id,
        "tokens":           final_state["tokens"],
        "latency_s":        final_state["latency_s"],
        "total_latency_s":  total_latency,
        "chunk_scores":     final_state.get("chunk_scores", []),
        "retrieval_mode":   final_state["retrieval_mode"],
        "attempts":         final_state["attempt"],
        "query_type":       final_state["query_type"],
    }


def run_graph_until_generate(graph, question, user_id, session_id, mode="hybrid"):
    initial_state = _initial_state(question, user_id, session_id, mode)
    final_state = graph.invoke(initial_state)
    return final_state


def _initial_state(question, user_id, session_id, mode):
    return {
        "question":        question,
        "user_id":         user_id,
        "session_id":      session_id,
        "query_type":      "NEEDS_RAG",
        "history":         [],
        "rewritten_query": question,
        "retrieval_mode":  mode,
        "attempt":         0,
        "candidates":      [],
        "chunks":          [],
        "chunk_scores":    [],
        "should_retry":    False,
        "memory_context":  "",
        "answer":          "",
        "sources":         [],
        "no_answer":       False,
        "tokens":          {"prompt": 0, "completion": 0, "total": 0},
        "latency_s":       0.0,
        "total_latency_s": 0.0,
    }