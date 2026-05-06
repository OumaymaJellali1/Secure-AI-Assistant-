"""
api/routes/conversations.py — With classifier + direct_chat for follow-ups.

Streaming endpoint:
  • DIRECT_ANSWER → direct_chat_stream (conversational LLM, no RAG)
  • NEEDS_RAG     → full pipeline (rewrite, retrieve, rerank, grade, generate)
"""
import asyncio
import json

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import text

from shared.db import engine
from api.auth import get_current_user
from api.schemas import (
    ConversationCreate,
    ConversationOut,
    ConversationDetail,
    ConversationUpdate,
    TurnOut,
    QueryRequest,
    QueryResponse,
    SourceOut,
)
from memory import SessionManager, MemoryManager
from answer_generation.pipeline import RAGPipeline


router = APIRouter(prefix="/conversations", tags=["conversations"])

_pipeline_cache: dict[tuple[str, str], RAGPipeline] = {}


def _get_pipeline(user_id: str, session_id: str) -> RAGPipeline:
    key = (user_id, session_id)
    if key not in _pipeline_cache:
        _pipeline_cache[key] = RAGPipeline(user_id=user_id, session_id=session_id)
    return _pipeline_cache[key]


def _session_to_out(session: dict) -> ConversationOut:
    return ConversationOut(
        session_id=session["session_id"],
        title=session.get("title"),
        is_active=not session.get("archived", False),
        created_at=session["created_at"],
        last_active=session["last_active"],
        turn_count=session.get("turn_count", 0),
    )


def _load_turns_from_db(session_id: str) -> list[TurnOut]:
    with engine.connect() as conn:
        rows = conn.execute(
            text("""
                SELECT role, content, created_at
                FROM conversation_turns
                WHERE session_id = :sid
                ORDER BY created_at ASC
            """),
            {"sid": session_id},
        ).fetchall()
    return [TurnOut(role=r[0], content=r[1], created_at=r[2]) for r in rows]


# ── CRUD ──────────────────────────────────────────────────────────

@router.post("", response_model=ConversationOut, status_code=201)
def create_conversation(
    body: ConversationCreate,
    current_user: dict = Depends(get_current_user),
):
    user_id = current_user["user_id"]
    sessions = SessionManager()
    session_id = sessions.create(user_id, title=body.title)
    return _session_to_out(sessions.get(session_id))


@router.get("", response_model=list[ConversationOut])
def list_conversations(
    current_user: dict = Depends(get_current_user),
    active_only: bool = True,
    limit: int = 50,
):
    user_id = current_user["user_id"]
    sessions = SessionManager()
    items = sessions.list_for_user(user_id=user_id, limit=limit, active_only=active_only)
    return [_session_to_out(item) for item in items]


@router.get("/{session_id}", response_model=ConversationDetail)
def get_conversation(
    session_id: str,
    current_user: dict = Depends(get_current_user),
):
    user_id = current_user["user_id"]
    sessions = SessionManager()
    session = sessions.get(session_id)
    if not session or session["user_id"] != user_id:
        raise HTTPException(404, "Conversation not found")
    return ConversationDetail(
        session_id=session_id,
        title=session.get("title"),
        is_active=not session.get("archived", False),
        created_at=session["created_at"],
        last_active=session["last_active"],
        turns=_load_turns_from_db(session_id),
        summary="",
    )


@router.patch("/{session_id}", response_model=ConversationOut)
def update_conversation(
    session_id: str,
    body: ConversationUpdate,
    current_user: dict = Depends(get_current_user),
):
    user_id = current_user["user_id"]
    sessions = SessionManager()
    session = sessions.get(session_id)
    if not session or session["user_id"] != user_id:
        raise HTTPException(404, "Conversation not found")
    if body.title is not None:
        sessions.update_title(session_id, body.title)
    return _session_to_out(sessions.get(session_id))


@router.delete("/{session_id}", status_code=204)
def delete_conversation(
    session_id: str,
    current_user: dict = Depends(get_current_user),
):
    user_id = current_user["user_id"]
    sessions = SessionManager()
    session = sessions.get(session_id)
    if not session or session["user_id"] != user_id:
        raise HTTPException(404, "Conversation not found")
    sessions.delete(session_id)
    _pipeline_cache.pop((user_id, session_id), None)
    return None


# ── QUERY (NON-STREAMING) ─────────────────────────────────────────

@router.post("/{session_id}/query", response_model=QueryResponse)
def query(
    session_id: str,
    body: QueryRequest,
    current_user: dict = Depends(get_current_user),
):
    user_id = current_user["user_id"]
    sessions = SessionManager()
    session = sessions.get(session_id)
    if not session or session["user_id"] != user_id:
        raise HTTPException(404, "Conversation not found")
    pipeline = _get_pipeline(user_id, session_id)
    try:
        result = pipeline.query(body.question, stream=False)
        sources = [
            SourceOut(source=s.get("source", "unknown"), page=s.get("page"), score=s.get("score"))
            for s in result.get("sources", [])
        ]
        return QueryResponse(
            answer=result.get("answer", ""),
            sources=sources,
            original_query=result.get("original_query", body.question),
            rewritten_query=result.get("rewritten_query", body.question),
            session_id=session_id,
            total_latency_s=result.get("total_latency_s"),
        )
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(500, f"Query failed: {e}")


# ── QUERY (STREAMING via SSE) ─────────────────────────────────────

@router.post("/{session_id}/query/stream")
async def query_stream(
    session_id: str,
    body: QueryRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    Streaming query with classifier routing.

    SSE events:
      data: {"type": "status", "stage": "classifying"|"retrieving"|"generating"}
      data: {"type": "route", "category": "NEEDS_RAG"|"DIRECT_ANSWER"}
      data: {"type": "token", "content": "..."}
      data: {"type": "done", "sources": [...], ...}
      data: {"type": "error", "message": "..."}
    """
    user_id = current_user["user_id"]
    sessions = SessionManager()
    session = sessions.get(session_id)
    if not session or session["user_id"] != user_id:
        raise HTTPException(404, "Conversation not found")

    pipeline = _get_pipeline(user_id, session_id)

    async def event_generator():
        loop = asyncio.get_event_loop()
        token_queue: asyncio.Queue = asyncio.Queue()
        final_result_holder = {}

        def run_pipeline_in_thread():
            try:
                asyncio.run_coroutine_threadsafe(
                    token_queue.put({"type": "status", "stage": "classifying"}),
                    loop,
                )

                from answer_generation.rag_graph import (
                    run_graph_until_generate,
                    _build_memory_chunk,
                )
                from answer_generation.direct_chat import direct_chat_stream

                state = run_graph_until_generate(
                    graph=pipeline.graph_streaming,
                    question=body.question,
                    user_id=user_id,
                    session_id=session_id,
                )

                query_type = state.get("query_type", "NEEDS_RAG")

                asyncio.run_coroutine_threadsafe(
                    token_queue.put({"type": "route", "category": query_type}),
                    loop,
                )

                # ── PATH A: DIRECT_ANSWER ────────────────────────
                if query_type == "DIRECT_ANSWER":
                    asyncio.run_coroutine_threadsafe(
                        token_queue.put({"type": "status", "stage": "generating"}),
                        loop,
                    )

                    history = state.get("history", [])

                    def on_token(token: str):
                        asyncio.run_coroutine_threadsafe(
                            token_queue.put({"type": "token", "content": token}),
                            loop,
                        )

                    result = direct_chat_stream(
                        question=body.question,
                        history=history,
                        on_token=on_token,
                    )

                    pipeline.memory.save_turn("user",      body.question, extract_facts=False)
                    pipeline.memory.save_turn("assistant", result.get("answer", ""), extract_facts=False)

                    final_result_holder["result"] = {
                        "answer":     result.get("answer", ""),
                        "sources":    [],
                        "no_answer":  False,
                        "query_type": "DIRECT_ANSWER",
                    }
                    return

                # ── PATH B: NEEDS_RAG ────────────────────────────
                asyncio.run_coroutine_threadsafe(
                    token_queue.put({"type": "status", "stage": "retrieving"}),
                    loop,
                )

                asyncio.run_coroutine_threadsafe(
                    token_queue.put({"type": "status", "stage": "generating"}),
                    loop,
                )

                chunks = state["chunks"]
                if state.get("memory_context"):
                    augmented = [_build_memory_chunk(state["memory_context"])] + chunks
                else:
                    augmented = chunks

                if not augmented:
                    answer = "I couldn't find relevant information in the knowledge base."
                    pipeline.memory.save_turn("user", body.question, extract_facts=False)
                    pipeline.memory.save_turn("assistant", answer, extract_facts=False)
                    asyncio.run_coroutine_threadsafe(
                        token_queue.put({"type": "token", "content": answer}),
                        loop,
                    )
                    final_result_holder["result"] = {
                        "answer": answer,
                        "sources": [],
                        "no_answer": True,
                        "query_type": "NEEDS_RAG",
                    }
                else:
                    def on_token(token: str):
                        asyncio.run_coroutine_threadsafe(
                            token_queue.put({"type": "token", "content": token}),
                            loop,
                        )

                    result = pipeline.generator.stream_with_metadata(
                        state["rewritten_query"],
                        augmented,
                        on_token=on_token,
                    )

                    pipeline.memory.save_turn("user",      body.question, extract_facts=True)
                    pipeline.memory.save_turn("assistant", result["answer"], extract_facts=False)

                    sources_serializable = [
                        {
                            "source": s.get("source", "unknown"),
                            "page": s.get("page"),
                            "score": s.get("score"),
                        }
                        for s in result.get("sources", [])
                    ]
                    final_result_holder["result"] = {
                        "answer":     result.get("answer", ""),
                        "sources":    sources_serializable,
                        "no_answer":  result.get("no_answer", False),
                        "query_type": "NEEDS_RAG",
                    }

            except Exception as e:
                import traceback; traceback.print_exc()
                asyncio.run_coroutine_threadsafe(
                    token_queue.put({"type": "error", "message": str(e)}),
                    loop,
                )
            finally:
                asyncio.run_coroutine_threadsafe(
                    token_queue.put({"type": "_done"}),
                    loop,
                )

        loop.run_in_executor(None, run_pipeline_in_thread)

        while True:
            event = await token_queue.get()
            if event.get("type") == "_done":
                final = final_result_holder.get("result", {})
                yield f"data: {json.dumps({'type': 'done', **final})}\n\n"
                break
            if event.get("type") == "error":
                yield f"data: {json.dumps(event)}\n\n"
                break
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )