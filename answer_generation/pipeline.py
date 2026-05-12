"""
answer_generation/pipeline.py — Full RAG pipeline with memory + multi-conversation.

UPDATED:
  - query() now accepts is_admin flag and passes it through to run_graph()
  - run_graph() passes is_admin → RAGState → make_retrieve_node → retriever.search()
  - ACL is enforced server-side in Qdrant — no post-filtering in Python
"""
from dotenv import load_dotenv
load_dotenv()
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from retrieve.retriever           import Retriever
from reranking.reranker           import Reranker
from answer_generation.generator  import Generator
from memory                       import MemoryManager, SessionManager

from answer_generation.rag_graph import (
    build_rag_graph,
    build_rag_graph_streaming,
    run_graph,
)


# ── CONFIG ────────────────────────────────────────────────────────
RETRIEVAL_POOL = 20
TOP_N          = 5


class RAGPipeline:
    """
    Full RAG pipeline with persistent memory.

    Has two compiled graphs:
      • self.graph           — full pipeline incl. generation (non-streaming)
      • self.graph_streaming — pipeline up to grading (for streaming endpoint)
    """

    def __init__(
        self,
        id    : str,
        session_id : str = None,
        top_n      : int = TOP_N,
        pool       : int = RETRIEVAL_POOL,
    ):
        self.id = id
        self.top_n   = top_n
        self.pool    = pool

        # Resolve is_admin from Postgres
        from auth.users import is_admin as _is_admin
        self.is_admin = _is_admin(id)
        print(f"[PIPELINE] User '{id}' — admin={self.is_admin}")

        # Session: resume or create
        sessions = SessionManager()
        if session_id is None:
            session_id = sessions.create(id, title=None)
            print(f"[PIPELINE] Created new conversation: {session_id}")
        else:
            existing = sessions.get(session_id)
            if existing is None:
                session_id = sessions.create(id, title=None)
                print(f"[PIPELINE] Created conversation: {session_id}")
            else:
                print(f"[PIPELINE] Resuming conversation: {session_id}")

        self.session_id = session_id

        print("[PIPELINE] Loading components...")
        self.memory    = MemoryManager(session_id, id)
        self.retriever = Retriever()
        self.reranker  = Reranker()
        self.generator = Generator()

        print("[PIPELINE] Building LangGraph (full)...")
        self.graph = build_rag_graph(
            retriever = self.retriever,
            reranker  = self.reranker,
            generator = self.generator,
            memory    = self.memory,
            pool      = self.pool,
            top_n     = self.top_n,
        )

        print("[PIPELINE] Building LangGraph (streaming)...")
        self.graph_streaming = build_rag_graph_streaming(
            retriever = self.retriever,
            reranker  = self.reranker,
            memory    = self.memory,
            pool      = self.pool,
            top_n     = self.top_n,
        )

        print("[PIPELINE] Ready.\n")

    def query(
        self,
        question         : str,
        *,
        stream           : bool       = True,
        retrieval_filter : dict | None = None,
    ) -> dict:
        """
        Non-streaming query (full graph).

        Args:
            question         : the user's question
            stream           : unused here (kept for API compat)
            retrieval_filter : optional Qdrant filter dict, e.g.:
                               {"must": [{"key": "document_id",
                                          "match": {"value": "doc_abc123"}}]}
                               Pass None (default) to search all accessible chunks.

        Access control is applied automatically:
            - Admin users (is_admin=True in Postgres) see all data.
            - Regular users see only their own Gmail chunks + shared docs.
        """
        result = run_graph(
            graph            = self.graph,
            question         = question,
            id          = self.id,
            session_id       = self.session_id,
            retrieval_filter = retrieval_filter,
            is_admin         = self.is_admin,
        )
        print(f"\n{result['answer']}\n")
        return result

    # ── Pass-through helpers ──────────────────────────────────────

    def list_my_conversations(self) -> list[dict]:
        return self.memory.list_my_conversations()

    def archive_current(self):
        self.memory.archive_current()

    def clear_buffer(self):
        self.memory.clear_buffer()

    def close(self):
        for component in (self.retriever, self.reranker, self.generator):
            if hasattr(component, "close"):
                try:
                    component.close()
                except Exception as e:
                    print(f"[PIPELINE] Warning during close: {e}")


# ── REPL ─────────────────────────────────────────────────────────
def run_repl(id: str = "dev_test", top_n: int = TOP_N):
    sessions = SessionManager()
    existing = sessions.list_for_user(id, limit=10)

    print("=" * 70)
    print(f"  RAG Memory REPL — user: {id}")
    print("=" * 70)

    if existing:
        print("\n  Your recent conversations:")
        for i, s in enumerate(existing, 1):
            title = s["title"] or "(untitled)"
            turns = s["turn_count"]
            print(f"    {i}. {title}  —  {turns} turns  ({s['last_active']:%Y-%m-%d %H:%M})")
        print(f"    N. Start a NEW conversation\n")
        choice = input("  Choose: ").strip().upper()

        if choice == "N" or not choice:
            session_id = None
        else:
            try:
                session_id = existing[int(choice) - 1]["session_id"]
            except (ValueError, IndexError):
                print("  Invalid choice — starting new conversation.")
                session_id = None
    else:
        print("\n  (no past conversations — creating new)\n")
        session_id = None

    pipeline = RAGPipeline(id=id, session_id=session_id, top_n=top_n)

    print(f"\n  Commands: list | state | clear | exit")
    print(f"  {'='*68}")

    try:
        while True:
            try:
                question = input("\nQ> ").strip()
            except EOFError:
                break

            if not question:
                continue
            if question.lower() in {"exit", "quit", ":q"}:
                break
            if question.lower() == "list":
                for s in pipeline.list_my_conversations():
                    print(f"  • {s['session_id'][:14]}  {s['title'] or '(untitled)':<40}  ({s['turn_count']} turns)")
                continue
            if question.lower() == "state":
                pipeline.memory.print_state()
                continue
            if question.lower() == "clear":
                pipeline.clear_buffer()
                print("[buffer cleared]")
                continue

            try:
                result = pipeline.query(question)
                if result.get("sources"):
                    print(f"\n  ─── Sources ({len(result['sources'])}) ───")
                    for i, src in enumerate(result["sources"], 1):
                        loc = src.get("source", "?")
                        if src.get("page"):
                            loc += f", page {src['page']}"
                        print(f"    [{i}] {loc}")
                print(f"\n  [latency={result.get('total_latency_s', '?')}s"
                      f"  |  attempts={result.get('attempts', 1)}"
                      f"  |  mode={result.get('retrieval_mode', 'hybrid')}]")
            except KeyboardInterrupt:
                print("\n[interrupted]")
            except Exception as e:
                print(f"\n[ERROR] {e}")
                import traceback; traceback.print_exc()

    except KeyboardInterrupt:
        pass
    finally:
        pipeline.close()
        print("\nGoodbye.\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--id",    default="dev_test")
    parser.add_argument("--session_id", default=None)
    parser.add_argument("--top_n",      type=int, default=TOP_N)
    parser.add_argument("question",     nargs="?", default=None)
    args = parser.parse_args()

    if args.question:
        pipeline = RAGPipeline(
            id    = args.id,
            session_id = args.session_id,
            top_n      = args.top_n,
        )
        try:
            result = pipeline.query(args.question)
        finally:
            pipeline.close()
    else:
        run_repl(id=args.id, top_n=args.top_n)