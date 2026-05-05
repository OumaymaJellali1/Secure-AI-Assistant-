"""
generation/pipeline.py — Full RAG pipeline with memory + multi-conversation.

Updated:
  • Builds TWO graphs: full (with generate) + streaming (without generate)
  • Streaming endpoint uses .graph_streaming + generator.stream_with_metadata()
"""
from dotenv import load_dotenv
load_dotenv()
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from retrieve.retriever  import Retriever
from reranking.reranker   import Reranker
from answer_generation.generator import Generator
from memory               import MemoryManager, SessionManager

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

    The streaming endpoint runs graph_streaming, then calls generator.stream_with_metadata
    to stream tokens directly to the client.
    """

    def __init__(
        self,
        user_id    : str,
        session_id : str = None,
        top_n      : int = TOP_N,
        pool       : int = RETRIEVAL_POOL,
    ):
        self.user_id = user_id
        self.top_n   = top_n
        self.pool    = pool

        # Session: resume or create
        sessions = SessionManager()
        if session_id is None:
            session_id = sessions.create(user_id, title=None)
            print(f"[PIPELINE] Created new conversation: {session_id}")
        else:
            existing = sessions.get(session_id)
            if existing is None:
                session_id = sessions.create(user_id, title=None)
                print(f"[PIPELINE] Created conversation: {session_id}")
            else:
                print(f"[PIPELINE] Resuming conversation: {session_id}")

        self.session_id = session_id

        print("[PIPELINE] Loading components...")
        self.memory    = MemoryManager(session_id, user_id)
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

    def query(self, question: str, *, stream: bool = True) -> dict:
        """Non-streaming query (full graph)."""
        result = run_graph(
            graph      = self.graph,
            question   = question,
            user_id    = self.user_id,
            session_id = self.session_id,
        )
        print(f"\n{result['answer']}\n")
        return result

    # Pass-through helpers
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
def run_repl(user_id: str = "dev_test", top_n: int = TOP_N):
    sessions = SessionManager()
    existing = sessions.list_for_user(user_id, limit=10)

    print("=" * 70)
    print(f"  RAG Memory REPL — user: {user_id}")
    print("=" * 70)

    if existing:
        print("\n  Your recent conversations:")
        for i, s in enumerate(existing, 1):
            title = s["title"] or "(untitled)"
            turns = s["turn_count"]
            print(f"    {i}. {title}  —  {turns} turns  ({s['last_active']:%Y-%m-%d %H:%M})")
        print(f"    N. Start a NEW conversation")
        print()
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

    pipeline = RAGPipeline(user_id=user_id, session_id=session_id, top_n=top_n)

    print(f"\n  Commands:")
    print(f"    list   — list your conversations")
    print(f"    state  — show memory state")
    print(f"    clear  — clear this conversation's buffer")
    print(f"    exit   — quit")
    print(f"  " + "=" * 68)

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
                convs = pipeline.list_my_conversations()
                for s in convs:
                    print(f"  • {s['session_id'][:14]}  "
                          f"{s['title'] or '(untitled)':<40}  ({s['turn_count']} turns)")
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
                      f"  |  final_mode={result.get('retrieval_mode', 'hybrid')}]")

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
    parser.add_argument("--user_id",    default="dev_test")
    parser.add_argument("--session_id", default=None)
    parser.add_argument("--top_n",      type=int, default=TOP_N)
    parser.add_argument("question",     nargs="?", default=None)
    args = parser.parse_args()

    if args.question:
        pipeline = RAGPipeline(
            user_id    = args.user_id,
            session_id = args.session_id,
            top_n      = args.top_n,
        )
        try:
            result = pipeline.query(args.question)
        finally:
            pipeline.close()
    else:
        run_repl(user_id=args.user_id, top_n=args.top_n)