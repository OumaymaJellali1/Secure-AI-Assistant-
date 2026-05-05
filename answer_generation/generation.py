"""
generator.py
════════════
RAG answer generation using Qwen2.5-7B via Ollama.

Pipeline:
  1. Retrieve relevant chunks (from retriever.py)
  2. Format context with source metadata
  3. Build grounded prompt with anti-hallucination instructions
  4. Call Qwen2.5-7B via Ollama
  5. Return answer with source citations

Prerequisites:
  - Ollama running: https://ollama.com
  - Model pulled: ollama pull qwen2.5:7b
  - Retriever configured (retrieval/retriever.py)

Usage:
  from generation.generator import Generator

  gen = Generator()
  result = gen.answer("What is BGE-M3?")
  print(result["answer"])
  print(result["sources"])
"""

from __future__ import annotations

import sys
import os
import json
import requests
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from retrieval.retriever import Retriever


# ── CONFIG ────────────────────────────────────────────────────────────────────

OLLAMA_URL   = "http://localhost:11434"
MODEL_NAME   ="llama-3.3-70b-versatile" 
# MODEL_NAME   = "qwen2.5:7b"
TOP_N_CHUNKS = 5
TEMPERATURE  = 0.1
MAX_TOKENS   = 1024
TIMEOUT      = 300


# ── PROMPT TEMPLATES ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a precise question-answering assistant. Your job is to answer questions using ONLY the context provided below.

STRICT RULES:
1. Answer ONLY from the provided context. Do not use external knowledge.
2. If the context doesn't contain the answer, respond EXACTLY: "I don't have enough information in the provided documents to answer this question."
3. Cite sources inline using this format: [Source: filename, page X]
4. Every factual claim MUST have a citation immediately after it.
5. Be concise and direct. No filler phrases like "Based on the context..."
6. Preserve the original language of the question in your answer.
7. If the context contains contradictions, mention them explicitly.
8. Never invent filenames, page numbers, or quotes."""


USER_PROMPT_TEMPLATE = """Context from documents:
{context}

────────────────────────────────
Question: {question}

Answer (with inline citations like [Source: filename, page X]):"""


# ── GENERATOR CLASS ───────────────────────────────────────────────────────────

class Generator:
    """
    RAG answer generator using Qwen2.5-7B via Ollama.

    Attributes:
        model_name   : Ollama model tag (default: "qwen2.5:7b")
        ollama_url   : Ollama server URL
        top_n        : chunks to retrieve
        temperature  : LLM sampling temperature (low = factual)
        retriever    : Retriever instance (hybrid search)
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        ollama_url: str = OLLAMA_URL,
        top_n: int = TOP_N_CHUNKS,
        temperature: float = TEMPERATURE,
        max_tokens: int = MAX_TOKENS,
    ) -> None:
        self.model_name  = model_name
        self.ollama_url  = ollama_url.rstrip("/")
        self.top_n       = top_n
        self.temperature = temperature
        self.max_tokens  = max_tokens
        self.retriever   = Retriever()

        self._check_ollama()

    # ── PUBLIC API ────────────────────────────────────────────────────────────

    def answer(
        self,
        question: str,
        top_n: Optional[int] = None,
        filters: Optional[dict] = None,
        mode: str = "hybrid",
        stream: bool = False,
    ) -> dict:
        """
        Answer a question using retrieved context.

        Args:
            question : user question (natural language)
            top_n    : number of chunks to retrieve
            filters  : metadata filters (e.g., {"source_type": "pdf"})
            mode     : retrieval mode ("hybrid", "dense", "sparse")
            stream   : if True, stream tokens as they generate

        Returns:
            {
              "question" : str,
              "answer"   : str,
              "sources"  : list[dict],
              "model"    : str,
              "n_chunks" : int,
            }
        """
        if not question or not question.strip():
            raise ValueError("[GENERATOR] Question is empty.")

        n = top_n or self.top_n

        print(f"\n{'='*70}")
        print(f"  QUESTION: {question}")
        print(f"{'='*70}")

        # Retrieve
        print(f"\n[GENERATOR] Retrieving top-{n} chunks...")
        chunks = self.retriever.search(question, mode=mode, top_n=n, filters=filters)

        if not chunks:
            return {
                "question" : question,
                "answer"   : "I don't have enough information in the provided documents to answer this question.",
                "sources"  : [],
                "model"    : self.model_name,
                "n_chunks" : 0,
            }

        # Format context
        print(f"[GENERATOR] Formatting context from {len(chunks)} chunks...")
        context = self._format_context(chunks)

        # Build prompt
        user_prompt = USER_PROMPT_TEMPLATE.format(
            context  = context,
            question = question.strip(),
        )

        # Call Ollama
        print(f"[GENERATOR] Calling {self.model_name} via Ollama...")
        if stream:
            answer_text = self._call_ollama_stream(SYSTEM_PROMPT, user_prompt)
        else:
            answer_text = self._call_ollama(SYSTEM_PROMPT, user_prompt)

        print(f"\n[GENERATOR] Answer generated ({len(answer_text)} chars)")

        return {
            "question" : question,
            "answer"   : answer_text.strip(),
            "sources"  : chunks,
            "model"    : self.model_name,
            "n_chunks" : len(chunks),
        }

    # ── CONTEXT FORMATTING ────────────────────────────────────────────────────

    def _format_context(self, chunks: list[dict]) -> str:
        """Format retrieved chunks into a clear labeled context block."""
        blocks = []

        for i, chunk in enumerate(chunks, start=1):
            source  = chunk.get("source", "unknown")
            page    = chunk.get("page")
            section = chunk.get("section")
            subject = chunk.get("subject")

            meta_parts = [f"filename: {source}"]
            if page:
                meta_parts.append(f"page: {page}")
            if section:
                meta_parts.append(f"section: {section}")
            if subject:
                meta_parts.append(f"email subject: {subject}")

            meta_line = " | ".join(meta_parts)
            content = chunk.get("content", "").strip()

            block = f"[Source {i}] ({meta_line})\n{content}"
            blocks.append(block)

        return "\n\n".join(blocks)

    # ── OLLAMA CALLS ──────────────────────────────────────────────────────────

    def _call_ollama(self, system_prompt: str, user_prompt: str) -> str:
        """Call Ollama /api/chat endpoint (non-streaming)."""
        try:
            response = requests.post(
                f"{self.ollama_url}/api/chat",
                json = {
                    "model"    : self.model_name,
                    "messages" : [
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_prompt},
                    ],
                    "stream"   : False,
                    "options"  : {
                        "temperature"    : self.temperature,
                        "num_predict"    : self.max_tokens,
                        "top_p"          : 0.9,
                        "top_k"          : 40,
                        "repeat_penalty" : 1.1,
                    },
                },
                timeout = TIMEOUT,
            )
            response.raise_for_status()
            return response.json()["message"]["content"]

        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"[GENERATOR] Cannot reach Ollama at {self.ollama_url}. "
                f"Is Ollama running? Start with: ollama serve"
            )
        except requests.exceptions.Timeout:
            raise RuntimeError(
                f"[GENERATOR] Ollama timeout after {TIMEOUT}s. "
                f"Try a smaller model (qwen2.5:3b) or increase TIMEOUT."
            )
        except Exception as e:
            raise RuntimeError(f"[GENERATOR] Ollama error: {e}")

    def _call_ollama_stream(self, system_prompt: str, user_prompt: str) -> str:
        """Stream tokens from Ollama, printing as they arrive."""
        print("\n[ANSWER]: ", end="", flush=True)
        full_answer = []

        try:
            response = requests.post(
                f"{self.ollama_url}/api/chat",
                json = {
                    "model"    : self.model_name,
                    "messages" : [
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_prompt},
                    ],
                    "stream"   : True,
                    "options"  : {
                        "temperature" : self.temperature,
                        "num_predict" : self.max_tokens,
                        "top_p"       : 0.9,
                    },
                },
                timeout = TIMEOUT,
                stream  = True,
            )
            response.raise_for_status()

            for line in response.iter_lines():
                if not line:
                    continue
                data = json.loads(line)
                if "message" in data and "content" in data["message"]:
                    token = data["message"]["content"]
                    print(token, end="", flush=True)
                    full_answer.append(token)
                if data.get("done"):
                    break

            print()
            return "".join(full_answer)

        except Exception as e:
            raise RuntimeError(f"[GENERATOR] Streaming error: {e}")

    # ── HEALTH CHECK ──────────────────────────────────────────────────────────

    def _check_ollama(self) -> None:
        """Verify Ollama is running and the model is available."""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            response.raise_for_status()
            models = response.json().get("models", [])
            available = [m["name"] for m in models]

            if not any(self.model_name in m for m in available):
                print(f"[GENERATOR] WARNING: Model '{self.model_name}' not found in Ollama.")
                print(f"[GENERATOR] Available models: {available}")
                print(f"[GENERATOR] Pull the model with: ollama pull {self.model_name}")
            else:
                print(f"[GENERATOR] Ollama OK, model '{self.model_name}' available.")

        except requests.exceptions.ConnectionError:
            print(
                f"[GENERATOR] Cannot reach Ollama at {self.ollama_url}.\n"
                f"[GENERATOR]   Start Ollama: `ollama serve`\n"
                f"[GENERATOR]   Pull model:   ollama pull {self.model_name}"
            )
        except Exception as e:
            print(f"[GENERATOR] Warning: {e}")


# ── PRETTY PRINT ──────────────────────────────────────────────────────────────

def print_answer(result: dict, show_sources: bool = True) -> None:
    """Pretty-print the RAG result."""
    print(f"\n{'='*70}")
    print(f"  ANSWER")
    print(f"{'='*70}")
    print(f"\n{result['answer']}\n")

    if show_sources and result.get("sources"):
        print(f"\n{'='*70}")
        print(f"  SOURCES ({result['n_chunks']} chunks)")
        print(f"{'='*70}")

        for i, src in enumerate(result["sources"], 1):
            source    = src.get("source", "unknown")
            page      = src.get("page")
            section   = src.get("section")
            retrieval = src.get("retrieval", "?")
            score     = src.get("score", 0)

            location = f"{source}"
            if page:
                location += f", page {page}"
            if section:
                location += f", section {section}"

            content_preview = src.get("content", "").replace("\n", " ")[:150]

            print(f"\n  [{i}] {location}")
            print(f"      score={score:.4f} | retrieval={retrieval}")
            print(f"      {content_preview}...")

    print(f"\n{'='*70}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Usage:
      python generator.py "What is BGE-M3?"
      python generator.py "What is RAG?" --stream
      python generator.py "summary" --filter source_type=pdf --top_n 10
      python generator.py --test
    """

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python generator.py <question>")
        print("  python generator.py <question> --stream")
        print("  python generator.py <question> --filter field=value --top_n N")
        print("  python generator.py --test")
        sys.exit(1)

    # Test mode
    if sys.argv[1] == "--test":
        gen = Generator()
        test_questions = [
            "What is BGE-M3?",
            "How does hybrid retrieval work?",
        ]
        for q in test_questions:
            result = gen.answer(q)
            print_answer(result)
        sys.exit(0)

    # Parse args
    question = sys.argv[1]
    stream   = False
    filters  = None
    top_n    = None

    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == "--stream":
            stream = True
            i += 1
        elif sys.argv[i] == "--filter" and i + 1 < len(sys.argv):
            key, val = sys.argv[i + 1].split("=", 1)
            filters = {key: val}
            i += 2
        elif sys.argv[i] == "--top_n" and i + 1 < len(sys.argv):
            top_n = int(sys.argv[i + 1])
            i += 2
        else:
            i += 1

    # Run
    try:
        gen = Generator()
        result = gen.answer(question, top_n=top_n, filters=filters, stream=stream)

        if not stream:
            print_answer(result)
        else:
            print_answer(
                {"answer": "", "sources": result["sources"], "n_chunks": result["n_chunks"]},
                show_sources=True,
            )

    except Exception as e:
        print(f"\n[GENERATOR] ERROR: {e}\n")
        sys.exit(1)