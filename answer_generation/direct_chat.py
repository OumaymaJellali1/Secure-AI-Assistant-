"""
generation/direct_chat.py — Direct conversational LLM call (no RAG).

Used by the classifier's DIRECT_ANSWER path:
  • Greetings: "hi", "hello", "thanks"
  • Meta-questions: "what can you do?", "who are you?"
  • Follow-ups: "summarize that", "translate to French"

Uses the SAME Ollama model as the Generator, but with a CONVERSATIONAL
system prompt — not the strict-RAG prompt.
"""
from __future__ import annotations

from typing import Iterator
import time
import ollama

from groq import Groq
import os

MODEL_NAME  = "llama-3.3-70b-versatile"
TEMPERATURE = 0.3
TOP_P       = 0.9

_client = None
def _get_client():
    global _client
    if _client is None:
        _client = Groq(api_key=os.environ["GROQ_API_KEY"])
    return _client
# Same Ollama config as generator.py
#OLLAMA_HOST = "http://localhost:11434"
MODEL_NAME   ="llama-3.3-70b-versatile" 
# MODEL_NAME  = "qwen2.5:7b"
NUM_CTX     = 4096
TEMPERATURE = 0.3   # slightly higher for friendliness
TOP_P       = 0.9
NUM_PREDICT = 1500
KEEP_ALIVE  = "30m"


SYSTEM_PROMPT = """You are a helpful, friendly AI assistant.

You have access to the conversation history below. Use it to:
- Answer follow-up questions about your previous responses
- Summarize, rephrase, translate, or modify previous answers
- Respond to greetings and casual conversation naturally
- Answer meta-questions about your capabilities

Be concise, natural, and helpful. You're not restricted to specific documents — 
you can answer based on the conversation context AND your general knowledge.

If asked about specific documents or information you don't have access to,
politely suggest the user ask a more specific question that would let you 
search the knowledge base."""



#loooocaallllllllll
"""
_client = None
def _get_client():
    global _client
    if _client is None:
        _client = ollama.Client(host=OLLAMA_HOST)
    return _client
"""

def _format_history(history: list[dict]) -> list[dict]:
    """Convert internal history format → Ollama messages format."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    for turn in history:
        role = turn.get("role", "user")
        # Map any non-standard roles
        if role not in ("user", "assistant", "system"):
            role = "user"
        messages.append({
            "role": role,
            "content": turn.get("content", ""),
        })
    
    return messages


def direct_chat_stream(
    question : str,
    history  : list[dict],
    on_token = None,
) -> dict:
    """
    Stream a direct conversational response.
    
    Args:
        question : the user's latest question
        history  : list of {"role", "content"} dicts (recent turns)
        on_token : optional callback called with each streamed token
    
    Returns:
        {
          "answer"    : full assembled answer (str),
          "tokens"    : {"prompt", "completion", "total"},
          "latency_s" : float,
          "no_answer" : False,  # always False for direct chat
          "sources"   : [],     # no documents used
        }
    """
    client = _get_client()
    
    # Build messages: system + history + new user question
    messages = _format_history(history)
    messages.append({"role": "user", "content": question})
    
    print(f"[DIRECT_CHAT] Calling {MODEL_NAME} | history_turns={len(history)}")
    
    full_answer: list[str] = []
    prompt_tokens = 0
    completion_tokens = 0
    
    t0 = time.perf_counter()

    #locaaaaaaaaaaaaaaaaal
    """
    stream = client.chat(
        model      = MODEL_NAME,
        messages   = messages,
        options    = {
            "num_ctx":     NUM_CTX,
            "temperature": TEMPERATURE,
            "top_p":       TOP_P,
            "num_predict": NUM_PREDICT,
        },
        stream     = True,
        keep_alive = KEEP_ALIVE,
    )
    """

    stream = client.chat.completions.create(
    model=MODEL_NAME,
    messages=messages,
    temperature=TEMPERATURE,
    top_p=TOP_P,
    max_tokens=NUM_PREDICT,
    stream=True,
)
    """
    for chunk in stream:
        token = chunk["message"]["content"]
        if token:
            full_answer.append(token)
            if on_token:
                on_token(token)
        if chunk.get("done"):
            prompt_tokens     = chunk.get("prompt_eval_count", 0)
            completion_tokens = chunk.get("eval_count", 0)
    """
    for chunk in stream:
     token = chunk.choices[0].delta.content or ""
     if token:
        full_answer.append(token)
        if on_token:
            on_token(token)
    latency = round(time.perf_counter() - t0, 3)
    answer  = "".join(full_answer).strip()
    
    print(f"[DIRECT_CHAT] Done | latency={latency}s | tokens={prompt_tokens}+{completion_tokens}")
    
    return {
        "answer":    answer,
        "sources":   [],
        "no_answer": False,
        "tokens":    {
            "prompt":     prompt_tokens,
            "completion": completion_tokens,
            "total":      prompt_tokens + completion_tokens,
        },
        "latency_s": latency,
    }


def direct_chat(question: str, history: list[dict]) -> dict:
    """Non-streaming version (collects all tokens before returning)."""
    return direct_chat_stream(question, history, on_token=None)