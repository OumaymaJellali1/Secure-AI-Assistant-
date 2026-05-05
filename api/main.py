"""
api/main.py — FastAPI app entry point.

Run with:
    uvicorn api.main:app --reload --port 8000

Then visit:
    http://localhost:8000/docs    (interactive Swagger UI)
    http://localhost:8000/health  (health check)
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import users, conversations, documents


# ─── APP ─────────────────────────────────────────────────────────

app = FastAPI(
    title="RAG Memory API",
    description="Backend for Fluent UI chat with multi-conversation memory",
    version="1.0.0",
)


# ─── CORS (allow frontend on localhost:3000 to call us) ─────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",   # Vite dev server (default)
        "http://localhost:5173",   # Vite alt port
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── ROUTES ──────────────────────────────────────────────────────

app.include_router(users.router)
app.include_router(conversations.router)
app.include_router(documents.router)


# ─── HEALTH CHECK ────────────────────────────────────────────────

@app.get("/health", tags=["meta"])
def health():
    """Quick check that the API is up."""
    return {"status": "ok", "service": "rag-memory-api"}


@app.get("/", tags=["meta"])
def root():
    return {
        "name": "RAG Memory API",
        "version": "1.0.0",
        "docs": "/docs",
    }