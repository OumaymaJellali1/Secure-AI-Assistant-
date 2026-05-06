"""
api/main.py — FastAPI app entry point with auth support.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import users, conversations, documents
from api.auth import router as auth_router

app = FastAPI(
    title="RAG Memory API",
    description="Backend for RAG chat with multi-conversation memory and real auth",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)       # NEW — /auth/register, /auth/login
app.include_router(users.router)
app.include_router(conversations.router)
app.include_router(documents.router)


@app.get("/health", tags=["meta"])
def health():
    return {"status": "ok", "service": "rag-memory-api", "version": "2.0.0"}


@app.get("/", tags=["meta"])
def root():
    return {
        "name": "RAG Memory API",
        "version": "2.0.0",
        "docs": "/docs",
        "auth": {
            "register": "POST /auth/register",
            "login": "POST /auth/login",
        },
    }