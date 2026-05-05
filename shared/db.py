"""
shared/db.py — Centralized Postgres engine.
Used by csv_processing, memory, and any other feature that needs the database.
"""
import os
import sys
from sqlalchemy import create_engine
from dotenv import load_dotenv
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()

# ── Postgres config (from .env file) ──────────────────────────────────────
PG_HOST     = os.environ.get("PG_HOST",     "localhost")
PG_PORT     = os.environ.get("PG_PORT",     "5432")
PG_DB       = os.environ.get("PG_DB",       "csvstore")
PG_USER     = os.environ.get("PG_USER",     "csvuser")
PG_PASSWORD = os.environ.get("PG_PASSWORD", "yourpassword")

DATABASE_URL = (
    f"postgresql+psycopg2://{PG_USER}:{PG_PASSWORD}"
    f"@{PG_HOST}:{PG_PORT}/{PG_DB}"
)

# ── Single shared engine (created once, reused everywhere) ────────────────
engine = create_engine(
    DATABASE_URL,
    future=True,
    pool_pre_ping=True,   # auto-detects stale connections
    pool_recycle=3600,    # recycle connections after 1 hour
)


def get_engine():
    """Returns the shared engine. Kept for backward compat with existing code."""
    return engine