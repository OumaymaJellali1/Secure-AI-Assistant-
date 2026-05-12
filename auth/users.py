# auth/users.py
# Full user registry backed by PostgreSQL.
# Replaces the old auth/users.json file entirely.

from __future__ import annotations
import sys
from datetime import datetime, timezone
from sqlalchemy import text
from shared.db import engine   # your existing SQLAlchemy engine


# ── READ ──────────────────────────────────────────────────────────────────────

def get_user(id: str) -> dict | None:
    """Return one user row as a dict, or None if not found."""
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT * FROM users WHERE id = :uid"),
            {"uid": id}
        ).fetchone()
    if row is None:
        return None
    return dict(row._mapping)


def get_user_by_email(email: str) -> dict | None:
    """Look up a user by their Gmail address."""
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT * FROM users WHERE email = :email"),
            {"email": email}
        ).fetchone()
    return dict(row._mapping) if row else None


def load_users() -> dict[str, dict]:
    """Return all users as {id: user_dict}."""
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT * FROM users ORDER BY id")
        ).fetchall()
    return {row._mapping["id"]: dict(row._mapping) for row in rows}


def is_admin(id: str) -> bool:
    user = get_user(id)
    return bool(user and user.get("is_admin"))


# ── WRITE ─────────────────────────────────────────────────────────────────────

def add_user(
    id: str,
    email: str,
    is_admin: bool = False,
) -> dict:
    """
    Insert a new user.
    Raises ValueError if id or email already exists.
    Returns the created user dict.
    """
    token_file = f"tokens/{id}.json"

    with engine.begin() as conn:
        existing = conn.execute(
            text("SELECT id FROM users WHERE id = :uid OR email = :email"),
            {"uid": id, "email": email}
        ).fetchone()

        if existing:
            raise ValueError(
                f"User already exists: id={id} or email={email}"
            )

        conn.execute(
            text("""
                INSERT INTO users (id, email, is_admin, token_file)
                VALUES (:uid, :email, :admin, :token)
            """),
            {
                "uid":   id,
                "email": email,
                "admin": is_admin,
                "token": token_file,
            }
        )

    print(f"[USERS] Added: {id} ({email}) admin={is_admin}")
    return get_user(id)


def upsert_user(
    id: str,
    email: str,
    is_admin: bool = False,
) -> dict:
    """Insert or update a user (safe to call repeatedly)."""
    token_file = f"tokens/{id}.json"

    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO users (id, email, is_admin, token_file)
                VALUES (:uid, :email, :admin, :token)
                ON CONFLICT (id) DO UPDATE
                    SET email      = EXCLUDED.email,
                        is_admin   = EXCLUDED.is_admin,
                        token_file = EXCLUDED.token_file
            """),
            {
                "uid":   id,
                "email": email,
                "admin": is_admin,
                "token": token_file,
            }
        )

    return get_user(id)


def update_crawl_status(
    id: str,
    status: str,                         # 'pending' | 'running' | 'done' | 'failed'
    last_crawl: datetime | None = None,
) -> None:
    """Update crawl status and optionally the last_crawl timestamp."""
    params: dict = {"uid": id, "status": status}

    if last_crawl is not None:
        sql = text("""
            UPDATE users
            SET crawl_status = :status,
                last_crawl   = :ts
            WHERE id = :uid
        """)
        params["ts"] = last_crawl
    else:
        sql = text("""
            UPDATE users
            SET crawl_status = :status
            WHERE id = :uid
        """)

    with engine.begin() as conn:
        conn.execute(sql, params)


def delete_user(id: str) -> None:
    """
    Remove a user from the registry.
    Does NOT delete their token file or Qdrant data — do that separately.
    """
    with engine.begin() as conn:
        conn.execute(
            text("DELETE FROM users WHERE id = :uid"),
            {"uid": id}
        )
    print(f"[USERS] Deleted: {id}")


# ── CLI ───────────────────────────────────────────────────────────────────────
# Manage users from the terminal:
#
#   python auth/users.py list
#   python auth/users.py add alice alice@gmail.com --admin
#   python auth/users.py add bob   bob@gmail.com
#   python auth/users.py delete bob

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "list"

    if cmd == "list":
        users = load_users()
        if not users:
            print("No users registered.")
        else:
            print(f"{'ID':<20} {'EMAIL':<35} {'ROLE':<8} {'STATUS':<10} LAST_CRAWL")
            print("-" * 90)
            for uid, u in users.items():
                role   = "ADMIN" if u["is_admin"] else "user"
                crawl  = str(u.get("last_crawl") or "never")[:19]
                status = u.get("crawl_status", "?")
                print(f"  {uid:<20} {u['email']:<35} [{role}]   {status:<10} {crawl}")

    elif cmd == "add":
        if len(sys.argv) < 4:
            print("Usage: python auth/users.py add <d> <email> [--admin]")
            sys.exit(1)
        uid    = sys.argv[2]
        email  = sys.argv[3]
        admin  = "--admin" in sys.argv
        user   = upsert_user(uid, email, is_admin=admin)
        print(f"Saved: {user}")

    elif cmd == "delete":
        if len(sys.argv) < 3:
            print("Usage: python auth/users.py delete <id>")
            sys.exit(1)
        delete_user(sys.argv[2])

    else:
        print("Commands:")
        print("  list")
        print("  add <id> <email> [--admin]")
        print("  delete <id>")