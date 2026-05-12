# db/drive_permissions.py

from sqlalchemy import text
from shared.db import engine


def create_tables() -> None:
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS drive_files (
                file_id       TEXT PRIMARY KEY,
                name          TEXT,
                mime_type     TEXT,
                owner_email   TEXT NOT NULL,
                allowed_users TEXT[],
                is_public     BOOLEAN DEFAULT false,
                crawled_at    TIMESTAMPTZ DEFAULT now()
            )
        """))
    print("[DRIVE DB] Table ready.")


def upsert_file(
    file_id       : str,
    name          : str,
    mime_type     : str,
    owner_email   : str,
    allowed_users : list[str],
    is_public     : bool,
) -> None:
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO drive_files 
                (file_id, name, mime_type, owner_email, allowed_users, is_public, crawled_at)
            VALUES 
                (:fid, :name, :mime, :owner, :allowed, :pub, now())
            ON CONFLICT (file_id) DO UPDATE
                SET name          = EXCLUDED.name,
                    mime_type     = EXCLUDED.mime_type,
                    owner_email   = EXCLUDED.owner_email,
                    allowed_users = EXCLUDED.allowed_users,
                    is_public     = EXCLUDED.is_public,
                    crawled_at    = now()
        """), {
            "fid"    : file_id,
            "name"   : name,
            "mime"   : mime_type,
            "owner"  : owner_email,
            "allowed": allowed_users,
            "pub"    : is_public,
        })


def get_allowed_users(file_id: str) -> list[str]:
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT allowed_users FROM drive_files WHERE file_id = :fid"),
            {"fid": file_id}
        ).fetchone()
    return row[0] if row else []