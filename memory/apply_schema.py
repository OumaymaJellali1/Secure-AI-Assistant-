"""
memory/apply_schema.py — Apply the memory schema to Postgres.

Run once: python -m memory.apply_schema
Safe to re-run (uses CREATE TABLE IF NOT EXISTS).
"""
from pathlib import Path
from sqlalchemy import text

from shared.db import engine


def apply_schema():
    schema_path = Path(__file__).parent / "schema.sql"

    if not schema_path.exists():
        print(f"❌ Cannot find {schema_path}")
        return

    sql = schema_path.read_text(encoding="utf-8")

    print(f"📄 Applying {schema_path.name}...\n")

    with engine.begin() as conn:
        conn.exec_driver_sql(sql)

    print("✅ Schema applied successfully.\n")

    # Verify tables exist
    expected_tables = [
        "users",
        "sessions",
        "conversation_turns",
        "session_summaries",
        "user_facts",
        "document_permissions",
    ]

    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
              AND table_name = ANY(:names)
            ORDER BY table_name
        """), {"names": expected_tables}).fetchall()

        found = [r[0] for r in rows]
        print(f"📋 Tables found ({len(found)}/{len(expected_tables)}):")
        for t in expected_tables:
            mark = "✅" if t in found else "❌"
            print(f"   {mark} {t}")
        print()

    # Verify dev users
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT id, display_name FROM users WHERE id LIKE 'dev_%' ORDER BY id")
        ).fetchall()

        print(f"👥 Dev users ({len(rows)}):")
        for uid, name in rows:
            print(f"   • {uid:<12} — {name}")
        print()

    print("🎉 Memory DB is ready!")


if __name__ == "__main__":
    apply_schema()