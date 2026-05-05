"""
api/routes/users.py — Endpoint to list dev users.

The frontend calls GET /users to populate the user-switcher dropdown.
In Phase 3 (MS Graph), this could return ME info instead.
"""
from fastapi import APIRouter, Depends
from sqlalchemy import text

from shared.db import engine
from api.auth import get_current_user
from api.schemas import UserOut


router = APIRouter(prefix="/users", tags=["users"])


@router.get("", response_model=list[UserOut])
def list_users():
    """
    Return all dev users for the dropdown.
    No auth required — public endpoint to populate login UI.
    """
    with engine.connect() as conn:
        rows = conn.execute(
            text("""
                SELECT id, email, display_name 
                FROM users 
                WHERE id LIKE 'dev_%'
                ORDER BY id
            """)
        ).fetchall()
    
    return [
        UserOut(id=r[0], email=r[1], display_name=r[2])
        for r in rows
    ]


@router.get("/me", response_model=UserOut)
def get_me(user_id: str = Depends(get_current_user)):
    """Return the currently 'logged in' user."""
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT id, email, display_name FROM users WHERE id = :uid"),
            {"uid": user_id},
        ).fetchone()
    
    return UserOut(id=row[0], email=row[1], display_name=row[2])