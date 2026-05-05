"""
api/auth.py — Authentication dependency.

PHASE 1 (now):
  Reads X-Dev-User-Id header to identify which dev user is "logged in."
  Frontend sends this based on the dropdown selection.

PHASE 3 (later, MS Graph):
  Replace get_current_user() body with:
    1. Read Authorization: Bearer <token>
    2. Verify token with Microsoft public keys
    3. Look up user in users table by ms_graph_id
    4. Return that user_id
  
  Function signature stays the same → no other code changes!
"""
from fastapi import Header, HTTPException, status
from sqlalchemy import text

from shared.db import engine


# Whitelist of dev users (matches users inserted by schema.sql)
DEV_USERS = {"dev_alice", "dev_bob", "dev_test"}


def get_current_user(
    x_dev_user_id: str | None = Header(default=None, alias="X-Dev-User-Id"),
) -> str:
    """
    Dependency that returns the current user_id.
    
    In dev mode: reads from X-Dev-User-Id header.
    In production: would verify a real JWT token.
    
    Returns:
        user_id (str) — guaranteed to exist in users table
    
    Raises:
        401 if header missing
        403 if user_id not in whitelist
        404 if user not found in DB
    """
    # 1. Header must be present
    if not x_dev_user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-Dev-User-Id header",
        )
    
    # 2. Must be a valid dev user
    if x_dev_user_id not in DEV_USERS:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Invalid dev user: {x_dev_user_id}",
        )
    
    # 3. Verify they exist in DB (paranoid but cheap)
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT id FROM users WHERE id = :uid"),
            {"uid": x_dev_user_id},
        ).fetchone()
    
    if not row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User {x_dev_user_id} not in database. Did you run apply_schema.py?",
        )
    
    return x_dev_user_id


def get_current_user_full(
    user_id: str = None,
) -> dict:
    """Return full user info (id, email, display_name)."""
    if user_id is None:
        # Dependency-injection compatible
        from fastapi import Depends
        user_id = Depends(get_current_user)
    
    with engine.connect() as conn:
        row = conn.execute(
            text("""
                SELECT id, email, display_name 
                FROM users WHERE id = :uid
            """),
            {"uid": user_id},
        ).fetchone()
    
    if not row:
        raise HTTPException(404, f"User {user_id} not found")
    
    return {
        "id": row[0],
        "email": row[1],
        "display_name": row[2],
    }