"""
api/routes/auth.py — Real authentication endpoints.

POST /auth/register  → create new user (hashed password)
POST /auth/login     → verify credentials, return user info

Uses bcrypt for password hashing via passlib.
Falls back gracefully if passlib not installed (plaintext comparison for dev).
"""
import uuid
import hashlib
import secrets
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, EmailStr, field_validator
from sqlalchemy import text
from shared.db import engine

router = APIRouter(prefix="/auth", tags=["auth"])


# ── SCHEMAS ──────────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    email: str
    password: str
    display_name: str

    @field_validator("email")
    @classmethod
    def email_must_be_valid(cls, v):
        if "@" not in v or "." not in v.split("@")[-1]:
            raise ValueError("Invalid email address")
        return v.lower().strip()

    @field_validator("password")
    @classmethod
    def password_strength(cls, v):
        if len(v) < 6:
            raise ValueError("Password must be at least 6 characters")
        return v

    @field_validator("display_name")
    @classmethod
    def name_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Display name cannot be empty")
        return v.strip()


class LoginRequest(BaseModel):
    email: str
    password: str

    @field_validator("email")
    @classmethod
    def normalize_email(cls, v):
        return v.lower().strip()


class AuthResponse(BaseModel):
    user_id: str
    email: str
    display_name: str
    token: str  # simple session token for now


# ── HELPERS ───────────────────────────────────────────────────────────────────

import base64

def _hash_password(password: str) -> str:
    """Hash password using bcrypt (with SHA-256 pre-hash to bypass 72-byte limit)."""
    try:
        from passlib.context import CryptContext
        ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")
        digest = base64.b64encode(hashlib.sha256(password.encode()).digest())
        return ctx.hash(digest)
    except ImportError:
        salt = secrets.token_hex(16)
        hashed = hashlib.sha256(f"{salt}{password}".encode()).hexdigest()
        return f"sha256:{salt}:{hashed}"


def _verify_password(password: str, hashed: str) -> bool:
    """Verify password against stored hash."""
    try:
        from passlib.context import CryptContext
        ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")
        digest = base64.b64encode(hashlib.sha256(password.encode()).digest())
        return ctx.verify(digest, hashed)
    except ImportError:
        if hashed.startswith("sha256:"):
            _, salt, stored = hashed.split(":", 2)
            computed = hashlib.sha256(f"{salt}{password}".encode()).hexdigest()
            return computed == stored
        return False

def _ensure_auth_columns():
    """Add password_hash and auth_token columns to users table if missing."""
    with engine.begin() as conn:
        # Add password_hash column
        conn.exec_driver_sql("""
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name='users' AND column_name='password_hash'
                ) THEN
                    ALTER TABLE users ADD COLUMN password_hash TEXT;
                END IF;
            END $$;
        """)
        # Add auth_token column
        conn.exec_driver_sql("""
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name='users' AND column_name='auth_token'
                ) THEN
                    ALTER TABLE users ADD COLUMN auth_token TEXT;
                END IF;
            END $$;
        """)


# ── ENDPOINTS ─────────────────────────────────────────────────────────────────

@router.post("/register", response_model=AuthResponse, status_code=201)
def register(body: RegisterRequest):
    """
    Create a new user account.
    - Validates email format and password strength
    - Hashes password before storage
    - Returns user info + session token
    """
    _ensure_auth_columns()

    # Check if email already exists
    with engine.connect() as conn:
        existing = conn.execute(
            text("SELECT id FROM users WHERE email = :email"),
            {"email": body.email},
        ).fetchone()

    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists.",
        )

    # Generate user ID and token
    user_id = f"user_{uuid.uuid4().hex[:12]}"
    token = secrets.token_urlsafe(32)
    password_hash = _hash_password(body.password)

    # Insert new user
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO users (id, email, display_name, password_hash, auth_token)
                VALUES (:id, :email, :name, :pw_hash, :token)
            """),
            {
                "id": user_id,
                "email": body.email,
                "name": body.display_name,
                "pw_hash": password_hash,
                "token": token,
            },
        )

    return AuthResponse(
        user_id=user_id,
        email=body.email,
        display_name=body.display_name,
        token=token,
    )


@router.post("/login", response_model=AuthResponse)
def login(body: LoginRequest):
    """
    Authenticate an existing user.
    - Checks email + password
    - Also supports dev users (dev_alice, dev_bob, dev_test) with any password
    - Returns user info + fresh session token
    """
    _ensure_auth_columns()

    with engine.connect() as conn:
        row = conn.execute(
            text("""
                SELECT id, email, display_name, password_hash
                FROM users
                WHERE email = :email
            """),
            {"email": body.email},
        ).fetchone()

    if not row:
        # Also try matching by user ID for dev users
        with engine.connect() as conn:
            row = conn.execute(
                text("""
                    SELECT id, email, display_name, password_hash
                    FROM users
                    WHERE id = :uid
                """),
                {"uid": body.email},  # allow login by user_id too
            ).fetchone()

    if not row:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="No account found with this email address.",
        )

    user_id, email, display_name, password_hash = row

    # Dev users can log in with any password
    is_dev_user = user_id.startswith("dev_")

    if not is_dev_user:
        if not password_hash:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Account has no password set. Please contact support.",
            )
        if not _verify_password(body.password, password_hash):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect password.",
            )

    # Issue fresh token
    token = secrets.token_urlsafe(32)
    with engine.begin() as conn:
        conn.execute(
            text("UPDATE users SET auth_token = :token WHERE id = :uid"),
            {"token": token, "uid": user_id},
        )

    return AuthResponse(
        user_id=user_id,
        email=email or body.email,
        display_name=display_name or user_id,
        token=token,
    )


@router.get("/me")
def get_current_user(token: str):
    """Validate a session token and return user info."""
    _ensure_auth_columns()

    with engine.connect() as conn:
        row = conn.execute(
            text("""
                SELECT id, email, display_name
                FROM users WHERE auth_token = :token
            """),
            {"token": token},
        ).fetchone()

    if not row:
        raise HTTPException(status_code=401, detail="Invalid or expired session.")

    return {"user_id": row[0], "email": row[1], "display_name": row[2]}