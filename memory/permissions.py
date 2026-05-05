"""
memory/permissions.py — Document access control (Phase 3 ready).

Until MS Graph sync is built, the document_permissions table is empty.
get_allowed_doc_ids() returns None when empty, meaning "no filter applied"
(backward compatible with current behavior).

When MS Graph populates the table, retrieval automatically filters
to only documents the user can access.
"""
from typing import Optional
from sqlalchemy import text

from shared.db import engine


class PermissionLoader:
    """
    Loads document permissions for the retriever.
    Designed to work seamlessly before AND after MS Graph integration.
    """

    def __init__(self):
        # Simple in-memory cache (can swap for Redis later)
        self._cache = {}

    def get_allowed_doc_ids(self, user_id: str) -> Optional[list[str]]:
        """
        Return list of document_ids the user can access.

        Returns:
          None  → no permissions configured (Phase 1: don't filter, return everything)
          []    → user has access to nothing
          [...] → user's allowed documents
        """
        # Check cache
        if user_id in self._cache:
            return self._cache[user_id]

        with engine.connect() as conn:
            # Are there ANY permissions in the system?
            total = conn.execute(
                text("SELECT COUNT(*) FROM document_permissions")
            ).scalar()

            # Phase 1 mode: empty permissions table means "no filtering yet"
            if total == 0:
                self._cache[user_id] = None
                return None

            # Phase 3+ mode: filter to user's allowed docs
            rows = conn.execute(
                text("""
                    SELECT document_id
                    FROM document_permissions
                    WHERE user_id = :uid
                      AND (expires_at IS NULL OR expires_at > NOW())
                """),
                {"uid": user_id},
            ).fetchall()

        doc_ids = [r[0] for r in rows]
        self._cache[user_id] = doc_ids
        return doc_ids

    def can_access(self, user_id: str, document_id: str) -> bool:
        """Check if a specific user can access a specific document."""
        allowed = self.get_allowed_doc_ids(user_id)
        if allowed is None:
            return True  # Phase 1: no enforcement yet
        return document_id in allowed

    def grant(
        self,
        user_id: str,
        document_id: str,
        permission_level: str = "read",
        granted_by: Optional[str] = None,
    ) -> None:
        """Grant a user access to a document. Used by MS Graph sync."""
        with engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO document_permissions
                        (user_id, document_id, permission_level, granted_by)
                    VALUES (:uid, :did, :lvl, :by)
                    ON CONFLICT (user_id, document_id) DO UPDATE
                    SET permission_level = EXCLUDED.permission_level,
                        granted_by       = EXCLUDED.granted_by,
                        granted_at       = NOW()
                """),
                {"uid": user_id, "did": document_id, "lvl": permission_level, "by": granted_by},
            )
        self._cache.pop(user_id, None)  # invalidate cache

    def revoke(self, user_id: str, document_id: str) -> None:
        """Revoke a user's access to a document."""
        with engine.begin() as conn:
            conn.execute(
                text("""
                    DELETE FROM document_permissions
                    WHERE user_id = :uid AND document_id = :did
                """),
                {"uid": user_id, "did": document_id},
            )
        self._cache.pop(user_id, None)

    def invalidate_cache(self, user_id: Optional[str] = None) -> None:
        """Clear cache for a user, or all users if None. Call after MS Graph sync."""
        if user_id:
            self._cache.pop(user_id, None)
        else:
            self._cache.clear()