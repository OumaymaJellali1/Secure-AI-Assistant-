-- ════════════════════════════════════════════════════════════════════
-- MEMORY SYSTEM — Production Schema
-- Combines best of both approaches:
--   • Approach 2's row-per-turn + sessions entity + permissions table
--   • Approach 1's SQLAlchemy compatibility + simpler structure
-- ════════════════════════════════════════════════════════════════════


-- ─── USERS ────────────────────────────────────────────────────────
-- Mirror of MS Graph users. Phase 3 will add ms_graph_id, tenant_id.
CREATE TABLE IF NOT EXISTS users (
    id           TEXT PRIMARY KEY,
    email        TEXT UNIQUE,
    display_name TEXT,
    ms_graph_id  TEXT UNIQUE,                      -- nullable until Phase 3
    tenant_id    TEXT,                              -- nullable until Phase 3
    last_sync    TIMESTAMPTZ,                       -- when MS Graph last synced
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_users_email     ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_ms_graph  ON users(ms_graph_id);


-- ─── SESSIONS (CONVERSATIONS) ────────────────────────────────────
-- Each row = one conversation thread (like a ChatGPT chat).
-- A user can have many. Sortable by last_active for sidebar UI.
CREATE TABLE IF NOT EXISTS sessions (
    session_id   TEXT PRIMARY KEY,
    user_id      TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title        TEXT,                              -- auto-generated or user-set
    is_active    BOOLEAN NOT NULL DEFAULT TRUE,    -- false = archived
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_active  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_sessions_user_active
    ON sessions(user_id, last_active DESC)
    WHERE is_active = TRUE;


-- ─── CONVERSATION TURNS (Layer 1 — recent buffer) ────────────────
-- One row per message. Atomic inserts, fast queries, easy debugging.
CREATE TABLE IF NOT EXISTS conversation_turns (
    id           BIGSERIAL PRIMARY KEY,
    session_id   TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    user_id      TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    role         TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content      TEXT NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_turns_session_created
    ON conversation_turns(session_id, created_at DESC);


-- ─── SESSION SUMMARIES (Layer 3 — compressed history) ───────────
-- One row per session. Updated when buffer overflows.
-- Capped at MAX_SUMMARY_WORDS (500) by re-summarization.
CREATE TABLE IF NOT EXISTS session_summaries (
    session_id   TEXT PRIMARY KEY REFERENCES sessions(session_id) ON DELETE CASCADE,
    summary      TEXT NOT NULL DEFAULT '',
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);


-- ─── USER FACTS (Layer 4 — persistent across all conversations) ──
-- One row per fact. Deduplicated by content. Cap enforced in code.
CREATE TABLE IF NOT EXISTS user_facts (
    id           BIGSERIAL PRIMARY KEY,
    user_id      TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    fact         TEXT NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (user_id, fact)
);
CREATE INDEX IF NOT EXISTS idx_facts_user
    ON user_facts(user_id, created_at DESC);


-- ─── DOCUMENT PERMISSIONS (Phase 3 — empty until MS Graph syncs) ─
-- Pre-built so Phase 3 is just "populate this table."
-- Already integrates with the retriever filter logic.
CREATE TABLE IF NOT EXISTS document_permissions (
    id               BIGSERIAL PRIMARY KEY,
    user_id          TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    document_id      TEXT NOT NULL,                -- matches Qdrant payload field
    permission_level TEXT NOT NULL DEFAULT 'read'
                          CHECK (permission_level IN ('read', 'write', 'admin')),
    granted_by       TEXT REFERENCES users(id),
    granted_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at       TIMESTAMPTZ,                  -- NULL = permanent
    UNIQUE (user_id, document_id)
);
CREATE INDEX IF NOT EXISTS idx_perms_user
    ON document_permissions(user_id);
CREATE INDEX IF NOT EXISTS idx_perms_document
    ON document_permissions(document_id);


-- ─── AUTO-UPDATE updated_at TRIGGERS ─────────────────────────────
CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_users_updated ON users;
CREATE TRIGGER trg_users_updated
    BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();


-- ─── DEV USERS ────────────────────────────────────────────────────
-- For testing before MS Graph integration.
INSERT INTO users (id, email, display_name) VALUES
    ('dev_alice', 'alice@dev.local', 'Alice (dev)'),
    ('dev_bob',   'bob@dev.local',   'Bob (dev)'),
    ('dev_test',  'test@dev.local',  'Test User')
ON CONFLICT (id) DO NOTHING;