-- auth/migrations/create_users.sql
-- Run once: psql $DATABASE_URL -f auth/migrations/create_users.sql

CREATE TABLE IF NOT EXISTS users (
    id      VARCHAR(64)  PRIMARY KEY,
    email        VARCHAR(255) NOT NULL UNIQUE,
    is_admin     BOOLEAN      NOT NULL DEFAULT FALSE,
    token_file   VARCHAR(255) NOT NULL,
    created_at   TIMESTAMP    NOT NULL DEFAULT NOW(),
    last_crawl   TIMESTAMP    NULL,
    crawl_status VARCHAR(32)  NOT NULL DEFAULT 'pending'
                              CHECK (crawl_status IN ('pending', 'running', 'done', 'failed'))
);

CREATE INDEX IF NOT EXISTS idx_users_is_admin ON users(is_admin);
CREATE INDEX IF NOT EXISTS idx_users_email    ON users(email);