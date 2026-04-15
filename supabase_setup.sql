-- Run this once in your Supabase SQL Editor
-- (Dashboard → SQL Editor → New query → paste → Run)

CREATE TABLE IF NOT EXISTS apex_state (
    id         TEXT        PRIMARY KEY,   -- always "singleton"
    data       JSONB       NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Disable RLS so the service-role key can read/write without policies
ALTER TABLE apex_state DISABLE ROW LEVEL SECURITY;

-- Optional: seed an empty row so the first upsert is always an UPDATE
-- (avoids a race on first boot; safe to skip if you prefer lazy creation)
INSERT INTO apex_state (id, data)
VALUES ('singleton', '{"india": {}, "us": {}, "started_at": null}'::jsonb)
ON CONFLICT (id) DO NOTHING;
