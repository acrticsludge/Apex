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

-- ── RL Agent decision log (separate from the main state blob) ────────────────
-- Each row is one PPO inference call: symbol, action probabilities, reasoning.
CREATE TABLE IF NOT EXISTS apex_rl_decisions (
    id         BIGSERIAL    PRIMARY KEY,
    ts         TIMESTAMPTZ  DEFAULT NOW(),
    symbol     TEXT         NOT NULL,
    market     TEXT         NOT NULL,
    action     SMALLINT     NOT NULL,   -- 0=hold  1=buy  2=sell
    prob_hold  REAL         NOT NULL,
    prob_buy   REAL         NOT NULL,
    prob_sell  REAL         NOT NULL,
    confidence REAL         NOT NULL,
    price      REAL         NOT NULL,
    reasoning  JSONB
);
ALTER TABLE apex_rl_decisions DISABLE ROW LEVEL SECURITY;
