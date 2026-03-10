-- Migration: Replace old tables with tracked_predictions
-- Run this in Supabase SQL Editor

-- Drop old tables (safe - backup branch exists)
DROP TABLE IF EXISTS predictions CASCADE;
DROP TABLE IF EXISTS backtest_runs CASCADE;
DROP TABLE IF EXISTS bankroll CASCADE;

-- Create new table
CREATE TABLE IF NOT EXISTS tracked_predictions (
    id SERIAL PRIMARY KEY,
    fixture_id INTEGER NOT NULL,
    league VARCHAR NOT NULL,
    home_team VARCHAR NOT NULL,
    away_team VARCHAR NOT NULL,
    match_date TIMESTAMPTZ NOT NULL,
    prediction_type VARCHAR NOT NULL,
    model_prob FLOAT,
    actual_outcome VARCHAR,
    correct BOOLEAN,
    home_score INTEGER,
    away_score INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT uq_fixture_prediction UNIQUE (fixture_id, prediction_type)
);

-- Index pro rychlé filtrování
CREATE INDEX IF NOT EXISTS idx_tp_league ON tracked_predictions (league);
CREATE INDEX IF NOT EXISTS idx_tp_correct ON tracked_predictions (correct);
CREATE INDEX IF NOT EXISTS idx_tp_match_date ON tracked_predictions (match_date);
