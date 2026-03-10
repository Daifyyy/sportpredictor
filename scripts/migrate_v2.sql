-- Migration v2: Add fixture_predictions table
-- Run this in Supabase SQL Editor

CREATE TABLE IF NOT EXISTS fixture_predictions (
    fixture_id INTEGER PRIMARY KEY,
    league VARCHAR NOT NULL,
    home_team VARCHAR NOT NULL,
    away_team VARCHAR NOT NULL,
    match_date TIMESTAMPTZ NOT NULL,
    prob_home FLOAT NOT NULL,
    prob_draw FLOAT NOT NULL,
    prob_away FLOAT NOT NULL,
    over2_5 FLOAT NOT NULL,
    under2_5 FLOAT NOT NULL,
    goals1_3 FLOAT NOT NULL,
    goals2_4 FLOAT NOT NULL,
    btts_yes FLOAT NOT NULL,
    btts_no FLOAT NOT NULL,
    computed_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_fp_league ON fixture_predictions (league);
CREATE INDEX IF NOT EXISTS idx_fp_match_date ON fixture_predictions (match_date);
