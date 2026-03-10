-- Migration v3: Add team logo columns to fixture_predictions
-- Run this in Supabase SQL Editor

ALTER TABLE fixture_predictions
    ADD COLUMN IF NOT EXISTS home_logo VARCHAR,
    ADD COLUMN IF NOT EXISTS away_logo VARCHAR;
