"""
Standalone predict script for GitHub Actions.
Trains ensemble DC model for each league, fetches upcoming fixtures,
and upserts pre-computed probabilities into fixture_predictions table.
"""
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.stats import poisson

from api.client import APIClient
from config.settings import settings
from data.fetcher import FootballFetcher
from db.models import Base, FixturePrediction
from db.session import SessionLocal, engine
from models.ensemble import EnsembleDCPredictor
from models.poisson import DixonColesPredictor

MODELS_DIR = Path("models/saved")


def compute_goal_probs(lam: float, mu: float) -> dict:
    max_g = 10
    h_pmf = poisson.pmf(range(max_g), lam)
    a_pmf = poisson.pmf(range(max_g), mu)
    mat = np.outer(h_pmf, a_pmf)
    tg = np.zeros(max_g * 2 - 1)
    for i in range(max_g):
        for j in range(max_g):
            tg[i + j] += mat[i, j]
    btts = float(sum(mat[i, j] for i in range(1, max_g) for j in range(1, max_g)))
    return {
        "over2_5": round(float(tg[3:].sum()), 4),
        "under2_5": round(float(tg[:3].sum()), 4),
        "goals1_3": round(float(tg[1:4].sum()), 4),
        "goals2_4": round(float(tg[2:5].sum()), 4),
        "btts_yes": round(btts, 4),
        "btts_no": round(1 - btts, 4),
    }


def train_ensemble(completed, cfg) -> EnsembleDCPredictor | None:
    if len(completed) < 50:
        print(f"  Not enough data ({len(completed)} matches), skipping.")
        return None

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # dc_all
    dc_all = DixonColesPredictor()
    dc_all.train(completed)
    dc_all.save(MODELS_DIR / f"dc_all_{cfg.name.lower().replace(' ', '_')}.joblib")
    print(f"  dc_all trained on {len(completed)} matches")

    # dc_season
    season_fixtures = [f for f in completed if f.season == cfg.season]
    if len(season_fixtures) >= 30:
        dc_season = DixonColesPredictor()
        dc_season.train(season_fixtures)
        dc_season.save(MODELS_DIR / f"dc_season_{cfg.name.lower().replace(' ', '_')}.joblib")
        print(f"  dc_season trained on {len(season_fixtures)} matches")
    else:
        dc_season = dc_all
        print(f"  dc_season fallback to dc_all ({len(season_fixtures)} season matches)")

    # dc_recent (last 60 days)
    cutoff = max(f.date for f in completed) - timedelta(days=60)
    recent_fixtures = [f for f in completed if f.date >= cutoff]
    if len(recent_fixtures) >= 30:
        dc_recent = DixonColesPredictor()
        dc_recent.train(recent_fixtures)
        print(f"  dc_recent trained on {len(recent_fixtures)} matches (last 60 days)")
    else:
        dc_recent = None
        print(f"  dc_recent skipped ({len(recent_fixtures)} recent matches)")

    return EnsembleDCPredictor(dc_all, dc_season, dc_recent)


def main():
    Base.metadata.create_all(engine)
    client = APIClient(settings)
    fetcher = FootballFetcher(client, settings)
    db = SessionLocal()

    try:
        for league_key, cfg in settings.leagues.items():
            print(f"\n[{cfg.name}]")

            history = fetcher.get_fixtures(cfg, status="FT")
            completed = [f for f in history if f.result is not None]
            model = train_ensemble(completed, cfg)
            if model is None:
                continue

            upcoming = fetcher.get_upcoming_fixtures(cfg, next_n=10)
            if not upcoming:
                print("  No upcoming fixtures found.")
                continue

            print(f"  Computing predictions for {len(upcoming)} upcoming fixtures...")

            # Delete stale rows for this league
            db.query(FixturePrediction).filter(FixturePrediction.league == league_key).delete()

            for fx in upcoming:
                pred = model.predict(fx)
                gp = compute_goal_probs(pred.expected_goals_home, pred.expected_goals_away)

                row = FixturePrediction(
                    fixture_id=fx.id,
                    league=league_key,
                    home_team=fx.home_team.name,
                    away_team=fx.away_team.name,
                    match_date=fx.date,
                    prob_home=round(pred.prob_home, 4),
                    prob_draw=round(pred.prob_draw, 4),
                    prob_away=round(pred.prob_away, 4),
                    over2_5=gp["over2_5"],
                    under2_5=gp["under2_5"],
                    goals1_3=gp["goals1_3"],
                    goals2_4=gp["goals2_4"],
                    btts_yes=gp["btts_yes"],
                    btts_no=gp["btts_no"],
                )
                db.add(row)
                print(f"  {fx.home_team.name} vs {fx.away_team.name} | H:{pred.prob_home:.0%} D:{pred.prob_draw:.0%} A:{pred.prob_away:.0%}")

            db.commit()
            print(f"  Saved {len(upcoming)} predictions.")

    finally:
        db.close()


if __name__ == "__main__":
    main()
