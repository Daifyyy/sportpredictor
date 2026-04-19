"""
Standalone script — enrich FT history with statistics and train corners models.
Does NOT touch the database or modify any predictions.
Safe to run at any time without side effects.
"""
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.client import APIClient
from config.settings import settings
from data.fetcher import FootballFetcher
from models.corners import train_corners_ensemble
from models.ensemble import CUP_LEAGUES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

MODELS_DIR = Path("models/saved")
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    client = APIClient(settings)
    fetcher = FootballFetcher(client, settings)

    for league_key, cfg in settings.leagues.items():
        if league_key in CUP_LEAGUES:
            print(f"\n[{cfg.name}] skipped (cup league)")
            continue

        print(f"\n[{cfg.name}]")
        history = fetcher.get_fixtures(cfg, status="FT")
        completed = [f for f in history if f.result is not None]
        print(f"  {len(completed)} completed fixtures loaded from cache")

        fetcher.enrich_full_history(completed)

        slug = cfg.name.lower().replace(" ", "_")
        cm = train_corners_ensemble(completed, cfg.season, MODELS_DIR, slug)
        if cm:
            print(f"  Corners ensemble saved to models/saved/corners_*_{slug}.joblib")
        else:
            print(f"  Corners ensemble not trained (insufficient enriched data)")

    print("\nDone. No database changes made.")


if __name__ == "__main__":
    main()
