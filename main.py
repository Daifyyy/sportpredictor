import logging
from pathlib import Path
from config.settings import settings
from api.client import APIClient
from data.fetcher import FootballFetcher
from models.poisson import DixonColesPredictor

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

MODELS_DIR = Path("models/saved")


def run():
    client  = APIClient(settings)
    fetcher = FootballFetcher(client, settings)

    for league_key, league in settings.leagues.items():
        print(f"\n{'='*50}")
        print(f"  {league.name} | Sezóna {league.season}")
        print(f"{'='*50}")

        history  = fetcher.get_fixtures(league, status="FT")
        upcoming = fetcher.get_upcoming_fixtures(league, next_n=10)

        if len(history) < 50:
            print(f"  Nedostatek dat ({len(history)} zápasů) — přeskakuji.")
            continue

        model_path = MODELS_DIR / f"dixon_coles_{league_key}.joblib"
        if model_path.exists():
            logging.info(f"Načítám uložený model: {model_path}")
            model = DixonColesPredictor.load(model_path)
        else:
            logging.info(f"Trénuji Dixon-Coles pro {league.name}...")
            model = DixonColesPredictor()
            model.train(history)
            model.save(model_path)
            logging.info(f"Model uložen: {model_path}")

        for fixture in upcoming:
            prediction = model.predict(fixture, history)
            print(f"\n  {fixture.home_team.name} vs {fixture.away_team.name}")
            print(f"  {fixture.date.strftime('%d.%m %H:%M')}")
            print(f"  P(H/D/A): {prediction.prob_home:.1%} / {prediction.prob_draw:.1%} / {prediction.prob_away:.1%}")
            print(f"  xG: {prediction.expected_goals_home:.2f} - {prediction.expected_goals_away:.2f}")


if __name__ == "__main__":
    run()
