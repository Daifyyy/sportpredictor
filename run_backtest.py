import logging
from config.settings import settings
from api.client import APIClient
from data.fetcher import FootballFetcher
from models.poisson import DixonColesPredictor
from models.xgboost_model import XGBoostPredictor
from backtesting.engine import BacktestEngine

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


def run():
    client  = APIClient(settings)
    fetcher = FootballFetcher(client, settings)

    models = [
        DixonColesPredictor(),
        XGBoostPredictor(),
    ]

    for league_key, league in settings.leagues.items():
        history = fetcher.get_fixtures(league, status="FT")

        if len(history) < 110:
            print(f"\nSkipping {league.name}: only {len(history)} matches (need 110+)")
            continue

        for model in models:
            engine = BacktestEngine(model, min_train_size=100, retrain_every=10)
            result = engine.run(history, league_name=league.name)
            result.print_report()


if __name__ == "__main__":
    run()
