# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env and set: API_FOOTBALL_KEY=your_key
```

## Commands

```bash
# Run predictions for all 5 leagues
python main.py

# Run walk-forward backtesting for all leagues
python run_backtest.py
```

There is no test suite yet. Validation is done by running the scripts above.

## Architecture

The data flow is: `APIClient` (with cache) → `FootballFetcher` → `FeatureEngineer` / `DixonColesPredictor` → `ValueBetDetector`.

**Layer responsibilities:**

- `config/settings.py` — Single `Settings` instance (`settings`) imported everywhere. Contains `LeagueConfig` per league and `CacheTTL` values.
- `api/cache.py` — `CacheManager`: SQLite TTL cache. TTL=-1 means never expires (used for finished match history).
- `api/client.py` — `APIClient`: wraps `requests.Session` with token-bucket `RateLimiter` (300 req/min) and auto-caching. Pass `force_refresh=True` to bypass cache.
- `data/fetcher.py` — `FootballFetcher`: orchestrates API calls. Returns typed dataclasses.
- `data/models.py` — Core dataclasses: `Team`, `Fixture`, `MatchResult`, `Odds`, `Prediction`.
- `features/engineer.py` — `FeatureEngineer`: produces feature dicts (form, H2H, Elo). Add new features as new methods; pipeline composes automatically.
- `models/base.py` — `BasePredictor` ABC with `train(fixtures)`, `predict(fixture, history)`, `name`. All models must implement this.
- `models/poisson.py` — `DixonColesPredictor`: MLE-trained Poisson model with DC correction (rho=-0.13) for low-score results.
- `betting/value.py` — `ValueBetDetector`: compares model probabilities to Bet365 implied odds. Edge threshold: 3%.
- `backtesting/engine.py` — `BacktestEngine`: walk-forward simulation with periodic retrain. Reports accuracy, Brier score, log loss, and calibration.

## Key design decisions

- **Cache-first**: all API responses are cached; completed match results (status="FT") use TTL=-1 and are never re-fetched.
- **BasePredictor interface**: add new models (e.g., XGBoost) by subclassing `BasePredictor` — no changes needed elsewhere.
- **Value bet focus**: the product is built around edge detection (model prob vs. implied odds), not raw outcome prediction.
- Odds are fetched from Bet365 (bookmaker ID 11 in API-Football).

## Planned next steps (from PROJEKT_KONTEXT.md)

1. XGBoost model (same `BasePredictor` interface)
2. Model ensemble (Dixon-Coles + XGBoost)
3. Model persistence (pickle/joblib)
4. FastAPI layer with endpoints `/predictions/{league}`, `/value-bets`, `/track-record`
5. PostgreSQL for prediction history (SQLAlchemy)
6. Frontend: Next.js or Streamlit dashboard
