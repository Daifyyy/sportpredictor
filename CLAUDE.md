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

## Current stack

- **Model**: `EnsembleDCPredictor` (`models/ensemble.py`) — blends 3 Dixon-Coles models (dc_all / dc_season / dc_recent) at λ/μ level
- **Weights (leagues)**: dc_all=0.25, dc_season=0.45, dc_recent=0.30
- **Weights (cups)**: dc_all=0.60, dc_season=0.30, dc_recent=0.10 — cups have fewer matches per team
- **Cup leagues**: `champions_league`, `europa_league`, `conference_league` (defined in `CUP_LEAGUES` in ensemble.py)
- **DB**: Supabase — `tracked_predictions` (user picks) + `fixture_predictions` (daily pre-computed cache)
- **Automation**: GitHub Actions — `predict.yml` (10:00 UTC) + `resolve.yml` (23:00 UTC)
- **Dashboard**: Streamlit Community Cloud (`dashboard.py`), reads DB directly

## Planned next steps

1. **Varianta C pro poháry** — použít parametry z domácích lig jako prior pro UCL/UEL MLE:
   - Natrénuj DC model na domácích ligách → získej `attack[tým]`, `defense[tým]`
   - Použij jako `x0` (startovací bod) pro UCL MLE místo nul
   - UCL data jen doladí parametry, týmy co hrají málo pohárových zápasů zdědí domácí sílu
   - Vyžaduje: mapování tým_id → liga, multi-league fetch před UCL tréninkem
