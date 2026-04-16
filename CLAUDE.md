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
# Run predictions for all 9 leagues
python scripts/predict.py

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
- `data/models.py` — Core dataclasses: `Team`, `Fixture`, `MatchResult`, `Odds`, `Prediction`, `PlayerInjury`, `FixtureStats`, `LineupPlayer`, `FixtureLineup`.
- `features/engineer.py` — `FeatureEngineer`: produces feature dicts (form, H2H, Elo, venue, streak, rest days, xG stats). `precompute()` builds indexes in O(n); `build_features()` is leakage-free per fixture.
- `models/base.py` — `BasePredictor` ABC with `train(fixtures)`, `predict(fixture, history)`, `name`. All models must implement this.
- `models/poisson.py` — `DixonColesPredictor`: MLE-trained Poisson model with DC correction (rho=-0.13) for low-score results. `train()` accepts `attack_prior`/`defence_prior` for Varianta C (cup leagues).
- `models/calibrator.py` — `ProbabilityCalibrator`: isotonic regression per H/D/A; `GoalCalibrator`: isotonic regression na λ+μ → skutečné celkové góly (opravuje Poissonův bias nezávislosti). Oba fittovány v jednom walk-forward průchodu (`build_calibrators()`), NaN filtered.
- `models/injury.py` — `InjuryAdjuster`: adjusts λ/μ for injuries. Uses `TYPE_CHECKING` guard for `PlayerInjury` import.
- `betting/value.py` — `ValueBetDetector`: compares model probabilities to Bet365 implied odds. Edge threshold: 3%.
- `backtesting/engine.py` — `BacktestEngine`: walk-forward simulation with periodic retrain. Reports accuracy, Brier score, log loss, and calibration.

## Key design decisions

- **Cache-first**: all API responses are cached; completed match results (status="FT") use TTL=-1 and are never re-fetched.
- **BasePredictor interface**: add new models (e.g., XGBoost) by subclassing `BasePredictor` — no changes needed elsewhere.
- **Value bet focus**: the product is built around edge detection (model prob vs. implied odds), not raw outcome prediction.
- Odds are fetched from Bet365 (bookmaker ID 11 in API-Football).
- **enrich_with_statistics is NOT called globally** — it runs only inside `archive_resolved_fixtures`, scoped to the last 8 fixtures of teams being archived. DC model training does not need stats. This keeps GitHub Actions API calls at ~100-300 per run instead of 4500+.
- **Calibration is independent of DB** — `build_calibrators()` uses only API history (thousands of samples). `resolved_fixture_predictions` is used only for the reliability diagram in the dashboard, not as a calibration input.
- **Dual calibration pipeline**: (1) `GoalCalibrator` scales λ/μ proportionally (zachovává ratio = relative team strength) tak, aby λ+μ odpovídalo empirickému průměru gólů → opravuje všechny goal markety najednou (over2.5, goals1-3, BTTS). (2) `ProbabilityCalibrator` isotonic regression na H/D/A probs navrch (nezávisle). Goal calibration probíhá před výpočtem `goal_probs`; H/D/A calibration po.
- **Kalibrátor cache na disku** — `models/saved/calibrator_{league}_hda.joblib` + `_goals.joblib` + `_meta.json`. Přetrénuje se jen pokud přibylo ≥50 nových FT zápasů od posledního fitu. GitHub Actions cachuje `models/saved/` spolu s `cache/football.db`.

## Current stack

- **Model**: `EnsembleDCPredictor` (`models/ensemble.py`) — blends 3 Dixon-Coles models (dc_all / dc_season / dc_recent) at λ/μ level
- **Weights (leagues)**: dc_all=0.25, dc_season=0.45, dc_recent=0.30
- **Weights (cups)**: dc_all=0.60, dc_season=0.30, dc_recent=0.10 — cups have fewer matches per team
- **Cup leagues**: `champions_league`, `europa_league`, `conference_league` (defined in `CUP_LEAGUES` in ensemble.py)
- **DB**: Supabase — `tracked_predictions` (user picks) + `fixture_predictions` (daily pre-computed cache) + `resolved_fixture_predictions` (10-day archive of played matches)
- **Automation**: GitHub Actions — `predict.yml` (10:00 UTC) + `resolve.yml` (23:00 UTC)
- **Dashboard**: Streamlit Community Cloud (`dashboard.py`), reads DB directly. **5 tabs**: Predikce / Sledované / Výsledky / Statistiky / Tabulka
- **API status sidebar**: `fetch_api_status()` volá `/status` endpoint přímo (TTL 5 min), zobrazuje plán, denní requesty/limit, upozornění na Free plán (100 req/den, odds/injuries nedostupné)
- **Výsledky tab**: tabulka obsahuje sloupce P(H)%, P(D)%, P(A)%, Tip modelu, Správně (1X2) + P(G1-3)%, G1-3 (✅/❌ zda zápas skončil 1–3 góly). Summary řádek zobrazuje accuracy pro oba typy.
- **Kalibrace (Statistiky tab)**: reliability diagramy renderovány vertikálně (ne 3 sloupce) — čitelné na mobilu
- **Tabulka tab**: ligová tabulka přes `/standings` endpoint. Přepínač Celková / Doma / Venku. TTL=24h (max 1 API volání/den na ligu). Pro poháry ve vyřazovací fázi vrací prázdný výsledek. Responsivní HTML tabulka — desktop zobrazí vše, mobil skryje název týmu + GF/GA/Forma.

## Dashboard tabs (5)

1. **Predikce** — HTML tabulka (responsivní: PC = logo+název, mobil = jen logo); sloupce H% D% A% O2.5 U2.5 G1-3 G2-4 BTTS_Y BTTS_N; zelená ≥65%. Pod tabulkou dvě sekce:
   - **📌 Rychlé sledování** — pro každý zápas: caption s názvem zápasu + 1 řada 9 tlačítek (`st.columns(9)`, `use_container_width=True`): H D A O2.5 U2.5 G1-3 G2-4 B+ B-. Jeden klik = okamžité přidání, `st.toast()` potvrzení.
   - **Detailní analýza** — expandery. Pořadí sekcí v každém expanderu: (1) `render_prediction_stats` — dual-bar chart předpovídaného průběhu, (2) `render_match_detail` — xG metriky + forma + H2H tabulky, (3) `render_bet_validation` — signály Goals1-3 / Výhra domácích, (4) `render_injuries` — zranění, (5) `render_lineups` — sestavy, (6) tracking form. Expander label: `🚑` pokud zranění, `⚽` pokud dostupné sestavy. Tracking používá `st.form` (selectbox + `st.form_submit_button`) — selectbox nehlásí rerun, expander zůstane otevřený při změně výběru, rerun nastane až po kliknutí Sledovat.
2. **Sledované** — filtr liga/stav; `tracked_prob` (přidáno) vs `model_prob` (aktuální) + delta s 🟢🟡🔴; resolve button. Vývoj pravděpodobnosti (sloupec Vývoj) se zobrazuje i u vyřešených predikcí — podmínka `r.correct is None` byla odstraněna (dashboard.py:1028).
3. **Výsledky** — archiv resolved_fixture_predictions, accuracy modelu
4. **Statistiky** — accuracy by type/league + reliability diagram
5. **Tabulka** — `/standings` endpoint; přepínač Celková/Doma/Venku; responsivní HTML tabulka s logem každého týmu; desktop: # | logo | název | Z V R P | GF GA | +/- | Forma | Body; mobil: skryje název, GF, GA, Forma; Forma barevná (W=zelená, D=šedá, L=červená); TTL=24h. `.sth-pts` má `color:inherit` (viditelné v light i dark mode)

## Validace sázky (render_bet_validation)

Funkce `render_bet_validation(fx_data, feats)` se volá v expanderu každého zápasu (po `render_match_detail`). Zobrazuje 4 signály pro každý typ sázky, jen pokud model překročí práh.

**Goals 1-3** (zobrazí se pokud `goals1_3 ≥ 0.40`):
- λ+μ (model xG celkem): 🟢 ≤2.3 / 🟡 ≤2.8 / 🔴 >2.8
- Domácí avg gólů/zápas (home_gf + home_ga): 🟢 ≤2.5 / 🟡 ≤3.0 / 🔴 >3.0
- Hosté avg gólů/zápas (away_gf + away_ga): 🟢 ≤2.5 / 🟡 ≤3.0 / 🔴 >3.0
- H2H avg gólů celkem (h2h_home_gf + h2h_away_gf): 🟢 ≤2.5 / 🟡 ≤3.0 / 🔴 >3.0

**Výhra domácích** (zobrazí se pokud `prob_home ≥ 0.40` nebo home je favorit modelu):
- λ/μ poměr: 🟢 ≥1.4 / 🟡 ≥1.1 / 🔴 <1.1
- Domácí forma DOMA (home_venue_form, pts/z): 🟢 ≥2.0 / 🟡 ≥1.2 / 🔴 <1.2
- H2H výhry domácích % (h2h_home_wins): 🟢 ≥50% / 🟡 ≥33% / 🔴 <33%
- Elo rozdíl doma−hosté (elo_diff): 🟢 ≥+50 / 🟡 ≥0 / 🔴 <0

Každá sekce zobrazí souhrn "X/4 signálů zelených" barevně (zelená ≥3, žlutá ≥2, červená <2).

## scripts/predict.py flow (GitHub Actions 10:00 UTC)

1. Phase 1: domestic leagues → train ensemble, collect `dc_all.attack/defence` as cup priors
2. Phase 2: cup leagues → train with domestic priors as MLE x0 (Varianta C)
3. Calibration: `build_calibrators()` — jeden walk-forward průchod dc_all; vrací `(ProbabilityCalibrator, GoalCalibrator)`; cachováno na disk, přetrénuje se jen při ≥50 nových zápasech
4. `archive_resolved_fixtures()` — for each league: check FT fixtures v `completed` listu (bez extra API volání), batch DB check na již archivované, enrich only involved teams' recent fixtures (last 8), save to `resolved_fixture_predictions` with `features_json`
5. DELETE stale rows + INSERT new upcoming predictions (calibrated probabilities)
6. `update_tracked_probs()` — update `model_prob` for unresolved tracked_predictions

## FootballFetcher methods

- `get_fixtures(league, status)` — all training seasons; FT: past seasons TTL=-1, current TTL=1h
- `get_fixtures_season(league, season, status)` — single season (used by resolve.py)
- `get_upcoming_fixtures(league, next_n)` — primary: `next` param; fallback: `status=NS` for cups
- `get_fixture_statistics(fixture_id)` — `{team_id: FixtureStats}`, TTL=-1
- `enrich_with_statistics(fixtures, max_per_team)` — in-place, last N per team
- `get_fixture_injuries(fixture, league_id, season)` — `(home_inj, away_inj, home_goals, away_goals)`
- `get_fixture_lineups(fixture)` — `(home_lineup, away_lineup)` as `Optional[FixtureLineup]`; TTL=30min (announced ~1h before kickoff, empty before that)
- `get_standings(league)` — list of standings groups, TTL=24h (1 group for domestic, N for cup phases)
- `get_odds(fixture_id)` — all bookmakers, TTL=15min

## DB tables (Supabase)

- `fixture_predictions` — upcoming fixtures, denně přepisováno, kalibrované pravděpodobnosti. Sloupce: fixture_id, league, home_team, away_team, logos, match_date, prob_home/draw/away, over2_5, under2_5, goals1_3, goals2_4, btts_yes, btts_no, expected_goals_home/away, computed_at.
- `resolved_fixture_predictions` — archiv odehraných zápasů s pre-match features JSON + výsledkem; TTL **10 dní** (resolve.py maže starší záznamy). Stejné sloupce jako fixture_predictions + home_score, away_score, actual_outcome, predicted_outcome, correct, features_json, resolved_at.
- `tracked_predictions` — manuálně sledované predikce; `tracked_prob` (imutabilní, při přidání), `model_prob` (denně update); UniqueConstraint(fixture_id, prediction_type).

## Gotchas

- Supabase: Transaction pooler URL (port 6543), NOT direct connection
- `Base.metadata.create_all(engine)` vytváří nové tabulky ale NEpřidává sloupce do existujících → ruční ALTER TABLE
- `@st.cache_data` vyžaduje pickle-serializable návratové hodnoty → PlayerInjury se převádí na dict přes `dataclasses.asdict()`, rekonstruuje se jako `PlayerInjury(**d)` jen při potřebě
- `models/injury.py`: `from __future__ import annotations` + `TYPE_CHECKING` guard pro PlayerInjury → žádný runtime import
- SQLite CacheManager: `threading.Lock()` + WAL mode pro thread safety
- `enrich_with_statistics()` voláno POUZE v `predict.py` (GitHub Actions s cache), NE v dashboardu (Streamlit Cloud nemá SQLite cache)
- Kalibrované pravděpodobnosti jsou přímo uloženy v DB — dashboard čte bez další transformace
- `expected_goals_home/away` v DB = goal-kalibrované λ/μ (po `GoalCalibrator.transform()`), ne raw ensemble výstup
- `ensemble.py` `goal_probs` používá long-form klíče (`over2_5`, `goals1_3`, `btts_yes`, …) shodné s DB sloupci — `predict_from_lam_mu()` také. Žádná konverzní vrstva není potřeba.
- GitHub Actions cache key: `predictor-cache-{os}-{run_id}`, restore prefix `predictor-cache-{os}-`. Cachuje `cache/football.db` + `models/saved/` (DC modely + kalibrátory).
- `get_fixture_injuries()`: API-Football může vrátit stejného hráče vícekrát → `raw_by_team` je `Dict[int, Dict[int, dict]]` (klíč = player_id), deduplikace zabraňuje duplicitám v UI
- `get_fixture_lineups()`: vrací `(Optional[FixtureLineup], Optional[FixtureLineup])`; před ohlášením (~1h pre-match) vrací `(None, None)` — dashboard to tiše přeskočí; `FixtureLineup` se serializuje přes `dataclasses.asdict()` pro `@st.cache_data` pickle kompatibilitu
- `render_prediction_stats()`: zobrazuje historické průměry jako predikci průběhu (xG=λ/μ, góly, forma, útočná síla, Elo, PPG); stats features (shots/corners) zobrazí se pouze pokud byly obohaceny přes `enrich_with_statistics` (tj. v praxi nedostupné pro nadcházející zápasy na dashboardu)

## Bookmaker IDs (API Football)

- ID 11 = 1xBet; ID 1 = Bet365; ID 23 = Pinnacle (prázdné odpovědi)

## Planned next steps

1. **Value bety** — betting/ je prázdná, predict.py netahá odds ani nepočítá edge
2. **Backtesting tab** v dashboardu
