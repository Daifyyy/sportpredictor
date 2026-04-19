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
- `models/injury.py` — `InjuryAdjuster`: adjusts λ/μ for injuries. Uses `TYPE_CHECKING` guard for `PlayerInjury` import. Per-90 attack contribution, dynamic midfielder G+A weight, defense quality factor.
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
- **Dashboard**: Streamlit Community Cloud (`dashboard.py`), reads DB directly. **6 tabs**: Predikce / Sledované / Výsledky / Statistiky / 🔄 Rohy / Tabulka
- **API status sidebar**: `fetch_api_status()` volá `/status` endpoint přímo (TTL 5 min), zobrazuje plán, denní requesty/limit, upozornění na Free plán (100 req/den, odds/injuries nedostupné)
- **Výsledky tab**: tabulka obsahuje sloupce P(H)%, P(D)%, P(A)%, Tip modelu, Správně (1X2) + P(G1-3)%, G1-3 (✅/❌ zda zápas skončil 1–3 góly). Summary řádek zobrazuje accuracy pro oba typy.
- **Kalibrace (Statistiky tab)**: reliability diagramy renderovány vertikálně (ne 3 sloupce) — čitelné na mobilu
- **Tabulka tab**: ligová tabulka přes `/standings` endpoint. Přepínač Celková / Doma / Venku. TTL=24h (max 1 API volání/den na ligu). Pro poháry ve vyřazovací fázi vrací prázdný výsledek. Responsivní HTML tabulka — desktop zobrazí vše, mobil skryje název týmu + GF/GA/Forma.

## Dashboard tabs (6)

1. **Predikce** — HTML tabulka (responsivní: PC = logo+název, mobil = jen logo); sloupce H% D% A% O2.5 U2.5 G1-3 G2-4 BTTS_Y BTTS_N; zelená ≥65%. Pod tabulkou dvě sekce:
   - **📌 Rychlé sledování** — pro každý zápas: caption s názvem zápasu + 1 řada 9 tlačítek (`st.columns(9)`, `use_container_width=True`): H D A O2.5 U2.5 G1-3 G2-4 B+ B-. Jeden klik = okamžité přidání, `st.toast()` potvrzení.
   - **Detailní analýza** — expandery. Pořadí sekcí v každém expanderu: (1) `render_prediction_stats` — dual-bar chart předpovídaného průběhu, (2) `render_match_detail` — xG metriky + forma + H2H tabulky, (3) `render_bet_validation` — signály Goals1-3 / Výhra domácích, (4) `render_injuries` — zranění, (5) `render_lineups` — sestavy, (6) tracking form. Expander label: `🚑` pokud zranění, `⚽` pokud dostupné sestavy. Tracking používá `st.form` (selectbox + `st.form_submit_button`) — selectbox nehlásí rerun, expander zůstane otevřený při změně výběru, rerun nastane až po kliknutí Sledovat.
2. **Sledované** — filtr liga/stav; `tracked_prob` (přidáno) vs `model_prob` (aktuální) + delta s 🟢🟡🔴; resolve button. Vývoj pravděpodobnosti (sloupec Vývoj) se zobrazuje i u vyřešených predikcí.
3. **Výsledky** — archiv resolved_fixture_predictions, accuracy modelu
4. **Statistiky** — 3 sekce: (1) Celkový přehled (metriky goals vs corners, avg P správných/špatných tipů), (2) ⚽ Model výsledků & gólů — type/league breakdown + reliability diagramy H/D/A + gólové markety, (3) 🔄 Model rohů — type/league breakdown + reliability diagram corners marketů (aktivní po ≥20 archivovaných zápasech s actual_corners)
5. **🔄 Rohy** — tabulka λ/μ/Σ + O/U 8.5–11.5 (zelená ≥65%); rychlé sledování C8+/C8-…C11+/C11-; expandery s barchart; `save_tracking()` volá se pro konzistenci se Sledované tab
6. **Tabulka** — `/standings` endpoint; přepínač Celková/Doma/Venku; responsivní HTML tabulka; TTL=24h. `.sth-pts` má `color:inherit` (viditelné v light i dark mode)

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

1. Phase 1: domestic leagues → train goals ensemble, collect `dc_all.attack/defence` as cup priors; `enrich_full_history(completed, max_new=100)` → `train_corners_ensemble()` — corners model uložen do `models/saved/`
2. Phase 2: cup leagues → train goals ensemble s domestic priors (Varianta C); corners přeskočeny
3. Calibration: `build_calibrators()` — walk-forward dc_all → `(ProbabilityCalibrator, GoalCalibrator)`; disk cache, přetrénuje se jen při ≥50 nových zápasech
4. `archive_resolved_fixtures()` — FT fixtures → `resolved_fixture_predictions` s features_json + `actual_corners_home/away` z `fx.home_stats/away_stats`
5. DELETE + INSERT upcoming predictions (calibrated probabilities + corners sloupce)
6. `update_tracked_probs()` — aktualizuje `model_prob` pro nevyřešené tracked_predictions (včetně Corners_ typů)

## FootballFetcher methods

- `get_fixtures(league, status)` — all training seasons; FT: past seasons TTL=-1, current TTL=1h
- `get_fixtures_season(league, season, status)` — single season (used by resolve.py)
- `get_upcoming_fixtures(league, next_n)` — primary: `next` param; fallback: `status=NS` for cups
- `get_fixture_statistics(fixture_id)` — `{team_id: FixtureStats}`, TTL=-1
- `enrich_with_statistics(fixtures, max_per_team)` — in-place, last N per team
- `get_fixture_injuries(fixture, league_id, season)` — `(home_inj, away_inj, home_gf, away_gf, home_ga, away_ga)` — 6 hodnot; goals_against z `_get_team_season_stats()`
- `get_fixture_lineups(fixture)` — `(home_lineup, away_lineup)` as `Optional[FixtureLineup]`; TTL=30min (announced ~1h before kickoff, empty before that)
- `enrich_full_history(fixtures, max_new=100, interval=2.0)` — fetch `/fixtures/statistics` pro neobohatené FT fixtures; newest-first; TTL=-1; pro corners model
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
- `get_fixture_injuries()`: vrací 6 hodnot `(home_inj, away_inj, home_gf, away_gf, home_ga, away_ga)`; API-Football může vrátit stejného hráče vícekrát → `raw_by_team` je `Dict[int, Dict[int, dict]]` (klíč = player_id), deduplikace zabraňuje duplicitám v UI. `_get_team_season_stats()` vrací `(goals_for, goals_against)` z jednoho `teams/statistics` volání.
- `get_fixture_lineups()`: vrací `(Optional[FixtureLineup], Optional[FixtureLineup])`; před ohlášením (~1h pre-match) vrací `(None, None)` — dashboard to tiše přeskočí; `FixtureLineup` se serializuje přes `dataclasses.asdict()` pro `@st.cache_data` pickle kompatibilitu
- `render_prediction_stats()`: zobrazuje historické průměry jako predikci průběhu (xG=λ/μ, góly, forma, útočná síla, Elo, PPG); stats features (shots/corners) zobrazí se pouze pokud byly obohaceny přes `enrich_with_statistics` (tj. v praxi nedostupné pro nadcházející zápasy na dashboardu)

## InjuryAdjuster (models/injury.py)

`InjuryAdjuster.adjust(lam, mu, home_inj, away_inj, home_gf, away_gf, home_ga, away_ga)` — vrací upravené `(λ, μ)`. Predikce v DB jsou **již po této úpravě** (není třeba počítat ručně).

**Attack impact** (→ snižuje λ/μ útočícího týmu):
- Pokud hráč odehrál ≥450 min: `g_per_90 / team_g_per_game` (per-90 rate; team_g_per_game = goals_for/38)
- Pokud <450 min (super-sub): fallback na `(goals + 0.7*assists) / team_goals_for`
- Záložník: násobí výsledek dynamickou vahou dle G+A: ≥12→70%, ≥6→50%, ≥2→30%, jinak→15%

**Defense impact** (→ zvyšuje λ/μ soupeře):
- Obránce: `(minutes/3420) * 0.15 * defense_quality`
- Brankář: `(minutes/3420) * 0.20 * defense_quality`
- `defense_quality = clamp(1.3 / (goals_against/38), 0.5, 2.0)` — elite obrany (málo GA) mají vyšší koeficient

**Caps**: MAX_ATTACK_REDUCTION=0.25, MAX_DEFENSE_INCREASE=0.20. Pochybný hráč: faktor 0.35.

## Bookmaker IDs (API Football)

- ID 11 = 1xBet; ID 1 = Bet365; ID 23 = Pinnacle (prázdné odpovědi)

## Corners model (implementováno)

- `models/corners.py` — `CornersPredictor` (čisté Poisson MLE, bez DC korekce) + `EnsembleCornersPredictor` (blend all/season/recent, stejné váhy jako goals) + `train_corners_ensemble()`
- `data/models.py` — `CornersPrediction` dataclass
- `data/fetcher.py` — `enrich_full_history(fixtures, max_new=100, interval=2.0)` — newest-first, TTL=-1, bezpečné pro paid plán
- `scripts/train_corners_only.py` — standalone enrichment + trénink bez zásahu do DB
- `api/client.py` — 429 retry s exponenciálním backoffem (60s / 120s / 180s)
- DB sloupce v `fixture_predictions` + `resolved_fixture_predictions`: `expected_corners_home/away`, `corners_over/under 8.5–11.5`, `actual_corners_home/away` ← Supabase ALTER TABLE **již provedeno**
- Dashboard: 6 tabů (+ 🔄 Rohy), Statistiky rozšířena o corners sekci, `save_tracking()` podporuje Corners_ typy, `_PROB_FIELD` + `PREDICTION_TYPES` rozšířeny

**Enrich stav:** probíhá postupně přes GitHub Actions (100 fixtures/liga/run, newest-first → c_season a c_recent jsou funkční od prvního runu).

## Planned next steps

1. **Value bety** — `betting/` je prázdná, predict.py netahá odds ani nepočítá edge
2. **Backtesting tab** v dashboardu
