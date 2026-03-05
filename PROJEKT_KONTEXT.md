# Football Predictor — Projektový kontext

## Co je to za projekt

Webová aplikace pro predikci fotbalových utkání Top 5 evropských lig s automatickou detekcí
value betů, Kelly criteriem a backtestingem. Backend + Streamlit dashboard s přímým napojením
na Supabase. Cíl: veřejný produkt s potenciálem monetizace (premium tier, API přístup).

---

## Technický stack

- **Jazyk:** Python 3.11+
- **Data:** API-Football (v3.football.api-sports.io)
- **Cache:** SQLite (lokální, TTL-based, `cache/football.db`)
- **Modely:** Dixon-Coles Poisson + XGBoost + Ensemble + izotonicka kalibrace
- **DB:** Supabase (PostgreSQL) přes SQLAlchemy + psycopg2
- **API:** FastAPI + APScheduler (scheduler 09:00 predikce / 23:00 resolve)
- **Frontend:** Streamlit dashboard (`dashboard.py`)
- **Deployment:** GitHub + Streamlit Community Cloud (dashboard); FastAPI běží lokálně

---

## Ligy (Top 5 evropských)

| Liga | ID | Sezóny (training) |
|---|---|---|
| Premier League | 39 | 2022, 2023, 2024, 2025 |
| La Liga | 140 | 2022, 2023, 2024, 2025 |
| Bundesliga | 78 | 2022, 2023, 2024, 2025 |
| Serie A | 135 | 2022, 2023, 2024, 2025 |
| Ligue 1 | 61 | 2022, 2023, 2024, 2025 |

Aktivní sezóna: **2025**

---

## Struktura projektu

```
football_predictor/
├── config/settings.py          # LeagueConfig, BookmakerConfig, CacheTTL, settings singleton
├── api/
│   ├── client.py               # APIClient + RateLimiter (300 req/min) + SQLite cache
│   ├── cache.py                # CacheManager (TTL-based, TTL=-1 = nikdy neexpiruje)
│   └── app.py                  # FastAPI aplikace — všechny endpointy
├── data/
│   ├── models.py               # Dataclasses: Team, Fixture, MatchResult, Odds, Prediction
│   └── fetcher.py              # FootballFetcher — multi-season fetch, multi-bookmaker odds
├── features/
│   └── engineer.py             # FeatureEngineer: O(n) precompute, 15+ features, bez data leakage
├── models/
│   ├── base.py                 # BasePredictor (ABC)
│   ├── poisson.py              # DixonColesPredictor — vektorizovaný (NumPy), rychlý
│   ├── xgboost_model.py        # XGBoostPredictor
│   ├── ensemble.py             # EnsemblePredictor (průměr pravděpodobností)
│   ├── calibrator.py           # ProbabilityCalibrator (isotonic regression)
│   └── saved/                  # Uložené modely joblib (gitignore)
├── betting/value.py            # ValueBetDetector (min_edge=3%)
├── backtesting/engine.py       # BacktestEngine — walk-forward, CalibrationResult
├── db/
│   ├── models.py               # ORM: PredictionRow, BacktestRunRow, BankrollRow
│   └── session.py              # SQLAlchemy engine + SessionLocal
├── dashboard.py                # Streamlit dashboard (7 tabů, čte přímo z Supabase)
├── main.py                     # CLI runner
├── run_backtest.py             # Backtest runner
├── .env                        # API_FOOTBALL_KEY + DATABASE_URL (gitignore)
├── .env.example                # Template
└── requirements.txt
```

---

## Co je hotovo

- [x] Data pipeline: fetch → SQLite cache → parse → dataclasses
- [x] Modely: Dixon-Coles (vektorizovaný MLE), XGBoost, Ensemble
- [x] Izotonicka kalibrace pravděpodobností
- [x] Feature engineering: form, venue form, H2H, Elo (snapshot, bez leakage), attack/defense strength, streak, rest days, trend, consistency, season PPG — O(n) via `precompute()`
- [x] Value bet detekce (edge nad implied probability, multi-bookmaker)
- [x] Kelly criterion (fractional 25%), bankroll tracking
- [x] Backtesting: walk-forward simulace, accuracy, Brier, log loss, ECE kalibrace
- [x] FastAPI: plná sada endpointů (viz níže)
- [x] Model persistence: joblib save/load
- [x] Supabase DB: predictions, backtest_runs, bankroll tabulky
- [x] Streamlit dashboard: 7 tabů, přímé čtení z Supabase, akce přes lokální API
- [x] Scheduler: APScheduler (09:00 predikce, 23:00 resolve)
- [x] Manuální retrain s live progress logem v dashboardu
- [x] GitHub repo: https://github.com/Daifyyy/sportpredictor

---

## FastAPI endpointy (`api/app.py`)

| Method | Endpoint | Popis |
|---|---|---|
| GET | `/predictions/{league}` | Fetch + ulož predikce pro nadcházející zápasy |
| GET | `/value-bets` | Value bety napříč všemi ligami |
| GET | `/track-record/{league}` | Spustí backtest, uloží výsledky |
| GET | `/track-record/{league}/history` | Historie backtest runů z DB |
| POST | `/retrain` | Spustí přetrénování na pozadí (202 Accepted) |
| GET | `/retrain/status` | Stav retrainingu + live log kroků |
| POST | `/resolve/{league}` | Doplní actual_outcome pro dokončené zápasy |
| GET | `/performance/{league}` | Reálná accuracy/VB statistiky |
| POST | `/calibrate/{league}` | Fit isotonic kalibrátoru |
| GET | `/calibration/{league}` | ECE kalibrace (backtest, 1-2 min) |
| GET | `/kelly/{league}` | Kelly stake suggestions |
| POST | `/bankroll/update/{league}` | Simulace bankrollu z vyřešených VB |
| GET | `/bankroll/{league}` | Aktuální bankroll stav |
| GET | `/config/leagues` | Seznam lig ze settings |
| GET | `/config/bookmakers` | Seznam bookmakerů ze settings |

---

## Streamlit dashboard (`dashboard.py`)

7 tabů:
1. **Predikce** — rozdělení na "právě hraje" (🔴) a "nadcházející" (📅), + value bety
2. **Výkonnost** — accuracy, VB hit rate per model (VB správně = vb_outcome == actual)
3. **Value Bety** — nadcházející value bety s kurzy
4. **Kelly** — stake suggestions (fractional Kelly 25%, bankroll 1000)
5. **Bankroll** — graf vývoje, P&L, hit rate
6. **Backtest** — history runů, accuracy v čase
7. **Kalibrace** — reliability diagram (lazy, načte se tlačítkem)

Sidebar akce (vyžadují lokální API):
- 🧠 Přetrénovat, 🔄 Načíst predikce, ✅ Resolve, 🎯 Kalibrovat, 💹 Update bankroll

Auto-chování:
- Přepnutí ligy → `cache_data.clear()` → načtení z Supabase
- Retrain log polluje každé 2s (jen když běží)
- `api_alive()` check každých 10s — tlačítka disabled když API offline

---

## DB schéma (Supabase)

**`predictions`** — unique: `(fixture_id, model_name)`
- probabilities, xG, value_bets (ARRAY), predicted_outcome, actual_outcome, correct, odds_*

**`backtest_runs`** — každý run = nový řádek (plná historie)

**`bankroll`** — Kelly simulace: bet_on, odds, stake_pct, outcome, pnl_pct, bankroll_after

---

## Lokální spuštění

```bash
# API backend
uvicorn api.app:app --reload

# Streamlit dashboard
streamlit run dashboard.py
```

---

## Kde příště začít

Projekt je funkční a pushnutý na GitHub. Streamlit Community Cloud deployment zatím
**nebyl nastaven** — to je logický další krok.

### Deployment na Streamlit Community Cloud:
1. Jdi na share.streamlit.io
2. Připoj repo: `Daifyyy/sportpredictor`, branch `main`, soubor `dashboard.py`
3. Přidej secret: `DATABASE_URL = postgresql://...` (Transaction pooler, port 6543)
4. Deploy

### Možné další kroky:
- Deployment dashboardu na Streamlit Community Cloud
- Více bookmakerů pro value bet porovnání (Pinnacle již v settings)
- Notifikace (email/Telegram) při value betu
- Monetizace / uživatelské účty (Stripe, premium tier)

---

## Supabase

- Free tier — 500 MB limit, projekt se pauze po 1 týdnu nečinnosti
- Data objem je zanedbatelný (~150 řádků/týden), cleanup není potřeba
- Připojení: Transaction pooler URL, port **6543** (ne 5432!)
- Při chybě "server closed connection": sessions nejsou cachované — `SASession(engine)` jako context manager per query

---

## Prostředí

- **OS:** Windows 11, shell: bash
- **Projekt:** `C:\Projekt\football_predictor\`
- **GitHub:** https://github.com/Daifyyy/sportpredictor
- **Streamlit:** spustit přes `streamlit run dashboard.py`

---

*Při startu nové session: "Přečti PROJEKT_KONTEXT.md" — tento soubor obsahuje aktuální stav.*
