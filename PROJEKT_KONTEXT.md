# Football Predictor — Projektový kontext

## Co je to za projekt

Webová aplikace / SaaS produkt pro predikci fotbalových utkání Top 5 evropských lig
s automatickou detekcí value betů. Cíl je veřejný produkt s potenciálem monetizace
(premium tier, API přístup). Projekt je ve fázi funkční kostry — backendová logika
napsaná, frontend a deployment teprve přijdou.

---

## Technický stack

- **Jazyk:** Python 3.11+
- **Data:** API-Football (přes RapidAPI) — placený plán
- **Cache:** SQLite (lokální, TTL-based)
- **Model:** Dixon-Coles Poisson model (1997)
- **Architektura:** OOP, modulární, připravená na rozšíření

---

## Ligy (Top 5 evropských)

| Liga | ID | Země |
|---|---|---|
| Premier League | 39 | England |
| La Liga | 140 | Spain |
| Bundesliga | 78 | Germany |
| Serie A | 135 | Italy |
| Ligue 1 | 61 | France |

Sezóna: **2024**

---

## Struktura projektu

```
football_predictor/
├── config/
│   └── settings.py        # Ligy, TTL, rate limits, API config
├── api/
│   ├── client.py          # HTTP client + RateLimiter + cache integrace
│   └── cache.py           # SQLite CacheManager (TTL-based, persistent)
├── data/
│   ├── models.py          # Dataclasses: Team, Fixture, MatchResult, Odds, Prediction
│   └── fetcher.py         # FootballFetcher — orchestrace API volání
├── features/
│   └── engineer.py        # FeatureEngineer: form, H2H, Elo rating
├── models/
│   ├── base.py            # BasePredictor (ABC)
│   └── poisson.py         # DixonColesPredictor
├── betting/
│   └── value.py           # ValueBetDetector (edge vs. implied odds)
├── cache/                 # SQLite databáze (gitignore)
├── main.py                # Hlavní runner
├── requirements.txt
├── .env.example
└── README.md
```

---

## Klíčové třídy a jejich role

### `Settings` (config/settings.py)
- Centrální konfigurace celého projektu
- Obsahuje `LeagueConfig` pro každou ligu a `CacheTTL` s různými TTL pro různé typy dat
- Načítá API klíč z `.env`

### `CacheManager` (api/cache.py)
- SQLite-based cache s TTL expirací
- TTL = -1 znamená "nikdy neexpiruje" (historická data)
- Metody: `get`, `set`, `delete`, `purge_expired`

### `APIClient` (api/client.py)
- Wrapper nad `requests.Session`
- Integrovaný `RateLimiter` (token bucket, 300 req/min)
- Automaticky cachuje každý response
- `force_refresh=True` přeskočí cache

### `FootballFetcher` (data/fetcher.py)
- `get_fixtures(league, status="FT")` — historické výsledky
- `get_upcoming_fixtures(league, next_n=10)` — nadcházející zápasy
- `get_odds(fixture_id)` — odds od Bet365 (bookmaker ID 11)

### `FeatureEngineer` (features/engineer.py)
- `build_features(fixture, history)` → Dict[str, float]
- Form features: průměrné body, góly dané/obdržené (posledních N zápasů)
- H2H features: win rate z posledních 10 vzájemných zápasů
- Elo rating: dynamicky builovaný z celé historie (K=32)

### `DixonColesPredictor` (models/poisson.py)
- Trénování: MLE optimalizace attack/defence parametrů pro každý tým
- Predikce: pravděpodobnostní matice výsledků skóre
- DC korekce: opravuje podhodnocení výsledků 0-0, 1-0, 0-1, 1-1 (rho=-0.13)
- Output: `Prediction` s prob_home, prob_draw, prob_away, xG

### `ValueBetDetector` (betting/value.py)
- Porovnává model probability vs. implied probability z odds
- `min_edge = 0.03` (3 % nad bookmaker implied prob)
- Počítá Expected Value pro každý value bet

---

## Cache TTL konfigurace

| Typ dat | TTL |
|---|---|
| Odds | 15 minut |
| Fixtures (upcoming) | 1 hodina |
| Standings | 1 hodina |
| Team stats | 24 hodin |
| Static data | 7 dní |
| Historické výsledky (FT) | nikdy neexpiruje |

---

## Co funguje (aktuální stav)

- [x] Kompletní data pipeline (fetch → cache → parse → dataclasses)
- [x] Rate limiter pro API volání
- [x] Dixon-Coles model s trénováním a predikcí
- [x] Value bet detekce s edge a EV výpočtem
- [x] Feature engineering (form, H2H, Elo)
- [x] `main.py` runner přes všech 5 lig

---

## Co chybí / další vývoj (prioritizovaný)

### Fáze 1 — Backend dokončení
- [ ] **Backtesting modul** — simulace predikcí na historických datech, ROI, Brier score, calibration
- [ ] **XGBoost model** — alternativa k Dixon-Colesovi, stejný `BasePredictor` interface
- [ ] **Model ensemble** — kombinace Dixon-Coles + XGBoost
- [ ] **Persistence modelu** — uložení natrénovaných parametrů (pickle / joblib)

### Fáze 2 — API vrstva
- [ ] **FastAPI backend** — endpointy `/predictions/{league}`, `/value-bets`, `/track-record`
- [ ] **Scheduled jobs** — automatický re-training, fetch nových dat (APScheduler / cron)
- [ ] **Databáze** — PostgreSQL pro uložení predikcí a track record (SQLAlchemy)

### Fáze 3 — Frontend
- [ ] **Dashboard** — Next.js nebo Streamlit (pro rychlý prototype)
- [ ] **Track record** — historická přesnost modelu, ROI graf
- [ ] **Value bet feed** — real-time aktualizace před zápasy

### Fáze 4 — Monetizace a deployment
- [ ] **Uživatelské účty** — free vs. premium tier
- [ ] **Deployment** — Railway/Render (backend) + Vercel (frontend)
- [ ] **Platební brána** — Stripe

---

## Závislosti (requirements.txt)

```
requests>=2.31
numpy>=1.26
scipy>=1.11
python-dotenv>=1.0
```

Budoucí přidání: `fastapi`, `uvicorn`, `sqlalchemy`, `xgboost`, `scikit-learn`, `apscheduler`

---

## Prostředí

- **OS:** Windows (PowerShell)
- **Claude Code:** v2.1.68, nainstalováno v `C:\Users\denis\.local\bin\claude.exe`
- **Projekt:** `C:\Projekt\football_predictor\`
- **API:** API-Football přes RapidAPI (placený plán)

---

## Designová rozhodnutí a principy

1. **OOP throughout** — každá vrstva je třída, žádný spaghetti kód
2. **Cache-first** — každé API volání jde přes cache, historická data se nikdy znovu nefetchují
3. **BasePredictor ABC** — nové modely lze přidat bez změny okolního kódu
4. **FeatureEngineer je rozšiřitelný** — nová feature = nová metoda, pipeline se složí automaticky
5. **ValueBet over výsledková predikce** — produkt stojí na edge detekci, ne na "kdo vyhraje"
6. **Transparentní track record** — důvěryhodnost produktu stojí na veřejné historii přesnosti

---

## Jak spustit projekt

```bash
# Instalace závislostí
pip install -r requirements.txt

# Nastavení API klíče
cp .env.example .env
# Edituj .env a doplň: API_FOOTBALL_KEY=tvuj_klic

# Spuštění
python main.py

# Claude Code
cd C:\Projekt\football_predictor
claude
```

---

*Tento soubor slouží jako kontext pro nové Claude konverzace / Claude Code sessions.*
*Při startu nové session: "Přečti PROJEKT_KONTEXT.md a navážeme na vývoj."*
