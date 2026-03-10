import os
from dataclasses import dataclass, field
from typing import Dict
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class BookmakerConfig:
    id: int
    name: str


@dataclass(frozen=True)
class LeagueConfig:
    id: int
    name: str
    country: str
    season: int = 2025
    seasons: tuple = (2022, 2023, 2024, 2025)  # seasons used for training history


@dataclass
class CacheTTL:
    fixtures: int = 3600       # 1h
    standings: int = 3600
    team_stats: int = 86400    # 24h
    odds: int = 900            # 15min
    static: int = 604800       # 7 dní
    historical: int = -1       # nikdy neexpiruje


@dataclass
class Settings:
    api_key: str = field(default_factory=lambda: os.getenv("API_FOOTBALL_KEY", "YOUR_KEY_HERE"))
    base_url: str = "https://v3.football.api-sports.io"
    cache_db: str = "cache/football.db"
    requests_per_minute: int = 300
    cache_ttl: CacheTTL = field(default_factory=CacheTTL)

    # Bookmakers used for value bet detection
    # Pinnacle = sharp reference (lowest margin = most accurate market probabilities)
    bookmakers: Dict[str, BookmakerConfig] = field(default_factory=lambda: {
        "pinnacle": BookmakerConfig(23, "Pinnacle"),
        # "bet365":   BookmakerConfig(1,   "Bet365"),   # ID 1, not 11 (11 = 1xBet)
        # "bwin":     BookmakerConfig(6,   "Bwin"),
        # "williamhill": BookmakerConfig(8, "William Hill"),
        # "unibet":   BookmakerConfig(16,  "Unibet"),
    })

    leagues: Dict[str, LeagueConfig] = field(default_factory=lambda: {
        "premier_league":    LeagueConfig(39,  "Premier League",      "England",      season=2025, seasons=(2022, 2023, 2024, 2025)),
        "la_liga":           LeagueConfig(140, "La Liga",             "Spain",        season=2025, seasons=(2022, 2023, 2024, 2025)),
        "bundesliga":        LeagueConfig(78,  "Bundesliga",          "Germany",      season=2025, seasons=(2022, 2023, 2024, 2025)),
        "serie_a":           LeagueConfig(135, "Serie A",             "Italy",        season=2025, seasons=(2022, 2023, 2024, 2025)),
        "ligue_1":           LeagueConfig(61,  "Ligue 1",             "France",       season=2025, seasons=(2022, 2023, 2024, 2025)),
        "czech_liga":        LeagueConfig(345, "Fortuna Liga",        "Czech Republic", season=2024, seasons=(2022, 2023, 2024)),
        "champions_league":  LeagueConfig(2,   "Champions League",   "Europe",       season=2024, seasons=(2022, 2023, 2024)),
        "europa_league":     LeagueConfig(3,   "Europa League",      "Europe",       season=2024, seasons=(2022, 2023, 2024)),
        "conference_league": LeagueConfig(848, "Conference League",  "Europe",       season=2024, seasons=(2022, 2023, 2024)),
    })


settings = Settings()
