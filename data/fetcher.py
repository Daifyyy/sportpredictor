from typing import List
from datetime import datetime
from api.client import APIClient
from config.settings import Settings, LeagueConfig
from data.models import Fixture, Team, MatchResult, Odds


class FootballFetcher:
    def __init__(self, client: APIClient, settings: Settings):
        self.client = client
        self.settings = settings
        self.ttl = settings.cache_ttl

    def get_fixtures(self, league: LeagueConfig, status: str = "FT") -> List[Fixture]:
        if status == "FT":
            # Fetch all training seasons for richer history
            all_fixtures = []
            for season in league.seasons:
                data = self.client.get(
                    "fixtures",
                    {"league": league.id, "season": season, "status": status},
                    ttl=self.ttl.historical,
                )
                if data:
                    all_fixtures.extend(self._parse_fixture(f) for f in data.get("response", []))
            return all_fixtures
        else:
            data = self.client.get(
                "fixtures",
                {"league": league.id, "season": league.season, "status": status},
                ttl=self.ttl.fixtures,
            )
            return [self._parse_fixture(f) for f in data.get("response", [])] if data else []

    def get_fixtures_season(self, league: LeagueConfig, season: int, status: str = "FT") -> List[Fixture]:
        """Fetch fixtures for a single specific season (used for resolve)."""
        data = self.client.get(
            "fixtures",
            {"league": league.id, "season": season, "status": status},
            ttl=self.ttl.historical if status == "FT" else self.ttl.fixtures,
        )
        return [self._parse_fixture(f) for f in data.get("response", [])] if data else []

    def get_upcoming_fixtures(self, league: LeagueConfig, next_n: int = 10) -> List[Fixture]:
        data = self.client.get(
            "fixtures",
            {"league": league.id, "season": league.season, "next": next_n},
            ttl=self.ttl.fixtures
        )
        return [self._parse_fixture(f) for f in data.get("response", [])] if data else []

    def get_odds(self, fixture_id: int) -> List[Odds]:
        """Fetch odds from all configured bookmakers."""
        all_odds = []
        for bm in self.settings.bookmakers.values():
            data = self.client.get(
                "odds",
                {"fixture": fixture_id, "bookmaker": bm.id},
                ttl=self.ttl.odds,
            )
            if data:
                all_odds.extend(self._parse_odds(fixture_id, data))
        return all_odds

    def _parse_fixture(self, raw: dict) -> Fixture:
        f = raw["fixture"]
        teams = raw["teams"]
        goals = raw.get("goals", {})
        return Fixture(
            id=f["id"],
            date=datetime.fromisoformat(f["date"].replace("Z", "+00:00")),
            league_id=raw["league"]["id"],
            season=raw["league"]["season"],
            home_team=Team(teams["home"]["id"], teams["home"]["name"], ""),
            away_team=Team(teams["away"]["id"], teams["away"]["name"], ""),
            result=MatchResult(goals["home"] or 0, goals["away"] or 0)
                   if goals.get("home") is not None else None,
            status=f["status"]["short"],
        )

    def _parse_odds(self, fixture_id: int, data: dict) -> List[Odds]:
        result = []
        response = data.get("response", [])
        for bookmaker in (response[0].get("bookmakers", []) if response else []):
            for bet in bookmaker.get("bets", []):
                if bet["name"] == "Match Winner":
                    vals = {v["value"]: float(v["odd"]) for v in bet["values"]}
                    result.append(Odds(
                        fixture_id=fixture_id,
                        bookmaker=bookmaker["name"],
                        home_win=vals.get("Home", 0),
                        draw=vals.get("Draw", 0),
                        away_win=vals.get("Away", 0),
                    ))
        return result
