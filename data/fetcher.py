from typing import List, Dict, Optional, Tuple
from datetime import datetime, timezone
from api.client import APIClient
from config.settings import Settings, LeagueConfig
from data.models import Fixture, FixtureLineup, FixtureStats, LineupPlayer, PlayerInjury, Team, MatchResult, Odds


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
                # Current season: cache 1h (new matches finish regularly)
                # Past seasons: cache forever (complete, never change)
                ttl = self.ttl.fixtures if season == league.season else self.ttl.historical
                data = self.client.get(
                    "fixtures",
                    {"league": league.id, "season": season, "status": status},
                    ttl=ttl,
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
        is_current = season == league.season
        ttl = (self.ttl.fixtures if is_current else self.ttl.historical) if status == "FT" else self.ttl.fixtures
        data = self.client.get(
            "fixtures",
            {"league": league.id, "season": season, "status": status},
            ttl=ttl,
        )
        return [self._parse_fixture(f) for f in data.get("response", [])] if data else []

    def get_upcoming_fixtures(self, league: LeagueConfig, next_n: int = 10) -> List[Fixture]:
        # Primary: use 'next' parameter
        data = self.client.get(
            "fixtures",
            {"league": league.id, "season": league.season, "next": next_n},
            ttl=self.ttl.fixtures,
        )
        fixtures = [self._parse_fixture(f) for f in data.get("response", [])] if data else []

        # Fallback: status=NS filtered to future dates (useful for cup competitions
        # where knockout fixtures may not appear in 'next' until confirmed)
        if not fixtures:
            data = self.client.get(
                "fixtures",
                {"league": league.id, "season": league.season, "status": "NS"},
                ttl=self.ttl.fixtures,
            )
            now = datetime.now(timezone.utc)
            all_ns = [self._parse_fixture(f) for f in data.get("response", [])] if data else []
            fixtures = sorted(
                [f for f in all_ns if f.date > now],
                key=lambda x: x.date,
            )[:next_n]

        return fixtures

    def get_fixture_statistics(self, fixture_id: int) -> Dict[int, FixtureStats]:
        """Returns {team_id: FixtureStats} for a finished fixture. Cached forever (TTL=-1)."""
        data = self.client.get("fixtures/statistics", {"fixture": fixture_id}, ttl=self.ttl.historical)
        if not data:
            return {}
        return {
            td["team"]["id"]: self._parse_team_stats(td)
            for td in data.get("response", [])
        }

    def enrich_with_statistics(self, fixtures: List[Fixture], max_per_team: int = 15) -> None:
        """Fetch and attach match statistics for the last max_per_team fixtures per team.

        Modifies Fixture objects in-place. Skips fixtures already enriched.
        TTL=-1 ensures FT stats are cached forever after first fetch.
        """
        # Collect last N fixture IDs per team
        team_recent: Dict[int, List[Fixture]] = {}
        for f in sorted((f for f in fixtures if f.result is not None), key=lambda x: x.date):
            team_recent.setdefault(f.home_team.id, []).append(f)
            team_recent.setdefault(f.away_team.id, []).append(f)

        ids_to_fetch: set = set()
        for team_fixtures in team_recent.values():
            for f in team_fixtures[-max_per_team:]:
                ids_to_fetch.add(f.id)

        fixture_map = {f.id: f for f in fixtures}
        fetched = 0
        for fid in ids_to_fetch:
            fx = fixture_map.get(fid)
            if fx is None or fx.home_stats is not None:
                continue  # already enriched (in-place from a previous call)
            stats = self.get_fixture_statistics(fid)
            if stats:
                fx.home_stats = stats.get(fx.home_team.id)
                fx.away_stats = stats.get(fx.away_team.id)
                fetched += 1

        if fetched:
            print(f"  Enriched {fetched} fixtures with match statistics")

    def get_fixture_injuries(
        self, fixture: Fixture, league_id: int, season: int
    ) -> Tuple[List[PlayerInjury], List[PlayerInjury], int, int, int, int]:
        """Fetch injuries for an upcoming fixture, enriched with player season stats.

        Returns (home_injuries, away_injuries, home_goals_for, away_goals_for,
                 home_goals_against, away_goals_against).
        goals_for/against are team season totals used for normalization in InjuryAdjuster.
        """
        data = self.client.get("injuries", {"fixture": fixture.id}, ttl=self.ttl.injuries)
        if not data or not data.get("response"):
            return [], [], 50, 50, 50, 50

        # Group raw entries by team_id, dedup by player_id
        raw_by_team: Dict[int, Dict[int, dict]] = {}
        for item in data["response"]:
            team_id = item["team"]["id"]
            p = item["player"]
            # Normalize "Missing Fixture" / "Questionable" — API uses player.type or player.reason
            raw_status = (p.get("status") or p.get("type") or "").lower()
            status = "Questionable" if any(w in raw_status for w in ("question", "doubtful", "doubt")) else "Missing"
            raw_by_team.setdefault(team_id, {})[p["id"]] = {
                "player_id": p["id"],
                "name": p["name"],
                "status": status,
            }

        def enrich(team_id: int) -> List[PlayerInjury]:
            injuries = []
            for p in raw_by_team.get(team_id, {}).values():
                stats = self._get_player_season_stats(p["player_id"], league_id, season)
                if not stats or stats.get("position", "Unknown") == "Unknown":
                    continue
                injuries.append(PlayerInjury(
                    player_id=p["player_id"],
                    player_name=p["name"],
                    team_id=team_id,
                    position=stats["position"],
                    status=p["status"],
                    goals=stats.get("goals", 0),
                    assists=stats.get("assists", 0),
                    minutes=stats.get("minutes", 0),
                ))
            return injuries

        home_injuries = enrich(fixture.home_team.id)
        away_injuries = enrich(fixture.away_team.id)
        home_gf, home_ga = self._get_team_season_stats(fixture.home_team.id, league_id, season)
        away_gf, away_ga = self._get_team_season_stats(fixture.away_team.id, league_id, season)
        return home_injuries, away_injuries, home_gf, away_gf, home_ga, away_ga

    def _get_player_season_stats(self, player_id: int, league_id: int, season: int) -> dict:
        """Returns {position, goals, assists, minutes}. TTL=24h (updates after each matchday)."""
        data = self.client.get(
            "players", {"id": player_id, "league": league_id, "season": season}, ttl=self.ttl.team_stats
        )
        if not data:
            return {}
        response = data.get("response", [])
        if not response:
            return {}
        stats_list = response[0].get("statistics", [])
        if not stats_list:
            return {}
        s = stats_list[0]
        games = s.get("games", {})
        goals = s.get("goals", {})
        return {
            "position": games.get("position") or "Unknown",
            "minutes":  games.get("minutes") or 0,
            "goals":    goals.get("total") or 0,
            "assists":  goals.get("assists") or 0,
        }

    def _get_team_season_stats(self, team_id: int, league_id: int, season: int) -> Tuple[int, int]:
        """Returns (goals_for, goals_against) for team in season. Fallback=(50, 50). TTL=6h."""
        data = self.client.get(
            "teams/statistics", {"team": team_id, "league": league_id, "season": season}, ttl=self.ttl.team_stats
        )
        if not data:
            return 50, 50
        response = data.get("response", {})
        goals = response.get("goals", {})
        gf = goals.get("for", {}).get("total", {}).get("total")
        ga = goals.get("against", {}).get("total", {}).get("total")
        return (int(gf) if gf else 50), (int(ga) if ga else 50)

    def get_fixture_lineups(
        self, fixture: Fixture
    ) -> Tuple[Optional[FixtureLineup], Optional[FixtureLineup]]:
        """Fetch official lineups for a fixture. Announced ~1h before kickoff.

        Returns (home_lineup, away_lineup). Either may be None if not yet announced.
        TTL=30min so empty pre-announcement responses are refreshed regularly.
        """
        data = self.client.get("fixtures/lineups", {"fixture": fixture.id}, ttl=self.ttl.lineups)
        if not data or not data.get("response"):
            return None, None

        home_id = fixture.home_team.id
        away_id = fixture.away_team.id
        lineups: Dict[int, FixtureLineup] = {}

        for entry in data["response"]:
            team_id = entry["team"]["id"]
            formation = entry.get("formation") or ""
            coach = entry.get("coach", {}).get("name") or ""

            starters = [
                LineupPlayer(
                    player_id=p["player"]["id"],
                    player_name=p["player"]["name"],
                    number=p["player"].get("number") or 0,
                    pos=p["player"].get("pos") or "?",
                    is_starter=True,
                )
                for p in entry.get("startXI", [])
            ]
            substitutes = [
                LineupPlayer(
                    player_id=p["player"]["id"],
                    player_name=p["player"]["name"],
                    number=p["player"].get("number") or 0,
                    pos=p["player"].get("pos") or "?",
                    is_starter=False,
                )
                for p in entry.get("substitutes", [])
            ]

            lineups[team_id] = FixtureLineup(
                team_id=team_id,
                formation=formation,
                coach=coach,
                starters=starters,
                substitutes=substitutes,
            )

        return lineups.get(home_id), lineups.get(away_id)

    def get_standings(self, league: LeagueConfig) -> list:
        """Returns list of standings tables (1 for domestic leagues, multiple for cup groups).
        Each entry contains rank, team, points, goalsDiff, form, and all/home/away stats.
        TTL=6h — updates after each matchday."""
        data = self.client.get(
            "standings",
            {"league": league.id, "season": league.season},
            ttl=86400,  # 24h — standings change only after a matchday
        )
        if not data:
            return []
        response = data.get("response", [])
        if not response:
            return []
        return response[0].get("league", {}).get("standings", [])

    def get_odds(self, fixture_id: int) -> List[Odds]:
        """Fetch all available bookmaker odds for a fixture in a single API call."""
        data = self.client.get("odds", {"fixture": fixture_id}, ttl=self.ttl.odds)
        return self._parse_odds(fixture_id, data) if data else []

    def _parse_team_stats(self, team_data: dict) -> FixtureStats:
        stats_map = {s["type"]: s["value"] for s in team_data.get("statistics", [])}

        def to_int(key):
            v = stats_map.get(key)
            try:
                return int(v) if v is not None else None
            except (ValueError, TypeError):
                return None

        possession = stats_map.get("Ball Possession")
        if isinstance(possession, str) and possession.endswith("%"):
            try:
                possession = float(possession[:-1]) / 100
            except ValueError:
                possession = None
        else:
            possession = None

        xg_raw = stats_map.get("expected_goals")
        xg = None
        if xg_raw is not None:
            try:
                xg = float(xg_raw)
            except (ValueError, TypeError):
                xg = None

        return FixtureStats(
            shots_on_target=to_int("Shots on Goal"),
            total_shots=to_int("Total Shots"),
            corners=to_int("Corner Kicks"),
            possession=possession,
            xg=xg,
        )

    def _parse_fixture(self, raw: dict) -> Fixture:
        f = raw["fixture"]
        teams = raw["teams"]
        goals = raw.get("goals") or {}
        home_g = goals.get("home")
        away_g = goals.get("away")
        return Fixture(
            id=f["id"],
            date=datetime.fromisoformat(f["date"].replace("Z", "+00:00")),
            league_id=raw["league"]["id"],
            season=raw["league"]["season"],
            home_team=Team(teams["home"]["id"], teams["home"]["name"], "", teams["home"].get("logo", "")),
            away_team=Team(teams["away"]["id"], teams["away"]["name"], "", teams["away"].get("logo", "")),
            # Both goals must be non-None — API sometimes returns None mid-match or for postponed fixtures
            result=MatchResult(int(home_g), int(away_g)) if (home_g is not None and away_g is not None) else None,
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
