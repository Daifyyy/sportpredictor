from dataclasses import dataclass, field
from typing import Optional, List, Dict
from datetime import datetime


@dataclass
class Team:
    id: int
    name: str
    country: str
    logo: str = ""


@dataclass
class MatchResult:
    home_goals: int
    away_goals: int

    @property
    def outcome(self) -> str:  # 'H', 'D', 'A'
        if self.home_goals > self.away_goals: return 'H'
        if self.home_goals < self.away_goals: return 'A'
        return 'D'


@dataclass
class FixtureStats:
    shots_on_target: Optional[int] = None
    total_shots: Optional[int] = None
    corners: Optional[int] = None
    possession: Optional[float] = None  # 0.0–1.0
    xg: Optional[float] = None          # only available for top leagues (PL, LaLiga, BL)


@dataclass
class Fixture:
    id: int
    date: datetime
    league_id: int
    season: int
    home_team: Team
    away_team: Team
    result: Optional[MatchResult] = None
    status: str = "NS"  # NS, FT, LIVE...
    home_stats: Optional[FixtureStats] = None
    away_stats: Optional[FixtureStats] = None
    referee: Optional[str] = None


@dataclass
class PlayerInjury:
    player_id: int
    player_name: str
    team_id: int
    position: str   # "Attacker", "Midfielder", "Defender", "Goalkeeper"
    status: str     # "Missing" or "Questionable"
    goals: int = 0
    assists: int = 0
    minutes: int = 0


@dataclass
class LineupPlayer:
    player_id: int
    player_name: str
    number: int
    pos: str        # "G", "D", "M", "F"
    is_starter: bool


@dataclass
class FixtureLineup:
    team_id: int
    formation: str          # e.g. "4-2-3-1"
    coach: str
    starters: List["LineupPlayer"]
    substitutes: List["LineupPlayer"]


@dataclass
class Odds:
    fixture_id: int
    bookmaker: str
    home_win: float
    draw: float
    away_win: float

    def implied_probs(self) -> dict:
        """Odstraní margin bookmakers."""
        raw = {
            'H': 1 / self.home_win,
            'D': 1 / self.draw,
            'A': 1 / self.away_win,
        }
        total = sum(raw.values())
        return {k: v / total for k, v in raw.items()}


@dataclass
class Prediction:
    fixture_id: int
    prob_home: float
    prob_draw: float
    prob_away: float
    model_name: str
    value_bets: List[str] = field(default_factory=list)
    expected_goals_home: Optional[float] = None
    expected_goals_away: Optional[float] = None
    goal_probs: Dict[str, float] = field(default_factory=dict)


@dataclass
class CornersPrediction:
    fixture_id: int
    lambda_home: float
    mu_away: float
    over8_5:  float
    under8_5: float
    over9_5:  float
    under9_5: float
    over10_5: float
    under10_5: float
    over11_5: float
    under11_5: float
