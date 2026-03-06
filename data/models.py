from dataclasses import dataclass, field
from typing import Optional, List, Dict
from datetime import datetime


@dataclass
class Team:
    id: int
    name: str
    country: str


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
class Fixture:
    id: int
    date: datetime
    league_id: int
    season: int
    home_team: Team
    away_team: Team
    result: Optional[MatchResult] = None
    status: str = "NS"  # NS, FT, LIVE...


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
