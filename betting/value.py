from typing import List, Optional
from data.models import Prediction, Odds

# Bookmakers ordered from sharpest to softest margin.
# The first one found in the available odds list is used as reference.
_SHARP_PRIORITY = ["pinnacle", "betfair", "sbobet", "marathonbet", "1xbet", "bet365", "bwin", "unibet"]


class ValueBetDetector:
    def __init__(self, min_edge: float = 0.03, min_prob: float = 0.60, max_odds: float = 2.7):
        self.min_edge = min_edge
        self.min_prob = min_prob
        self.max_odds = max_odds

    def detect(self, prediction: Prediction, odds: List[Odds]) -> Prediction:
        if not odds:
            prediction.value_bets = []
            return prediction

        reference = self._find_sharpest(odds)
        if not reference:
            prediction.value_bets = []
            return prediction

        ref_implied = reference.implied_probs()
        ref_odds_map = {"H": reference.home_win, "D": reference.draw, "A": reference.away_win}

        checks = [
            ("H", prediction.prob_home),
            ("D", prediction.prob_draw),
            ("A", prediction.prob_away),
        ]

        value_bets = []
        for outcome, model_prob in checks:
            if model_prob < self.min_prob:
                continue
            decimal_odd = ref_odds_map[outcome]
            if decimal_odd > self.max_odds:
                continue
            edge = model_prob - ref_implied[outcome]
            if edge >= self.min_edge:
                ev = model_prob * decimal_odd - 1
                value_bets.append(
                    f"{outcome} @ {decimal_odd} | edge={edge:.1%} | EV={ev:.1%} ({reference.bookmaker})"
                )

        prediction.value_bets = value_bets
        return prediction

    @classmethod
    def _find_sharpest(cls, odds: List[Odds]) -> Optional[Odds]:
        """Return the sharpest available bookmaker by priority list."""
        for name in _SHARP_PRIORITY:
            found = cls._find_bookmaker(odds, name)
            if found:
                return found
        return odds[0] if odds else None

    @staticmethod
    def _find_bookmaker(odds: List[Odds], name: str) -> Optional[Odds]:
        return next((o for o in odds if name.lower() in o.bookmaker.lower()), None)
