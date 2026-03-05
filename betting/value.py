from typing import List
from data.models import Prediction, Odds


class ValueBetDetector:
    def __init__(self, min_edge: float = 0.03):
        """min_edge = minimální edge nad implied probability bookmakers (3 %)"""
        self.min_edge = min_edge

    def detect(self, prediction: Prediction, odds: List[Odds]) -> Prediction:
        if not odds:
            return prediction

        value_bets = []
        for odd in odds:
            implied = odd.implied_probs()
            checks = [
                ("H", prediction.prob_home, odd.home_win),
                ("D", prediction.prob_draw, odd.draw),
                ("A", prediction.prob_away, odd.away_win),
            ]
            for outcome, model_prob, decimal_odd in checks:
                edge = model_prob - implied[outcome]
                if edge >= self.min_edge:
                    ev = model_prob * decimal_odd - 1
                    value_bets.append(
                        f"{outcome} @ {decimal_odd} | edge={edge:.1%} | EV={ev:.1%} ({odd.bookmaker})"
                    )

        prediction.value_bets = value_bets
        return prediction
