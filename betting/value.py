from typing import List, Optional
from data.models import Prediction, Odds


class ValueBetDetector:
    def __init__(self, min_edge: float = 0.03, min_prob: float = 0.60, max_odds: float = 2.7):
        """
        min_edge = minimální edge nad Pinnacle implied probability (3 %).
        min_prob = minimální pravděpodobnost modelu — sázíme jen na výsledky s 60%+ jistotou.
        max_odds = maximální kurz pro sázení — nic nad 2.7 (favorité, ne long-shoty).
        Pinnacle slouží jako sharp reference (nejnižší margin = nejpřesnější tržní prob).
        Sázecí kurzy jsou brány z Bet365 (pokud dostupné), jinak z Pinnacle.
        """
        self.min_edge = min_edge
        self.min_prob = min_prob
        self.max_odds = max_odds

    def detect(self, prediction: Prediction, odds: List[Odds]) -> Prediction:
        if not odds:
            return prediction

        pinnacle = self._find_bookmaker(odds, "pinnacle")
        if not pinnacle:
            # Bez Pinnacle nelze spolehlivě detekovat value — přeskočíme
            prediction.value_bets = []
            return prediction

        # Bet365 pro zobrazení kurzu (kde fyzicky sázíme)
        bet365 = self._find_bookmaker(odds, "bet365")
        display = bet365 or pinnacle

        pinnacle_implied = pinnacle.implied_probs()
        display_odds_map = {"H": display.home_win, "D": display.draw, "A": display.away_win}

        checks = [
            ("H", prediction.prob_home),
            ("D", prediction.prob_draw),
            ("A", prediction.prob_away),
        ]

        value_bets = []
        for outcome, model_prob in checks:
            if model_prob < self.min_prob:
                continue
            decimal_odd = display_odds_map[outcome]
            if decimal_odd > self.max_odds:
                continue
            edge = model_prob - pinnacle_implied[outcome]
            if edge >= self.min_edge:
                ev = model_prob * decimal_odd - 1
                value_bets.append(
                    f"{outcome} @ {decimal_odd} ({display.bookmaker}) | edge={edge:.1%} | EV={ev:.1%} [vs Pinnacle]"
                )

        prediction.value_bets = value_bets
        return prediction

    @staticmethod
    def _find_bookmaker(odds: List[Odds], name: str) -> Optional[Odds]:
        return next((o for o in odds if name.lower() in o.bookmaker.lower()), None)
