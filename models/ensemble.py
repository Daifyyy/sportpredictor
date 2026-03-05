from typing import List, Optional

from data.models import Fixture, Prediction
from models.base import BasePredictor


class EnsemblePredictor(BasePredictor):
    """Averages probabilities from multiple models. Not persisted — composed at runtime."""

    def __init__(self, models: List[BasePredictor]):
        self._models = models

    @property
    def name(self) -> str:
        return "ensemble"

    def train(self, fixtures) -> None:
        for m in self._models:
            m.train(fixtures)

    def predict(self, fixture: Fixture, history: Optional[List[Fixture]] = None) -> Prediction:
        preds = [m.predict(fixture, history) for m in self._models]
        n = len(preds)
        xg_home = next((p.expected_goals_home for p in preds if p.expected_goals_home is not None), None)
        xg_away = next((p.expected_goals_away for p in preds if p.expected_goals_away is not None), None)
        return Prediction(
            fixture_id=fixture.id,
            prob_home=sum(p.prob_home for p in preds) / n,
            prob_draw=sum(p.prob_draw for p in preds) / n,
            prob_away=sum(p.prob_away for p in preds) / n,
            model_name=self.name,
            expected_goals_home=xg_home,
            expected_goals_away=xg_away,
        )
