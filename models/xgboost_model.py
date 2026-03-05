from typing import List, Optional

import numpy as np
from xgboost import XGBClassifier

from data.models import Fixture, Prediction
from features.engineer import FeatureEngineer
from models.base import BasePredictor

LABEL_MAP = {'H': 0, 'D': 1, 'A': 2}


class XGBoostPredictor(BasePredictor):
    def __init__(self, form_window: int = 5, **xgb_kwargs):
        self._engineer = FeatureEngineer(form_window=form_window)
        self._clf = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='mlogloss',
            **xgb_kwargs,
        )
        self._history: List[Fixture] = []
        self._feature_names: List[str] = []
        self._fitted = False

    @property
    def name(self) -> str:
        return "xgboost"

    def train(self, fixtures: List[Fixture]) -> None:
        completed = [f for f in fixtures if f.result is not None]
        if not completed:
            return

        raw = [self._engineer.build_features(f, completed) for f in completed]
        self._feature_names = sorted(raw[0].keys())
        X = np.array([[row[k] for k in self._feature_names] for row in raw])
        y = np.array([LABEL_MAP[f.result.outcome] for f in completed])

        self._clf.fit(X, y)
        self._history = completed
        self._fitted = True

    def predict(self, fixture: Fixture, history: Optional[List[Fixture]] = None) -> Prediction:
        if not self._fitted:
            raise RuntimeError("Model není natrénovaný. Zavolej .train() nejdříve.")

        hist = history if history is not None else self._history
        feats = self._engineer.build_features(fixture, hist)
        x = np.array([[feats[k] for k in self._feature_names]])
        proba = self._clf.predict_proba(x)[0]  # [p_H, p_D, p_A]

        return Prediction(
            fixture_id=fixture.id,
            prob_home=float(proba[0]),
            prob_draw=float(proba[1]),
            prob_away=float(proba[2]),
            model_name=self.name,
        )
