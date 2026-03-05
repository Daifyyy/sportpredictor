"""
Post-hoc probability calibration using isotonic regression.

Trained from resolved predictions in Supabase — no full model retrain needed.
Applied automatically in predictions if a saved calibrator exists.

Usage:
    cal = ProbabilityCalibrator()
    cal.fit(probs_h, probs_d, probs_a, actuals)   # actuals: list of 'H'/'D'/'A'
    cal.save(path)

    cal = ProbabilityCalibrator.load(path)
    p_h, p_d, p_a = cal.transform(prob_h, prob_d, prob_a)
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression


class ProbabilityCalibrator:
    def __init__(self):
        self._iso: dict[str, IsotonicRegression] = {}
        self._fitted = False

    def fit(
        self,
        probs_h: List[float],
        probs_d: List[float],
        probs_a: List[float],
        actuals: List[str],          # 'H', 'D', 'A'
        min_samples: int = 50,
    ) -> None:
        if len(actuals) < min_samples:
            raise ValueError(f"Need at least {min_samples} resolved predictions, got {len(actuals)}")

        for outcome, probs in [("H", probs_h), ("D", probs_d), ("A", probs_a)]:
            labels = [1.0 if a == outcome else 0.0 for a in actuals]
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(probs, labels)
            self._iso[outcome] = iso

        self._fitted = True

    def transform(self, prob_h: float, prob_d: float, prob_a: float) -> Tuple[float, float, float]:
        if not self._fitted:
            return prob_h, prob_d, prob_a

        cal_h = float(self._iso["H"].predict([prob_h])[0])
        cal_d = float(self._iso["D"].predict([prob_d])[0])
        cal_a = float(self._iso["A"].predict([prob_a])[0])

        # Renormalize so probabilities sum to 1
        total = cal_h + cal_d + cal_a
        if total <= 0:
            return prob_h, prob_d, prob_a
        return cal_h / total, cal_d / total, cal_a / total

    @property
    def fitted(self) -> bool:
        return self._fitted

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "ProbabilityCalibrator":
        return joblib.load(path)
