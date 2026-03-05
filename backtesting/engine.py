import math
import logging
from dataclasses import dataclass, field
from typing import List, Dict

import numpy as np

from data.models import Fixture, Prediction
from models.base import BasePredictor

logger = logging.getLogger(__name__)


@dataclass
class BacktestMatch:
    fixture_id: int
    home_team: str
    away_team: str
    date: str
    actual: str        # 'H', 'D', 'A'
    predicted: str     # most likely outcome from model
    prob_home: float
    prob_draw: float
    prob_away: float


@dataclass
class BacktestResult:
    league_name: str
    n_train: int
    n_test: int
    matches: List[BacktestMatch] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        if not self.matches:
            return 0.0
        return sum(1 for m in self.matches if m.predicted == m.actual) / len(self.matches)

    @property
    def brier_score(self) -> float:
        """
        Multi-class Brier score (mean squared error over outcome probabilities).
        Random baseline ≈ 0.667. Lower is better.
        """
        if not self.matches:
            return 0.0
        total = 0.0
        for m in self.matches:
            for outcome, prob in [('H', m.prob_home), ('D', m.prob_draw), ('A', m.prob_away)]:
                o = 1.0 if m.actual == outcome else 0.0
                total += (prob - o) ** 2
        return total / len(self.matches)

    @property
    def log_loss(self) -> float:
        """
        Cross-entropy loss on correct outcome probability.
        Lower is better. Random baseline ≈ 1.099 (ln 3).
        """
        if not self.matches:
            return 0.0
        total = 0.0
        for m in self.matches:
            p = {'H': m.prob_home, 'D': m.prob_draw, 'A': m.prob_away}[m.actual]
            total -= math.log(max(p, 1e-10))
        return total / len(self.matches)

    def print_report(self) -> None:
        print(f"\n{'='*55}")
        print(f"  BACKTEST: {self.league_name}")
        print(f"{'='*55}")
        print(f"  Train:       {self.n_train} matches")
        print(f"  Test:        {self.n_test} matches  (actual: {len(self.matches)})")
        print(f"  Accuracy:    {self.accuracy:.1%}  (baseline H~45%, random~33%)")
        print(f"  Brier Score: {self.brier_score:.4f}  (random baseline ~0.6667)")
        print(f"  Log Loss:    {self.log_loss:.4f}  (random baseline ~1.0986)")

        print(f"\n  Calibration (predicted vs actual):")
        print(f"  {'Outcome':<10} {'Predicted as':>14} {'Avg model prob':>15} {'Actual rate':>12}")
        print(f"  {'-'*54}")
        for label in ('H', 'D', 'A'):
            label_name = {'H': 'Home win', 'D': 'Draw', 'A': 'Away win'}[label]
            subset = [m for m in self.matches if m.predicted == label]
            if not subset:
                continue
            prob_key = {'H': 'prob_home', 'D': 'prob_draw', 'A': 'prob_away'}[label]
            avg_prob = sum(getattr(m, prob_key) for m in subset) / len(subset)
            actual_rate = sum(1 for m in subset if m.actual == label) / len(subset)
            print(f"  {label_name:<10} {len(subset):>14}  {avg_prob:>14.1%}  {actual_rate:>11.1%}")

        # Outcome distribution in test set
        print(f"\n  Actual outcome distribution:")
        for label, name in [('H', 'Home win'), ('D', 'Draw'), ('A', 'Away win')]:
            n = sum(1 for m in self.matches if m.actual == label)
            print(f"  {name:<10} {n:>5}  ({n/len(self.matches):.1%})")


@dataclass
class CalibrationBin:
    bin_lower: float       # e.g. 0.3
    bin_upper: float       # e.g. 0.4
    predicted_prob: float  # mean predicted prob in this bin
    actual_freq: float     # actual win rate in this bin
    count: int             # number of samples in bin


@dataclass
class OutcomeCalibration:
    outcome: str                      # 'H', 'D', 'A'
    ece: float                        # Expected Calibration Error
    bins: List[CalibrationBin]


@dataclass
class CalibrationResult:
    model: str
    n_samples: int
    overall_ece: float
    outcomes: List[OutcomeCalibration]


def compute_calibration(matches: List[BacktestMatch], model_name: str,
                        n_bins: int = 10) -> CalibrationResult:
    """
    Reliability diagram data + ECE for each outcome (H, D, A).
    Lower ECE = better calibrated. Perfect calibration = ECE 0.
    Typical range: 0.02 (great) – 0.10 (poor).
    """
    outcome_configs = [
        ("H", [m.prob_home for m in matches]),
        ("D", [m.prob_draw for m in matches]),
        ("A", [m.prob_away for m in matches]),
    ]
    actuals = [m.actual for m in matches]
    calibrations = []
    ece_sum = 0.0

    for outcome, probs in outcome_configs:
        labels = [1 if a == outcome else 0 for a in actuals]
        bins: List[CalibrationBin] = []
        ece = 0.0
        edges = np.linspace(0, 1, n_bins + 1)

        for i in range(n_bins):
            lo, hi = edges[i], edges[i + 1]
            idxs = [j for j, p in enumerate(probs) if lo <= p < hi]
            if not idxs:
                continue
            mean_pred = float(np.mean([probs[j] for j in idxs]))
            actual_freq = float(np.mean([labels[j] for j in idxs]))
            count = len(idxs)
            bins.append(CalibrationBin(
                bin_lower=round(lo, 2),
                bin_upper=round(hi, 2),
                predicted_prob=round(mean_pred, 4),
                actual_freq=round(actual_freq, 4),
                count=count,
            ))
            ece += (count / len(matches)) * abs(mean_pred - actual_freq)

        calibrations.append(OutcomeCalibration(outcome=outcome, ece=round(ece, 4), bins=bins))
        ece_sum += ece

    return CalibrationResult(
        model=model_name,
        n_samples=len(matches),
        overall_ece=round(ece_sum / 3, 4),
        outcomes=calibrations,
    )


class BacktestEngine:
    """
    Walk-forward backtesting with periodic retrain.

    Fixtures are sorted by date. The model is trained on all fixtures before
    each test match, with retraining every `retrain_every` matches to balance
    accuracy and speed.
    """

    def __init__(
        self,
        predictor: BasePredictor,
        min_train_size: int = 100,
        retrain_every: int = 10,
    ):
        self.predictor = predictor
        self.min_train_size = min_train_size
        self.retrain_every = retrain_every

    def run(self, fixtures: List[Fixture], league_name: str = "") -> BacktestResult:
        completed = sorted(
            [f for f in fixtures if f.result is not None],
            key=lambda f: f.date,
        )

        n = len(completed)
        if n < self.min_train_size + 10:
            raise ValueError(
                f"Not enough data: {n} matches (need at least {self.min_train_size + 10})"
            )

        n_test = n - self.min_train_size
        result = BacktestResult(league_name=league_name, n_train=self.min_train_size, n_test=n_test)
        logger.info(f"Backtesting {league_name}: {n} total, {n_test} test matches")

        last_trained_at = None

        for i in range(self.min_train_size, n):
            # Retrain if first test match or every `retrain_every` steps
            if last_trained_at is None or (i - self.min_train_size) % self.retrain_every == 0:
                train_data = completed[:i]
                try:
                    self.predictor.train(train_data)
                    last_trained_at = i
                    logger.info(f"  Retrained at match {i}/{n} ({len(train_data)} train fixtures)")
                except Exception as e:
                    logger.warning(f"  Training failed at index {i}: {e}")
                    continue

            test_fixture = completed[i]
            try:
                prediction = self.predictor.predict(test_fixture)
            except Exception as e:
                logger.warning(f"  Prediction failed for fixture {test_fixture.id}: {e}")
                continue

            probs = {'H': prediction.prob_home, 'D': prediction.prob_draw, 'A': prediction.prob_away}
            predicted = max(probs, key=probs.get)

            result.matches.append(BacktestMatch(
                fixture_id=test_fixture.id,
                home_team=test_fixture.home_team.name,
                away_team=test_fixture.away_team.name,
                date=test_fixture.date.strftime('%Y-%m-%d'),
                actual=test_fixture.result.outcome,
                predicted=predicted,
                prob_home=prediction.prob_home,
                prob_draw=prediction.prob_draw,
                prob_away=prediction.prob_away,
            ))

        return result
