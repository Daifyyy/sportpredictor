from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson

from data.models import Fixture

# Weights — same as goals ensemble (domestic leagues)
_W_ALL    = 0.25
_W_SEASON = 0.45
_W_RECENT = 0.30


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


def _ou(total_lam: float, threshold: float) -> tuple[float, float]:
    """P(X > threshold) and P(X <= threshold) for Poisson(total_lam)."""
    over = round(1.0 - float(poisson.cdf(int(threshold), total_lam)), 4)
    return over, round(1.0 - over, 4)


class CornersPredictor:
    """Pure Poisson MLE for corners — no DC correction (corners avg ~10, never 0)."""

    def __init__(self, home_advantage: float = 0.05):
        self.home_advantage = home_advantage
        self.attack: Dict[int, float] = {}
        self.defence: Dict[int, float] = {}
        self._fitted = False

    def train(self, fixtures: List[Fixture]) -> None:
        completed = [
            f for f in fixtures
            if f.result is not None
            and f.home_stats is not None
            and f.away_stats is not None
            and f.home_stats.corners is not None
            and f.away_stats.corners is not None
        ]
        if len(completed) < 30:
            return

        teams = list({f.home_team.id for f in completed} | {f.away_team.id for f in completed})
        team_idx = {t: i for i, t in enumerate(teams)}
        n = len(teams)

        hi = np.array([team_idx[f.home_team.id] for f in completed])
        ai = np.array([team_idx[f.away_team.id] for f in completed])
        hc = np.array([f.home_stats.corners for f in completed], dtype=np.int32)
        ac = np.array([f.away_stats.corners for f in completed], dtype=np.int32)

        def neg_ll(params):
            att = params[:n]
            dfc = params[n:2 * n]
            ha  = params[2 * n]
            lam = np.exp(att[hi] - dfc[ai] + ha)
            mu  = np.exp(att[ai] - dfc[hi])
            return -(poisson.logpmf(hc, lam).sum() + poisson.logpmf(ac, mu).sum())

        x0 = np.zeros(2 * n + 1)
        x0[2 * n] = self.home_advantage
        res = minimize(neg_ll, x0, method="L-BFGS-B")

        for i, tid in enumerate(teams):
            self.attack[tid]  = res.x[i]
            self.defence[tid] = res.x[n + i]
        self.home_advantage = res.x[2 * n]
        self._fitted = True

    def _lam_mu(self, h_id: int, a_id: int) -> tuple[float, float]:
        lam = float(np.exp(self.attack.get(h_id, 0) - self.defence.get(a_id, 0) + self.home_advantage))
        mu  = float(np.exp(self.attack.get(a_id, 0) - self.defence.get(h_id, 0)))
        return lam, mu

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "CornersPredictor":
        return joblib.load(path)


class EnsembleCornersPredictor:
    """Weighted blend of three CornersPredictor models at the λ/μ level."""

    def __init__(
        self,
        c_all: CornersPredictor,
        c_season: CornersPredictor,
        c_recent: Optional[CornersPredictor] = None,
    ):
        self.c_all    = c_all
        self.c_season = c_season
        self.c_recent = c_recent

        if c_recent is not None:
            self._w_all, self._w_season, self._w_recent = _W_ALL, _W_SEASON, _W_RECENT
        else:
            total = _W_ALL + _W_SEASON
            self._w_all    = _W_ALL / total
            self._w_season = _W_SEASON / total
            self._w_recent = 0.0

    def predict_corners(self, fixture: Fixture) -> Optional[CornersPrediction]:
        if not self.c_all._fitted:
            return None
        h_id = fixture.home_team.id
        a_id = fixture.away_team.id

        lam_a, mu_a = self.c_all._lam_mu(h_id, a_id)
        lam_s, mu_s = self.c_season._lam_mu(h_id, a_id)
        lam_r = mu_r = 0.0
        if self.c_recent is not None:
            lam_r, mu_r = self.c_recent._lam_mu(h_id, a_id)

        lam = self._w_all * lam_a + self._w_season * lam_s + self._w_recent * lam_r
        mu  = self._w_all * mu_a  + self._w_season * mu_s  + self._w_recent * mu_r
        total = lam + mu

        o8,  u8  = _ou(total, 8.5)
        o9,  u9  = _ou(total, 9.5)
        o10, u10 = _ou(total, 10.5)
        o11, u11 = _ou(total, 11.5)

        return CornersPrediction(
            fixture_id=fixture.id,
            lambda_home=round(lam, 3),
            mu_away=round(mu, 3),
            over8_5=o8,   under8_5=u8,
            over9_5=o9,   under9_5=u9,
            over10_5=o10, under10_5=u10,
            over11_5=o11, under11_5=u11,
        )


def corners_prediction_from_lam_mu(fixture_id: int, lam: float, mu: float) -> CornersPrediction:
    """Build a CornersPrediction from calibrated λ/μ, recomputing all O/U probabilities."""
    total = lam + mu
    o8,  u8  = _ou(total, 8.5)
    o9,  u9  = _ou(total, 9.5)
    o10, u10 = _ou(total, 10.5)
    o11, u11 = _ou(total, 11.5)
    return CornersPrediction(
        fixture_id=fixture_id,
        lambda_home=round(lam, 3),
        mu_away=round(mu, 3),
        over8_5=o8,   under8_5=u8,
        over9_5=o9,   under9_5=u9,
        over10_5=o10, under10_5=u10,
        over11_5=o11, under11_5=u11,
    )


def train_corners_ensemble(
    completed: List[Fixture],
    season: int,
    models_dir: Path,
    league_slug: str,
) -> Optional[EnsembleCornersPredictor]:
    """Train all/season/recent CornersPredictor models and return ensemble.

    Returns None if not enough enriched fixtures with corners data.
    """
    with_corners = [
        f for f in completed
        if f.home_stats and f.home_stats.corners is not None
        and f.away_stats and f.away_stats.corners is not None
    ]
    if len(with_corners) < 50:
        print(f"  Corners: only {len(with_corners)} enriched fixtures, skipping.")
        return None

    c_all = CornersPredictor()
    c_all.train(with_corners)
    if not c_all._fitted:
        return None
    c_all.save(models_dir / f"corners_all_{league_slug}.joblib")
    print(f"  Corners c_all trained on {len(with_corners)} fixtures")

    season_c = [f for f in with_corners if f.season == season]
    if len(season_c) >= 30:
        c_season = CornersPredictor()
        c_season.train(season_c)
        if not c_season._fitted:
            c_season = c_all
        else:
            c_season.save(models_dir / f"corners_season_{league_slug}.joblib")
            print(f"  Corners c_season trained on {len(season_c)} fixtures")
    else:
        c_season = c_all
        print(f"  Corners c_season fallback to c_all ({len(season_c)} season fixtures)")

    cutoff = max(f.date for f in with_corners) - timedelta(days=60)
    recent_c = [f for f in with_corners if f.date >= cutoff]
    if len(recent_c) >= 30:
        c_recent = CornersPredictor()
        c_recent.train(recent_c)
        if not c_recent._fitted:
            c_recent = None
        else:
            print(f"  Corners c_recent trained on {len(recent_c)} fixtures (last 60 days)")
    else:
        c_recent = None
        print(f"  Corners c_recent skipped ({len(recent_c)} recent fixtures)")

    return EnsembleCornersPredictor(c_all, c_season, c_recent)
