from typing import Optional

import numpy as np
from scipy.stats import poisson

from data.models import Fixture, Prediction
from models.poisson import DixonColesPredictor

# Weights for domestic leagues (enough data per team)
W_ALL = 0.25
W_SEASON = 0.45
W_RECENT = 0.30

# Weights for cup competitions (few matches per team → rely more on full history)
W_ALL_CUP = 0.60
W_SEASON_CUP = 0.30
W_RECENT_CUP = 0.10

CUP_LEAGUES = {"champions_league", "europa_league", "conference_league"}


class EnsembleDCPredictor:
    """
    Weighted blend of three Dixon-Coles models:
      dc_all    — full 3-season history  (baseline)
      dc_season — current season only    (transfers/new coach)
      dc_recent — last ~60 days          (current form)

    Blending happens at the λ/μ level, then DC correction is applied once.
    If dc_recent lacks data (None), its weight is redistributed to the other two.
    """

    def __init__(
        self,
        dc_all: DixonColesPredictor,
        dc_season: DixonColesPredictor,
        dc_recent: Optional[DixonColesPredictor],
        league_key: str = "",
    ):
        self.dc_all = dc_all
        self.dc_season = dc_season
        self.dc_recent = dc_recent

        is_cup = league_key in CUP_LEAGUES
        w_all = W_ALL_CUP if is_cup else W_ALL
        w_season = W_SEASON_CUP if is_cup else W_SEASON
        w_recent = W_RECENT_CUP if is_cup else W_RECENT

        if dc_recent is not None:
            self._w_all, self._w_season, self._w_recent = w_all, w_season, w_recent
        else:
            total = w_all + w_season
            self._w_all = w_all / total
            self._w_season = w_season / total
            self._w_recent = 0.0

    def _lam_mu(self, model: DixonColesPredictor, h_id: int, a_id: int):
        lam = np.exp(model.attack.get(h_id, 0) - model.defence.get(a_id, 0) + model.home_advantage)
        mu = np.exp(model.attack.get(a_id, 0) - model.defence.get(h_id, 0))
        return float(lam), float(mu)

    def predict(self, fixture: Fixture, history=None) -> Prediction:
        h_id = fixture.home_team.id
        a_id = fixture.away_team.id

        lam_a, mu_a = self._lam_mu(self.dc_all, h_id, a_id)
        lam_s, mu_s = self._lam_mu(self.dc_season, h_id, a_id)

        if self.dc_recent is not None:
            lam_r, mu_r = self._lam_mu(self.dc_recent, h_id, a_id)
        else:
            lam_r = mu_r = 0.0

        lam = self._w_all * lam_a + self._w_season * lam_s + self._w_recent * lam_r
        mu = self._w_all * mu_a + self._w_season * mu_s + self._w_recent * mu_r

        max_g = 10
        prob_matrix = np.outer(
            poisson.pmf(range(max_g), lam),
            poisson.pmf(range(max_g), mu),
        )
        prob_matrix = self.dc_all._dc_correction(prob_matrix, lam, mu, self.dc_all.rho)

        prob_home = float(np.sum(np.tril(prob_matrix, -1)))
        prob_draw = float(np.sum(np.diag(prob_matrix)))
        prob_away = float(np.sum(np.triu(prob_matrix, 1)))

        tg = np.zeros(max_g * 2 - 1)
        for i in range(max_g):
            for j in range(max_g):
                tg[i + j] += prob_matrix[i, j]

        btts = float(sum(prob_matrix[i, j] for i in range(1, max_g) for j in range(1, max_g)))

        goal_probs = {
            "u2.5": round(float(tg[:3].sum()), 4),
            "o2.5": round(float(tg[3:].sum()), 4),
            "1-3":  round(float(tg[1:4].sum()), 4),
            "2-4":  round(float(tg[2:5].sum()), 4),
            "btts": round(btts, 4),
        }

        return Prediction(
            fixture_id=fixture.id,
            prob_home=prob_home,
            prob_draw=prob_draw,
            prob_away=prob_away,
            model_name="ensemble_dc",
            expected_goals_home=float(lam),
            expected_goals_away=float(mu),
            goal_probs=goal_probs,
        )
