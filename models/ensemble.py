from datetime import timedelta
from typing import Optional

import numpy as np
from scipy.optimize import minimize as sp_minimize
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
        delta = getattr(model, 'home_adv_delta', {}).get(h_id, 0.0)
        lam = np.exp(model.attack.get(h_id, 0) - model.defence.get(a_id, 0) + model.home_advantage + delta)
        mu = np.exp(model.attack.get(a_id, 0) - model.defence.get(h_id, 0))
        return float(lam), float(mu)

    def optimize_weights(
        self,
        completed: list,
        min_train: int = 200,
        retrain_every: int = 100,
        xi_fixed_all: float = 0.0,
        xi_fixed_season: float = 0.003,
        xi_fixed_recent: float = 0.0,
    ) -> None:
        """Walk-forward Brier score minimization to set optimal blend weights.

        All three models use fixed xi matching production to ensure the walk-forward
        holdout reflects the real model setup. Free xi optimization causes all models
        to converge to xi=0.02 (upper bound), making them indistinguishable and
        collapsing weights to [1, 0, 0].

        xi_fixed_season=0.003 → half-life ~231 days: full season stays relevant
        with mild recency bias, keeping dc_season distinct from dc_recent.

        Modifies self._w_all / _w_season / _w_recent in-place.
        """
        if len(completed) < min_train + 50:
            return

        completed_sorted = sorted(completed, key=lambda f: f.date)

        lam_a_list, mu_a_list = [], []
        lam_s_list, mu_s_list = [], []
        lam_r_list, mu_r_list = [], []
        y_h_list, y_d_list, y_a_list = [], [], []
        has_r_list = []

        dc_a = dc_s = dc_r = None

        for i in range(min_train, len(completed_sorted)):
            if dc_a is None or (i - min_train) % retrain_every == 0:
                train_slice = completed_sorted[:i]
                cur_season = completed_sorted[i].season
                cur_date = completed_sorted[i].date

                dc_a = DixonColesPredictor()
                dc_a.train(train_slice, xi_fixed=xi_fixed_all)

                season_slice = [f for f in train_slice if f.season == cur_season]
                if len(season_slice) >= 30:
                    dc_s = DixonColesPredictor()
                    dc_s.train(season_slice, xi_fixed=xi_fixed_season)
                else:
                    dc_s = dc_a

                cutoff = cur_date - timedelta(days=60)
                recent_slice = [f for f in train_slice if f.date >= cutoff]
                if len(recent_slice) >= 30:
                    dc_r = DixonColesPredictor()
                    dc_r.train(recent_slice, xi_fixed=xi_fixed_recent)
                else:
                    dc_r = None

            fx = completed_sorted[i]
            h_id, a_id = fx.home_team.id, fx.away_team.id
            try:
                lam_a, mu_a = self._lam_mu(dc_a, h_id, a_id)
                lam_s, mu_s = self._lam_mu(dc_s, h_id, a_id)
                lam_r, mu_r = self._lam_mu(dc_r, h_id, a_id) if dc_r else (0.0, 0.0)
            except Exception:
                continue

            outcome = fx.result.outcome
            lam_a_list.append(lam_a); mu_a_list.append(mu_a)
            lam_s_list.append(lam_s); mu_s_list.append(mu_s)
            lam_r_list.append(lam_r); mu_r_list.append(mu_r)
            y_h_list.append(1.0 if outcome == "H" else 0.0)
            y_d_list.append(1.0 if outcome == "D" else 0.0)
            y_a_list.append(1.0 if outcome == "A" else 0.0)
            has_r_list.append(dc_r is not None)

        if len(y_h_list) < 80:
            return

        lam_a_arr = np.array(lam_a_list); mu_a_arr = np.array(mu_a_list)
        lam_s_arr = np.array(lam_s_list); mu_s_arr = np.array(mu_s_list)
        lam_r_arr = np.array(lam_r_list); mu_r_arr = np.array(mu_r_list)
        y_h = np.array(y_h_list); y_d = np.array(y_d_list); y_a = np.array(y_a_list)
        recent_available = float(np.mean(has_r_list)) > 0.5

        max_g = 10
        goals = np.arange(max_g)
        tril_r, tril_c = np.tril_indices(max_g, -1)
        diag_r, diag_c = np.diag_indices(max_g)
        triu_r, triu_c = np.triu_indices(max_g, 1)

        def brier(w):
            lam = w[0] * lam_a_arr + w[1] * lam_s_arr + w[2] * lam_r_arr
            mu  = w[0] * mu_a_arr  + w[1] * mu_s_arr  + w[2] * mu_r_arr
            pmf_h = poisson.pmf(goals[:, None], lam[None, :])  # (max_g, N)
            pmf_a = poisson.pmf(goals[:, None], mu[None, :])
            pm = pmf_h[:, None, :] * pmf_a[None, :, :]          # (max_g, max_g, N)
            ph  = pm[tril_r, tril_c].sum(axis=0)
            pd_ = pm[diag_r, diag_c].sum(axis=0)
            pa  = pm[triu_r, triu_c].sum(axis=0)
            return float(np.mean((ph - y_h) ** 2 + (pd_ - y_d) ** 2 + (pa - y_a) ** 2))

        w0 = np.array([self._w_all, self._w_season, self._w_recent if recent_available else 0.0])

        if not recent_available:
            def obj_2d(w):
                return brier([w[0], 1.0 - w[0], 0.0])
            res = sp_minimize(obj_2d, [w0[0]], method="L-BFGS-B", bounds=[(0.0, 1.0)])
            self._w_all    = float(np.clip(res.x[0], 0.0, 1.0))
            self._w_season = 1.0 - self._w_all
            self._w_recent = 0.0
        else:
            res = sp_minimize(
                brier, w0,
                method="SLSQP",
                bounds=[(0.0, 1.0)] * 3,
                constraints=[{"type": "eq", "fun": lambda w: float(w.sum()) - 1.0}],
            )
            w_opt = np.clip(res.x, 0.0, 1.0)
            w_opt /= w_opt.sum()
            self._w_all, self._w_season, self._w_recent = float(w_opt[0]), float(w_opt[1]), float(w_opt[2])

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
            "over2_5":  round(float(tg[3:].sum()), 4),
            "under2_5": round(float(tg[:3].sum()), 4),
            "goals1_3": round(float(tg[1:4].sum()), 4),
            "goals2_4": round(float(tg[2:5].sum()), 4),
            "btts_yes": round(btts, 4),
            "btts_no":  round(1 - btts, 4),
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
