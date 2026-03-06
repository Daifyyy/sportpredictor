import numpy as np
from scipy.stats import poisson
from scipy.optimize import minimize
from typing import List, Dict
from data.models import Fixture, Prediction
from models.base import BasePredictor


class DixonColesPredictor(BasePredictor):
    """
    Dixon-Coles model (1997) — lepší než základní Poisson,
    opravuje podhodnocení low-score výsledků (0-0, 1-0, 0-1, 1-1).
    """

    @property
    def name(self) -> str:
        return "dixon_coles"

    def __init__(self, home_advantage: float = 0.1):
        self.home_advantage = home_advantage
        self.rho: float = -0.13
        self.attack: Dict[int, float] = {}
        self.defence: Dict[int, float] = {}
        self._fitted = False

    @staticmethod
    def _tau(x: int, y: int, lam: float, mu: float, rho: float) -> float:
        """Dixon-Coles correction factor for low-score results."""
        if x == 0 and y == 0: return 1 - lam * mu * rho
        if x == 0 and y == 1: return 1 + lam * rho
        if x == 1 and y == 0: return 1 + mu * rho
        if x == 1 and y == 1: return 1 - rho
        return 1.0

    def train(self, fixtures: List[Fixture]) -> None:
        completed = [f for f in fixtures if f.result is not None]
        teams = list({f.home_team.id for f in completed} | {f.away_team.id for f in completed})
        team_idx = {t: i for i, t in enumerate(teams)}
        n = len(teams)

        # Pre-compute index/goal arrays once — avoids Python loop inside optimizer
        hi_arr = np.array([team_idx[f.home_team.id] for f in completed])
        ai_arr = np.array([team_idx[f.away_team.id] for f in completed])
        hg_arr = np.array([f.result.home_goals for f in completed], dtype=np.int32)
        ag_arr = np.array([f.result.away_goals for f in completed], dtype=np.int32)

        # Boolean masks for DC tau correction (only 0-0, 0-1, 1-0, 1-1)
        m00 = (hg_arr == 0) & (ag_arr == 0)
        m01 = (hg_arr == 0) & (ag_arr == 1)
        m10 = (hg_arr == 1) & (ag_arr == 0)
        m11 = (hg_arr == 1) & (ag_arr == 1)

        def log_likelihood(params):
            attack  = params[:n]
            defence = params[n:2*n]
            home    = params[2*n]
            rho     = params[2*n + 1]

            lam = np.exp(attack[hi_arr] - defence[ai_arr] + home)
            mu  = np.exp(attack[ai_arr] - defence[hi_arr])

            ll = -(poisson.logpmf(hg_arr, lam).sum() + poisson.logpmf(ag_arr, mu).sum())

            tau = np.ones(len(completed))
            tau[m00] = 1 - lam[m00] * mu[m00] * rho
            tau[m01] = 1 + lam[m01] * rho
            tau[m10] = 1 + mu[m10] * rho
            tau[m11] = 1 - rho
            if np.any(tau <= 0):
                return 1e15
            ll -= np.log(tau).sum()
            return ll

        x0 = np.zeros(2 * n + 2)
        x0[2 * n]     = self.home_advantage
        x0[2 * n + 1] = -0.13
        bounds = [(None, None)] * (2 * n + 1) + [(-0.99, 0.99)]
        res = minimize(log_likelihood, x0, method='L-BFGS-B', bounds=bounds)

        for i, team_id in enumerate(teams):
            self.attack[team_id]  = res.x[i]
            self.defence[team_id] = res.x[n + i]
        self.home_advantage = res.x[2 * n]
        self.rho            = res.x[2 * n + 1]
        self._fitted = True

    def predict(self, fixture: Fixture, history: List[Fixture] = None) -> Prediction:
        if not self._fitted:
            raise RuntimeError("Model není natrénovaný. Zavolej .train() nejdříve.")

        h_id = fixture.home_team.id
        a_id = fixture.away_team.id
        lam = np.exp(self.attack.get(h_id, 0) - self.defence.get(a_id, 0) + self.home_advantage)
        mu  = np.exp(self.attack.get(a_id, 0) - self.defence.get(h_id, 0))

        max_goals = 10
        prob_matrix = np.outer(
            poisson.pmf(range(max_goals), lam),
            poisson.pmf(range(max_goals), mu)
        )
        prob_matrix = self._dc_correction(prob_matrix, lam, mu, self.rho)

        prob_home = float(np.sum(np.tril(prob_matrix, -1)))
        prob_draw = float(np.sum(np.diag(prob_matrix)))
        prob_away = float(np.sum(np.triu(prob_matrix, 1)))

        # Total goals distribution (index = total goals in match)
        tg = np.zeros(max_goals * 2 - 1)
        for i in range(max_goals):
            for j in range(max_goals):
                tg[i + j] += prob_matrix[i, j]

        goal_probs = {
            "u1.5": round(float(tg[:2].sum()), 4),
            "o1.5": round(float(tg[2:].sum()), 4),
            "u2.5": round(float(tg[:3].sum()), 4),
            "o2.5": round(float(tg[3:].sum()), 4),
            "u3.5": round(float(tg[:4].sum()), 4),
            "o3.5": round(float(tg[4:].sum()), 4),
            "u4.5": round(float(tg[:5].sum()), 4),
            "o4.5": round(float(tg[5:].sum()), 4),
            "1-3":  round(float(tg[1:4].sum()), 4),
            "2-4":  round(float(tg[2:5].sum()), 4),
        }

        return Prediction(
            fixture_id=fixture.id,
            prob_home=prob_home,
            prob_draw=prob_draw,
            prob_away=prob_away,
            model_name=self.name,
            expected_goals_home=float(lam),
            expected_goals_away=float(mu),
            goal_probs=goal_probs,
        )

    def _dc_correction(self, matrix, lam, mu, rho: float):
        for i in range(2):
            for j in range(2):
                matrix[i, j] *= self._tau(i, j, lam, mu, rho)
        return matrix / matrix.sum()
