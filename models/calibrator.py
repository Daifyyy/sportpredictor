"""
Calibrators for football match predictions.

ProbabilityCalibrator — isotonic regression on H/D/A outcome probabilities.
GoalCalibrator        — isotonic regression on λ+μ → actual total goals.
                        Corrects the Poisson independence bias: real football
                        has negative goal correlation (teams defend more when
                        trailing) → actual totals are lower than λ+μ predicts.
GoalMarketCalibrator  — per-market isotonic regression on goal market probs.
                        Each market (over2_5, goals1_3, btts_yes, …) has its
                        own independent bias — unlike GoalCalibrator which
                        applies a single λ+μ scaling to all markets at once.

All are fit on out-of-sample (walk-forward) ensemble predictions to avoid overfitting.
"""
import numpy as np
from sklearn.isotonic import IsotonicRegression

GOAL_MARKETS = ("over2_5", "under2_5", "goals1_3", "goals2_4", "btts_yes", "btts_no")


class GoalCalibrator:
    """Calibrates predicted total goals (λ+μ) to match empirical distribution.

    Preserves the λ/μ ratio (relative team strength) — only scales the total.
    transform(lam, mu) → (lam_cal, mu_cal) with lam_cal + mu_cal = calibrated total.
    """

    def __init__(self):
        self._ir = IsotonicRegression(out_of_bounds="clip")
        self._fitted = False
        self.n_samples = 0
        self.mean_bias: float = 0.0  # avg(predicted) - avg(actual), positive = model overcounts

    def fit(self, predicted_totals, actual_totals) -> None:
        x = np.array(predicted_totals, dtype=float)
        y = np.array(actual_totals, dtype=float)
        self._ir.fit(x, y)
        self._fitted = True
        self.n_samples = len(y)
        self.mean_bias = round(float(x.mean() - y.mean()), 4)

    def transform(self, lam: float, mu: float) -> tuple[float, float]:
        """Scale λ/μ proportionally so λ+μ matches calibrated expected total."""
        if not self._fitted:
            return lam, mu
        total = lam + mu
        if total <= 0:
            return lam, mu
        cal_total = float(self._ir.predict([total])[0])
        if cal_total <= 0:
            return lam, mu
        scale = cal_total / total
        return lam * scale, mu * scale


class ProbabilityCalibrator:

    def __init__(self):
        self._ir_h = IsotonicRegression(out_of_bounds="clip")
        self._ir_d = IsotonicRegression(out_of_bounds="clip")
        self._ir_a = IsotonicRegression(out_of_bounds="clip")
        self._fitted = False
        self.n_samples = 0
        self.ece_before: float = 0.0
        self.ece_after: float = 0.0

    def fit(self, probs_h, probs_d, probs_a, actuals) -> None:
        """
        probs_*  : array-like of model probabilities for each outcome
        actuals  : array-like of 'H' / 'D' / 'A' strings
        """
        ph = np.array(probs_h, dtype=float)
        pd = np.array(probs_d, dtype=float)
        pa = np.array(probs_a, dtype=float)
        y  = np.array(actuals)

        self._ir_h.fit(ph, (y == "H").astype(float))
        self._ir_d.fit(pd, (y == "D").astype(float))
        self._ir_a.fit(pa, (y == "A").astype(float))
        self._fitted = True
        self.n_samples = len(y)

        self.ece_before = _ece(ph, pd, pa, y)
        cal_h = self._ir_h.predict(ph)
        cal_d = self._ir_d.predict(pd)
        cal_a = self._ir_a.predict(pa)
        totals = cal_h + cal_d + cal_a
        safe   = totals > 0
        cal_h[safe] /= totals[safe]
        cal_d[safe] /= totals[safe]
        cal_a[safe] /= totals[safe]
        self.ece_after = _ece(cal_h, cal_d, cal_a, y)

    def transform(self, prob_h: float, prob_d: float, prob_a: float) -> tuple[float, float, float]:
        """Apply calibration + renormalize. Falls back to raw probs if not fitted."""
        if not self._fitted:
            return prob_h, prob_d, prob_a

        c_h = float(self._ir_h.predict([prob_h])[0])
        c_d = float(self._ir_d.predict([prob_d])[0])
        c_a = float(self._ir_a.predict([prob_a])[0])

        total = c_h + c_d + c_a
        if total <= 0:
            return prob_h, prob_d, prob_a
        return c_h / total, c_d / total, c_a / total


class GoalMarketCalibrator:
    """Per-market isotonic regression calibration for goal market probabilities.

    Unlike GoalCalibrator (which scales λ+μ globally), calibrates each of the
    six goal markets independently so per-market biases are corrected separately.
    Trained on out-of-sample ensemble predictions; applied before DB insert.
    """

    def __init__(self):
        self._irs: dict = {m: IsotonicRegression(out_of_bounds="clip") for m in GOAL_MARKETS}
        self._fitted_markets: set = set()
        self.n_samples: int = 0
        self.biases: dict = {}  # market -> mean_bias (positive = model overcounts)

    def fit(self, preds: dict, actuals: dict) -> None:
        """
        preds:   {market: [predicted_prob, ...]}
        actuals: {market: [0.0 / 1.0, ...]}
        """
        n = 0
        for m in GOAL_MARKETS:
            x = np.array(preds.get(m, []), dtype=float)
            y = np.array(actuals.get(m, []), dtype=float)
            mask = ~(np.isnan(x) | np.isnan(y))
            if mask.sum() < 20:
                continue
            self._irs[m].fit(x[mask], y[mask])
            self._fitted_markets.add(m)
            self.biases[m] = round(float(x[mask].mean() - y[mask].mean()), 4)
            n = max(n, int(mask.sum()))
        self.n_samples = n

    def transform(self, goal_probs: dict) -> dict:
        """Calibrate per-market. Falls back to raw prob for unfitted markets."""
        if not self._fitted_markets:
            return goal_probs
        result = dict(goal_probs)
        for m in self._fitted_markets:
            if m in goal_probs:
                cal = float(self._irs[m].predict([goal_probs[m]])[0])
                result[m] = round(max(0.0, min(1.0, cal)), 4)
        # Renormalize complementary pairs so they sum to 1
        for over, under in (("over2_5", "under2_5"), ("btts_yes", "btts_no")):
            if over in result and under in result:
                total = result[over] + result[under]
                if total > 0:
                    result[over] = round(result[over] / total, 4)
                    result[under] = round(1.0 - result[over], 4)
        return result


class CornersCalibrator(GoalCalibrator):
    """Calibrates predicted total corners (λ+μ) to match empirical distribution.

    Identical logic to GoalCalibrator — separate class for distinct disk storage.
    """


def _ece(probs_h, probs_d, probs_a, actuals, n_bins: int = 10) -> float:
    """Mean Expected Calibration Error across H/D/A."""
    edges = np.linspace(0, 1, n_bins + 1)
    n = len(actuals)
    ece_sum = 0.0

    for probs, outcome in [(probs_h, "H"), (probs_d, "D"), (probs_a, "A")]:
        labels = (np.array(actuals) == outcome).astype(float)
        ece = 0.0
        for i in range(n_bins):
            mask = (probs >= edges[i]) & (probs < edges[i + 1])
            if not mask.any():
                continue
            ece += (mask.sum() / n) * abs(probs[mask].mean() - labels[mask].mean())
        ece_sum += ece

    return round(ece_sum / 3, 4)
