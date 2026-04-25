"""
Calibrators for football match predictions.

ProbabilityCalibrator — isotonic regression on H/D/A outcome probabilities.
GoalCalibrator        — isotonic regression on λ+μ → actual total goals.
                        Corrects the Poisson independence bias: real football
                        has negative goal correlation (teams defend more when
                        trailing) → actual totals are lower than λ+μ predicts.
GoalMarketCalibrator  — Platt scaling (logistic regression in logit space) per market.
                        Smooth monotone mapping — avoids isotonic step-function collapse
                        when input range is narrow (post-GoalCalibrator scaled probs).
                        Each market fitted independently; complementary pairs renormalized.

All are fit on out-of-sample (walk-forward) ensemble predictions to avoid overfitting.
"""
import numpy as np
from scipy.special import expit, logit as _logit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

# IsotonicRegression is still used by GoalCalibrator and CornersCalibrator

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

    def fit(self, predicted_totals, actual_totals, sample_weight=None) -> None:
        x = np.array(predicted_totals, dtype=float)
        y = np.array(actual_totals, dtype=float)
        self._ir.fit(x, y, sample_weight=sample_weight)
        self._fitted = True
        self.n_samples = len(y)
        w = np.array(sample_weight) if sample_weight is not None else None
        self.mean_bias = round(float(np.average(x, weights=w) - np.average(y, weights=w)), 4)

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
    """Platt scaling (logistic regression in logit space) per outcome.

    Replaces isotonic regression to avoid step-function collapse: isotonic maps
    a range of raw probabilities to a single constant output (plateau), causing
    different fixtures to receive identical calibrated values. Logistic regression
    gives a smooth, strictly monotone mapping that extrapolates continuously.
    """

    def __init__(self):
        self._lr_h = LogisticRegression(C=1e9, solver="lbfgs", max_iter=300)
        self._lr_d = LogisticRegression(C=1e9, solver="lbfgs", max_iter=300)
        self._lr_a = LogisticRegression(C=1e9, solver="lbfgs", max_iter=300)
        self._fitted = False
        self.n_samples = 0
        self.ece_before: float = 0.0
        self.ece_after: float = 0.0

    def _fit_one(self, lr: LogisticRegression, probs: np.ndarray, labels: np.ndarray, w) -> None:
        x = _logit(np.clip(probs, 1e-6, 1 - 1e-6)).reshape(-1, 1)
        lr.fit(x, labels, sample_weight=w)

    def _predict_one(self, lr: LogisticRegression, probs: np.ndarray) -> np.ndarray:
        x = _logit(np.clip(probs, 1e-6, 1 - 1e-6)).reshape(-1, 1)
        return lr.predict_proba(x)[:, 1]

    def fit(self, probs_h, probs_d, probs_a, actuals, sample_weight=None) -> None:
        ph = np.array(probs_h, dtype=float)
        pd = np.array(probs_d, dtype=float)
        pa = np.array(probs_a, dtype=float)
        y  = np.array(actuals)
        w  = np.array(sample_weight) if sample_weight is not None else None

        self._fit_one(self._lr_h, ph, (y == "H").astype(int), w)
        self._fit_one(self._lr_d, pd, (y == "D").astype(int), w)
        self._fit_one(self._lr_a, pa, (y == "A").astype(int), w)
        self._fitted = True
        self.n_samples = len(y)

        self.ece_before = _ece(ph, pd, pa, y)
        cal_h = self._predict_one(self._lr_h, ph)
        cal_d = self._predict_one(self._lr_d, pd)
        cal_a = self._predict_one(self._lr_a, pa)
        totals = cal_h + cal_d + cal_a
        safe   = totals > 0
        cal_h[safe] /= totals[safe]
        cal_d[safe] /= totals[safe]
        cal_a[safe] /= totals[safe]
        self.ece_after = _ece(cal_h, cal_d, cal_a, y)

    def transform(self, prob_h: float, prob_d: float, prob_a: float) -> tuple[float, float, float]:
        """Apply Platt scaling + renormalize. Falls back to raw probs if not fitted."""
        if not self._fitted:
            return prob_h, prob_d, prob_a

        c_h = float(self._predict_one(self._lr_h, np.array([prob_h]))[0])
        c_d = float(self._predict_one(self._lr_d, np.array([prob_d]))[0])
        c_a = float(self._predict_one(self._lr_a, np.array([prob_a]))[0])

        total = c_h + c_d + c_a
        if total <= 0:
            return prob_h, prob_d, prob_a
        return c_h / total, c_d / total, c_a / total


class GoalMarketCalibrator:
    """Per-market Platt scaling for goal market probabilities.

    Fits logistic regression in logit space per market:
        logit(cal) = a * logit(pred) + b
    Smooth monotone mapping — avoids isotonic step-function collapse when
    input probabilities cluster in a narrow range (e.g. post-GoalCalibrator).
    Extrapolates continuously beyond the training range.
    """

    def __init__(self):
        self._lrs: dict = {}  # market -> LogisticRegression
        self._fitted_markets: set = set()
        self.n_samples: int = 0
        self.biases: dict = {}  # market -> weighted mean_bias before calibration

    def fit(self, preds: dict, actuals: dict, sample_weight=None) -> None:
        """
        preds:   {market: [predicted_prob, ...]}
        actuals: {market: [0.0 / 1.0, ...]}
        """
        n = 0
        w_arr = np.array(sample_weight) if sample_weight is not None else None
        for m in GOAL_MARKETS:
            x = np.array(preds.get(m, []), dtype=float)
            y = np.array(actuals.get(m, []), dtype=float)
            mask = ~(np.isnan(x) | np.isnan(y))
            if mask.sum() < 20:
                continue
            w_m = w_arr[mask] if w_arr is not None else None
            x_logit = _logit(np.clip(x[mask], 1e-6, 1 - 1e-6)).reshape(-1, 1)
            lr = LogisticRegression(C=1e9, solver="lbfgs", max_iter=300)
            lr.fit(x_logit, y[mask].astype(int), sample_weight=w_m)
            self._lrs[m] = lr
            self._fitted_markets.add(m)
            self.biases[m] = round(float(
                np.average(x[mask], weights=w_m) - np.average(y[mask], weights=w_m)
            ), 4)
            n = max(n, int(mask.sum()))
        self.n_samples = n

    def transform(self, goal_probs: dict) -> dict:
        """Calibrate per-market via Platt scaling. Falls back to raw for unfitted markets."""
        if not self._fitted_markets:
            return goal_probs
        result = dict(goal_probs)
        for m in self._fitted_markets:
            if m in goal_probs:
                p = float(goal_probs[m])
                x_logit = np.array([[_logit(np.clip(p, 1e-6, 1 - 1e-6))]])
                cal = float(self._lrs[m].predict_proba(x_logit)[0, 1])
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
