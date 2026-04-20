"""
Standalone predict script for GitHub Actions.
Trains ensemble DC model for each league, fetches upcoming fixtures,
and upserts pre-computed probabilities into fixture_predictions table.
"""
import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import numpy as np
from scipy.stats import poisson

from api.client import APIClient
from config.settings import settings
from data.fetcher import FootballFetcher
from data.models import Prediction
from db.models import Base, FixturePrediction, ResolvedFixturePrediction, TrackedPrediction
from db.session import SessionLocal, engine
from features.engineer import FeatureEngineer
from models.calibrator import CornersCalibrator, GoalCalibrator, ProbabilityCalibrator
from models.corners import (
    CornersPredictor,
    EnsembleCornersPredictor,
    corners_prediction_from_lam_mu,
    train_corners_ensemble,
)
from models.ensemble import CUP_LEAGUES, EnsembleDCPredictor
from models.injury import InjuryAdjuster
from models.poisson import DixonColesPredictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

MODELS_DIR = Path("models/saved")


def predict_from_lam_mu(fixture_id: int, lam: float, mu: float, rho: float) -> Prediction:
    """Recompute a Prediction from injury-adjusted λ/μ with DC correction.

    Mirrors EnsembleDCPredictor.predict() but skips model blending — takes
    final λ/μ directly. goal_probs uses long-form keys (over2_5, goals1_3, …).
    """
    max_g = 10
    prob_matrix = np.outer(
        poisson.pmf(range(max_g), lam),
        poisson.pmf(range(max_g), mu),
    )
    # DC correction for low-score results (0-0, 1-0, 0-1, 1-1)
    for i in range(2):
        for j in range(2):
            prob_matrix[i, j] *= DixonColesPredictor._tau(i, j, lam, mu, rho)
    prob_matrix /= prob_matrix.sum()

    prob_home = float(np.sum(np.tril(prob_matrix, -1)))
    prob_draw = float(np.sum(np.diag(prob_matrix)))
    prob_away = float(np.sum(np.triu(prob_matrix, 1)))

    tg = np.zeros(max_g * 2 - 1)
    for i in range(max_g):
        for j in range(max_g):
            tg[i + j] += prob_matrix[i, j]
    btts = float(sum(prob_matrix[i, j] for i in range(1, max_g) for j in range(1, max_g)))

    return Prediction(
        fixture_id=fixture_id,
        prob_home=prob_home,
        prob_draw=prob_draw,
        prob_away=prob_away,
        model_name="ensemble_dc+injuries",
        expected_goals_home=lam,
        expected_goals_away=mu,
        goal_probs={
            "over2_5":  round(float(tg[3:].sum()), 4),
            "under2_5": round(float(tg[:3].sum()), 4),
            "goals1_3": round(float(tg[1:4].sum()), 4),
            "goals2_4": round(float(tg[2:5].sum()), 4),
            "btts_yes": round(btts, 4),
            "btts_no":  round(1 - btts, 4),
        },
    )



def train_ensemble(
    completed,
    cfg,
    league_key: str = "",
    attack_prior: dict = None,
    defence_prior: dict = None,
) -> EnsembleDCPredictor | None:
    if len(completed) < 50:
        print(f"  Not enough data ({len(completed)} matches), skipping.")
        return None

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    using_prior = bool(attack_prior)
    if using_prior:
        covered = sum(1 for f in completed if f.home_team.id in attack_prior or f.away_team.id in attack_prior)
        print(f"  Using domestic prior — {len(attack_prior)} teams in prior, {covered}/{len(completed)*2} appearances covered")

    # dc_all
    dc_all = DixonColesPredictor()
    dc_all.train(completed, attack_prior=attack_prior, defence_prior=defence_prior)
    dc_all.save(MODELS_DIR / f"dc_all_{cfg.name.lower().replace(' ', '_')}.joblib")
    print(f"  dc_all trained on {len(completed)} matches")

    # dc_season
    season_fixtures = [f for f in completed if f.season == cfg.season]
    if len(season_fixtures) >= 30:
        dc_season = DixonColesPredictor()
        dc_season.train(season_fixtures, attack_prior=attack_prior, defence_prior=defence_prior)
        dc_season.save(MODELS_DIR / f"dc_season_{cfg.name.lower().replace(' ', '_')}.joblib")
        print(f"  dc_season trained on {len(season_fixtures)} matches")
    else:
        dc_season = dc_all
        print(f"  dc_season fallback to dc_all ({len(season_fixtures)} season matches)")

    # dc_recent (last 60 days)
    cutoff = max(f.date for f in completed) - timedelta(days=60)
    recent_fixtures = [f for f in completed if f.date >= cutoff]
    if len(recent_fixtures) >= 30:
        dc_recent = DixonColesPredictor()
        dc_recent.train(recent_fixtures, attack_prior=attack_prior, defence_prior=defence_prior)
        print(f"  dc_recent trained on {len(recent_fixtures)} matches (last 60 days)")
    else:
        dc_recent = None
        print(f"  dc_recent skipped ({len(recent_fixtures)} recent matches)")

    return EnsembleDCPredictor(dc_all, dc_season, dc_recent, league_key=league_key)


def archive_resolved_fixtures(db, fetcher: "FootballFetcher", league_key: str, completed) -> int:
    """Before wiping fixture_predictions for a league, archive rows whose match is now FT.

    Features are computed from `completed` history — FeatureEngineer correctly returns
    pre-match stats for any fixture in the history (uses position-based past filtering).
    Returns number of newly archived rows.
    """
    existing = db.query(FixturePrediction).filter(
        FixturePrediction.league == league_key
    ).all()
    if not existing:
        return 0

    now = datetime.now(timezone.utc)
    to_archive = []

    # Build lookup from already-fetched completed history — avoids one API call per fixture
    fixture_by_id = {f.id: f for f in completed}

    # Batch-check already-archived IDs to avoid N+1 DB queries
    candidate_ids = [row.fixture_id for row in existing if row.match_date <= now]
    already_archived = {
        r.fixture_id
        for r in db.query(ResolvedFixturePrediction.fixture_id).filter(
            ResolvedFixturePrediction.fixture_id.in_(candidate_ids)
        ).all()
    }

    for row in existing:
        if row.match_date > now:
            continue  # Not played yet
        if row.fixture_id in already_archived:
            continue
        fx = fixture_by_id.get(row.fixture_id)
        if fx is None or fx.result is None:
            continue  # Not in FT history or result missing
        to_archive.append((row, fx.result.home_goals, fx.result.away_goals))

    if not to_archive:
        return 0

    # Targeted enrichment: fetch stats only for recent fixtures of teams being archived.
    # Avoids fetching stats for all 1000+ historical matches — stats are optional in features.
    teams_needed: set = set()
    for row, _, _ in to_archive:
        fx_chk = fixture_by_id.get(row.fixture_id)
        if fx_chk:
            teams_needed.add(fx_chk.home_team.id)
            teams_needed.add(fx_chk.away_team.id)
    if teams_needed:
        team_recent_map: dict = {}
        for f in sorted(completed, key=lambda x: x.date):
            for tid in (f.home_team.id, f.away_team.id):
                if tid in teams_needed:
                    team_recent_map.setdefault(tid, []).append(f)
        subset_ids: set = set()
        for tid_fixtures in team_recent_map.values():
            for f in tid_fixtures[-8:]:
                subset_ids.add(f.id)
        subset = [f for f in completed if f.id in subset_ids]
        fetcher.enrich_with_statistics(subset)

    # Compute pre-match features for all fixtures being archived
    fe = FeatureEngineer()
    fe.precompute(completed)

    for row, hs, as_ in to_archive:
        fx = fixture_by_id.get(row.fixture_id)
        feats = fe.build_features(fx, completed) if fx else {}

        outcome = "H" if hs > as_ else ("D" if hs == as_ else "A")
        predicted = max(
            [("H", row.prob_home), ("D", row.prob_draw), ("A", row.prob_away)],
            key=lambda x: x[1],
        )[0]

        actual_corners_home = fx.home_stats.corners if fx.home_stats else None
        actual_corners_away = fx.away_stats.corners if fx.away_stats else None
        db.add(ResolvedFixturePrediction(
            fixture_id=row.fixture_id,
            league=league_key,
            home_team=row.home_team,
            away_team=row.away_team,
            home_logo=row.home_logo,
            away_logo=row.away_logo,
            match_date=row.match_date,
            prob_home=row.prob_home,
            prob_draw=row.prob_draw,
            prob_away=row.prob_away,
            over2_5=row.over2_5,
            under2_5=row.under2_5,
            goals1_3=row.goals1_3,
            goals2_4=row.goals2_4,
            btts_yes=row.btts_yes,
            btts_no=row.btts_no,
            home_score=hs,
            away_score=as_,
            actual_outcome=outcome,
            predicted_outcome=predicted,
            correct=(outcome == predicted),
            features_json=json.dumps(feats) if feats else None,
            computed_at=row.computed_at,
            expected_corners_home=getattr(row, "expected_corners_home", None),
            expected_corners_away=getattr(row, "expected_corners_away", None),
            corners_over8_5=getattr(row, "corners_over8_5", None),
            corners_under8_5=getattr(row, "corners_under8_5", None),
            corners_over9_5=getattr(row, "corners_over9_5", None),
            corners_under9_5=getattr(row, "corners_under9_5", None),
            corners_over10_5=getattr(row, "corners_over10_5", None),
            corners_under10_5=getattr(row, "corners_under10_5", None),
            corners_over11_5=getattr(row, "corners_over11_5", None),
            corners_under11_5=getattr(row, "corners_under11_5", None),
            actual_corners_home=actual_corners_home,
            actual_corners_away=actual_corners_away,
        ))
        print(f"  Archived: {row.home_team} vs {row.away_team}  {hs}-{as_} ({outcome})")

    db.commit()
    return len(to_archive)


_PROB_FIELD = {
    "H": "prob_home", "D": "prob_draw", "A": "prob_away",
    "Under2.5": "under2_5", "Over2.5": "over2_5",
    "Goals1-3": "goals1_3", "Goals2-4": "goals2_4",
    "BTTS_Yes": "btts_yes", "BTTS_No": "btts_no",
    "Corners_Over8.5":   "corners_over8_5",
    "Corners_Under8.5":  "corners_under8_5",
    "Corners_Over9.5":   "corners_over9_5",
    "Corners_Under9.5":  "corners_under9_5",
    "Corners_Over10.5":  "corners_over10_5",
    "Corners_Under10.5": "corners_under10_5",
    "Corners_Over11.5":  "corners_over11_5",
    "Corners_Under11.5": "corners_under11_5",
}


def update_tracked_probs(db, saved: dict) -> int:
    """Update model_prob for unresolved tracked_predictions whose fixture was just recalculated.

    saved: {fixture_id: FixturePrediction} — rows committed in current run.
    Returns number of updated rows.
    """
    if not saved:
        return 0
    pending = db.query(TrackedPrediction).filter(
        TrackedPrediction.correct.is_(None),
        TrackedPrediction.fixture_id.in_(saved.keys()),
    ).all()
    updated = 0
    for tp in pending:
        fp = saved.get(tp.fixture_id)
        field = _PROB_FIELD.get(tp.prediction_type)
        if fp and field:
            tp.model_prob = getattr(fp, field, None)
            updated += 1
    if updated:
        db.commit()
    return updated


def build_calibrators(
    completed, min_train: int = 100, retrain_every: int = 50
) -> tuple[ProbabilityCalibrator | None, GoalCalibrator | None]:
    """Walk-forward calibration using dc_all only (single pass for both calibrators).

    Retrains dc_all every `retrain_every` matches, predicts the next fixture.
    Collects out-of-sample samples for:
      - ProbabilityCalibrator: (prob_H, prob_D, prob_A, actual_outcome)
      - GoalCalibrator:        (λ+μ, actual_total_goals)

    Returns (None, None) if < 80 samples collected.
    """
    if len(completed) < min_train + 30:
        return None, None

    samples_ph, samples_pd, samples_pa, actuals = [], [], [], []
    predicted_totals, actual_totals = [], []
    model: DixonColesPredictor | None = None

    for i in range(min_train, len(completed)):
        if model is None or (i - min_train) % retrain_every == 0:
            dc = DixonColesPredictor()
            dc.train(completed[:i])
            model = dc

        fx = completed[i]
        try:
            pred = model.predict(fx)
        except Exception:
            continue

        lam = pred.expected_goals_home
        mu = pred.expected_goals_away
        if any(np.isnan(v) for v in (pred.prob_home, pred.prob_draw, pred.prob_away, lam, mu)):
            continue

        samples_ph.append(pred.prob_home)
        samples_pd.append(pred.prob_draw)
        samples_pa.append(pred.prob_away)
        actuals.append(fx.result.outcome)

        predicted_totals.append(lam + mu)
        actual_totals.append(fx.result.home_goals + fx.result.away_goals)

    if len(actuals) < 80:
        return None, None

    prob_cal = ProbabilityCalibrator()
    prob_cal.fit(samples_ph, samples_pd, samples_pa, actuals)

    goal_cal = GoalCalibrator()
    goal_cal.fit(predicted_totals, actual_totals)

    return prob_cal, goal_cal


def build_corners_calibrator(
    with_corners: list, min_train: int = 80, retrain_every: int = 30
) -> CornersCalibrator | None:
    """Walk-forward calibration for corners λ+μ → actual total corners.

    Returns None if < 40 out-of-sample samples collected.
    """
    if len(with_corners) < min_train + 20:
        return None

    sorted_c = sorted(with_corners, key=lambda f: f.date)
    predicted_totals, actual_totals = [], []
    model: CornersPredictor | None = None

    for i in range(min_train, len(sorted_c)):
        if model is None or (i - min_train) % retrain_every == 0:
            c = CornersPredictor()
            c.train(sorted_c[:i])
            model = c

        if not model._fitted:
            continue
        fx = sorted_c[i]
        lam, mu = model._lam_mu(fx.home_team.id, fx.away_team.id)
        if lam <= 0 or mu <= 0:
            continue

        predicted_totals.append(lam + mu)
        actual_totals.append(fx.home_stats.corners + fx.away_stats.corners)

    if len(predicted_totals) < 40:
        return None

    cal = CornersCalibrator()
    cal.fit(predicted_totals, actual_totals)
    return cal


def main():
    Base.metadata.create_all(engine)
    client = APIClient(settings)
    fetcher = FootballFetcher(client, settings)
    db = SessionLocal()

    injury_adjuster = InjuryAdjuster()

    try:
        # Phase 1: train domestic leagues, collect attack/defence parameters as cup priors
        domestic_attack: dict = {}
        domestic_defence: dict = {}
        domestic_results: dict = {}      # league_key -> (completed, model)
        corners_models: dict = {}        # league_key -> EnsembleCornersPredictor | None
        corners_calibrators: dict = {}   # league_key -> CornersCalibrator | None

        for league_key, cfg in settings.leagues.items():
            if league_key in CUP_LEAGUES:
                continue
            print(f"\n[{cfg.name}]")
            history = fetcher.get_fixtures(cfg, status="FT")
            completed = [f for f in history if f.result is not None]
            model = train_ensemble(completed, cfg, league_key=league_key)
            if model is None:
                continue
            domestic_results[league_key] = (completed, model)
            # Merge team parameters into global prior dicts (team IDs are unique across API-Football)
            domestic_attack.update(model.dc_all.attack)
            domestic_defence.update(model.dc_all.defence)

            # Corners: enrich history with statistics, then train corners ensemble
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            league_slug = cfg.name.lower().replace(" ", "_")
            fetcher.enrich_full_history(completed)
            cm = train_corners_ensemble(completed, cfg.season, MODELS_DIR, league_slug)
            corners_models[league_key] = cm
            if cm:
                print(f"  Corners ensemble ready for {cfg.name}")

            # Corners calibrator: walk-forward λ+μ → actual corners scaling
            with_corners = [
                f for f in completed
                if f.home_stats and f.home_stats.corners is not None
                and f.away_stats and f.away_stats.corners is not None
            ]
            c_cal_path  = MODELS_DIR / f"calibrator_{league_key}_corners.joblib"
            c_cal_meta  = MODELS_DIR / f"calibrator_{league_key}_corners_meta.json"
            corners_cal = None
            if c_cal_path.exists() and c_cal_meta.exists():
                try:
                    meta = json.loads(c_cal_meta.read_text())
                    if len(with_corners) - meta.get("n_corners", 0) < 50:
                        corners_cal = joblib.load(c_cal_path)
                        print(f"  Corners calibrator: loaded from cache (+{len(with_corners) - meta.get('n_corners', 0)} new)")
                except Exception:
                    pass
            if corners_cal is None:
                corners_cal = build_corners_calibrator(with_corners)
                if corners_cal:
                    MODELS_DIR.mkdir(parents=True, exist_ok=True)
                    joblib.dump(corners_cal, c_cal_path)
                    c_cal_meta.write_text(json.dumps({"n_corners": len(with_corners)}))
                    print(f"  Corners calibrator: {corners_cal.n_samples} samples | bias {corners_cal.mean_bias:+.3f} corners")
                else:
                    print(f"  Corners calibrator: not enough data ({len(with_corners)} enriched fixtures)")
            corners_calibrators[league_key] = corners_cal

        print(f"\n[Prior] Collected {len(domestic_attack)} teams from domestic leagues")

        # Phase 2: cup leagues — use domestic priors as MLE starting point
        cup_results: dict = {}
        for league_key, cfg in settings.leagues.items():
            if league_key not in CUP_LEAGUES:
                continue
            print(f"\n[{cfg.name}]")
            history = fetcher.get_fixtures(cfg, status="FT")
            completed = [f for f in history if f.result is not None]
            model = train_ensemble(
                completed, cfg, league_key=league_key,
                attack_prior=domestic_attack,
                defence_prior=domestic_defence,
            )
            if model is None:
                continue
            cup_results[league_key] = (completed, model)

        # Build per-league calibrators — load from disk cache if < 50 new matches since last fit
        print("\n[Calibration] Fitting calibrators (H/D/A + goals)...")
        calibrators: dict[str, ProbabilityCalibrator | None] = {}
        goal_calibrators: dict[str, GoalCalibrator | None] = {}

        for league_key, (completed, _) in {**domestic_results, **cup_results}.items():
            hda_path  = MODELS_DIR / f"calibrator_{league_key}_hda.joblib"
            goal_path = MODELS_DIR / f"calibrator_{league_key}_goals.joblib"
            meta_path = MODELS_DIR / f"calibrator_{league_key}_meta.json"
            n_completed = len(completed)
            prob_cal = goal_cal = None

            if hda_path.exists() and goal_path.exists() and meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text())
                    new_matches = n_completed - meta.get("n_completed", 0)
                    if new_matches < 50:
                        prob_cal = joblib.load(hda_path)
                        goal_cal = joblib.load(goal_path)
                        print(f"  {settings.leagues[league_key].name}: loaded from cache (+{new_matches} new matches)")
                except Exception:
                    pass  # corrupted cache — fall through to retrain

            if prob_cal is None:
                prob_cal, goal_cal = build_calibrators(completed)
                if prob_cal:
                    MODELS_DIR.mkdir(parents=True, exist_ok=True)
                    joblib.dump(prob_cal, hda_path)
                    joblib.dump(goal_cal, goal_path)
                    meta_path.write_text(json.dumps({"n_completed": n_completed}))
                    print(
                        f"  {settings.leagues[league_key].name}: {prob_cal.n_samples} samples"
                        f" | ECE {prob_cal.ece_before:.4f}→{prob_cal.ece_after:.4f}"
                        f" | goal bias {goal_cal.mean_bias:+.3f} xG"
                    )
                else:
                    print(f"  {settings.leagues[league_key].name}: not enough samples, skipping")

            calibrators[league_key] = prob_cal
            goal_calibrators[league_key] = goal_cal

        # Save predictions for all leagues
        all_results = {**domestic_results, **cup_results}
        for league_key, (completed, model) in all_results.items():
            cfg = settings.leagues[league_key]
            # Archive completed fixtures before wiping the table (must happen before upcoming check
            # so fixtures are preserved even when no upcoming matches exist for the league)
            archived = archive_resolved_fixtures(db, fetcher, league_key, completed)
            if archived:
                print(f"  → Archived {archived} resolved fixture(s)")

            # Capture current predictions as prev_ before overwriting (change tracking)
            prev_vals: dict = {}
            for r in db.query(FixturePrediction).filter(FixturePrediction.league == league_key).all():
                prev_vals[r.fixture_id] = (r.prob_home, r.prob_draw, r.prob_away, r.computed_at)

            # Delete stale rows for this league
            db.query(FixturePrediction).filter(FixturePrediction.league == league_key).delete()

            upcoming = fetcher.get_upcoming_fixtures(cfg, next_n=10)
            if not upcoming:
                print(f"\n[{cfg.name}] No upcoming fixtures found.")
                db.commit()
                continue

            print(f"\n[{cfg.name}] Computing predictions for {len(upcoming)} upcoming fixtures...")

            prob_cal = calibrators.get(league_key)
            goal_cal = goal_calibrators.get(league_key)

            corners_model = corners_models.get(league_key)       # None for cups
            corners_cal   = corners_calibrators.get(league_key)  # None for cups
            saved_predictions: dict = {}  # fixture_id -> FixturePrediction, for tracked prob update
            for fx in upcoming:
                pred = model.predict(fx)

                # Injury adjustment: fetch injuries, adjust λ/μ, recompute probabilities
                home_inj, away_inj, home_goals, away_goals, home_goals_against, away_goals_against = (
                    fetcher.get_fixture_injuries(fx, cfg.id, cfg.season)
                )
                if home_inj or away_inj:
                    lam_orig, mu_orig = pred.expected_goals_home, pred.expected_goals_away
                    lam_adj, mu_adj = injury_adjuster.adjust(
                        lam_orig, mu_orig, home_inj, away_inj, home_goals, away_goals,
                        home_goals_against, away_goals_against
                    )
                    pred = predict_from_lam_mu(fx.id, lam_adj, mu_adj, model.dc_all.rho)
                    inj_names = (
                        [f"{i.player_name}({i.position[0]})" for i in home_inj] +
                        [f"{i.player_name}({i.position[0]})*" for i in away_inj]
                    )
                    print(f"    Injuries: {', '.join(inj_names)} | λ {lam_orig:.2f}→{lam_adj:.2f} μ {mu_orig:.2f}→{mu_adj:.2f}")

                # Goal calibration: scale λ/μ proportionally to match empirical total goals
                # Corrects Poisson independence bias (model overcounts goals vs reality)
                lam_final = pred.expected_goals_home
                mu_final = pred.expected_goals_away
                if goal_cal is not None:
                    lam_final, mu_final = goal_cal.transform(lam_final, mu_final)
                    gp = predict_from_lam_mu(fx.id, lam_final, mu_final, model.dc_all.rho).goal_probs
                else:
                    gp = pred.goal_probs

                # H/D/A isotonic calibration (independent of goal calibration)
                if prob_cal is not None:
                    ph, pd_val, pa = prob_cal.transform(pred.prob_home, pred.prob_draw, pred.prob_away)
                else:
                    ph, pd_val, pa = pred.prob_home, pred.prob_draw, pred.prob_away

                cp = corners_model.predict_corners(fx) if corners_model else None
                if cp and corners_cal:
                    lam_c, mu_c = corners_cal.transform(cp.lambda_home, cp.mu_away)
                    cp = corners_prediction_from_lam_mu(cp.fixture_id, lam_c, mu_c)

                prev = prev_vals.get(fx.id)
                row = FixturePrediction(
                    fixture_id=fx.id,
                    league=league_key,
                    home_team=fx.home_team.name,
                    away_team=fx.away_team.name,
                    home_logo=fx.home_team.logo or None,
                    away_logo=fx.away_team.logo or None,
                    match_date=fx.date,
                    prob_home=round(ph, 4),
                    prob_draw=round(pd_val, 4),
                    prob_away=round(pa, 4),
                    over2_5=gp["over2_5"],
                    under2_5=gp["under2_5"],
                    goals1_3=gp["goals1_3"],
                    goals2_4=gp["goals2_4"],
                    btts_yes=gp["btts_yes"],
                    btts_no=gp["btts_no"],
                    expected_goals_home=round(lam_final, 3),
                    expected_goals_away=round(mu_final, 3),
                    prev_prob_home=prev[0] if prev else None,
                    prev_prob_draw=prev[1] if prev else None,
                    prev_prob_away=prev[2] if prev else None,
                    prev_computed_at=prev[3] if prev else None,
                    expected_corners_home=cp.lambda_home if cp else None,
                    expected_corners_away=cp.mu_away if cp else None,
                    corners_over8_5=cp.over8_5 if cp else None,
                    corners_under8_5=cp.under8_5 if cp else None,
                    corners_over9_5=cp.over9_5 if cp else None,
                    corners_under9_5=cp.under9_5 if cp else None,
                    corners_over10_5=cp.over10_5 if cp else None,
                    corners_under10_5=cp.under10_5 if cp else None,
                    corners_over11_5=cp.over11_5 if cp else None,
                    corners_under11_5=cp.under11_5 if cp else None,
                )
                db.add(row)
                saved_predictions[fx.id] = row
                corners_str = f" | C:{cp.lambda_home:.1f}+{cp.mu_away:.1f}" if cp else ""
                print(f"  {fx.home_team.name} vs {fx.away_team.name} | H:{ph:.0%} D:{pd_val:.0%} A:{pa:.0%} | λ={lam_final:.2f} μ={mu_final:.2f}{corners_str}")

            db.commit()
            print(f"  Saved {len(upcoming)} predictions.")

            n_updated = update_tracked_probs(db, saved_predictions)
            if n_updated:
                print(f"  Updated model_prob for {n_updated} tracked prediction(s).")

    finally:
        db.close()


if __name__ == "__main__":
    main()
