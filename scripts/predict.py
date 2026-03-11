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

import numpy as np
from scipy.stats import poisson

from api.client import APIClient
from config.settings import settings
from data.fetcher import FootballFetcher
from data.models import Prediction
from db.models import Base, FixturePrediction, ResolvedFixturePrediction, TrackedPrediction
from db.session import SessionLocal, engine
from features.engineer import FeatureEngineer
from models.calibrator import ProbabilityCalibrator
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
    final λ/μ directly. goal_probs keys match compute_goal_probs() for consistency.
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


def compute_goal_probs(lam: float, mu: float) -> dict:
    max_g = 10
    h_pmf = poisson.pmf(range(max_g), lam)
    a_pmf = poisson.pmf(range(max_g), mu)
    mat = np.outer(h_pmf, a_pmf)
    tg = np.zeros(max_g * 2 - 1)
    for i in range(max_g):
        for j in range(max_g):
            tg[i + j] += mat[i, j]
    btts = float(sum(mat[i, j] for i in range(1, max_g) for j in range(1, max_g)))
    return {
        "over2_5": round(float(tg[3:].sum()), 4),
        "under2_5": round(float(tg[:3].sum()), 4),
        "goals1_3": round(float(tg[1:4].sum()), 4),
        "goals2_4": round(float(tg[2:5].sum()), 4),
        "btts_yes": round(btts, 4),
        "btts_no": round(1 - btts, 4),
    }


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


def archive_resolved_fixtures(db, client, league_key: str, completed) -> int:
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

    for row in existing:
        if row.match_date > now:
            continue  # Not played yet
        already = db.query(ResolvedFixturePrediction).filter(
            ResolvedFixturePrediction.fixture_id == row.fixture_id
        ).first()
        if already:
            continue
        data = client.get("fixtures", {"id": row.fixture_id}, ttl=3600)
        if not data:
            continue
        resp = data.get("response", [])
        if not resp:
            continue
        raw = resp[0]
        status = raw["fixture"]["status"]["short"]
        goals = raw.get("goals", {})
        if status == "FT" and goals.get("home") is not None and goals.get("away") is not None:
            to_archive.append((row, int(goals["home"]), int(goals["away"])))

    if not to_archive:
        return 0

    # Compute pre-match features for all fixtures being archived
    fe = FeatureEngineer()
    fe.precompute(completed)
    fixture_by_id = {f.id: f for f in completed}

    for row, hs, as_ in to_archive:
        fx = fixture_by_id.get(row.fixture_id)
        feats = fe.build_features(fx, completed) if fx else {}

        outcome = "H" if hs > as_ else ("D" if hs == as_ else "A")
        predicted = max(
            [("H", row.prob_home), ("D", row.prob_draw), ("A", row.prob_away)],
            key=lambda x: x[1],
        )[0]

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
        ))
        print(f"  Archived: {row.home_team} vs {row.away_team}  {hs}-{as_} ({outcome})")

    db.commit()
    return len(to_archive)


_PROB_FIELD = {
    "H": "prob_home", "D": "prob_draw", "A": "prob_away",
    "Under2.5": "under2_5", "Over2.5": "over2_5",
    "Goals1-3": "goals1_3", "Goals2-4": "goals2_4",
    "BTTS_Yes": "btts_yes", "BTTS_No": "btts_no",
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


def build_calibrator(completed, league_key: str, min_train: int = 100, retrain_every: int = 50) -> ProbabilityCalibrator | None:
    """Walk-forward calibration data collection using dc_all only (fast).

    Retrains dc_all every `retrain_every` matches, predicts the next fixture.
    Collects out-of-sample (prob_H, prob_D, prob_A, actual) pairs, then fits
    an isotonic regression calibrator. Returns None if < 80 samples collected.
    """
    if len(completed) < min_train + 30:
        return None

    samples_ph, samples_pd, samples_pa, actuals = [], [], [], []
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

        samples_ph.append(pred.prob_home)
        samples_pd.append(pred.prob_draw)
        samples_pa.append(pred.prob_away)
        actuals.append(fx.result.outcome)

    if len(actuals) < 80:
        return None

    cal = ProbabilityCalibrator()
    cal.fit(samples_ph, samples_pd, samples_pa, actuals)
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
        domestic_results: dict = {}  # league_key -> (completed, model)

        for league_key, cfg in settings.leagues.items():
            if league_key in CUP_LEAGUES:
                continue
            print(f"\n[{cfg.name}]")
            history = fetcher.get_fixtures(cfg, status="FT")
            completed = [f for f in history if f.result is not None]
            fetcher.enrich_with_statistics(completed)
            model = train_ensemble(completed, cfg, league_key=league_key)
            if model is None:
                continue
            domestic_results[league_key] = (completed, model)
            # Merge team parameters into global prior dicts (team IDs are unique across API-Football)
            domestic_attack.update(model.dc_all.attack)
            domestic_defence.update(model.dc_all.defence)

        print(f"\n[Prior] Collected {len(domestic_attack)} teams from domestic leagues")

        # Phase 2: cup leagues — use domestic priors as MLE starting point
        cup_results: dict = {}
        for league_key, cfg in settings.leagues.items():
            if league_key not in CUP_LEAGUES:
                continue
            print(f"\n[{cfg.name}]")
            history = fetcher.get_fixtures(cfg, status="FT")
            completed = [f for f in history if f.result is not None]
            fetcher.enrich_with_statistics(completed)
            model = train_ensemble(
                completed, cfg, league_key=league_key,
                attack_prior=domestic_attack,
                defence_prior=domestic_defence,
            )
            if model is None:
                continue
            cup_results[league_key] = (completed, model)

        # Build per-league calibrators from walk-forward backtest on historical data
        print("\n[Calibration] Fitting isotonic regression calibrators...")
        calibrators: dict[str, ProbabilityCalibrator | None] = {}
        for league_key, (completed, _) in {**domestic_results, **cup_results}.items():
            cal = build_calibrator(completed, league_key)
            calibrators[league_key] = cal
            if cal:
                print(f"  {settings.leagues[league_key].name}: {cal.n_samples} samples | ECE {cal.ece_before:.4f} → {cal.ece_after:.4f}")
            else:
                print(f"  {settings.leagues[league_key].name}: not enough samples, skipping")

        # Save predictions for all leagues
        all_results = {**domestic_results, **cup_results}
        for league_key, (completed, model) in all_results.items():
            cfg = settings.leagues[league_key]
            upcoming = fetcher.get_upcoming_fixtures(cfg, next_n=10)
            if not upcoming:
                print(f"\n[{cfg.name}] No upcoming fixtures found.")
                continue

            print(f"\n[{cfg.name}] Computing predictions for {len(upcoming)} upcoming fixtures...")

            # Archive completed fixtures before wiping the table
            archived = archive_resolved_fixtures(db, client, league_key, completed)
            if archived:
                print(f"  → Archived {archived} resolved fixture(s)")

            # Delete stale rows for this league
            db.query(FixturePrediction).filter(FixturePrediction.league == league_key).delete()

            cal = calibrators.get(league_key)

            saved_predictions: dict = {}  # fixture_id -> FixturePrediction, for tracked prob update
            for fx in upcoming:
                pred = model.predict(fx)

                # Injury adjustment: fetch injuries, adjust λ/μ, recompute probabilities
                home_inj, away_inj, home_goals, away_goals = fetcher.get_fixture_injuries(
                    fx, cfg.id, cfg.season
                )
                if home_inj or away_inj:
                    lam_orig, mu_orig = pred.expected_goals_home, pred.expected_goals_away
                    lam_adj, mu_adj = injury_adjuster.adjust(
                        lam_orig, mu_orig, home_inj, away_inj, home_goals, away_goals
                    )
                    pred = predict_from_lam_mu(fx.id, lam_adj, mu_adj, model.dc_all.rho)
                    inj_names = (
                        [f"{i.player_name}({i.position[0]})" for i in home_inj] +
                        [f"{i.player_name}({i.position[0]})*" for i in away_inj]
                    )
                    print(f"    Injuries: {', '.join(inj_names)} | λ {lam_orig:.2f}→{lam_adj:.2f} μ {mu_orig:.2f}→{mu_adj:.2f}")

                # Apply isotonic calibration if available
                if cal is not None:
                    ph, pd_val, pa = cal.transform(pred.prob_home, pred.prob_draw, pred.prob_away)
                else:
                    ph, pd_val, pa = pred.prob_home, pred.prob_draw, pred.prob_away

                gp = compute_goal_probs(pred.expected_goals_home, pred.expected_goals_away)

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
                    expected_goals_home=round(pred.expected_goals_home, 3),
                    expected_goals_away=round(pred.expected_goals_away, 3),
                )
                db.add(row)
                saved_predictions[fx.id] = row
                print(f"  {fx.home_team.name} vs {fx.away_team.name} | H:{pred.prob_home:.0%} D:{pred.prob_draw:.0%} A:{pred.prob_away:.0%} | λ={pred.expected_goals_home:.2f} μ={pred.expected_goals_away:.2f}")

            db.commit()
            print(f"  Saved {len(upcoming)} predictions.")

            n_updated = update_tracked_probs(db, saved_predictions)
            if n_updated:
                print(f"  Updated model_prob for {n_updated} tracked prediction(s).")

    finally:
        db.close()


if __name__ == "__main__":
    main()
