import logging
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel
from scipy.stats import poisson
from sqlalchemy.orm import Session

from api.client import APIClient
from config.settings import settings
from data.fetcher import FootballFetcher
from db.models import Base, TrackedPrediction
from db.session import SessionLocal, engine, get_db
from models.base import BasePredictor
from models.poisson import DixonColesPredictor

logger = logging.getLogger(__name__)

MODELS_DIR = Path("models/saved")
MODEL_CLASSES: dict[str, type[BasePredictor]] = {
    "dixon_coles": DixonColesPredictor,
}

_models: dict[str, dict[str, BasePredictor]] = {}
_fetcher: Optional[FootballFetcher] = None

_retrain_status: dict = {
    "running": False,
    "steps": [],
    "error": None,
}


def _retrain_log(msg: str, level: str = "info") -> None:
    _retrain_status["steps"].append({
        "time": datetime.now().strftime("%H:%M:%S"),
        "msg": msg,
        "level": level,
    })
    logger.info(f"[retrain] {msg}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _fetcher
    client = APIClient(settings)
    _fetcher = FootballFetcher(client, settings)

    Base.metadata.create_all(engine)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    any_model_loaded = False
    for league_key in settings.leagues:
        _models[league_key] = {}
        for model_name, cls in MODEL_CLASSES.items():
            path = MODELS_DIR / f"{model_name}_{league_key}.joblib"
            if path.exists():
                try:
                    m = cls.load(path)
                    _models[league_key][model_name] = m
                    any_model_loaded = True
                    logger.info(f"Loaded {model_name} for {league_key}")
                except Exception as e:
                    logger.warning(f"Failed to load {path}: {e}")

    if not any_model_loaded:
        logger.info("No trained models found — starting auto-retrain in background")
        threading.Thread(target=_run_retrain, daemon=True).start()

    yield


app = FastAPI(title="Football Predictor API", lifespan=lifespan)


def _require_league(league: str) -> None:
    if league not in settings.leagues:
        raise HTTPException(status_code=404, detail=f"Unknown league '{league}'. "
                            f"Valid keys: {list(settings.leagues)}")


def _require_models(league: str) -> dict[str, BasePredictor]:
    loaded = _models.get(league, {})
    if not loaded:
        raise HTTPException(status_code=503,
                            detail=f"No trained models for '{league}'. Run retrain first.")
    return loaded


def _compute_goal_probs(lam: float, mu: float) -> dict:
    max_goals = 10
    home_pmf = poisson.pmf(range(max_goals), lam)
    away_pmf = poisson.pmf(range(max_goals), mu)
    prob_matrix = np.outer(home_pmf, away_pmf)

    tg = np.zeros(max_goals * 2 - 1)
    for i in range(max_goals):
        for j in range(max_goals):
            tg[i + j] += prob_matrix[i, j]

    btts = float(sum(
        prob_matrix[i, j]
        for i in range(1, max_goals)
        for j in range(1, max_goals)
    ))

    return {
        "over2_5": round(float(tg[3:].sum()), 4),
        "under2_5": round(float(tg[:3].sum()), 4),
        "goals1_3": round(float(tg[1:4].sum()), 4),
        "goals2_4": round(float(tg[2:5].sum()), 4),
        "btts_yes": round(btts, 4),
        "btts_no": round(1 - btts, 4),
    }


def _run_retrain() -> None:
    _retrain_status["running"] = True
    _retrain_status["steps"] = []
    _retrain_status["error"] = None

    try:
        for league_key, league_cfg in settings.leagues.items():
            _retrain_log(f"[{league_cfg.name}] Fetching history...")
            history = _fetcher.get_fixtures(league_cfg, status="FT")
            completed = [f for f in history if f.result is not None]

            if len(completed) < 50:
                _retrain_log(f"[{league_cfg.name}] Skipped — only {len(completed)} matches", "warning")
                continue

            _models[league_key] = {}
            for model_name, cls in MODEL_CLASSES.items():
                _retrain_log(f"[{league_cfg.name}] Training {model_name}...")
                path = MODELS_DIR / f"{model_name}_{league_key}.joblib"
                m = cls()
                m.train(completed)
                m.save(path)
                _models[league_key][model_name] = m
                _retrain_log(f"[{league_cfg.name}] {model_name} done", "success")

        _retrain_log("Retrain complete", "success")
    except Exception as e:
        _retrain_status["error"] = str(e)
        _retrain_log(f"Error: {e}", "error")
    finally:
        _retrain_status["running"] = False


@app.post("/retrain", status_code=202)
def retrain():
    if _retrain_status["running"]:
        raise HTTPException(status_code=409, detail="Retraining already in progress.")
    threading.Thread(target=_run_retrain, daemon=True).start()
    return {"status": "started"}


@app.get("/retrain/status")
def retrain_status():
    return _retrain_status


@app.get("/health")
def health():
    loaded = {lg: list(m.keys()) for lg, m in _models.items() if m}
    return {
        "status": "ok",
        "models_loaded": len(loaded),
        "leagues_ready": list(loaded.keys()),
        "retraining": _retrain_status["running"],
    }


# --- Upcoming fixtures with model probabilities ---

class FixtureProbs(BaseModel):
    fixture_id: int
    date: str
    home_team: str
    away_team: str
    prob_home: float
    prob_draw: float
    prob_away: float
    over2_5: float
    under2_5: float
    goals1_3: float
    goals2_4: float
    btts_yes: float
    btts_no: float


@app.get("/upcoming/{league}", response_model=list[FixtureProbs])
def upcoming(league: str):
    _require_league(league)
    league_models = _require_models(league)
    league_cfg = settings.leagues[league]

    fixtures = _fetcher.get_upcoming_fixtures(league_cfg, next_n=10)
    results = []

    model = next(iter(league_models.values()))

    for fixture in fixtures:
        pred = model.predict(fixture)
        lam = pred.expected_goals_home or 1.3
        mu = pred.expected_goals_away or 1.0
        gp = _compute_goal_probs(lam, mu)

        results.append(FixtureProbs(
            fixture_id=fixture.id,
            date=fixture.date.isoformat(),
            home_team=fixture.home_team.name,
            away_team=fixture.away_team.name,
            prob_home=round(pred.prob_home, 4),
            prob_draw=round(pred.prob_draw, 4),
            prob_away=round(pred.prob_away, 4),
            over2_5=gp["over2_5"],
            under2_5=gp["under2_5"],
            goals1_3=gp["goals1_3"],
            goals2_4=gp["goals2_4"],
            btts_yes=gp["btts_yes"],
            btts_no=gp["btts_no"],
        ))

    return results


# --- Track a prediction ---

class TrackRequest(BaseModel):
    fixture_id: int
    league: str
    home_team: str
    away_team: str
    match_date: str
    prediction_type: str
    model_prob: Optional[float] = None


@app.post("/track")
def track(body: TrackRequest, db: Session = Depends(get_db)):
    existing = db.query(TrackedPrediction).filter(
        TrackedPrediction.fixture_id == body.fixture_id,
        TrackedPrediction.prediction_type == body.prediction_type,
    ).first()

    if existing:
        raise HTTPException(status_code=409, detail="Already tracked.")

    row = TrackedPrediction(
        fixture_id=body.fixture_id,
        league=body.league,
        home_team=body.home_team,
        away_team=body.away_team,
        match_date=datetime.fromisoformat(body.match_date),
        prediction_type=body.prediction_type,
        model_prob=body.model_prob,
    )
    db.add(row)
    db.commit()
    return {"status": "ok", "id": row.id}


# --- Resolve finished matches ---

def _compute_correct(prediction_type: str, home_score: int, away_score: int) -> bool:
    total = home_score + away_score
    if prediction_type == "H":
        return home_score > away_score
    if prediction_type == "D":
        return home_score == away_score
    if prediction_type == "A":
        return away_score > home_score
    if prediction_type == "Under2.5":
        return total <= 2
    if prediction_type == "Over2.5":
        return total >= 3
    if prediction_type == "Goals1-3":
        return 1 <= total <= 3
    if prediction_type == "Goals2-4":
        return 2 <= total <= 4
    if prediction_type == "BTTS_Yes":
        return home_score >= 1 and away_score >= 1
    if prediction_type == "BTTS_No":
        return home_score == 0 or away_score == 0
    return False


@app.post("/resolve")
def resolve(db: Session = Depends(get_db)):
    now = datetime.now(timezone.utc)

    pending = db.query(TrackedPrediction).filter(
        TrackedPrediction.correct.is_(None),
        TrackedPrediction.match_date < now,
    ).all()

    if not pending:
        return {"resolved": 0}

    fixture_ids = list({r.fixture_id for r in pending})
    results_by_id: dict[int, tuple[int, int]] = {}

    client = APIClient(settings)
    for fid in fixture_ids:
        data = client.get("fixtures", {"id": fid}, ttl=3600)
        if not data:
            continue
        response = data.get("response", [])
        if not response:
            continue
        raw = response[0]
        goals = raw.get("goals", {})
        status = raw["fixture"]["status"]["short"]
        if status == "FT" and goals.get("home") is not None and goals.get("away") is not None:
            results_by_id[fid] = (int(goals["home"]), int(goals["away"]))

    resolved = 0
    for row in pending:
        if row.fixture_id not in results_by_id:
            continue
        hs, as_ = results_by_id[row.fixture_id]
        row.home_score = hs
        row.away_score = as_
        row.actual_outcome = f"{hs}-{as_}"
        row.correct = _compute_correct(row.prediction_type, hs, as_)
        resolved += 1

    db.commit()
    return {"resolved": resolved}


# --- Read tracked predictions ---

@app.get("/tracked")
def tracked(
    league: Optional[str] = None,
    resolved: Optional[bool] = None,
    db: Session = Depends(get_db),
):
    q = db.query(TrackedPrediction)
    if league:
        q = q.filter(TrackedPrediction.league == league)
    if resolved is True:
        q = q.filter(TrackedPrediction.correct.isnot(None))
    elif resolved is False:
        q = q.filter(TrackedPrediction.correct.is_(None))
    rows = q.order_by(TrackedPrediction.match_date.desc()).all()

    return [
        {
            "id": r.id,
            "fixture_id": r.fixture_id,
            "league": r.league,
            "home_team": r.home_team,
            "away_team": r.away_team,
            "match_date": r.match_date.isoformat(),
            "prediction_type": r.prediction_type,
            "model_prob": r.model_prob,
            "actual_outcome": r.actual_outcome,
            "correct": r.correct,
            "home_score": r.home_score,
            "away_score": r.away_score,
        }
        for r in rows
    ]


# --- Stats ---

@app.get("/stats")
def stats(db: Session = Depends(get_db)):
    rows = db.query(TrackedPrediction).filter(
        TrackedPrediction.correct.isnot(None)
    ).all()

    by_type: dict[str, dict] = {}
    by_league: dict[str, dict] = {}

    for r in rows:
        for key, bucket in [(r.prediction_type, by_type), (r.league, by_league)]:
            if key not in bucket:
                bucket[key] = {"count": 0, "correct": 0}
            bucket[key]["count"] += 1
            if r.correct:
                bucket[key]["correct"] += 1

    def to_list(d):
        return [
            {
                "key": k,
                "count": v["count"],
                "correct": v["correct"],
                "success_pct": round(v["correct"] / v["count"] * 100, 1) if v["count"] else 0,
            }
            for k, v in d.items()
        ]

    total = len(rows)
    total_correct = sum(1 for r in rows if r.correct)
    return {
        "total": total,
        "total_correct": total_correct,
        "overall_pct": round(total_correct / total * 100, 1) if total else 0,
        "by_prediction_type": to_list(by_type),
        "by_league": to_list(by_league),
    }
