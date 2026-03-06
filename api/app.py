import logging
import threading
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import func, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from api.client import APIClient
from backtesting.engine import BacktestEngine, compute_calibration
from betting.value import ValueBetDetector
from config.settings import settings
from data.fetcher import FootballFetcher
from db.models import BacktestRunRow, BankrollRow, Base, PredictionRow
from db.session import SessionLocal, engine, get_db
from models.base import BasePredictor
from models.calibrator import ProbabilityCalibrator
from models.poisson import DixonColesPredictor

logger = logging.getLogger(__name__)

MODELS_DIR = Path("models/saved")
MODEL_CLASSES: dict[str, type[BasePredictor]] = {
    "dixon_coles": DixonColesPredictor,
}

_models: dict[str, dict[str, BasePredictor]] = {}
_calibrators: dict[str, dict[str, ProbabilityCalibrator]] = {}
_fetcher: Optional[FootballFetcher] = None
_detector = ValueBetDetector(min_edge=0.03)

_retrain_status: dict = {
    "running": False,
    "steps": [],
    "error": None,
    "started_at": None,
    "finished_at": None,
}


def _retrain_log(msg: str, level: str = "info") -> None:
    _retrain_status["steps"].append({
        "time": datetime.now().strftime("%H:%M:%S"),
        "msg": msg,
        "level": level,
    })
    logger.info(f"[retrain] {msg}")

_MIGRATIONS = """
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS predicted_outcome VARCHAR(1);
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS actual_outcome VARCHAR(1);
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS correct BOOLEAN;
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS odds_home FLOAT;
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS odds_draw FLOAT;
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS odds_away FLOAT;
"""

KELLY_FRACTION = 0.25   # fractional Kelly (bezpečnější — 25% plného Kelly)
INITIAL_BANKROLL = 1000.0  # startovní bankroll v jednotkách (€ nebo body)


def _kelly_stake(prob: float, odds: float) -> float:
    """Full Kelly fraction: (p*b - q) / b  kde b = odds-1."""
    b = odds - 1
    if b <= 0:
        return 0.0
    q = 1 - prob
    kelly = (prob * b - q) / b
    return max(0.0, kelly * KELLY_FRACTION)  # fractional Kelly, never negative


def _scheduled_predictions():
    """Run every morning: fetch upcoming fixtures and save predictions to DB."""
    db = SessionLocal()
    try:
        for league_key, league_cfg in settings.leagues.items():
            league_models = _models.get(league_key, {})
            if not league_models:
                logger.warning(f"Scheduler: no models for {league_key}, skipping")
                continue
            upcoming = _fetcher.get_upcoming_fixtures(league_cfg, next_n=10)
            for fixture in upcoming:
                odds_list = _fetcher.get_odds(fixture.id)
                bet365 = next((o for o in odds_list if "bet365" in o.bookmaker.lower()), None)
                for model_name, model in league_models.items():
                    pred = model.predict(fixture)
                    pred = _detector.detect(pred, odds_list)
                    po = _predicted_outcome(pred.prob_home, pred.prob_draw, pred.prob_away)
                    stmt = pg_insert(PredictionRow).values(
                        fixture_id=fixture.id,
                        league_key=league_key,
                        model_name=model_name,
                        match_date=fixture.date,
                        home_team=fixture.home_team.name,
                        away_team=fixture.away_team.name,
                        prob_home=pred.prob_home,
                        prob_draw=pred.prob_draw,
                        prob_away=pred.prob_away,
                        xg_home=pred.expected_goals_home,
                        xg_away=pred.expected_goals_away,
                        value_bets=pred.value_bets or [],
                        goal_probs=pred.goal_probs or {},
                        predicted_outcome=po,
                        odds_home=bet365.home_win if bet365 else None,
                        odds_draw=bet365.draw if bet365 else None,
                        odds_away=bet365.away_win if bet365 else None,
                    ).on_conflict_do_nothing(constraint="uq_prediction_fixture_model")
                    db.execute(stmt)
            db.commit()
            logger.info(f"Scheduler: predictions saved for {league_key} ({len(upcoming)} fixtures)")
    except Exception as e:
        logger.error(f"Scheduler predictions failed: {e}")
        db.rollback()
    finally:
        db.close()


def _scheduled_resolve():
    """Run every evening: fill actual_outcome for finished matches."""
    db = SessionLocal()
    try:
        total_resolved = 0
        for league_key, league_cfg in settings.leagues.items():
            finished = _fetcher.get_fixtures_season(league_cfg, league_cfg.season, status="FT")
            finished_by_id = {f.id: f for f in finished if f.result is not None}
            pending = (
                db.query(PredictionRow)
                .filter(
                    PredictionRow.league_key == league_key,
                    PredictionRow.actual_outcome.is_(None),
                    PredictionRow.fixture_id.in_(finished_by_id.keys()),
                )
                .all()
            )
            for row in pending:
                actual = finished_by_id[row.fixture_id].result.outcome
                row.actual_outcome = actual
                row.correct = row.predicted_outcome == actual
                total_resolved += 1
        db.commit()
        logger.info(f"Scheduler: resolved {total_resolved} predictions")
    except Exception as e:
        logger.error(f"Scheduler resolve failed: {e}")
        db.rollback()
    finally:
        db.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _fetcher
    client = APIClient(settings)
    _fetcher = FootballFetcher(client, settings)

    Base.metadata.create_all(engine)

    for league_key in settings.leagues:
        _models[league_key] = {}
        base_models = []
        for model_name, cls in MODEL_CLASSES.items():
            path = MODELS_DIR / f"{model_name}_{league_key}.joblib"
            if path.exists():
                try:
                    m = cls.load(path)
                    _models[league_key][model_name] = m
                    base_models.append(m)
                    logger.info(f"Loaded {model_name} for {league_key}")
                except Exception as e:
                    logger.warning(f"Failed to load {path}: {e}")
            else:
                logger.warning(f"Model not found: {path} — run main.py first")

        _calibrators[league_key] = {}
        for model_name in MODEL_CLASSES.keys():
            cal_path = MODELS_DIR / f"calibrator_{model_name}_{league_key}.joblib"
            if cal_path.exists():
                try:
                    _calibrators[league_key][model_name] = ProbabilityCalibrator.load(cal_path)
                    logger.info(f"Loaded calibrator for {model_name}/{league_key}")
                except Exception as e:
                    logger.warning(f"Failed to load calibrator {cal_path}: {e}")

    scheduler = AsyncIOScheduler()
    scheduler.add_job(_scheduled_predictions, CronTrigger(hour=9, minute=0),
                      name="daily_predictions", misfire_grace_time=3600)
    scheduler.add_job(_scheduled_resolve, CronTrigger(hour=23, minute=0),
                      name="daily_resolve", misfire_grace_time=3600)
    scheduler.start()
    logger.info("Scheduler started — predictions @ 09:00, resolve @ 23:00")

    yield

    scheduler.shutdown()


app = FastAPI(title="Football Predictor API", lifespan=lifespan)


# --- Helpers ---

def _require_league(league: str) -> None:
    if league not in settings.leagues:
        raise HTTPException(status_code=404, detail=f"Unknown league '{league}'. "
                            f"Valid keys: {list(settings.leagues)}")


def _require_models(league: str) -> dict[str, BasePredictor]:
    loaded = _models.get(league, {})
    if not loaded:
        raise HTTPException(status_code=503,
                            detail=f"No trained models for '{league}'. Run main.py first.")
    return loaded


def _predicted_outcome(prob_home: float, prob_draw: float, prob_away: float) -> str:
    return max({"H": prob_home, "D": prob_draw, "A": prob_away}, key=lambda k: {"H": prob_home, "D": prob_draw, "A": prob_away}[k])


def _apply_calibration(league: str, model_name: str,
                       prob_h: float, prob_d: float, prob_a: float
                       ) -> tuple[float, float, float]:
    cal = _calibrators.get(league, {}).get(model_name)
    if cal and cal.fitted:
        return cal.transform(prob_h, prob_d, prob_a)
    return prob_h, prob_d, prob_a


# --- Pydantic schemas ---

class ModelPrediction(BaseModel):
    model: str
    prob_home: float
    prob_draw: float
    prob_away: float
    predicted_outcome: str
    expected_goals_home: Optional[float]
    expected_goals_away: Optional[float]
    value_bets: list[str]
    goal_probs: dict[str, float]


class FixturePrediction(BaseModel):
    fixture_id: int
    date: str
    home_team: str
    away_team: str
    predictions: list[ModelPrediction]


class ValueBet(BaseModel):
    league: str
    fixture_id: int
    date: str
    home_team: str
    away_team: str
    model: str
    prob_home: float
    prob_draw: float
    prob_away: float
    value_bets: list[str]


class TrackRecord(BaseModel):
    model: str
    n_train: int
    n_test: int
    accuracy: float
    brier_score: float
    log_loss: float


class TrackRecordHistory(TrackRecord):
    run_at: str


class ResolveResult(BaseModel):
    resolved: int
    already_resolved: int


class ModelPerformance(BaseModel):
    model: str
    predictions_total: int
    predictions_resolved: int
    accuracy: Optional[float]
    value_bets_total: int
    value_bets_correct: int
    value_bet_accuracy: Optional[float]


# --- Endpoints ---

@app.post("/calibrate/{league}")
def calibrate(league: str, db: Session = Depends(get_db)):
    """
    Train post-hoc calibration (isotonic regression) from resolved predictions in DB.
    Requires at least 50 resolved predictions per model.
    Calibrators are saved to disk and hot-loaded into memory.
    """
    _require_league(league)

    rows = (
        db.query(PredictionRow)
        .filter(
            PredictionRow.league_key == league,
            PredictionRow.actual_outcome.isnot(None),
        )
        .all()
    )

    if not rows:
        raise HTTPException(status_code=400, detail="No resolved predictions found. Run /resolve first.")

    by_model: dict[str, list] = {}
    for row in rows:
        by_model.setdefault(row.model_name, []).append(row)

    results = {}
    for model_name, model_rows in by_model.items():
        try:
            probs_h = [r.prob_home for r in model_rows]
            probs_d = [r.prob_draw for r in model_rows]
            probs_a = [r.prob_away for r in model_rows]
            actuals = [r.actual_outcome for r in model_rows]

            cal = ProbabilityCalibrator()
            cal.fit(probs_h, probs_d, probs_a, actuals, min_samples=50)

            cal_path = MODELS_DIR / f"calibrator_{model_name}_{league}.joblib"
            cal.save(cal_path)

            _calibrators.setdefault(league, {})[model_name] = cal
            results[model_name] = {"status": "ok", "samples": len(model_rows)}
            logger.info(f"Calibrator trained for {model_name}/{league} on {len(model_rows)} samples")
        except ValueError as e:
            results[model_name] = {"status": "skipped", "reason": str(e)}
        except Exception as e:
            results[model_name] = {"status": "error", "reason": str(e)}

    return results


@app.get("/calibrate/status/{league}")
def calibrate_status(league: str, db: Session = Depends(get_db)):
    """Show calibration status per model — how many samples, whether calibrator is active."""
    _require_league(league)

    rows = (
        db.query(PredictionRow)
        .filter(PredictionRow.league_key == league)
        .all()
    )

    by_model: dict[str, list] = {}
    for row in rows:
        by_model.setdefault(row.model_name, []).append(row)

    status = {}
    for model_name, model_rows in by_model.items():
        resolved = [r for r in model_rows if r.actual_outcome is not None]
        cal_active = model_name in _calibrators.get(league, {})
        status[model_name] = {
            "resolved_predictions": len(resolved),
            "needed_for_calibration": max(0, 50 - len(resolved)),
            "calibrator_active": cal_active,
        }
    return status


def _run_retrain() -> None:
    _retrain_status["running"] = True
    _retrain_status["steps"] = []
    _retrain_status["error"] = None
    _retrain_status["started_at"] = datetime.now().isoformat()
    _retrain_status["finished_at"] = None

    try:
        leagues = list(settings.leagues.items())
        _retrain_log(f"Spouštím přetrénování — {len(leagues)} liga/y")

        for league_key, league_cfg in leagues:
            _retrain_log(f"[{league_cfg.name}] Stahuji historii zápasů...")
            history = _fetcher.get_fixtures(league_cfg, status="FT")
            completed = [f for f in history if f.result is not None]

            if len(completed) < 50:
                _retrain_log(f"[{league_cfg.name}] Přeskočeno — pouze {len(completed)} zápasů", "warning")
                continue

            _retrain_log(f"[{league_cfg.name}] {len(completed)} zápasů načteno, trénuji modely...")
            _models[league_key] = {}
            base_models = []

            for model_name, cls in MODEL_CLASSES.items():
                _retrain_log(f"[{league_cfg.name}] Trénuji {model_name}...")
                path = MODELS_DIR / f"{model_name}_{league_key}.joblib"
                m = cls()
                m.train(completed)
                m.save(path)
                _models[league_key][model_name] = m
                base_models.append(m)
                _retrain_log(f"[{league_cfg.name}] ✓ {model_name} hotovo", "success")

        _retrain_log("Přetrénování dokončeno ✓", "success")

    except Exception as e:
        _retrain_status["error"] = str(e)
        _retrain_log(f"Chyba: {e}", "error")
        logger.error(f"Retrain failed: {e}")
    finally:
        _retrain_status["running"] = False
        _retrain_status["finished_at"] = datetime.now().isoformat()


@app.post("/retrain", status_code=202)
def retrain():
    """Start retraining in background. Poll /retrain/status for progress."""
    if _retrain_status["running"]:
        raise HTTPException(status_code=409, detail="Retraining already in progress.")
    threading.Thread(target=_run_retrain, daemon=True).start()
    return {"status": "started"}


@app.get("/retrain/status")
def retrain_status():
    return _retrain_status


@app.get("/config/leagues")
def config_leagues():
    """League keys and display names for the dashboard."""
    flags = {"England": "🏴󠁧󠁢󠁥󠁮󠁧󠁿", "Spain": "🇪🇸", "Germany": "🇩🇪", "Italy": "🇮🇹", "France": "🇫🇷"}
    return {
        key: f"{flags.get(cfg.country, '')} {cfg.name}"
        for key, cfg in settings.leagues.items()
    }


@app.get("/config/bookmakers")
def config_bookmakers():
    """Bookmaker keys and display names for the dashboard."""
    return {key: bm.name for key, bm in settings.bookmakers.items()}


@app.get("/predictions/{league}", response_model=list[FixturePrediction])
def predictions(league: str, db: Session = Depends(get_db)):
    _require_league(league)
    league_models = _require_models(league)
    league_cfg = settings.leagues[league]

    upcoming = _fetcher.get_upcoming_fixtures(league_cfg, next_n=10)
    results = []

    for fixture in upcoming:
        odds_list = _fetcher.get_odds(fixture.id)
        bet365 = next((o for o in odds_list if "bet365" in o.bookmaker.lower()), None)
        model_preds = []

        for model_name, model in league_models.items():
            pred = model.predict(fixture)
            # Apply post-hoc calibration if available
            ph, pd_, pa = _apply_calibration(
                league, model_name, pred.prob_home, pred.prob_draw, pred.prob_away)
            pred.prob_home, pred.prob_draw, pred.prob_away = ph, pd_, pa
            pred = _detector.detect(pred, odds_list)
            po = _predicted_outcome(pred.prob_home, pred.prob_draw, pred.prob_away)

            stmt = pg_insert(PredictionRow).values(
                fixture_id=fixture.id,
                league_key=league,
                model_name=model_name,
                match_date=fixture.date,
                home_team=fixture.home_team.name,
                away_team=fixture.away_team.name,
                prob_home=pred.prob_home,
                prob_draw=pred.prob_draw,
                prob_away=pred.prob_away,
                xg_home=pred.expected_goals_home,
                xg_away=pred.expected_goals_away,
                value_bets=pred.value_bets or [],
                goal_probs=pred.goal_probs or {},
                predicted_outcome=po,
                odds_home=bet365.home_win if bet365 else None,
                odds_draw=bet365.draw if bet365 else None,
                odds_away=bet365.away_win if bet365 else None,
            ).on_conflict_do_nothing(constraint="uq_prediction_fixture_model")
            db.execute(stmt)

            model_preds.append(ModelPrediction(
                model=model_name,
                prob_home=pred.prob_home,
                prob_draw=pred.prob_draw,
                prob_away=pred.prob_away,
                predicted_outcome=po,
                expected_goals_home=pred.expected_goals_home,
                expected_goals_away=pred.expected_goals_away,
                value_bets=pred.value_bets,
                goal_probs=pred.goal_probs,
            ))

        results.append(FixturePrediction(
            fixture_id=fixture.id,
            date=fixture.date.isoformat(),
            home_team=fixture.home_team.name,
            away_team=fixture.away_team.name,
            predictions=model_preds,
        ))

    db.commit()
    return results


@app.get("/value-bets", response_model=list[ValueBet])
def value_bets():
    any_loaded = any(_models.get(lk) for lk in settings.leagues)
    if not any_loaded:
        raise HTTPException(status_code=503, detail="No trained models found. Run main.py first.")

    results = []
    for league_key, league_cfg in settings.leagues.items():
        league_models = _models.get(league_key, {})
        if not league_models:
            continue
        upcoming = _fetcher.get_upcoming_fixtures(league_cfg, next_n=10)
        for fixture in upcoming:
            odds_list = _fetcher.get_odds(fixture.id)
            for model_name, model in league_models.items():
                pred = model.predict(fixture)
                pred = _detector.detect(pred, odds_list)
                if pred.value_bets:
                    results.append(ValueBet(
                        league=league_key,
                        fixture_id=fixture.id,
                        date=fixture.date.isoformat(),
                        home_team=fixture.home_team.name,
                        away_team=fixture.away_team.name,
                        model=model_name,
                        prob_home=pred.prob_home,
                        prob_draw=pred.prob_draw,
                        prob_away=pred.prob_away,
                        value_bets=pred.value_bets,
                    ))
    return results


@app.post("/resolve/{league}", response_model=ResolveResult)
def resolve(league: str, db: Session = Depends(get_db)):
    """
    Match finished fixtures to saved predictions and fill actual_outcome + correct.
    Call this periodically (e.g. daily) after matches have been played.
    """
    _require_league(league)
    league_cfg = settings.leagues[league]

    finished = _fetcher.get_fixtures_season(league_cfg, league_cfg.season, status="FT")
    finished_by_id = {f.id: f for f in finished if f.result is not None}

    pending = (
        db.query(PredictionRow)
        .filter(
            PredictionRow.league_key == league,
            PredictionRow.actual_outcome.is_(None),
            PredictionRow.fixture_id.in_(finished_by_id.keys()),
        )
        .all()
    )

    resolved = 0
    for row in pending:
        actual = finished_by_id[row.fixture_id].result.outcome
        row.actual_outcome = actual
        row.correct = row.predicted_outcome == actual
        resolved += 1

    db.commit()
    already = (
        db.query(PredictionRow)
        .filter(
            PredictionRow.league_key == league,
            PredictionRow.actual_outcome.isnot(None),
        )
        .count()
    )
    return ResolveResult(resolved=resolved, already_resolved=already - resolved)


@app.get("/performance/{league}", response_model=list[ModelPerformance])
def performance(league: str, db: Session = Depends(get_db)):
    """Real-world accuracy and value-bet success rate per model based on resolved predictions."""
    _require_league(league)

    rows = (
        db.query(PredictionRow)
        .filter(PredictionRow.league_key == league)
        .all()
    )

    by_model: dict[str, list[PredictionRow]] = {}
    for row in rows:
        by_model.setdefault(row.model_name, []).append(row)

    results = []
    for model_name, model_rows in by_model.items():
        resolved = [r for r in model_rows if r.actual_outcome is not None]
        correct = [r for r in resolved if r.correct]

        vb_rows = [r for r in resolved if r.value_bets]
        vb_correct = [r for r in vb_rows if r.correct]

        results.append(ModelPerformance(
            model=model_name,
            predictions_total=len(model_rows),
            predictions_resolved=len(resolved),
            accuracy=round(len(correct) / len(resolved), 4) if resolved else None,
            value_bets_total=len(vb_rows),
            value_bets_correct=len(vb_correct),
            value_bet_accuracy=round(len(vb_correct) / len(vb_rows), 4) if vb_rows else None,
        ))

    return sorted(results, key=lambda x: x.model)


class CalibrationBinOut(BaseModel):
    bin_lower: float
    bin_upper: float
    predicted_prob: float
    actual_freq: float
    count: int


class OutcomeCalibrationOut(BaseModel):
    outcome: str
    ece: float
    bins: list[CalibrationBinOut]


class CalibrationOut(BaseModel):
    model: str
    n_samples: int
    overall_ece: float
    outcomes: list[OutcomeCalibrationOut]


@app.get("/calibration/{league}", response_model=list[CalibrationOut])
def calibration(league: str):
    """
    Reliability diagram data + ECE per model.
    ECE < 0.03 = well calibrated | ECE > 0.07 = poorly calibrated.
    """
    _require_league(league)
    league_models = _require_models(league)
    league_cfg = settings.leagues[league]

    history = _fetcher.get_fixtures(league_cfg, status="FT")
    results = []

    for model_name, model in league_models.items():
        bt_engine = BacktestEngine(model, min_train_size=100, retrain_every=10)
        bt = bt_engine.run(history, league_name=league_cfg.name)
        cal = compute_calibration(bt.matches, model_name)
        results.append(CalibrationOut(
            model=cal.model,
            n_samples=cal.n_samples,
            overall_ece=cal.overall_ece,
            outcomes=[
                OutcomeCalibrationOut(
                    outcome=oc.outcome,
                    ece=oc.ece,
                    bins=[CalibrationBinOut(**b.__dict__) for b in oc.bins],
                )
                for oc in cal.outcomes
            ],
        ))

    return results


@app.get("/track-record/{league}", response_model=list[TrackRecord])
def track_record(league: str, db: Session = Depends(get_db)):
    _require_league(league)
    league_models = _require_models(league)
    league_cfg = settings.leagues[league]

    history = _fetcher.get_fixtures(league_cfg, status="FT")
    results = []

    for model_name, model in league_models.items():
        engine_bt = BacktestEngine(model, min_train_size=100, retrain_every=10)
        bt = engine_bt.run(history, league_name=league_cfg.name)

        db.add(BacktestRunRow(
            league_key=league,
            league_name=league_cfg.name,
            model_name=model_name,
            n_train=bt.n_train,
            n_test=bt.n_test,
            accuracy=round(bt.accuracy, 4),
            brier_score=round(bt.brier_score, 4),
            log_loss=round(bt.log_loss, 4),
        ))
        results.append(TrackRecord(
            model=model_name,
            n_train=bt.n_train,
            n_test=bt.n_test,
            accuracy=round(bt.accuracy, 4),
            brier_score=round(bt.brier_score, 4),
            log_loss=round(bt.log_loss, 4),
        ))

    db.commit()
    return results


@app.get("/track-record/{league}/history", response_model=list[TrackRecordHistory])
def track_record_history(league: str, db: Session = Depends(get_db)):
    _require_league(league)
    rows = (
        db.query(BacktestRunRow)
        .filter(BacktestRunRow.league_key == league)
        .order_by(BacktestRunRow.run_at.desc())
        .all()
    )
    return [
        TrackRecordHistory(
            model=row.model_name,
            n_train=row.n_train,
            n_test=row.n_test,
            accuracy=row.accuracy,
            brier_score=row.brier_score,
            log_loss=row.log_loss,
            run_at=row.run_at.isoformat(),
        )
        for row in rows
    ]


# --- Kelly / Bankroll ---

class KellySuggestion(BaseModel):
    fixture_id: int
    date: str
    home_team: str
    away_team: str
    model: str
    bet_on: str
    prob: float
    odds: float
    kelly_fraction: float
    stake_pct: float
    stake_units: float  # at INITIAL_BANKROLL


class BankrollPoint(BaseModel):
    match_date: str
    home_team: str
    away_team: str
    model: str
    bet_on: str
    odds: float
    stake_pct: float
    outcome: Optional[str]
    pnl_pct: Optional[float]
    bankroll_after: Optional[float]


@app.get("/kelly/{league}", response_model=list[KellySuggestion])
def kelly(league: str, db: Session = Depends(get_db)):
    """Kelly-sized stake suggestions for upcoming value bets."""
    _require_league(league)
    league_models = _require_models(league)
    league_cfg = settings.leagues[league]

    upcoming = _fetcher.get_upcoming_fixtures(league_cfg, next_n=10)
    results = []

    for fixture in upcoming:
        odds_list = _fetcher.get_odds(fixture.id)
        bet365 = next((o for o in odds_list if o.bookmaker == "Bet365"), None)
        if not bet365:
            continue

        odds_map = {"H": bet365.home_win, "D": bet365.draw, "A": bet365.away_win}

        for model_name, model in league_models.items():
            pred = model.predict(fixture)
            pred = _detector.detect(pred, odds_list)
            if not pred.value_bets:
                continue

            prob_map = {"H": pred.prob_home, "D": pred.prob_draw, "A": pred.prob_away}

            for outcome in pred.value_bets:
                o = outcome[0]  # 'H', 'D', or 'A'
                prob = prob_map[o]
                odds = odds_map[o]
                stake = _kelly_stake(prob, odds)

                if stake > 0:
                    results.append(KellySuggestion(
                        fixture_id=fixture.id,
                        date=fixture.date.isoformat(),
                        home_team=fixture.home_team.name,
                        away_team=fixture.away_team.name,
                        model=model_name,
                        bet_on=o,
                        prob=round(prob, 4),
                        odds=round(odds, 2),
                        kelly_fraction=round(stake, 4),
                        stake_pct=round(stake * 100, 2),
                        stake_units=round(stake * INITIAL_BANKROLL, 2),
                    ))

    return results


@app.post("/bankroll/update/{league}")
def bankroll_update(league: str, db: Session = Depends(get_db)):
    """
    Sync bankroll table from resolved predictions with value bets.
    Calculates running bankroll starting from INITIAL_BANKROLL.
    """
    _require_league(league)

    resolved_vb = (
        db.query(PredictionRow)
        .filter(
            PredictionRow.league_key == league,
            PredictionRow.actual_outcome.isnot(None),
            PredictionRow.value_bets.isnot(None),
            func.cardinality(PredictionRow.value_bets) > 0,
        )
        .order_by(PredictionRow.match_date)
        .all()
    )

    existing_ids = {r.fixture_id for r in db.query(BankrollRow.fixture_id)
                    .filter(BankrollRow.league_key == league).all()}

    bankroll = (
        db.query(BankrollRow)
        .filter(BankrollRow.league_key == league)
        .order_by(BankrollRow.match_date.desc())
        .first()
    )
    current_bankroll = bankroll.bankroll_after if bankroll else INITIAL_BANKROLL

    added = 0
    for row in resolved_vb:
        if row.fixture_id in existing_ids:
            continue
        prob_map = {"H": row.prob_home, "D": row.prob_draw, "A": row.prob_away}
        odds_map = {"H": row.odds_home, "D": row.odds_draw, "A": row.odds_away}

        for vb in (row.value_bets or []):
            o = vb[0]
            prob = prob_map.get(o)
            odds = odds_map.get(o)
            if not prob or not odds:
                continue

            stake = _kelly_stake(prob, odds)
            if stake <= 0:
                continue

            won = row.actual_outcome == o
            pnl = stake * (odds - 1) if won else -stake
            current_bankroll = current_bankroll * (1 + pnl)

            db.add(BankrollRow(
                league_key=league,
                model_name=row.model_name,
                fixture_id=row.fixture_id,
                match_date=row.match_date,
                home_team=row.home_team,
                away_team=row.away_team,
                bet_on=o,
                kelly_fraction=round(stake, 4),
                odds=round(odds, 2),
                stake_pct=round(stake * 100, 2),
                outcome="win" if won else "loss",
                pnl_pct=round(pnl * 100, 2),
                bankroll_after=round(current_bankroll, 2),
            ))
            added += 1

    db.commit()
    return {"added": added, "bankroll": round(current_bankroll, 2)}


@app.get("/bankroll/{league}", response_model=list[BankrollPoint])
def bankroll(league: str, db: Session = Depends(get_db)):
    """Bankroll evolution for the given league."""
    _require_league(league)
    rows = (
        db.query(BankrollRow)
        .filter(BankrollRow.league_key == league)
        .order_by(BankrollRow.match_date)
        .all()
    )
    return [
        BankrollPoint(
            match_date=r.match_date.isoformat(),
            home_team=r.home_team,
            away_team=r.away_team,
            model=r.model_name,
            bet_on=r.bet_on,
            odds=r.odds,
            stake_pct=r.stake_pct,
            outcome=r.outcome,
            pnl_pct=r.pnl_pct,
            bankroll_after=r.bankroll_after,
        )
        for r in rows
    ]
