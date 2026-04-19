"""
Standalone resolve script for GitHub Actions.
Fetches results for all pending tracked_predictions and updates the DB.
"""
import logging
import sys
import os

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timedelta, timezone

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

from api.client import APIClient
from config.settings import settings
from data.fetcher import FootballFetcher
from db.models import ResolvedFixturePrediction, TrackedPrediction
from db.session import SessionLocal


def compute_correct(prediction_type: str, home_score: int, away_score: int,
                    actual_corners: int | None = None) -> bool:
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
    # Corners markets — need actual total corners
    if prediction_type.startswith("Corners_"):
        if actual_corners is None:
            return False
        if prediction_type == "Corners_Over8.5":  return actual_corners > 8
        if prediction_type == "Corners_Under8.5": return actual_corners <= 8
        if prediction_type == "Corners_Over9.5":  return actual_corners > 9
        if prediction_type == "Corners_Under9.5": return actual_corners <= 9
        if prediction_type == "Corners_Over10.5": return actual_corners > 10
        if prediction_type == "Corners_Under10.5":return actual_corners <= 10
        if prediction_type == "Corners_Over11.5": return actual_corners > 11
        if prediction_type == "Corners_Under11.5":return actual_corners <= 11
    return False


def main():
    client = APIClient(settings)
    fetcher = FootballFetcher(client, settings)
    db = SessionLocal()

    try:
        now = datetime.now(timezone.utc)

        pending = db.query(TrackedPrediction).filter(
            TrackedPrediction.correct.is_(None),
            TrackedPrediction.match_date < now,
        ).all()

        if not pending:
            print("No pending predictions to resolve.")
            return

        print(f"Found {len(pending)} pending predictions.")

        fixture_ids = list({r.fixture_id for r in pending})
        results_by_id: dict[int, tuple[int, int]] = {}
        corners_by_id: dict[int, int] = {}  # fixture_id -> total corners

        for fid in fixture_ids:
            data = client.get("fixtures", {"id": fid}, ttl=settings.cache_ttl.fixtures)
            if not data:
                print(f"No data for fixture {fid}")
                continue
            response = data.get("response", [])
            if not response:
                print(f"Empty response for fixture {fid}")
                continue
            raw = response[0]
            goals = raw.get("goals", {})
            status = raw["fixture"]["status"]["short"]
            if status == "FT" and goals.get("home") is not None and goals.get("away") is not None:
                results_by_id[fid] = (int(goals["home"]), int(goals["away"]))
                print(f"Fixture {fid}: {goals['home']}-{goals['away']}")
            else:
                print(f"Fixture {fid}: status={status}, not finished yet")

        # Fetch corners for fixtures that have corners predictions pending
        corners_fixture_ids = {
            r.fixture_id for r in pending
            if r.prediction_type.startswith("Corners_") and r.fixture_id in results_by_id
        }
        for fid in corners_fixture_ids:
            stats = fetcher.get_fixture_statistics(fid)
            if stats:
                total = sum(
                    (s.corners or 0) for s in stats.values() if s.corners is not None
                )
                corners_by_id[fid] = total
                print(f"Fixture {fid}: {total} total corners")

        resolved = 0
        for row in pending:
            if row.fixture_id not in results_by_id:
                continue
            hs, as_ = results_by_id[row.fixture_id]
            actual_corners = corners_by_id.get(row.fixture_id)
            row.home_score = hs
            row.away_score = as_
            row.actual_outcome = f"{hs}-{as_}"
            row.correct = compute_correct(row.prediction_type, hs, as_, actual_corners)
            correct_str = "✓" if row.correct else "✗"
            print(f"  {row.home_team} vs {row.away_team} | {row.prediction_type} | {hs}-{as_} | {correct_str}")
            resolved += 1

        db.commit()
        print(f"Resolved {resolved} predictions.")

        # Clean up resolved_fixture_predictions older than 60 days
        cutoff = now - timedelta(days=60)
        deleted = db.query(ResolvedFixturePrediction).filter(
            ResolvedFixturePrediction.match_date < cutoff
        ).delete()
        if deleted:
            db.commit()
            print(f"Cleaned up {deleted} resolved fixture(s) older than 60 days.")

    finally:
        db.close()


if __name__ == "__main__":
    main()
