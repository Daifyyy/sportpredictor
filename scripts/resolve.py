"""
Standalone resolve script for GitHub Actions.
Fetches results for all pending tracked_predictions and updates the DB.
"""
import sys
import os

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timedelta, timezone

from api.client import APIClient
from config.settings import settings
from db.models import ResolvedFixturePrediction, TrackedPrediction
from db.session import SessionLocal


def compute_correct(prediction_type: str, home_score: int, away_score: int) -> bool:
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


def main():
    client = APIClient(settings)
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

        for fid in fixture_ids:
            data = client.get("fixtures", {"id": fid}, ttl=3600)
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

        resolved = 0
        for row in pending:
            if row.fixture_id not in results_by_id:
                continue
            hs, as_ = results_by_id[row.fixture_id]
            row.home_score = hs
            row.away_score = as_
            row.actual_outcome = f"{hs}-{as_}"
            row.correct = compute_correct(row.prediction_type, hs, as_)
            correct_str = "✓" if row.correct else "✗"
            print(f"  {row.home_team} vs {row.away_team} | {row.prediction_type} | {hs}-{as_} | {correct_str}")
            resolved += 1

        db.commit()
        print(f"Resolved {resolved} predictions.")

        # Clean up resolved_fixture_predictions older than 10 days
        cutoff = now - timedelta(days=10)
        deleted = db.query(ResolvedFixturePrediction).filter(
            ResolvedFixturePrediction.match_date < cutoff
        ).delete()
        if deleted:
            db.commit()
            print(f"Cleaned up {deleted} resolved fixture(s) older than 10 days.")

    finally:
        db.close()


if __name__ == "__main__":
    main()
