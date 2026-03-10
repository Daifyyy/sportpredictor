from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.sql import func


class Base(DeclarativeBase):
    pass


class FixturePrediction(Base):
    """Pre-computed model predictions for upcoming fixtures (refreshed daily by GitHub Actions)."""
    __tablename__ = "fixture_predictions"

    fixture_id = Column(Integer, primary_key=True)
    league = Column(String, nullable=False)
    home_team = Column(String, nullable=False)
    away_team = Column(String, nullable=False)
    match_date = Column(DateTime(timezone=True), nullable=False)
    prob_home = Column(Float, nullable=False)
    prob_draw = Column(Float, nullable=False)
    prob_away = Column(Float, nullable=False)
    over2_5 = Column(Float, nullable=False)
    under2_5 = Column(Float, nullable=False)
    goals1_3 = Column(Float, nullable=False)
    goals2_4 = Column(Float, nullable=False)
    btts_yes = Column(Float, nullable=False)
    btts_no = Column(Float, nullable=False)
    computed_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class TrackedPrediction(Base):
    __tablename__ = "tracked_predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    fixture_id = Column(Integer, nullable=False)
    league = Column(String, nullable=False)
    home_team = Column(String, nullable=False)
    away_team = Column(String, nullable=False)
    match_date = Column(DateTime(timezone=True), nullable=False)
    prediction_type = Column(String, nullable=False)  # H/D/A/Under2.5/Over2.5/Goals1-3/Goals2-4/BTTS_Yes/BTTS_No
    model_prob = Column(Float, nullable=True)
    actual_outcome = Column(String, nullable=True)
    correct = Column(Boolean, nullable=True)
    home_score = Column(Integer, nullable=True)
    away_score = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint("fixture_id", "prediction_type", name="uq_fixture_prediction"),
    )
