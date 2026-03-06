from datetime import datetime
from typing import Optional

from sqlalchemy import BigInteger, Boolean, DateTime, Float, Integer, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import DeclarativeBase, mapped_column, Mapped


class Base(DeclarativeBase):
    pass


class PredictionRow(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    fixture_id: Mapped[int] = mapped_column(Integer, nullable=False)
    league_key: Mapped[str] = mapped_column(String(50), nullable=False)
    model_name: Mapped[str] = mapped_column(String(50), nullable=False)
    match_date: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    home_team: Mapped[str] = mapped_column(String(100), nullable=False)
    away_team: Mapped[str] = mapped_column(String(100), nullable=False)
    prob_home: Mapped[float] = mapped_column(Float, nullable=False)
    prob_draw: Mapped[float] = mapped_column(Float, nullable=False)
    prob_away: Mapped[float] = mapped_column(Float, nullable=False)
    xg_home: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    xg_away: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    value_bets: Mapped[list] = mapped_column(ARRAY(String), nullable=True)
    goal_probs: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    # Outcome tracking
    predicted_outcome: Mapped[Optional[str]] = mapped_column(String(1), nullable=True)  # H / D / A
    actual_outcome: Mapped[Optional[str]] = mapped_column(String(1), nullable=True)     # filled after FT
    correct: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    # Bet365 odds at prediction time (for value bet ROI tracking)
    odds_home: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    odds_draw: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    odds_away: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("fixture_id", "model_name", name="uq_prediction_fixture_model"),
    )


class BankrollRow(Base):
    """Simulated bankroll evolution based on Kelly-sized value bets."""
    __tablename__ = "bankroll"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    league_key: Mapped[str] = mapped_column(String(50), nullable=False)
    model_name: Mapped[str] = mapped_column(String(50), nullable=False)
    fixture_id: Mapped[int] = mapped_column(Integer, nullable=False)
    match_date: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    home_team: Mapped[str] = mapped_column(String(100), nullable=False)
    away_team: Mapped[str] = mapped_column(String(100), nullable=False)
    bet_on: Mapped[str] = mapped_column(String(1), nullable=False)       # H / D / A
    kelly_fraction: Mapped[float] = mapped_column(Float, nullable=False)  # 0–1
    odds: Mapped[float] = mapped_column(Float, nullable=False)
    stake_pct: Mapped[float] = mapped_column(Float, nullable=False)       # % bankrollu
    outcome: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)  # win / loss
    pnl_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)     # % change
    bankroll_after: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class BacktestRunRow(Base):
    __tablename__ = "backtest_runs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    league_key: Mapped[str] = mapped_column(String(50), nullable=False)
    league_name: Mapped[str] = mapped_column(String(100), nullable=False)
    model_name: Mapped[str] = mapped_column(String(50), nullable=False)
    n_train: Mapped[int] = mapped_column(Integer, nullable=False)
    n_test: Mapped[int] = mapped_column(Integer, nullable=False)
    accuracy: Mapped[float] = mapped_column(Float, nullable=False)
    brier_score: Mapped[float] = mapped_column(Float, nullable=False)
    log_loss: Mapped[float] = mapped_column(Float, nullable=False)
    run_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
