import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import poisson
from sqlalchemy.orm import Session as SASession

for _k in ["DATABASE_URL", "API_FOOTBALL_KEY"]:
    if _k in st.secrets and not os.getenv(_k):
        os.environ[_k] = st.secrets[_k]

from api.client import APIClient
from config.settings import settings
from data.fetcher import FootballFetcher
from db.models import Base, FixturePrediction, ResolvedFixturePrediction, TrackedPrediction
from db.session import engine
from features.engineer import FeatureEngineer
from models.ensemble import EnsembleDCPredictor
from models.injury import InjuryAdjuster
from models.poisson import DixonColesPredictor

MODELS_DIR = Path("models/saved")
PREDICTION_TYPES = ["H", "D", "A", "Under2.5", "Over2.5", "Goals1-3", "Goals2-4", "BTTS_Yes", "BTTS_No"]
_injury_adjuster = InjuryAdjuster()

flags = {"England": "🏴󠁧󠁢󠁥󠁮󠁧󠁿", "Spain": "🇪🇸", "Germany": "🇩🇪", "Italy": "🇮🇹", "France": "🇫🇷", "Czech Republic": "🇨🇿", "Europe": "🇪🇺"}
leagues_display = {
    k: f"{flags.get(cfg.country, '')} {cfg.name}"
    for k, cfg in settings.leagues.items()
}
league_keys = list(leagues_display.keys())


# ── Cached resources ───────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Inicializuji API klienta...")
def get_fetcher():
    client = APIClient(settings)
    return FootballFetcher(client, settings)


@st.cache_resource(show_spinner=False)
def get_model(league_key: str) -> EnsembleDCPredictor | None:
    from datetime import timedelta
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    fetcher = get_fetcher()
    cfg = settings.leagues[league_key]
    history = fetcher.get_fixtures(cfg, status="FT")
    completed = [f for f in history if f.result is not None]

    if len(completed) < 50:
        return None

    # dc_all — full history
    path_all = MODELS_DIR / f"dc_all_{league_key}.joblib"
    if path_all.exists():
        try:
            dc_all = DixonColesPredictor.load(path_all)
        except Exception:
            dc_all = None
    else:
        dc_all = None
    if dc_all is None:
        dc_all = DixonColesPredictor()
        dc_all.train(completed)
        dc_all.save(path_all)

    # dc_season — current season only
    season_fixtures = [f for f in completed if f.season == cfg.season]
    dc_season = None
    if len(season_fixtures) >= 30:
        path_season = MODELS_DIR / f"dc_season_{league_key}.joblib"
        if path_season.exists():
            try:
                dc_season = DixonColesPredictor.load(path_season)
            except Exception:
                dc_season = None
        if dc_season is None:
            dc_season = DixonColesPredictor()
            dc_season.train(season_fixtures)
            dc_season.save(path_season)
    if dc_season is None:
        dc_season = dc_all  # fallback

    # dc_recent — last 60 days
    cutoff = max(f.date for f in completed) - timedelta(days=60)
    recent_fixtures = [f for f in completed if f.date >= cutoff]
    dc_recent = None
    if len(recent_fixtures) >= 30:
        dc_recent = DixonColesPredictor()
        dc_recent.train(recent_fixtures)

    return EnsembleDCPredictor(dc_all, dc_season, dc_recent, league_key=league_key)


@st.cache_data(ttl=3600, show_spinner=False)
def get_league_features(league_key: str) -> tuple[dict, dict]:
    """Compute FeatureEngineer stats for all upcoming fixtures + league averages."""
    fetcher = get_fetcher()
    cfg = settings.leagues[league_key]
    history = fetcher.get_fixtures(cfg, status="FT")
    completed = [f for f in history if f.result is not None]
    upcoming = fetcher.get_upcoming_fixtures(cfg, next_n=10)

    if len(completed) < 20 or not upcoming:
        return {}, {}

    fe = FeatureEngineer()
    fe.precompute(completed)
    features_by_id = {fx.id: fe.build_features(fx, completed) for fx in upcoming}

    n = len(completed)
    total_goals = sum(f.result.home_goals + f.result.away_goals for f in completed)
    wins = sum(1 for f in completed if f.result.outcome in ("H", "A"))
    draws = sum(1 for f in completed if f.result.outcome == "D")
    league_avg = {
        "avg_gf": round(total_goals / (2 * n), 2),
        "avg_pts": round((3 * wins + 2 * draws) / (2 * n), 2),
    }
    return features_by_id, league_avg


@st.cache_data(ttl=3600, show_spinner=False)
def get_league_injuries(league_key: str) -> dict:
    """Returns {fixture_id: {home, away, home_goals, away_goals}} for all upcoming fixtures."""
    fetcher = get_fetcher()
    cfg = settings.leagues[league_key]
    upcoming = fetcher.get_upcoming_fixtures(cfg, next_n=10)
    result = {}
    for fx in upcoming:
        try:
            home_inj, away_inj, home_goals, away_goals = fetcher.get_fixture_injuries(
                fx, cfg.id, cfg.season
            )
        except Exception:
            home_inj, away_inj, home_goals, away_goals = [], [], 50, 50
        result[fx.id] = {
            "home": [asdict(i) for i in home_inj],
            "away": [asdict(i) for i in away_inj],
            "home_goals": home_goals,
            "away_goals": away_goals,
        }
    return result


def _fv(val, fmt=".2f") -> str:
    if isinstance(val, (int, float)):
        try:
            return f"{val:{fmt}}"
        except (ValueError, TypeError):
            return str(round(val, 2))
    return "—"


def render_match_detail(fx_data: dict, feats: dict, league_avg: dict) -> None:
    if not feats:
        st.caption("Statistiky nejsou k dispozici (nedostatek dat).")
        return

    home = fx_data["home_team"]
    away = fx_data["away_team"]
    avg_gf = league_avg.get("avg_gf", 1.4)
    avg_pts = league_avg.get("avg_pts", 1.2)

    # ── Expected goals (λ/μ) — model's raw scoring rate estimates ──────────────
    lam = fx_data.get("expected_goals_home")
    mu  = fx_data.get("expected_goals_away")
    if lam is not None and mu is not None:
        eg_col1, eg_col2, eg_col3 = st.columns(3)
        eg_col1.metric(f"λ  ({home})", f"{lam:.2f}", help="Očekávaný počet gólů domácích (model)")
        eg_col2.metric("Celkem gólů (model)", f"{lam + mu:.2f}", help="λ + μ = celkový očekávaný počet gólů v zápase")
        eg_col3.metric(f"μ  ({away})", f"{mu:.2f}", help="Očekávaný počet gólů hostů (model)")

    H = f"🏠 {home}"
    A = f"✈️ {away}"

    def tbl(caption: str, rows: list) -> None:
        st.caption(caption)
        st.dataframe(
            pd.DataFrame(rows, columns=[H, "Statistika", A]),
            use_container_width=True,
            hide_index=True,
        )

    streak_h = feats.get("home_streak", 0)
    streak_a = feats.get("away_streak", 0)

    tbl("⚡ Forma (posl. 5 zápasů)", [
        [_fv(feats.get("home_form")),          f"Pts/zápas  ·  ø {avg_pts:.2f}",   _fv(feats.get("away_form"))],
        [_fv(feats.get("home_gf")),             f"GF/zápas  ·  ø {avg_gf:.2f}",    _fv(feats.get("away_gf"))],
        [_fv(feats.get("home_ga")),             f"GA/zápas  ·  ø {avg_gf:.2f}",    _fv(feats.get("away_ga"))],
        [_fv(feats.get("home_attack_str")),     "Útočná síla  ·  ø 1.00",           _fv(feats.get("away_attack_str"))],
        [_fv(feats.get("home_defense_str")),    "Obranná síla  ·  ø 1.00",          _fv(feats.get("away_defense_str"))],
    ])

    tbl("🏠 Domácí / ✈️ Venkovní forma (posl. 5 zápasů na vlastním hřišti / venku)", [
        [_fv(feats.get("home_venue_form")),  f"Pts/zápas  ·  ø {avg_pts:.2f}",  _fv(feats.get("away_venue_form"))],
        [_fv(feats.get("home_venue_gf")),    f"GF/zápas  ·  ø {avg_gf:.2f}",   _fv(feats.get("away_venue_gf"))],
        [_fv(feats.get("home_venue_ga")),    f"GA/zápas  ·  ø {avg_gf:.2f}",   _fv(feats.get("away_venue_ga"))],
    ])

    tbl("📅 Sezóna · Trend · Elo", [
        [_fv(feats.get("home_season_ppg")),   f"PPG sezóna  ·  ø {avg_pts:.2f}",  _fv(feats.get("away_season_ppg"))],
        [_fv(feats.get("home_form_short")),   "Forma (posl. 3 zápasy)",            _fv(feats.get("away_form_short"))],
        [_fv(feats.get("home_trend"), "+.2f"), "Trend (krátká − dlouhá forma)",   _fv(feats.get("away_trend"), "+.2f")],
        [f"{streak_h:+.0f}" if isinstance(streak_h, (int, float)) else "—",
         "Aktuální série  (W = +, L = −)",
         f"{streak_a:+.0f}" if isinstance(streak_a, (int, float)) else "—"],
        [_fv(feats.get("elo_home", 1500), ".0f"),  "Elo rating  ·  ø 1500",  _fv(feats.get("elo_away", 1500), ".0f")],
        [f"{feats.get('home_rest_days', 7):.0f} dní", "Odpočinek od posl. zápasu",
         f"{feats.get('away_rest_days', 7):.0f} dní"],
    ])

    draws_pct = feats.get("h2h_draws", 0.33)
    tbl("🤝 Head to Head (posl. 10 vzájemných zápasů)", [
        [f"{feats.get('h2h_home_wins', 0.5) * 100:.0f} %",  "Výhry",              f"{feats.get('h2h_away_wins', 0.17) * 100:.0f} %"],
        [f"{draws_pct * 100:.0f} %",                          "Remízy  ·  (shodně)", f"{draws_pct * 100:.0f} %"],
        [_fv(feats.get("h2h_home_gf", 1.5)),                  f"Avg GF  ·  ø {avg_gf:.2f}", _fv(feats.get("h2h_away_gf", 1.2))],
    ])

    # ── Zápasové statistiky (jen pokud jsou data — velké ligy s fixture/statistics) ──
    has_shots = feats.get("home_avg_shots_on_target") is not None
    has_xg    = feats.get("home_avg_xg") is not None
    if has_shots or has_xg:
        stats_rows = []
        if has_shots:
            stats_rows += [
                [_fv(feats.get("home_avg_shots_on_target")), "Střely na bránu / zápas (posl. 5)",  _fv(feats.get("away_avg_shots_on_target"))],
                [_fv(feats.get("home_avg_total_shots")),     "Celkové střely / zápas (posl. 5)",   _fv(feats.get("away_avg_total_shots"))],
                [_fv(feats.get("home_avg_corners")),         "Rohy / zápas (posl. 5)",             _fv(feats.get("away_avg_corners"))],
            ]
        if has_xg:
            stats_rows.append(
                [_fv(feats.get("home_avg_xg")), "xG / zápas (posl. 5)  ·  model očekávaná kvalita šancí", _fv(feats.get("away_avg_xg"))]
            )
        tbl("📊 Zápasové statistiky (průměr posl. 5 zápasů)", stats_rows)


_POS_CZ = {
    "Attacker":   "Útočník",
    "Midfielder": "Záložník",
    "Defender":   "Obránce",
    "Goalkeeper": "Brankář",
}
_STATUS_ICON = {"Missing": "🔴", "Questionable": "🟡"}


def render_injuries(
    home_name: str, away_name: str,
    home_injuries: list, away_injuries: list,
    home_goals: int, away_goals: int,
) -> None:
    """Render injured/suspended player lists with per-player impact estimate."""
    if not home_injuries and not away_injuries:
        return

    st.caption("🚑 Zranění a absence")

    def fmt_player(inj: dict, team_goals: int) -> str:
        from data.models import PlayerInjury
        icon = _STATUS_ICON.get(inj["status"], "🔴")
        pos  = _POS_CZ.get(inj["position"], inj["position"])
        atk, dfn = _injury_adjuster.player_impact(PlayerInjury(**inj), team_goals)
        impact = atk or dfn

        if inj["position"] in ("Attacker", "Midfielder"):
            stat_str = f"{inj['goals']}G {inj['assists']}A"
        else:
            stat_str = f"{inj['minutes']} min"

        impact_str = f" · **−{impact*100:.0f}% λ**" if impact >= 0.01 else ""
        status_str = "Chybí" if inj["status"] == "Missing" else "Pochybný"
        return f"{icon} {inj['player_name']} *({pos})* · {stat_str}{impact_str} · {status_str}"

    col_h, col_a = st.columns(2)
    with col_h:
        st.markdown(f"**🏠 {home_name}**")
        if home_injuries:
            for inj in home_injuries:
                st.markdown(fmt_player(inj, home_goals))
        else:
            st.caption("Žádná hlášená zranění")
    with col_a:
        st.markdown(f"**✈️ {away_name}**")
        if away_injuries:
            for inj in away_injuries:
                st.markdown(fmt_player(inj, away_goals))
        else:
            st.caption("Žádná hlášená zranění")


def get_db() -> SASession:
    return SASession(engine)


# ── Helpers ────────────────────────────────────────────────────────────────────

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


def compute_correct(prediction_type: str, hs: int, as_: int) -> bool:
    total = hs + as_
    if prediction_type == "H":
        return hs > as_
    if prediction_type == "D":
        return hs == as_
    if prediction_type == "A":
        return as_ > hs
    if prediction_type == "Under2.5":
        return total <= 2
    if prediction_type == "Over2.5":
        return total >= 3
    if prediction_type == "Goals1-3":
        return 1 <= total <= 3
    if prediction_type == "Goals2-4":
        return 2 <= total <= 4
    if prediction_type == "BTTS_Yes":
        return hs >= 1 and as_ >= 1
    if prediction_type == "BTTS_No":
        return hs == 0 or as_ == 0
    return False


def run_resolve() -> int:
    now = datetime.now(timezone.utc)
    client = APIClient(settings)
    resolved = 0
    with get_db() as db:
        pending = db.query(TrackedPrediction).filter(
            TrackedPrediction.correct.is_(None),
            TrackedPrediction.match_date < now,
        ).all()
        if not pending:
            return 0
        fixture_ids = list({r.fixture_id for r in pending})
        results_by_id: dict[int, tuple[int, int]] = {}
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
            if status == "FT" and goals.get("home") is not None:
                results_by_id[fid] = (int(goals["home"]), int(goals["away"]))
        for row in pending:
            if row.fixture_id not in results_by_id:
                continue
            hs, as_ = results_by_id[row.fixture_id]
            row.home_score = hs
            row.away_score = as_
            row.actual_outcome = f"{hs}-{as_}"
            row.correct = compute_correct(row.prediction_type, hs, as_)
            resolved += 1
        db.commit()
    return resolved


def save_tracking(fixture: dict, league: str, prediction_type: str, model_prob: float | None) -> str:
    prob_map = {
        "H": fixture.get("prob_home"),
        "D": fixture.get("prob_draw"),
        "A": fixture.get("prob_away"),
        "Under2.5": fixture.get("under2_5"),
        "Over2.5": fixture.get("over2_5"),
        "Goals1-3": fixture.get("goals1_3"),
        "Goals2-4": fixture.get("goals2_4"),
        "BTTS_Yes": fixture.get("btts_yes"),
        "BTTS_No": fixture.get("btts_no"),
    }
    with get_db() as db:
        existing = db.query(TrackedPrediction).filter(
            TrackedPrediction.fixture_id == fixture["fixture_id"],
            TrackedPrediction.prediction_type == prediction_type,
        ).first()
        if existing:
            return "duplicate"
        orig_prob = prob_map.get(prediction_type)
        row = TrackedPrediction(
            fixture_id=fixture["fixture_id"],
            league=league,
            home_team=fixture["home_team"],
            away_team=fixture["away_team"],
            match_date=datetime.fromisoformat(fixture["date"]),
            prediction_type=prediction_type,
            tracked_prob=orig_prob,
            model_prob=orig_prob,
        )
        db.add(row)
        db.commit()
    return "ok"


# ── App layout ─────────────────────────────────────────────────────────────────

Base.metadata.create_all(engine)

st.set_page_config(page_title="Football Tracker", page_icon="⚽", layout="wide")

# ── Mobile-first CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Tighter padding — more usable width on small screens */
.block-container {
    padding-left: 1rem !important;
    padding-right: 1rem !important;
    padding-top: 1rem !important;
    max-width: 100% !important;
}
/* Touch targets: Apple HIG ≥ 44px */
button[kind="secondary"], button[kind="primary"], [data-testid="baseButton-secondary"] {
    min-height: 44px !important;
}
/* Selectbox touch-friendly height */
[data-testid="stSelectbox"] > div > div {
    min-height: 44px !important;
}
/* Tabs */
[data-testid="stTabs"] [role="tab"] {
    padding-left: 0.75rem !important;
    padding-right: 0.75rem !important;
}
/* Mobile: compact tabs and metrics */
@media (max-width: 768px) {
    [data-testid="stTabs"] [role="tab"] {
        padding-left: 0.4rem !important;
        padding-right: 0.4rem !important;
    }
    [data-testid="stTabs"] [role="tab"] p {
        font-size: 12px !important;
        white-space: nowrap;
    }
    [data-testid="stMetricValue"] { font-size: 1.3rem !important; }
    [data-testid="stMetricLabel"] { font-size: 0.75rem !important; }
    [data-testid="stHorizontalBlock"] > div { min-width: 130px; }
}
/* Dataframe horizontal scroll on mobile */
[data-testid="stDataFrame"] > div { overflow-x: auto !important; }
</style>
""", unsafe_allow_html=True)

st.title("⚽ Football Tracker")

tab_pred, tab_tracked, tab_results, tab_stats = st.tabs(["📅 Predikce", "📋 Sledované", "🔍 Výsledky", "📊 Statistiky"])


# ── TAB 1: Predikce ────────────────────────────────────────────────────────────

with tab_pred:
    st.subheader("Nadcházející zápasy")

    league = st.selectbox(
        "Liga",
        options=league_keys,
        format_func=lambda k: leagues_display[k],
        key="pred_league",
    )

    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        refresh = st.button("Obnovit", key="load_pred")

    if refresh or st.session_state.get("upcoming_league") != league:
        st.session_state.pop("upcoming_data", None)

    if "upcoming_data" not in st.session_state:
        # Primary: load from DB (pre-computed by GitHub Actions)
        with get_db() as db:
            rows = (
                db.query(FixturePrediction)
                .filter(FixturePrediction.league == league)
                .order_by(FixturePrediction.match_date)
                .all()
            )

        if rows:
            computed_at = rows[0].computed_at
            data = [{
                "fixture_id": r.fixture_id,
                "date": r.match_date.isoformat(),
                "home_team": r.home_team,
                "away_team": r.away_team,
                "home_logo": getattr(r, "home_logo", None) or "",
                "away_logo": getattr(r, "away_logo", None) or "",
                "prob_home": r.prob_home,
                "prob_draw": r.prob_draw,
                "prob_away": r.prob_away,
                "over2_5": r.over2_5,
                "under2_5": r.under2_5,
                "goals1_3": r.goals1_3,
                "goals2_4": r.goals2_4,
                "btts_yes": r.btts_yes,
                "btts_no": r.btts_no,
                "expected_goals_home": getattr(r, "expected_goals_home", None),
                "expected_goals_away": getattr(r, "expected_goals_away", None),
            } for r in rows]
            st.session_state["upcoming_data"] = data
            st.session_state["upcoming_league"] = league
            if computed_at:
                st.caption(f"Naposledy přepočítáno: {computed_at.strftime('%d.%m.%Y %H:%M')} UTC")
        else:
            # Fallback: compute live (DB not yet populated)
            st.info("DB není naplněna — počítám live (GitHub Actions ještě neproběhly).")
            with st.spinner("Načítám predikce..."):
                model = get_model(league)
                if model is None:
                    st.warning("Model není dostupný — nedostatek dat pro tuto ligu.")
                    st.session_state["upcoming_data"] = []
                else:
                    fetcher = get_fetcher()
                    cfg = settings.leagues[league]
                    fixtures = fetcher.get_upcoming_fixtures(cfg, next_n=10)
                    data = []
                    for fx in fixtures:
                        pred = model.predict(fx)
                        gp = compute_goal_probs(pred.expected_goals_home or 1.3, pred.expected_goals_away or 1.0)
                        data.append({
                            "fixture_id": fx.id,
                            "date": fx.date.isoformat(),
                            "home_team": fx.home_team.name,
                            "away_team": fx.away_team.name,
                            "prob_home": round(pred.prob_home, 4),
                            "prob_draw": round(pred.prob_draw, 4),
                            "prob_away": round(pred.prob_away, 4),
                            **gp,
                        })
                    st.session_state["upcoming_data"] = data
                    st.session_state["upcoming_league"] = league

    fixtures = st.session_state.get("upcoming_data", [])

    if not fixtures:
        st.info("Žádné nadcházející zápasy.")
    else:
        # ── Přehledová tabulka (HTML — responsivní: PC=logo+název, mobil=jen logo) ──
        def _pct_cell(val: float) -> str:
            s = "background:#1a6e3c;color:#fff;font-weight:700;" if val >= 65 else ""
            return f'<td style="{s}">{val:.1f}</td>'

        rows_html = []
        for f in fixtures:
            h_logo = f.get("home_logo") or ""
            a_logo = f.get("away_logo") or ""
            h_img = f'<img src="{h_logo}" width="22" height="22" style="vertical-align:middle">' if h_logo else "🏠"
            a_img = f'<img src="{a_logo}" width="22" height="22" style="vertical-align:middle">' if a_logo else "✈️"
            date = f["date"][5:16].replace("T", " ")
            rows_html.append(f"""<tr>
<td>{date}</td>
<td class="logo">{h_img}</td><td class="tname">{f["home_team"]}</td>
<td class="logo">{a_img}</td><td class="tname">{f["away_team"]}</td>
{_pct_cell(f["prob_home"]*100)}{_pct_cell(f["prob_draw"]*100)}{_pct_cell(f["prob_away"]*100)}
{_pct_cell(f.get("over2_5",0)*100)}{_pct_cell(f.get("btts_yes",0)*100)}
{_pct_cell(f.get("goals1_3",0)*100)}{_pct_cell(f.get("goals2_4",0)*100)}
</tr>""")

        st.markdown(f"""
<style>
.pt{{width:100%;border-collapse:collapse;font-size:14px}}
.pt th,.pt td{{padding:6px 8px;text-align:center;border-bottom:1px solid #2a2a2a;white-space:nowrap}}
.pt th{{color:#aaa;font-size:11px;font-weight:600;text-transform:uppercase}}
.pt td.logo{{width:30px;padding:4px 4px}}
.pt td.tname{{text-align:left;max-width:140px;overflow:hidden;text-overflow:ellipsis}}
@media(max-width:768px){{
  .pt .tname{{display:none}}
  .pt{{font-size:12px}}
  .pt th,.pt td{{padding:5px 5px}}
}}
</style>
<div style="overflow-x:auto">
<table class="pt"><thead><tr>
<th>Datum</th>
<th class="logo"></th><th class="tname">Domácí</th>
<th class="logo"></th><th class="tname">Hosté</th>
<th>H%</th><th>D%</th><th>A%</th>
<th>O2.5</th><th>BTTS</th><th>G1-3</th><th>G2-4</th>
</tr></thead><tbody>{"".join(rows_html)}</tbody></table>
</div>""", unsafe_allow_html=True)

        # ── Detailní analýza (expandery) ────────────────────────────────────
        st.subheader("Detailní analýza")
        with st.spinner("Načítám statistiky a zranění..."):
            features_by_id, league_avg = get_league_features(league)
            injuries_by_id = get_league_injuries(league)

        for fx in fixtures:
            home = fx["home_team"]
            away = fx["away_team"]
            date_str = fx["date"][:16].replace("T", " ")
            inj_data = injuries_by_id.get(fx["fixture_id"], {})
            has_injuries = bool(inj_data.get("home") or inj_data.get("away"))
            label = f"📅 {date_str}  ·  {home} vs {away}" + ("  🚑" if has_injuries else "")
            with st.expander(label):
                feats = features_by_id.get(fx["fixture_id"], {})
                render_match_detail(fx, feats, league_avg)

                render_injuries(
                    home, away,
                    inj_data.get("home", []),
                    inj_data.get("away", []),
                    inj_data.get("home_goals", 50),
                    inj_data.get("away_goals", 50),
                )

                st.divider()
                tr_col1, tr_col2 = st.columns([3, 1])
                with tr_col1:
                    track_type = st.selectbox(
                        "Typ predikce ke sledování",
                        PREDICTION_TYPES,
                        key=f"track_type_{fx['fixture_id']}",
                    )
                with tr_col2:
                    st.write("")
                    if st.button("📌 Sledovat", key=f"track_btn_{fx['fixture_id']}"):
                        result = save_tracking(fx, league, track_type, None)
                        if result == "ok":
                            st.success("Přidáno ke sledování.")
                        elif result == "duplicate":
                            st.warning("Již sledováno.")
                        else:
                            st.error("Chyba při ukládání.")


# ── TAB 2: Sledované predikce ──────────────────────────────────────────────────

with tab_tracked:
    st.subheader("Sledované predikce")

    col_filter1, col_filter2 = st.columns(2)
    with col_filter1:
        filter_league = st.selectbox(
            "Liga",
            options=["Všechny"] + league_keys,
            format_func=lambda k: "Všechny" if k == "Všechny" else leagues_display[k],
            key="track_league_filter",
        )
    with col_filter2:
        filter_status = st.selectbox(
            "Stav",
            options=["Všechny", "Čekající", "Vyřešené"],
            key="track_status_filter",
        )

    if st.button("Resolve výsledků", key="resolve_btn"):
        with st.spinner("Resolvuji..."):
            count = run_resolve()
            st.success(f"Vyřešeno: {count} predikcí")

    with get_db() as db:
        q = db.query(TrackedPrediction)
        if filter_league != "Všechny":
            q = q.filter(TrackedPrediction.league == filter_league)
        if filter_status == "Čekající":
            q = q.filter(TrackedPrediction.correct.is_(None))
        elif filter_status == "Vyřešené":
            q = q.filter(TrackedPrediction.correct.isnot(None))
        rows = q.order_by(TrackedPrediction.match_date.desc()).all()

    if not rows:
        st.info("Žádné sledované predikce.")
    else:
        table_data = []
        for r in rows:
            if r.correct is True:
                status_icon = "✅"
            elif r.correct is False:
                status_icon = "❌"
            else:
                status_icon = "⏳"
            score = f"{r.home_score}-{r.away_score}" if r.home_score is not None else "—"
            prob_str = f"{r.model_prob*100:.1f}%" if r.model_prob is not None else "—"

            # Probability drift: tracked_prob (at time of adding) vs model_prob (latest recalc)
            tracked = getattr(r, "tracked_prob", None)
            current = r.model_prob
            if r.correct is None and tracked is not None and current is not None:
                delta = (current - tracked) * 100
                if abs(delta) >= 1:
                    icon = "🟢" if delta >= 3 else ("🔴" if delta <= -3 else "🟡")
                    drift = f"{tracked*100:.1f}→{current*100:.1f} ({delta:+.1f}pp) {icon}"
                else:
                    drift = f"{current*100:.1f}% →"
            else:
                drift = "—"

            table_data.append({
                "Datum": r.match_date.strftime("%d.%m.%Y %H:%M"),
                "Liga": leagues_display.get(r.league, r.league),
                "Zápas": f"{r.home_team} vs {r.away_team}",
                "Typ": r.prediction_type,
                "Přidáno": f"{tracked*100:.1f}%" if tracked is not None else prob_str,
                "Vývoj": drift,
                "Skóre": score,
                "✓": status_icon,
            })
        st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)


# ── TAB 3: Výsledky ────────────────────────────────────────────────────────────

with tab_results:
    st.subheader("Výsledky posledního kola")
    st.caption("Archiv predikcí odehraných zápasů s předzápasovými statistikami. Data se mažou po 10 dnech.")

    res_league = st.selectbox(
        "Liga",
        options=["Všechny"] + league_keys,
        format_func=lambda k: "Všechny" if k == "Všechny" else leagues_display[k],
        key="results_league",
    )

    with get_db() as db:
        q = db.query(ResolvedFixturePrediction)
        if res_league != "Všechny":
            q = q.filter(ResolvedFixturePrediction.league == res_league)
        resolved_rows = q.order_by(ResolvedFixturePrediction.match_date.desc()).all()

    if not resolved_rows:
        st.info("Žádné archivované výsledky. Data se plní automaticky při každém denním běhu predikcí.")
    else:
        # Overview table
        prob_cols_r = ["P(H)%", "P(D)%", "P(A)%"]
        overview = pd.DataFrame([{
            "Datum": r.match_date.strftime("%d.%m %H:%M"),
            "Liga": leagues_display.get(r.league, r.league),
            "Domácí": r.home_team,
            "Hosté": r.away_team,
            "Výsledek": f"{r.home_score}–{r.away_score}",
            "P(H)%": round(r.prob_home * 100, 1),
            "P(D)%": round(r.prob_draw * 100, 1),
            "P(A)%": round(r.prob_away * 100, 1),
            "Tip modelu": r.predicted_outcome,
            "Správně": "✅" if r.correct else "❌",
        } for r in resolved_rows])

        def highlight_correct(row):
            color = "#1a6e3c" if row["Správně"] == "✅" else "#6e1a1a"
            return [f"background-color: {color}; color: white" if col == "Správně" else "" for col in row.index]

        styled_r = (
            overview.style
            .apply(highlight_correct, axis=1)
            .format({col: "{:.1f}" for col in prob_cols_r})
        )
        st.dataframe(styled_r, use_container_width=True, hide_index=True)

        # Accuracy summary
        total_r = len(resolved_rows)
        correct_r = sum(1 for r in resolved_rows if r.correct)
        st.caption(f"Úspěšnost modelu (1X2): **{correct_r}/{total_r}** = {correct_r/total_r*100:.1f}%")

        st.subheader("Detailní analýza")
        # Group league_avg per league for feature display
        league_avgs_cache: dict = {}

        for r in resolved_rows:
            date_str = r.match_date.strftime("%d.%m.%Y %H:%M")
            result_icon = "✅" if r.correct else "❌"
            header = (
                f"{result_icon}  {date_str}  ·  {r.home_team} {r.home_score}–{r.away_score} {r.away_team}"
                f"  ·  tip: {r.predicted_outcome}  ·  skutečnost: {r.actual_outcome}"
            )
            with st.expander(header):
                # Reconstruct fx_data dict compatible with render_match_detail
                fx_data = {
                    "home_team": r.home_team,
                    "away_team": r.away_team,
                    "prob_home": r.prob_home,
                    "prob_draw": r.prob_draw,
                    "prob_away": r.prob_away,
                    "over2_5": r.over2_5,
                    "under2_5": r.under2_5,
                    "goals1_3": r.goals1_3,
                    "goals2_4": r.goals2_4,
                    "btts_yes": r.btts_yes,
                    "btts_no": r.btts_no,
                }
                feats = json.loads(r.features_json) if r.features_json else {}
                # Lazy-load league avg per league (cached in dict for this page run)
                if r.league not in league_avgs_cache:
                    _, league_avgs_cache[r.league] = get_league_features(r.league)
                render_match_detail(fx_data, feats, league_avgs_cache.get(r.league, {}))


# ── TAB 4: Statistiky ──────────────────────────────────────────────────────────

with tab_stats:
    st.subheader("Statistiky")

    with get_db() as db:
        all_rows = db.query(TrackedPrediction).filter(
            TrackedPrediction.correct.isnot(None)
        ).all()

    if not all_rows:
        st.info("Žádné vyřešené predikce pro statistiky.")
    else:
        total = len(all_rows)
        total_correct = sum(1 for r in all_rows if r.correct)
        overall_pct = total_correct / total * 100 if total else 0

        st.metric("Celková úspěšnost", f"{overall_pct:.1f}%", f"{total_correct}/{total} správně")
        st.divider()

        by_type: dict[str, dict] = {}
        by_league: dict[str, dict] = {}

        for r in all_rows:
            for key, bucket in [(r.prediction_type, by_type), (r.league, by_league)]:
                if key not in bucket:
                    bucket[key] = {"count": 0, "correct": 0}
                bucket[key]["count"] += 1
                if r.correct:
                    bucket[key]["correct"] += 1

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Podle typu predikce")
            type_rows = [
                {
                    "Typ": k,
                    "Počet": v["count"],
                    "Správně": v["correct"],
                    "Úspěšnost %": round(v["correct"] / v["count"] * 100, 1),
                }
                for k, v in by_type.items()
            ]
            df_type = pd.DataFrame(type_rows).sort_values("Úspěšnost %", ascending=False)
            st.dataframe(df_type, use_container_width=True, hide_index=True)
            if not df_type.empty:
                st.bar_chart(df_type.set_index("Typ")["Úspěšnost %"])

        with col2:
            st.subheader("Podle ligy")
            league_rows = [
                {
                    "Liga": leagues_display.get(k, k),
                    "Počet": v["count"],
                    "Správně": v["correct"],
                    "Úspěšnost %": round(v["correct"] / v["count"] * 100, 1),
                }
                for k, v in by_league.items()
            ]
            df_league = pd.DataFrame(league_rows).sort_values("Úspěšnost %", ascending=False)
            st.dataframe(df_league, use_container_width=True, hide_index=True)
            if not df_league.empty:
                st.bar_chart(df_league.set_index("Liga")["Úspěšnost %"])

        # ── Reliability diagram from resolved_fixture_predictions ─────────────
        st.divider()
        st.subheader("Kalibrace modelu")
        st.caption(
            "Reliability diagram: pokud je model dobře zkalibrován, měly by body ležet na diagonále. "
            "Data z posledních 10 dní (resolved_fixture_predictions)."
        )

        with get_db() as db:
            cal_rows = db.query(ResolvedFixturePrediction).all()

        if len(cal_rows) < 20:
            st.info("Nedostatek dat pro reliability diagram (potřeba ≥ 20 odehraných zápasů v archivu).")
        else:
            n_bins = 5
            edges = [i / n_bins for i in range(n_bins + 1)]

            bin_data = []
            for outcome, prob_attr, label in [
                ("H", "prob_home", "Výhra domácích"),
                ("D", "prob_draw", "Remíza"),
                ("A", "prob_away", "Výhra hostů"),
            ]:
                probs  = [getattr(r, prob_attr) for r in cal_rows]
                labels = [1 if r.actual_outcome == outcome else 0 for r in cal_rows]
                for i in range(n_bins):
                    lo, hi = edges[i], edges[i + 1]
                    idxs = [j for j, p in enumerate(probs) if lo <= p < hi]
                    if len(idxs) < 3:
                        continue
                    mean_pred  = sum(probs[j] for j in idxs) / len(idxs)
                    actual_freq = sum(labels[j] for j in idxs) / len(idxs)
                    bin_data.append({
                        "Outcome": label,
                        "Predicted %": round(mean_pred * 100, 1),
                        "Actual %": round(actual_freq * 100, 1),
                        "N": len(idxs),
                    })

            if bin_data:
                df_cal = pd.DataFrame(bin_data)
                # One chart per outcome
                cal_col1, cal_col2, cal_col3 = st.columns(3)
                for col_widget, outcome_label in zip(
                    [cal_col1, cal_col2, cal_col3],
                    ["Výhra domácích", "Remíza", "Výhra hostů"],
                ):
                    subset = df_cal[df_cal["Outcome"] == outcome_label].copy()
                    if subset.empty:
                        continue
                    with col_widget:
                        st.caption(f"**{outcome_label}**")
                        subset = subset.set_index("Predicted %")[["Actual %"]].sort_index()
                        # Add perfect calibration reference
                        perfect = pd.DataFrame(
                            {"Actual %": subset.index.tolist()},
                            index=subset.index,
                        )
                        combined = subset.rename(columns={"Actual %": "Model"})
                        combined["Ideál"] = combined.index
                        st.line_chart(combined, use_container_width=True)
            else:
                st.info("Nedostatek dat v jednotlivých binech.")
