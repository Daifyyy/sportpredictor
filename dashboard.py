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
PREDICTION_TYPES = [
    "H", "D", "A", "Under2.5", "Over2.5", "Goals1-3", "Goals2-4", "BTTS_Yes", "BTTS_No",
    "Corners_Over8.5", "Corners_Under8.5",
    "Corners_Over9.5", "Corners_Under9.5",
    "Corners_Over10.5", "Corners_Under10.5",
    "Corners_Over11.5", "Corners_Under11.5",
]
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
            home_inj, away_inj, home_goals, away_goals, home_goals_against, away_goals_against = (
                fetcher.get_fixture_injuries(fx, cfg.id, cfg.season)
            )
        except Exception:
            home_inj, away_inj, home_goals, away_goals, home_goals_against, away_goals_against = [], [], 50, 50, 50, 50
        # Deduplicate by player_id (API-Football may return the same player multiple times)
        result[fx.id] = {
            "home": list({i.player_id: asdict(i) for i in home_inj}.values()),
            "away": list({i.player_id: asdict(i) for i in away_inj}.values()),
            "home_goals": home_goals,
            "away_goals": away_goals,
            "home_goals_against": home_goals_against,
            "away_goals_against": away_goals_against,
        }
    return result


@st.cache_data(ttl=1800, show_spinner=False)
def get_league_lineups(league_key: str) -> dict:
    """Returns {fixture_id: {home: dict|None, away: dict|None}} for all upcoming fixtures.

    lineup dict: {formation, coach, starters: [{player_id, player_name, number, pos}], substitutes: [...]}
    TTL=30min — lineups announced ~1h before kickoff, empty before that.
    """
    from dataclasses import asdict

    fetcher = get_fetcher()
    cfg = settings.leagues[league_key]
    upcoming = fetcher.get_upcoming_fixtures(cfg, next_n=10)
    result = {}
    for fx in upcoming:
        try:
            home_lu, away_lu = fetcher.get_fixture_lineups(fx)
        except Exception:
            home_lu, away_lu = None, None
        result[fx.id] = {
            "home": asdict(home_lu) if home_lu is not None else None,
            "away": asdict(away_lu) if away_lu is not None else None,
        }
    return result


@st.cache_data(ttl=86400, show_spinner="Načítám tabulku...")
def fetch_standings(league_key: str) -> list:
    """Returns list of standings groups from API. TTL=24h — standings change only after a matchday."""
    fetcher = get_fetcher()
    cfg = settings.leagues[league_key]
    return fetcher.get_standings(cfg)


@st.cache_data(ttl=3600, show_spinner=False)
def get_prob_changes(min_delta: float = 0.02) -> list:
    """Returns list of fixtures with significant H/D/A probability changes since last predict run.

    Only includes rows where prev_prob_home IS NOT NULL (populated after first run post-ALTER TABLE).
    Sorted by max absolute delta descending. min_delta=0.02 = 2 percentage points.
    """
    try:
        with get_db() as db:
            rows = (
                db.query(FixturePrediction)
                .filter(FixturePrediction.prev_prob_home.isnot(None))
                .order_by(FixturePrediction.match_date)
                .all()
            )
    except Exception:
        return []

    result = []
    for r in rows:
        dh = r.prob_home - r.prev_prob_home
        dd = r.prob_draw - r.prev_prob_draw
        da = r.prob_away - r.prev_prob_away
        max_delta = max(abs(dh), abs(dd), abs(da))
        if max_delta < min_delta:
            continue
        result.append({
            "fixture_id": r.fixture_id,
            "league": r.league,
            "home_team": r.home_team,
            "away_team": r.away_team,
            "match_date": r.match_date,
            "prob_home": r.prob_home,
            "prob_draw": r.prob_draw,
            "prob_away": r.prob_away,
            "prev_prob_home": r.prev_prob_home,
            "prev_prob_draw": r.prev_prob_draw,
            "prev_prob_away": r.prev_prob_away,
            "delta_home": dh,
            "delta_draw": dd,
            "delta_away": da,
            "max_delta": max_delta,
            "prev_computed_at": r.prev_computed_at,
        })

    return sorted(result, key=lambda x: x["max_delta"], reverse=True)


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

    # ── Expected goals (λ/μ) ───────────────────────────────────────────────────
    lam = fx_data.get("expected_goals_home")
    mu  = fx_data.get("expected_goals_away")
    if lam is not None and mu is not None:
        eg_col1, eg_col2, eg_col3 = st.columns(3)
        eg_col1.metric(f"λ  ({home})", f"{lam:.2f}", help="Očekávaný počet gólů domácích (model)")
        eg_col2.metric("Σ gólů", f"{lam + mu:.2f}", help="λ + μ = celkový očekávaný počet gólů v zápase")
        eg_col3.metric(f"μ  ({away})", f"{mu:.2f}", help="Očekávaný počet gólů hostů (model)")

    # ── Comparison table ───────────────────────────────────────────────────────
    def _val(key: str) -> float | None:
        v = feats.get(key)
        return v if isinstance(v, (int, float)) else None

    def _row(label: str, hv: float | None, av: float | None,
             fmt: str = ".2f", higher_is_better: bool = True,
             avg: float | None = None, neutral: bool = False) -> str:
        hs = f"{hv:{fmt}}" if hv is not None else "—"
        as_ = f"{av:{fmt}}" if av is not None else "—"
        avg_s = (f" <span style='color:#555;font-size:10px'>ø{avg:.2f}</span>"
                 if avg is not None else "")
        h_style = a_style = "color:#ccc"
        if not neutral and hv is not None and av is not None and abs(hv - av) > 0.02:
            h_better = (hv > av) == higher_is_better
            h_style = "color:#4ade80;font-weight:600" if h_better else "color:#f87171"
            a_style = "color:#f87171" if h_better else "color:#4ade80;font-weight:600"
        return (
            f'<tr>'
            f'<td style="text-align:right;padding:4px 10px;{h_style}">{hs}</td>'
            f'<td style="text-align:center;padding:4px 6px;color:#777;font-size:12px;white-space:nowrap">'
            f'{label}{avg_s}</td>'
            f'<td style="padding:4px 10px;{a_style}">{as_}</td>'
            f'</tr>'
        )

    def _sec(title: str) -> str:
        return (
            f'<tr><td colspan="3" style="padding:10px 10px 3px;color:#888;font-size:11px;'
            f'text-transform:uppercase;letter-spacing:0.06em;border-top:1px solid #2a2a2a">'
            f'{title}</td></tr>'
        )

    streak_h = feats.get("home_streak", 0)
    streak_a = feats.get("away_streak", 0)
    sh = streak_h if isinstance(streak_h, (int, float)) else 0
    sa = streak_a if isinstance(streak_a, (int, float)) else 0

    rows: list[str] = [
        f'<tr>'
        f'<th style="text-align:right;padding:6px 10px;color:#e2e8f0;font-size:13px">🏠 {home}</th>'
        f'<th style="padding:6px 6px"></th>'
        f'<th style="padding:6px 10px;color:#e2e8f0;font-size:13px">✈️ {away}</th>'
        f'</tr>',

        _sec("Forma — posl. 5 zápasů"),
        _row("Pts/zápas", _val("home_form"), _val("away_form"), avg=avg_pts),
        _row("GF/zápas", _val("home_gf"), _val("away_gf"), avg=avg_gf),
        _row("GA/zápas", _val("home_ga"), _val("away_ga"), avg=avg_gf, higher_is_better=False),
        _row("Útočná síla", _val("home_attack_str"), _val("away_attack_str"), avg=1.0),
        _row("Obranná síla", _val("home_defense_str"), _val("away_defense_str"), avg=1.0, higher_is_better=False),

        _sec("Domácí / Venkovní forma — posl. 5"),
        _row("Pts/zápas", _val("home_venue_form"), _val("away_venue_form"), avg=avg_pts),
        _row("GF/zápas", _val("home_venue_gf"), _val("away_venue_gf"), avg=avg_gf),
        _row("GA/zápas", _val("home_venue_ga"), _val("away_venue_ga"), avg=avg_gf, higher_is_better=False),

        _sec("Sezóna · Trend · Elo"),
        _row("PPG sezóna", _val("home_season_ppg"), _val("away_season_ppg"), avg=avg_pts),
        _row("Forma posl. 3", _val("home_form_short"), _val("away_form_short")),
        _row("Trend (krátká−dlouhá)", _val("home_trend"), _val("away_trend"), fmt="+.2f"),
        _row("Série (W=+, L=−)", sh, sa, fmt="+.0f"),
        _row("Elo rating", _val("elo_home") or 1500, _val("elo_away") or 1500, fmt=".0f", avg=1500),
        _row("Odpočinek (dny)", _val("home_rest_days"), _val("away_rest_days"), fmt=".0f"),

        _sec("Head to Head — posl. 10 vzájemných"),
        _row("Výhry %", round(feats.get("h2h_home_wins", 0.5) * 100),
             round(feats.get("h2h_away_wins", 0.17) * 100), fmt=".0f"),
        _row("Remízy %", round(feats.get("h2h_draws", 0.33) * 100),
             round(feats.get("h2h_draws", 0.33) * 100), fmt=".0f", neutral=True),
        _row("Avg GF", _val("h2h_home_gf"), _val("h2h_away_gf"), avg=avg_gf),
    ]

    has_shots = feats.get("home_avg_shots_on_target") is not None
    has_xg    = feats.get("home_avg_xg") is not None
    if has_shots or has_xg:
        rows.append(_sec("Statistiky — průměr posl. 5 zápasů"))
        if has_shots:
            rows += [
                _row("Střely na bránu", _val("home_avg_shots_on_target"), _val("away_avg_shots_on_target")),
                _row("Celkové střely", _val("home_avg_total_shots"), _val("away_avg_total_shots")),
                _row("Rohy", _val("home_avg_corners"), _val("away_avg_corners")),
            ]
        if has_xg:
            rows.append(_row("xG/zápas", _val("home_avg_xg"), _val("away_avg_xg")))

    st.markdown(
        f'<div style="overflow-x:auto">'
        f'<table style="width:100%;border-collapse:collapse;font-size:14px">'
        f'{"".join(rows)}'
        f'</table></div>',
        unsafe_allow_html=True,
    )


def render_bet_validation(fx_data: dict, feats: dict) -> None:
    """Render 4-signal validation for Goals 1-3 and Home Win bets."""
    prob_home = fx_data.get("prob_home") or 0
    goals1_3  = fx_data.get("goals1_3") or 0
    lam = fx_data.get("expected_goals_home")
    mu  = fx_data.get("expected_goals_away")

    sections = []

    # ── Goals 1-3 ─────────────────────────────────────────────────────────────
    if goals1_3 >= 0.40:
        lam_mu     = (lam + mu) if (lam is not None and mu is not None) else None
        home_total = (feats.get("home_gf") or 0) + (feats.get("home_ga") or 0)
        away_total = (feats.get("away_gf") or 0) + (feats.get("away_ga") or 0)
        h2h_total  = (feats.get("h2h_home_gf") or 0) + (feats.get("h2h_away_gf") or 0)

        def _g(val, g, y):   # lower is better
            if val is None: return "⚪"
            return "🟢" if val <= g else ("🟡" if val <= y else "🔴")

        sections.append((
            f"Goals 1–3  ({goals1_3 * 100:.0f}%)",
            [
                ("λ+μ (model xG celkem)",      f"{lam_mu:.2f}" if lam_mu else "—",    _g(lam_mu, 2.3, 2.8)),
                ("Domácí avg gólů/zápas",       f"{home_total:.2f}",                   _g(home_total, 2.5, 3.0)),
                ("Hosté avg gólů/zápas",         f"{away_total:.2f}",                   _g(away_total, 2.5, 3.0)),
                ("H2H avg gólů celkem",          f"{h2h_total:.2f}",                    _g(h2h_total, 2.5, 3.0)),
            ],
        ))

    # ── Home Win ──────────────────────────────────────────────────────────────
    prob_draw = fx_data.get("prob_draw") or 0
    prob_away = fx_data.get("prob_away") or 0
    home_is_favorite = prob_home >= prob_draw and prob_home >= prob_away
    if prob_home >= 0.40 or home_is_favorite:
        ratio      = (lam / mu) if (lam is not None and mu is not None and mu > 0) else None
        venue_form = feats.get("home_venue_form")
        h2h_hw     = feats.get("h2h_home_wins")
        elo_diff   = feats.get("elo_diff")

        def _h(val, g, y):   # higher is better
            if val is None: return "⚪"
            return "🟢" if val >= g else ("🟡" if val >= y else "🔴")

        sections.append((
            f"Výhra domácích  ({prob_home * 100:.0f}%)",
            [
                ("λ/μ poměr (model)",           f"{ratio:.2f}" if ratio else "—",                 _h(ratio, 1.4, 1.1)),
                ("Domácí forma DOMA (pts/z)",    f"{venue_form:.2f}" if venue_form is not None else "—",  _h(venue_form, 2.0, 1.2)),
                ("H2H výhry domácích %",         f"{h2h_hw * 100:.0f}%" if h2h_hw is not None else "—", _h(h2h_hw, 0.50, 0.33)),
                ("Elo rozdíl (doma − hosté)",    f"{elo_diff:+.0f}" if elo_diff is not None else "—",   _h(elo_diff, 50, 0)),
            ],
        ))

    if not sections:
        return

    st.caption("📊 Validace sázky")
    for title, signals in sections:
        green = sum(1 for *_, icon in signals if icon == "🟢")
        color = "#4ade80" if green >= 3 else ("#facc15" if green >= 2 else "#f87171")
        st.markdown(
            f'<div style="margin:6px 0 2px;font-size:13px;font-weight:600;color:{color}">'
            f'{title} — {green}/4 signálů zelených</div>',
            unsafe_allow_html=True,
        )
        rows_html = "".join(
            f'<tr>'
            f'<td style="padding:2px 6px;font-size:14px">{icon}</td>'
            f'<td style="padding:2px 8px;color:#aaa;font-size:12px">{name}</td>'
            f'<td style="padding:2px 8px;color:#e2e8f0;font-size:12px;font-weight:600;text-align:right">{val}</td>'
            f'</tr>'
            for name, val, icon in signals
        )
        st.markdown(
            f'<table style="width:100%;border-collapse:collapse;margin-bottom:4px">{rows_html}</table>',
            unsafe_allow_html=True,
        )


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
    home_goals_against: int = 50, away_goals_against: int = 50,
) -> None:
    """Render injured/suspended player lists with per-player impact estimate."""
    if not home_injuries and not away_injuries:
        return

    st.caption("🚑 Zranění a absence")

    def fmt_player(inj: dict, team_goals: int, team_goals_against: int) -> str:
        from data.models import PlayerInjury
        icon = _STATUS_ICON.get(inj["status"], "🔴")
        pos  = _POS_CZ.get(inj["position"], inj["position"])
        atk, dfn = _injury_adjuster.player_impact(PlayerInjury(**inj), team_goals, team_goals_against)
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
                st.markdown(fmt_player(inj, home_goals, home_goals_against))
        else:
            st.caption("Žádná hlášená zranění")
    with col_a:
        st.markdown(f"**✈️ {away_name}**")
        if away_injuries:
            for inj in away_injuries:
                st.markdown(fmt_player(inj, away_goals, away_goals_against))
        else:
            st.caption("Žádná hlášená zranění")


def render_referee(feats: dict) -> None:
    """Display referee historical goal stats as a neutral info badge."""
    name = feats.get("referee_name")
    if not name:
        return
    n = feats.get("referee_n_games", 0)
    avg_goals = feats.get("referee_avg_goals")
    factor = feats.get("referee_goal_factor", 1.0)

    if avg_goals is None or n < 5:
        st.caption(f"🧑‍⚖️ Rozhodčí: **{name}** &nbsp;·&nbsp; nedostatek dat ({n} zápasů)")
        return

    pct = (factor - 1.0) * 100
    if pct > 7:
        badge = "🔴 výrazně více gólů"
    elif pct > 3:
        badge = "🟡 mírně více gólů"
    elif pct < -7:
        badge = "🔵 výrazně méně gólů"
    elif pct < -3:
        badge = "🟡 mírně méně gólů"
    else:
        badge = "🟢 průměrný"

    sign = "+" if pct >= 0 else ""
    st.caption(
        f"🧑‍⚖️ Rozhodčí: **{name}** &nbsp;·&nbsp; "
        f"avg {avg_goals:.1f} gólů/zápas &nbsp;·&nbsp; "
        f"{n} zápasů v historii &nbsp;·&nbsp; "
        f"{sign}{pct:.0f}% vs průměr ligy &nbsp;·&nbsp; {badge}"
    )


_POS_ORDER = {"G": 0, "D": 1, "M": 2, "F": 3}
_POS_LABEL = {"G": "Brankář", "D": "Obránci", "M": "Záložníci", "F": "Útočníci"}


def render_lineups(
    home_name: str, away_name: str,
    home_lineup: dict | None, away_lineup: dict | None,
) -> None:
    """Render official starting lineups with formation and bench."""
    if home_lineup is None and away_lineup is None:
        return

    st.caption("⚽ Sestavy")

    def fmt_lineup(lu: dict | None, team_name: str) -> None:
        if lu is None:
            st.caption("Sestava ještě nebyla ohlášena")
            return

        formation = lu.get("formation") or "?"
        coach = lu.get("coach") or "?"
        st.markdown(f"**Rozestavení:** {formation}  ·  **Trenér:** {coach}")

        starters = sorted(lu.get("starters", []), key=lambda p: _POS_ORDER.get(p["pos"], 9))

        # Group by position
        current_pos = None
        for p in starters:
            pos = p.get("pos", "?")
            if pos != current_pos:
                current_pos = pos
                st.markdown(f"*{_POS_LABEL.get(pos, pos)}*")
            num = p.get("number") or ""
            st.markdown(f"&nbsp;&nbsp;{num}. {p['player_name']}")

        subs = lu.get("substitutes", [])
        if subs:
            st.markdown("*Náhradníci*")
            names = ", ".join(
                f"{p.get('number') or ''}. {p['player_name']}" for p in subs
            )
            st.caption(names)

    col_h, col_a = st.columns(2)
    with col_h:
        st.markdown(f"**🏠 {home_name}**")
        fmt_lineup(home_lineup, home_name)
    with col_a:
        st.markdown(f"**✈️ {away_name}**")
        fmt_lineup(away_lineup, away_name)


def render_prediction_stats(fx_data: dict, feats: dict) -> None:
    """Render predicted match statistics as a dual-bar visual comparison (similar to live match stats screens).

    Always shows: xG, goals/game, form pts/game, home/away form, attack strength, season PPG, Elo.
    Conditionally shows shots/corners if enrich_with_statistics data is available.
    """
    if not feats:
        return

    home = fx_data["home_team"]
    away = fx_data["away_team"]
    lam = fx_data.get("expected_goals_home")
    mu = fx_data.get("expected_goals_away")

    def row(label: str, h, a, fmt: str = "{:.1f}"):
        if h is None or a is None:
            return None
        try:
            h, a = float(h), float(a)
        except (TypeError, ValueError):
            return None
        return (label, h, a, fmt)

    rows = []
    if lam is not None and mu is not None:
        rows.append(("xG (očekávané góly)", float(lam), float(mu), "{:.2f}"))

    r = row("Góly vstřelené / zápas", feats.get("home_gf"), feats.get("away_gf"))
    if r: rows.append(r)

    r = row("Forma — body / zápas", feats.get("home_form"), feats.get("away_form"))
    if r: rows.append(r)

    r = row("Forma doma / venku", feats.get("home_venue_form"), feats.get("away_venue_form"))
    if r: rows.append(r)

    r = row("Útočná síla (index)", feats.get("home_attack_str"), feats.get("away_attack_str"), "{:.2f}")
    if r: rows.append(r)

    r = row("Sezónní PPG", feats.get("home_season_ppg"), feats.get("away_season_ppg"), "{:.2f}")
    if r: rows.append(r)

    # Elo: raw values are ~1300-1800, percentages still meaningful
    r = row("Elo rating", feats.get("elo_home"), feats.get("elo_away"), "{:.0f}")
    if r: rows.append(r)

    # Stats-enriched features — only present when enrich_with_statistics ran (GitHub Actions cache)
    r = row("Střely na bránu (prům.)", feats.get("home_avg_shots_on_target"), feats.get("away_avg_shots_on_target"))
    if r: rows.append(r)

    r = row("Střely celkem (prům.)", feats.get("home_avg_total_shots"), feats.get("away_avg_total_shots"))
    if r: rows.append(r)

    r = row("Rohy (prům.)", feats.get("home_avg_corners"), feats.get("away_avg_corners"))
    if r: rows.append(r)

    if not rows:
        return

    HC = "#00c9a7"   # teal — home
    AC = "#ff6b6b"   # coral — away

    html_rows = []
    for label, h_val, a_val, fmt in rows:
        total = h_val + a_val
        h_pct = (h_val / total * 100) if total > 0 else 50.0
        a_pct = 100.0 - h_pct
        h_str = fmt.format(h_val)
        a_str = fmt.format(a_val)
        html_rows.append(
            f'<div style="display:grid;grid-template-columns:58px 1fr 148px 1fr 58px;'
            f'align-items:center;padding:5px 0;border-bottom:1px solid #2d2d2d;">'
            f'<div style="text-align:right;font-weight:600;font-size:0.88rem;color:{HC};padding-right:6px;">{h_str}</div>'
            f'<div style="display:flex;justify-content:flex-end;">'
            f'<div style="width:{h_pct:.1f}%;height:7px;background:{HC};border-radius:3px 0 0 3px;min-width:3px;"></div>'
            f'</div>'
            f'<div style="text-align:center;font-size:0.77rem;color:#9a9a9a;padding:0 8px;">{label}</div>'
            f'<div>'
            f'<div style="width:{a_pct:.1f}%;height:7px;background:{AC};border-radius:0 3px 3px 0;min-width:3px;"></div>'
            f'</div>'
            f'<div style="text-align:left;font-weight:600;font-size:0.88rem;color:{AC};padding-left:6px;">{a_str}</div>'
            f'</div>'
        )

    header = (
        f'<div style="display:grid;grid-template-columns:58px 1fr 148px 1fr 58px;padding:6px 0 8px 0;">'
        f'<div></div>'
        f'<div style="text-align:right;font-size:0.85rem;font-weight:700;color:{HC};padding-right:6px;">🏠 {home}</div>'
        f'<div></div>'
        f'<div style="font-size:0.85rem;font-weight:700;color:{AC};padding-left:6px;">✈️ {away}</div>'
        f'<div></div>'
        f'</div>'
    )

    st.caption("📊 Predikce průběhu zápasu")
    st.markdown(
        f'<div style="margin:4px 0 12px 0;">{header}{"".join(html_rows)}</div>',
        unsafe_allow_html=True,
    )


def generate_ai_analysis(fx: dict, feats: dict, league_avg: dict, inj_data: dict) -> str:
    """Call Gemini API and return markdown analysis of the match."""
    try:
        from google import genai
    except ImportError:
        return "❌ Knihovna `google-genai` není nainstalována."

    api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "❌ Chybí `GEMINI_API_KEY` v st.secrets."

    client = genai.Client(api_key=api_key)

    home = fx["home_team"]
    away = fx["away_team"]
    avg_gf = league_avg.get("avg_gf", 1.4)
    avg_pts = league_avg.get("avg_pts", 1.2)

    def fv(key, default="N/A", mult=1, fmt=".2f"):
        v = feats.get(key)
        if isinstance(v, (int, float)):
            return f"{v * mult:{fmt}}"
        return str(default)

    lam = fx.get("expected_goals_home")
    mu  = fx.get("expected_goals_away")
    xg_line = (f"- Očekávané góly (model): {home} λ={lam:.2f}, {away} μ={mu:.2f}, celkem={lam+mu:.2f}"
               if lam and mu else "")

    home_inj = [i["player_name"] for i in inj_data.get("home", []) if i.get("status") == "Missing"]
    away_inj = [i["player_name"] for i in inj_data.get("away", []) if i.get("status") == "Missing"]
    inj_line = ""
    if home_inj:
        inj_line += f"- Zranění {home}: {', '.join(home_inj)}\n"
    if away_inj:
        inj_line += f"- Zranění {away}: {', '.join(away_inj)}\n"

    league_name = settings.leagues[fx["league"]].name if fx.get("league") and fx.get("league") in settings.leagues else ""
    h2h_home_wins = feats.get("h2h_home_wins")
    h2h_away_wins = feats.get("h2h_away_wins")
    h2h_draws     = feats.get("h2h_draws")
    h2h_line = (
        f"- Výhry {home}: {h2h_home_wins*100:.0f}%, remízy: {h2h_draws*100:.0f}%, výhry {away}: {h2h_away_wins*100:.0f}%"
        if None not in (h2h_home_wins, h2h_draws, h2h_away_wins)
        else "- H2H data nejsou k dispozici"
    )

    prompt = f"""Jsi fotbalový analytik. Analyzuj nadcházející zápas pouze na základě níže uvedených dat.
Nepřidávej žádné informace, které v datech nejsou. Piš česky, stručně a věcně.

## Zápas: {home} vs {away}{f" ({league_name})" if league_name else ""}

### Pravděpodobnosti modelu
- Výhra {home}: {fx.get("prob_home", 0)*100:.1f}%
- Remíza: {fx.get("prob_draw", 0)*100:.1f}%
- Výhra {away}: {fx.get("prob_away", 0)*100:.1f}%
- Over 2.5 gólu: {fx.get("over2_5", 0)*100:.1f}%
- BTTS: {fx.get("btts_yes", 0)*100:.1f}%
{xg_line}

### Forma (posl. 5 zápasů) | průměr ligy: {avg_pts:.2f} pts/z, {avg_gf:.2f} GF/z
- {home}: {fv("home_form")} pts/z, {fv("home_gf")} GF, {fv("home_ga")} GA, útočná síla {fv("home_attack_str")}, obranná síla {fv("home_defense_str")}
- {away}: {fv("away_form")} pts/z, {fv("away_gf")} GF, {fv("away_ga")} GA, útočná síla {fv("away_attack_str")}, obranná síla {fv("away_defense_str")}

### Domácí / Venkovní forma (posl. 5)
- {home} doma: {fv("home_venue_form")} pts/z, {fv("home_venue_gf")} GF, {fv("home_venue_ga")} GA
- {away} venku: {fv("away_venue_form")} pts/z, {fv("away_venue_gf")} GF, {fv("away_venue_ga")} GA

### Sezóna a Elo
- {home}: PPG {fv("home_season_ppg")}, trend {fv("home_trend", fmt="+.2f")}, série {feats.get("home_streak", 0):+.0f}, Elo {fv("elo_home", fmt=".0f")} (průměr 1500), odpočinek {feats.get("home_rest_days", 7):.0f} dní
- {away}: PPG {fv("away_season_ppg")}, trend {fv("away_trend", fmt="+.2f")}, série {feats.get("away_streak", 0):+.0f}, Elo {fv("elo_away", fmt=".0f")} (průměr 1500), odpočinek {feats.get("away_rest_days", 7):.0f} dní

### Head to Head (posl. 10 vzájemných zápasů)
{h2h_line}
- Průměrné góly: {home} {fv("h2h_home_gf", default="N/A")}, {away} {fv("h2h_away_gf", default="N/A")}

{("### Zranění (chybějící hráči)\n" + inj_line) if inj_line else ""}
---
Napiš analýzu v tomto formátu (přibližně 150–250 slov):
1. **Klíčové faktory** — 2–3 bullet pointy, co rozhodne zápas
2. **Silné/slabé stránky** — stručně pro každý tým (1–2 věty)
3. **Závěr** — celkové vyznění zápasu a na co si dát pozor
"""

    errors = []
    for model_name in ("gemini-2.5-flash", "gemini-2.5-pro"):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
            )
            return response.text
        except Exception as e:
            errors.append(f"**{model_name}**: {e}")
    return "❌ Gemini API selhalo:\n\n" + "\n\n".join(errors)


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
    # Corners markets cannot be resolved from goals alone — leave unresolved
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
        "Corners_Over8.5":   fixture.get("corners_over8_5"),
        "Corners_Under8.5":  fixture.get("corners_under8_5"),
        "Corners_Over9.5":   fixture.get("corners_over9_5"),
        "Corners_Under9.5":  fixture.get("corners_under9_5"),
        "Corners_Over10.5":  fixture.get("corners_over10_5"),
        "Corners_Under10.5": fixture.get("corners_under10_5"),
        "Corners_Over11.5":  fixture.get("corners_over11_5"),
        "Corners_Under11.5": fixture.get("corners_under11_5"),
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


# ── Sidebar: API status ─────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def fetch_api_status(api_key: str) -> dict | None:
    """Volá /status endpoint přímo (bez cache klienta) — TTL 5 min."""
    try:
        r = __import__("requests").get(
            "https://v3.football.api-sports.io/status",
            headers={"x-apisports-key": api_key},
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
        if data.get("errors"):
            return {"error": str(data["errors"])}
        return data.get("response", {})
    except Exception as e:
        return {"error": str(e)}


with st.sidebar:
    st.markdown("### API-Football")
    api_key = settings.api_key
    if not api_key or api_key == "YOUR_KEY_HERE":
        st.error("API klíč není nastaven (API_FOOTBALL_KEY).")
    else:
        status_data = fetch_api_status(api_key)
        if status_data is None or "error" in (status_data or {}):
            err = (status_data or {}).get("error", "Neznámá chyba")
            st.error(f"Chyba připojení: {err}")
        else:
            sub = status_data.get("subscription", {})
            reqs = status_data.get("requests", {})

            plan = sub.get("plan", "?")
            active = sub.get("active", False)
            end_date = sub.get("end", "?")

            used = reqs.get("current", "?")
            limit = reqs.get("limit_day", "?")
            remaining = reqs.get("remaining_day", None)

            # Stav plánu
            if active:
                st.success(f"Plán: **{plan}**")
            else:
                st.warning(f"Plán: **{plan}** — neaktivní (do {end_date})")

            # Requesty dnes
            if isinstance(used, int) and isinstance(limit, int):
                pct = used / limit if limit > 0 else 0
                st.progress(min(pct, 1.0), text=f"Requesty dnes: {used} / {limit}")
                if plan.lower() == "free" or limit <= 100:
                    st.caption(
                        "Free plán: limit 100 req/den. "
                        "Odds a zranění mohou být nedostupné."
                    )
            else:
                st.caption(f"Requesty dnes: {used} / {limit}")

            st.caption(f"Platnost: {end_date}  ·  Obnoveno před < 5 min")


tab_pred, tab_tracked, tab_results, tab_stats, tab_corners, tab_standings = st.tabs(["📅 Predikce", "📋 Sledované", "🔍 Výsledky", "📊 Statistiky", "🔄 Rohy", "🏆 Tabulka"])


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
                "expected_corners_home": getattr(r, "expected_corners_home", None),
                "expected_corners_away": getattr(r, "expected_corners_away", None),
                "corners_over8_5":   getattr(r, "corners_over8_5", None),
                "corners_under8_5":  getattr(r, "corners_under8_5", None),
                "corners_over9_5":   getattr(r, "corners_over9_5", None),
                "corners_under9_5":  getattr(r, "corners_under9_5", None),
                "corners_over10_5":  getattr(r, "corners_over10_5", None),
                "corners_under10_5": getattr(r, "corners_under10_5", None),
                "corners_over11_5":  getattr(r, "corners_over11_5", None),
                "corners_under11_5": getattr(r, "corners_under11_5", None),
                "league": league,
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

        # ── Rychlé sledování ────────────────────────────────────────────────
        st.subheader("📌 Rychlé sledování")
        _QUICK_BTNS = [
            ("H", "H"), ("D", "D"), ("A", "A"),
            ("O2.5", "Over2.5"), ("U2.5", "Under2.5"),
            ("G1-3", "Goals1-3"), ("G2-4", "Goals2-4"),
            ("B+", "BTTS_Yes"), ("B-", "BTTS_No"),
        ]
        for fx in fixtures:
            st.caption(f"{fx['home_team']} vs {fx['away_team']}")
            _cols = st.columns(9)
            for _qi, (_qlabel, _qtype) in enumerate(_QUICK_BTNS):
                with _cols[_qi]:
                    if st.button(_qlabel, key=f"qs_{_qtype}_{fx['fixture_id']}", use_container_width=True):
                        _qresult = save_tracking(fx, league, _qtype, None)
                        if _qresult == "ok":
                            st.toast(f"✅ {_qlabel} · {fx['home_team']} vs {fx['away_team']}", icon="📌")
                        elif _qresult == "duplicate":
                            st.toast("⚠️ Již sledováno.", icon="⚠️")
                        else:
                            st.toast("❌ Chyba při ukládání.", icon="❌")

        # ── Detailní analýza (expandery) ────────────────────────────────────
        st.subheader("Detailní analýza")
        with st.spinner("Načítám statistiky, zranění a sestavy..."):
            features_by_id, league_avg = get_league_features(league)
            injuries_by_id = get_league_injuries(league)
            lineups_by_id = get_league_lineups(league)

        for fx in fixtures:
            home = fx["home_team"]
            away = fx["away_team"]
            date_str = fx["date"][:16].replace("T", " ")
            inj_data = injuries_by_id.get(fx["fixture_id"], {})
            lu_data = lineups_by_id.get(fx["fixture_id"], {})
            has_injuries = bool(inj_data.get("home") or inj_data.get("away"))
            has_lineups = bool(lu_data.get("home") or lu_data.get("away"))
            suffix = ("  🚑" if has_injuries else "") + ("  ⚽" if has_lineups else "")
            label = f"📅 {date_str}  ·  {home} vs {away}" + suffix
            with st.expander(label):
                feats = features_by_id.get(fx["fixture_id"], {})
                render_prediction_stats(fx, feats)
                render_match_detail(fx, feats, league_avg)
                render_bet_validation(fx, feats)
                render_referee(feats)

                render_injuries(
                    home, away,
                    inj_data.get("home", []),
                    inj_data.get("away", []),
                    inj_data.get("home_goals", 50),
                    inj_data.get("away_goals", 50),
                    inj_data.get("home_goals_against", 50),
                    inj_data.get("away_goals_against", 50),
                )

                render_lineups(
                    home, away,
                    lu_data.get("home"),
                    lu_data.get("away"),
                )

                st.divider()
                with st.form(key=f"track_form_{fx['fixture_id']}"):
                    tr_col1, tr_col2 = st.columns([3, 1])
                    with tr_col1:
                        track_type = st.selectbox(
                            "Typ predikce ke sledování",
                            PREDICTION_TYPES,
                            key=f"track_type_{fx['fixture_id']}",
                        )
                    with tr_col2:
                        st.write("")
                        submitted = st.form_submit_button("📌 Sledovat")
                    if submitted:
                        result = save_tracking(fx, league, track_type, None)
                        if result == "ok":
                            st.toast("✅ Přidáno ke sledování.", icon="📌")
                        elif result == "duplicate":
                            st.toast("⚠️ Již sledováno.", icon="⚠️")
                        else:
                            st.toast("❌ Chyba při ukládání.", icon="❌")

                ai_key = f"ai_{fx['fixture_id']}"
                if st.button("🤖 AI analýza", key=f"ai_btn_{fx['fixture_id']}"):
                    with st.spinner("Generuji analýzu..."):
                        st.session_state[ai_key] = generate_ai_analysis(
                            fx, feats, league_avg, inj_data
                        )
                if ai_key in st.session_state:
                    st.markdown(st.session_state[ai_key])


# ── Největší pohyby ────────────────────────────────────────────────────────────

with tab_pred:
    st.divider()
    st.subheader("📈 Největší pohyby od posledního přepočtu")

    changes = get_prob_changes(min_delta=0.02)

    if not changes:
        st.caption("Žádné výrazné změny — data budou dostupná od druhého běhu GitHub Actions po nasazení.")
    else:
        ref_time = changes[0].get("prev_computed_at")
        if ref_time:
            st.caption(f"Porovnání s přepočtem: {ref_time.strftime('%d.%m.%Y %H:%M')} UTC · zobrazeny změny ≥ 2 %")

        def _delta_html(val: float, delta: float) -> str:
            pct = f"{val * 100:.0f}%"
            if abs(delta) < 0.005:
                return f'<span style="color:inherit">{pct}</span>'
            sign = "+" if delta > 0 else ""
            color = "#4ade80" if delta > 0 else "#f87171"
            return (
                f'<span style="color:inherit">{pct}</span>'
                f'<span style="color:{color};font-size:0.78rem;margin-left:4px">'
                f'({sign}{delta * 100:.0f}%)</span>'
            )

        rows_html = []
        for c in changes[:15]:
            flag = flags.get(settings.leagues[c["league"]].country, "")
            liga = f"{flag} {settings.leagues[c['league']].name}"
            date_s = c["match_date"].strftime("%d.%m") if c["match_date"] else ""
            max_d = c["max_delta"]
            badge_color = "#4ade80" if max_d >= 0.08 else "#facc15" if max_d >= 0.04 else "#94a3b8"
            rows_html.append(
                f"<tr>"
                f"<td style='text-align:left;color:#9a9a9a;font-size:0.78rem;white-space:nowrap'>{liga}</td>"
                f"<td style='text-align:left;white-space:nowrap'>{c['home_team']} vs {c['away_team']}</td>"
                f"<td style='color:#9a9a9a;font-size:0.82rem'>{date_s}</td>"
                f"<td>{_delta_html(c['prob_home'], c['delta_home'])}</td>"
                f"<td>{_delta_html(c['prob_draw'], c['delta_draw'])}</td>"
                f"<td>{_delta_html(c['prob_away'], c['delta_away'])}</td>"
                f"<td style='font-weight:700;color:{badge_color}'>{max_d * 100:.0f}%</td>"
                f"</tr>"
            )

        st.markdown(f"""
<style>
.chg{{width:100%;border-collapse:collapse;font-size:14px}}
.chg th,.chg td{{padding:5px 10px;border-bottom:1px solid #2a2a2a;text-align:center}}
.chg th{{color:#aaa;font-size:11px;font-weight:600;text-transform:uppercase}}
</style>
<div style="overflow-x:auto">
<table class="chg">
<thead><tr>
  <th style="text-align:left">Liga</th>
  <th style="text-align:left">Zápas</th>
  <th>Datum</th>
  <th>H%</th><th>D%</th><th>A%</th>
  <th>Max Δ</th>
</tr></thead>
<tbody>{"".join(rows_html)}</tbody>
</table>
</div>""", unsafe_allow_html=True)


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
            if tracked is not None and current is not None:
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

    # ── Odebrat sledovanou predikci ──────────────────────────────────────────
    pending_rows = [r for r in rows if r.correct is None]
    if pending_rows:
        st.divider()
        st.caption("Odebrat sledovanou predikci")
        options = {
            f"{r.home_team} vs {r.away_team}  –  {r.prediction_type}  ({r.match_date.strftime('%d.%m.')})": r.id
            for r in pending_rows
        }
        to_delete_label = st.selectbox("Vyber predikci", list(options.keys()), key="delete_select")
        if st.button("🗑️ Odebrat", key="delete_btn"):
            row_id = options[to_delete_label]
            with get_db() as db:
                db.query(TrackedPrediction).filter(TrackedPrediction.id == row_id).delete()
                db.commit()
            st.success("Predikce odebrána.")
            st.rerun()


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
        prob_cols_r = ["P(H)%", "P(D)%", "P(A)%", "P(G1-3)%"]
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
            "P(G1-3)%": round(r.goals1_3 * 100, 1),
            "G1-3": "✅" if 1 <= (r.home_score + r.away_score) <= 3 else "❌",
        } for r in resolved_rows])

        def highlight_correct(row):
            colors = []
            for col in row.index:
                if col == "Správně":
                    colors.append(f"background-color: {'#1a6e3c' if row['Správně'] == '✅' else '#6e1a1a'}; color: white")
                elif col == "G1-3":
                    colors.append(f"background-color: {'#1a6e3c' if row['G1-3'] == '✅' else '#6e1a1a'}; color: white")
                else:
                    colors.append("")
            return colors

        styled_r = (
            overview.style
            .apply(highlight_correct, axis=1)
            .format({col: "{:.1f}" for col in prob_cols_r})
        )
        st.dataframe(styled_r, use_container_width=True, hide_index=True)

        # Accuracy summary
        total_r = len(resolved_rows)
        correct_r = sum(1 for r in resolved_rows if r.correct)
        correct_g13 = sum(1 for r in resolved_rows if 1 <= (r.home_score + r.away_score) <= 3)
        st.caption(
            f"Úspěšnost modelu (1X2): **{correct_r}/{total_r}** = {correct_r/total_r*100:.1f}%"
            f"  ·  Goals 1–3 (skutečnost): **{correct_g13}/{total_r}** = {correct_g13/total_r*100:.1f}%"
        )

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
        all_tracked = db.query(TrackedPrediction).filter(
            TrackedPrediction.correct.isnot(None)
        ).all()
        cal_rows = db.query(ResolvedFixturePrediction).all()

    _CORNERS_TYPES = {t for t in PREDICTION_TYPES if t.startswith("Corners_")}
    goals_rows   = [r for r in all_tracked if r.prediction_type not in _CORNERS_TYPES]
    corners_rows = [r for r in all_tracked if r.prediction_type in _CORNERS_TYPES]

    # ── Celkový přehled ───────────────────────────────────────────────────────
    st.markdown("### Celkový přehled")

    def _pct(rows):
        if not rows:
            return 0.0, 0, 0
        c = sum(1 for r in rows if r.correct)
        return c / len(rows) * 100, c, len(rows)

    g_pct, g_ok, g_tot   = _pct(goals_rows)
    c_pct, c_ok, c_tot   = _pct(corners_rows)
    a_pct, a_ok, a_tot   = _pct(all_tracked)

    m1, m2, m3 = st.columns(3)
    m1.metric("Celková úspěšnost",           f"{a_pct:.1f}%", f"{a_ok}/{a_tot}")
    m2.metric("Model výsledků & gólů",       f"{g_pct:.1f}%", f"{g_ok}/{g_tot}")
    m3.metric("Model rohů",                  f"{c_pct:.1f}%", f"{c_ok}/{c_tot}" if c_tot else "žádná data")

    if all_tracked:
        # Avg tracked_prob for correct vs incorrect (confidence when right/wrong)
        correct_probs   = [r.tracked_prob for r in all_tracked if r.correct     and r.tracked_prob is not None]
        incorrect_probs = [r.tracked_prob for r in all_tracked if not r.correct and r.tracked_prob is not None]
        if correct_probs and incorrect_probs:
            p1, p2 = st.columns(2)
            p1.metric("Průměrná P(správné tipy)",   f"{sum(correct_probs)/len(correct_probs):.1%}")
            p2.metric("Průměrná P(špatné tipy)",    f"{sum(incorrect_probs)/len(incorrect_probs):.1%}")

    if not all_tracked:
        st.info("Žádné vyřešené predikce pro statistiky.")
        st.stop()

    # ── Helper: reliability diagram sekce ────────────────────────────────────
    from scipy.stats import beta as beta_dist

    def _reliability_section(markets, resolved, n_bins=5):
        """Render reliability line charts + tables for a list of markets.

        markets: list of (attr, actual_fn, label)
        resolved: list of ResolvedFixturePrediction rows
        """
        edges = [i / n_bins for i in range(n_bins + 1)]
        for attr, actual_fn, label in markets:
            probs  = [getattr(r, attr, None) for r in resolved]
            valid  = [(p, actual_fn(r)) for p, r in zip(probs, resolved) if p is not None]
            if not valid:
                st.caption(f"**{label}** — žádná data")
                continue
            ps, ls = zip(*valid)

            rows_r = []
            for i in range(n_bins):
                lo, hi = edges[i], edges[i + 1]
                idxs = [j for j, p in enumerate(ps) if lo <= p < hi]
                n = len(idxs)
                if n < 5:
                    continue
                k = sum(ls[j] for j in idxs)
                mean_pred   = sum(ps[j] for j in idxs) / n
                actual_freq = k / n
                ci_lo, ci_hi = beta_dist.interval(0.90, k + 0.5, n - k + 0.5)
                rows_r.append({
                    "Predicted %": round(mean_pred * 100, 1),
                    "Model":       round(actual_freq * 100, 1),
                    "CI low":      round(ci_lo * 100, 1),
                    "CI high":     round(ci_hi * 100, 1),
                    "N":           n,
                })

            n_total = sum(r["N"] for r in rows_r)
            st.caption(f"**{label}** ({n_total} zápasů v binech)")
            if not rows_r:
                st.caption("Nedostatek dat v binech.")
                continue
            df_r = pd.DataFrame(rows_r).set_index("Predicted %").sort_index()
            chart = df_r[["Model"]].copy()
            chart["Ideál"] = chart.index
            st.line_chart(chart, use_container_width=True)
            tbl = df_r[["Model", "CI low", "CI high", "N"]].copy()
            tbl.index = [f"{x}%" for x in tbl.index]
            tbl.columns = ["Actual %", "CI 90% low", "CI 90% high", "N"]
            st.dataframe(tbl, use_container_width=True)

    # ── Helper: typ/liga breakdown ────────────────────────────────────────────
    def _color_pct(val):
        if val >= 60:
            return "background-color: #1a7a3a; color: white"
        elif val >= 45:
            return "background-color: #7a6a1a; color: white"
        else:
            return "background-color: #7a1a1a; color: white"

    def _breakdown_section(rows, label_fn=None, key_suffix=""):
        by_type: dict[str, dict]   = {}
        by_league: dict[str, dict] = {}
        for r in rows:
            for key, bucket in [(r.prediction_type, by_type), (r.league, by_league)]:
                if key not in bucket:
                    bucket[key] = {"count": 0, "correct": 0}
                bucket[key]["count"]   += 1
                if r.correct:
                    bucket[key]["correct"] += 1

        view = st.radio(
            "Zobrazit podle",
            ["Podle typu", "Podle ligy"],
            horizontal=True,
            key=f"breakdown_view_{key_suffix}",
        )

        if view == "Podle typu":
            data = [
                {
                    "Typ":          label_fn(k) if label_fn else k,
                    "Počet":        v["count"],
                    "Správně":      v["correct"],
                    "Špatně":       v["count"] - v["correct"],
                    "Úspěšnost %":  round(v["correct"] / v["count"] * 100, 1),
                }
                for k, v in by_type.items()
            ]
        else:
            data = [
                {
                    "Liga":         leagues_display.get(k, k),
                    "Počet":        v["count"],
                    "Správně":      v["correct"],
                    "Špatně":       v["count"] - v["correct"],
                    "Úspěšnost %":  round(v["correct"] / v["count"] * 100, 1),
                }
                for k, v in by_league.items()
            ]

        if not data:
            return
        df = pd.DataFrame(data).sort_values("Úspěšnost %", ascending=False)
        styled = df.style.map(_color_pct, subset=["Úspěšnost %"])
        st.dataframe(styled, use_container_width=True, hide_index=True)

    # ── Helper: metriky modelu ────────────────────────────────────────────────
    import math as _math

    def _metrics_table(markets, resolved):
        """Brier Score + Log Loss + Mean Pred + Actual Freq pro každý market."""
        eps = 1e-9
        rows_m = []
        for attr, actual_fn, label in markets:
            pairs = [
                (getattr(r, attr), actual_fn(r))
                for r in resolved
                if getattr(r, attr) is not None
            ]
            if len(pairs) < 5:
                continue
            ps, ys = zip(*pairs)
            n   = len(ps)
            yi  = [int(y) for y in ys]
            brier   = sum((p - y) ** 2 for p, y in zip(ps, yi)) / n
            logloss = -sum(
                y * _math.log(p + eps) + (1 - y) * _math.log(1 - p + eps)
                for p, y in zip(ps, yi)
            ) / n
            rows_m.append({
                "Market":        label,
                "N":             n,
                "Brier ↓":       round(brier, 4),
                "Log Loss ↓":    round(logloss, 4),
                "Průměr modelu": f"{sum(ps)/n:.1%}",
                "Skutečná freq": f"{sum(yi)/n:.1%}",
            })
        if not rows_m:
            st.caption("Nedostatek dat pro výpočet metrik.")
            return
        df_m = pd.DataFrame(rows_m)

        def _color_brier(val):
            if val <= 0.20:
                return "background-color: #1a7a3a; color: white"
            elif val <= 0.25:
                return "background-color: #7a6a1a; color: white"
            else:
                return "background-color: #7a1a1a; color: white"

        def _color_logloss(val):
            if val <= 0.60:
                return "background-color: #1a7a3a; color: white"
            elif val <= 0.70:
                return "background-color: #7a6a1a; color: white"
            else:
                return "background-color: #7a1a1a; color: white"

        styled_m = (
            df_m.style
            .map(_color_brier,   subset=["Brier ↓"])
            .map(_color_logloss, subset=["Log Loss ↓"])
        )
        st.dataframe(styled_m, use_container_width=True, hide_index=True)

    # ── Sub-taby: Výsledky & Góly | Rohy ─────────────────────────────────────
    st.divider()
    subtab_goals, subtab_corners = st.tabs(["⚽ Výsledky & Góly", "🔄 Rohy"])

    with subtab_goals:
        if not goals_rows:
            st.info("Žádné vyřešené predikce výsledků/gólů.")
        else:
            _breakdown_section(goals_rows, key_suffix="goals")

            if len(cal_rows) >= 10:
                st.markdown("#### Metriky modelu")
                st.caption("Brier Score a Log Loss — nižší = lepší. Průměr modelu vs. skutečná frekvence ověřuje kalibraci.")
                _metrics_table(
                    [
                        ("prob_home", lambda r: r.actual_outcome == "H", "Výhra domácích"),
                        ("prob_draw", lambda r: r.actual_outcome == "D", "Remíza"),
                        ("prob_away", lambda r: r.actual_outcome == "A", "Výhra hostů"),
                        ("over2_5",  lambda r: (r.home_score + r.away_score) > 2,        "Over 2.5"),
                        ("under2_5", lambda r: (r.home_score + r.away_score) <= 2,       "Under 2.5"),
                        ("goals1_3", lambda r: 1 <= (r.home_score + r.away_score) <= 3,  "Goals 1-3"),
                        ("goals2_4", lambda r: 2 <= (r.home_score + r.away_score) <= 4,  "Goals 2-4"),
                        ("btts_yes", lambda r: r.home_score > 0 and r.away_score > 0,    "BTTS Ano"),
                        ("btts_no",  lambda r: not (r.home_score > 0 and r.away_score > 0), "BTTS Ne"),
                    ],
                    cal_rows,
                )

            if len(cal_rows) >= 50:
                st.markdown("#### Kalibrace — výsledky (H/D/A)")
                st.caption(
                    "Reliability diagram: dobře zkalibrovaný model leží na diagonále. "
                    "Data z archivu resolved_fixture_predictions · 90% Clopper-Pearson CI."
                )
                _reliability_section(
                    [
                        ("prob_home", lambda r: r.actual_outcome == "H", "Výhra domácích"),
                        ("prob_draw", lambda r: r.actual_outcome == "D", "Remíza"),
                        ("prob_away", lambda r: r.actual_outcome == "A", "Výhra hostů"),
                    ],
                    cal_rows,
                )

                st.markdown("#### Kalibrace — gólové markety")
                st.caption("Ověřuje přesnost GoalCalibrátoru — body by měly ležet na diagonále.")
                _reliability_section(
                    [
                        ("over2_5",  lambda r: (r.home_score + r.away_score) > 2,        "Over 2.5"),
                        ("under2_5", lambda r: (r.home_score + r.away_score) <= 2,       "Under 2.5"),
                        ("goals1_3", lambda r: 1 <= (r.home_score + r.away_score) <= 3,  "Goals 1-3"),
                        ("goals2_4", lambda r: 2 <= (r.home_score + r.away_score) <= 4,  "Goals 2-4"),
                        ("btts_yes", lambda r: r.home_score > 0 and r.away_score > 0,    "BTTS Ano"),
                        ("btts_no",  lambda r: not (r.home_score > 0 and r.away_score > 0), "BTTS Ne"),
                    ],
                    cal_rows,
                )
            else:
                st.info(f"Reliability diagram: potřeba ≥ 50 archivovaných zápasů (aktuálně {len(cal_rows)}).")

    with subtab_corners:
        def _corners_label(k):
            return k.replace("Corners_", "").replace("Over", "O").replace("Under", "U")

        if not corners_rows:
            st.info("Žádné vyřešené predikce rohů. Sleduj rohy v záložce 🔄 Rohy.")
        else:
            _breakdown_section(corners_rows, label_fn=_corners_label, key_suffix="corners")

        cal_corners = [
            r for r in cal_rows
            if getattr(r, "actual_corners_home", None) is not None
            and getattr(r, "actual_corners_away", None) is not None
            and getattr(r, "corners_over9_5", None) is not None
        ]

        def _actual_corners(r):
            return (r.actual_corners_home or 0) + (r.actual_corners_away or 0)

        if len(cal_corners) >= 10:
            st.markdown("#### Metriky modelu")
            st.caption("Brier Score a Log Loss — nižší = lepší.")
            _metrics_table(
                [
                    ("corners_over8_5",   lambda r: _actual_corners(r) > 8,   "Over 8.5"),
                    ("corners_under8_5",  lambda r: _actual_corners(r) <= 8,  "Under 8.5"),
                    ("corners_over9_5",   lambda r: _actual_corners(r) > 9,   "Over 9.5"),
                    ("corners_under9_5",  lambda r: _actual_corners(r) <= 9,  "Under 9.5"),
                    ("corners_over10_5",  lambda r: _actual_corners(r) > 10,  "Over 10.5"),
                    ("corners_under10_5", lambda r: _actual_corners(r) <= 10, "Under 10.5"),
                    ("corners_over11_5",  lambda r: _actual_corners(r) > 11,  "Over 11.5"),
                    ("corners_under11_5", lambda r: _actual_corners(r) <= 11, "Under 11.5"),
                ],
                cal_corners,
            )

        if len(cal_corners) >= 20:
            st.markdown("#### Kalibrace — rohové markety")
            st.caption(
                "Reliability diagram pro Over/Under rohových trhů. "
                "actual = skutečný počet rohů v zápase z archivu."
            )
            _reliability_section(
                [
                    ("corners_over8_5",   lambda r: _actual_corners(r) > 8,   "Over 8.5"),
                    ("corners_under8_5",  lambda r: _actual_corners(r) <= 8,  "Under 8.5"),
                    ("corners_over9_5",   lambda r: _actual_corners(r) > 9,   "Over 9.5"),
                    ("corners_under9_5",  lambda r: _actual_corners(r) <= 9,  "Under 9.5"),
                    ("corners_over10_5",  lambda r: _actual_corners(r) > 10,  "Over 10.5"),
                    ("corners_under10_5", lambda r: _actual_corners(r) <= 10, "Under 10.5"),
                    ("corners_over11_5",  lambda r: _actual_corners(r) > 11,  "Over 11.5"),
                    ("corners_under11_5", lambda r: _actual_corners(r) <= 11, "Under 11.5"),
                ],
                cal_corners,
            )
        else:
            st.info(
                f"Reliability diagram rohů: potřeba ≥ 20 archivovaných zápasů s corners daty "
                f"(aktuálně {len(cal_corners)}). Data se nahromadí automaticky po dalších bězích GitHub Actions."
            )


# ── TAB 5: Rohy ────────────────────────────────────────────────────────────────

with tab_corners:
    st.subheader("Rohové kopy")

    cr_league = st.selectbox(
        "Liga",
        options=league_keys,
        format_func=lambda k: leagues_display[k],
        key="corners_league",
    )

    with get_db() as db:
        cr_rows = (
            db.query(FixturePrediction)
            .filter(
                FixturePrediction.league == cr_league,
                FixturePrediction.corners_over8_5.isnot(None),
            )
            .order_by(FixturePrediction.match_date)
            .all()
        )

    if not cr_rows:
        st.info("Pro tuto ligu nejsou k dispozici predikce rohů. Rohy jsou dostupné pouze pro domácí ligy po obohacení historických dat.")
    else:
        def _c(val, threshold=0.65):
            if val is None:
                return "—"
            pct = f"{val:.0%}"
            if val >= threshold:
                return f'<span style="color:#4ade80;font-weight:700">{pct}</span>'
            return pct

        thead_c = (
            "<tr>"
            "<th style='text-align:left'>Zápas</th>"
            "<th>Datum</th>"
            "<th>λ</th><th>μ</th><th>Σ</th>"
            "<th>O8.5</th><th>U8.5</th>"
            "<th>O9.5</th><th>U9.5</th>"
            "<th>O10.5</th><th>U10.5</th>"
            "<th>O11.5</th><th>U11.5</th>"
            "</tr>"
        )
        rows_c = []
        for r in cr_rows:
            lh = getattr(r, "expected_corners_home", None)
            la = getattr(r, "expected_corners_away", None)
            total_c = round(lh + la, 1) if lh is not None and la is not None else None
            dt = r.match_date.strftime("%d.%m %H:%M") if r.match_date else ""
            rows_c.append(
                f"<tr>"
                f"<td style='text-align:left'>{r.home_team} – {r.away_team}</td>"
                f"<td>{dt}</td>"
                f"<td>{lh:.1f}</td><td>{la:.1f}</td>"
                f"<td><b>{total_c}</b></td>"
                f"<td>{_c(getattr(r,'corners_over8_5',None))}</td>"
                f"<td>{_c(getattr(r,'corners_under8_5',None))}</td>"
                f"<td>{_c(getattr(r,'corners_over9_5',None))}</td>"
                f"<td>{_c(getattr(r,'corners_under9_5',None))}</td>"
                f"<td>{_c(getattr(r,'corners_over10_5',None))}</td>"
                f"<td>{_c(getattr(r,'corners_under10_5',None))}</td>"
                f"<td>{_c(getattr(r,'corners_over11_5',None))}</td>"
                f"<td>{_c(getattr(r,'corners_under11_5',None))}</td>"
                f"</tr>"
            )

        st.markdown(f"""
<style>
.crt{{width:100%;border-collapse:collapse;font-size:13px}}
.crt th,.crt td{{padding:5px 8px;text-align:center;border-bottom:1px solid #2a2a2a;white-space:nowrap}}
.crt th{{color:#aaa;font-size:11px;font-weight:600;text-transform:uppercase}}
@media(max-width:768px){{.crt{{font-size:11px}}.crt th,.crt td{{padding:4px 5px}}}}
</style>
<div style="overflow-x:auto">
<table class="crt"><thead>{thead_c}</thead><tbody>{"".join(rows_c)}</tbody></table>
</div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### Rychlé sledování")
        st.caption("Klikni pro sledování konkrétního rohu.")

        _C_BTNS = [
            ("C8+", "Corners_Over8.5"),  ("C8-", "Corners_Under8.5"),
            ("C9+", "Corners_Over9.5"),  ("C9-", "Corners_Under9.5"),
            ("C10+","Corners_Over10.5"), ("C10-","Corners_Under10.5"),
            ("C11+","Corners_Over11.5"), ("C11-","Corners_Under11.5"),
        ]

        for r in cr_rows:
            lh = getattr(r, "expected_corners_home", None)
            la = getattr(r, "expected_corners_away", None)
            total_c = round(lh + la, 1) if lh is not None and la is not None else "?"
            st.caption(f"**{r.home_team} – {r.away_team}** | Σ={total_c} rohů")
            # Build fixture dict compatible with save_tracking()
            cr_fixture = {
                "fixture_id": r.fixture_id,
                "home_team": r.home_team,
                "away_team": r.away_team,
                "date": r.match_date.isoformat(),
                "corners_over8_5":   getattr(r, "corners_over8_5", None),
                "corners_under8_5":  getattr(r, "corners_under8_5", None),
                "corners_over9_5":   getattr(r, "corners_over9_5", None),
                "corners_under9_5":  getattr(r, "corners_under9_5", None),
                "corners_over10_5":  getattr(r, "corners_over10_5", None),
                "corners_under10_5": getattr(r, "corners_under10_5", None),
                "corners_over11_5":  getattr(r, "corners_over11_5", None),
                "corners_under11_5": getattr(r, "corners_under11_5", None),
            }
            btn_cols = st.columns(len(_C_BTNS))
            for col, (label, ptype) in zip(btn_cols, _C_BTNS):
                prob_field = ptype.lower().replace(".", "_")
                prob_val = getattr(r, prob_field, None)
                btn_label = f"{label} {prob_val:.0%}" if prob_val else label
                if col.button(btn_label, key=f"cr_{r.fixture_id}_{ptype}", use_container_width=True):
                    result = save_tracking(cr_fixture, r.league, ptype, prob_val)
                    if result == "ok":
                        st.toast(f"✅ {r.home_team} – {r.away_team}: {ptype} přidáno")
                        st.rerun()
                    elif result == "duplicate":
                        st.toast("⚠️ Již sledováno.", icon="⚠️")
                    else:
                        st.toast("❌ Chyba při ukládání.", icon="❌")

        st.markdown("---")
        st.markdown("#### Detail zápasů")
        for r in cr_rows:
            lh = getattr(r, "expected_corners_home", None)
            la = getattr(r, "expected_corners_away", None)
            total_c = lh + la if lh is not None and la is not None else None
            with st.expander(f"{r.home_team} – {r.away_team}"):
                if total_c is not None:
                    col1, col2, col3 = st.columns(3)
                    col1.metric("λ domácí", f"{lh:.2f}")
                    col2.metric("μ hosté", f"{la:.2f}")
                    col3.metric("Celkem rohů (λ+μ)", f"{total_c:.2f}")

                    # Simple bar chart comparing lambda / mu
                    bar_data = {"Domácí": lh, "Hosté": la}
                    st.bar_chart(bar_data)

                markets = {
                    "Over 8.5": getattr(r, "corners_over8_5", None),
                    "Under 8.5": getattr(r, "corners_under8_5", None),
                    "Over 9.5": getattr(r, "corners_over9_5", None),
                    "Under 9.5": getattr(r, "corners_under9_5", None),
                    "Over 10.5": getattr(r, "corners_over10_5", None),
                    "Under 10.5": getattr(r, "corners_under10_5", None),
                    "Over 11.5": getattr(r, "corners_over11_5", None),
                    "Under 11.5": getattr(r, "corners_under11_5", None),
                }
                mdf = {k: f"{v:.1%}" if v else "—" for k, v in markets.items()}
                st.dataframe(pd.DataFrame(mdf, index=["P"]).T.rename(columns={"P": "Pravděp."}), use_container_width=True)


# ── TAB 6: Tabulka ─────────────────────────────────────────────────────────────

with tab_standings:
    st.subheader("Ligová tabulka")

    std_league = st.selectbox(
        "Liga",
        options=league_keys,
        format_func=lambda k: leagues_display[k],
        key="standings_league",
    )

    view = st.radio(
        "Zobrazení",
        ["Celková", "Doma", "Venku"],
        horizontal=True,
        key="standings_view",
    )
    view_key = {"Celková": "all", "Doma": "home", "Venku": "away"}[view]

    groups = fetch_standings(std_league)

    if not groups:
        st.info("Tabulka není pro tuto ligu k dispozici (pohárové ligy ve vyřazovací fázi nemají skupinovou tabulku).")
    else:
        def _form_html(form: str) -> str:
            """Colorize form string: W=green, D=gray, L=red."""
            colors = {"W": "#4ade80", "D": "#888", "L": "#f87171"}
            return "".join(
                f'<span style="color:{colors.get(c, "#ccc")};font-weight:600">{c}</span>'
                for c in (form or "")
            )

        for group_idx, group in enumerate(groups):
            if len(groups) > 1:
                group_name = group[0].get("group", f"Skupina {group_idx + 1}") if group else f"Skupina {group_idx + 1}"
                st.markdown(f"**{group_name}**")

            show_form = view_key == "all"
            header_extra = "<th class='sth-extra'>GF</th><th class='sth-extra'>GA</th>"
            header_form  = "<th class='sth-extra'>Forma</th>" if show_form else ""
            header_pts   = "<th class='sth-pts'>Body</th>"

            thead = (
                f"<tr><th class='sth-rank'>#</th><th class='sth-logo'></th>"
                f"<th class='sth-name'>Tým</th>"
                f"<th>Z</th><th>V</th><th>R</th><th>P</th>"
                f"{header_extra}<th>+/-</th>{header_form}{header_pts}</tr>"
            )

            # For home/away views sort by home/away points (API order = total standings)
            sorted_group = group if show_form else sorted(
                group,
                key=lambda e: e.get(view_key, {}).get("win", 0) * 3 + e.get(view_key, {}).get("draw", 0),
                reverse=True,
            )

            rows_html = []
            for rank_idx, entry in enumerate(sorted_group, start=1):
                stats  = entry.get(view_key, {})
                goals  = stats.get("goals", {})
                gf     = goals.get("for", 0) or 0
                ga     = goals.get("against", 0) or 0
                pts    = entry.get("points", 0) if show_form else stats.get("win", 0) * 3 + stats.get("draw", 0)
                diff   = gf - ga
                diff_s = f"+{diff}" if diff > 0 else str(diff)
                logo   = entry["team"].get("logo", "")
                img    = f'<img src="{logo}" width="20" height="20" style="vertical-align:middle">' if logo else "•"
                form_td = f"<td class='sth-extra'>{_form_html(entry.get('form', ''))}</td>" if show_form else ""
                rank_display = entry.get("rank", rank_idx) if show_form else rank_idx
                rows_html.append(
                    f"<tr>"
                    f"<td class='sth-rank'>{rank_display}</td>"
                    f"<td class='sth-logo'>{img}</td>"
                    f"<td class='sth-name'>{entry['team']['name']}</td>"
                    f"<td>{stats.get('played', 0)}</td>"
                    f"<td>{stats.get('win', 0)}</td>"
                    f"<td>{stats.get('draw', 0)}</td>"
                    f"<td>{stats.get('lose', 0)}</td>"
                    f"<td class='sth-extra'>{gf}</td>"
                    f"<td class='sth-extra'>{ga}</td>"
                    f"<td>{diff_s}</td>"
                    f"{form_td}"
                    f"<td class='sth-pts'>{pts}</td>"
                    f"</tr>"
                )

            st.markdown(f"""
<style>
.sth{{width:100%;border-collapse:collapse;font-size:14px}}
.sth th,.sth td{{padding:5px 7px;text-align:center;border-bottom:1px solid #2a2a2a;white-space:nowrap}}
.sth th{{color:#aaa;font-size:11px;font-weight:600;text-transform:uppercase}}
.sth-rank{{width:24px;color:#888}}
.sth-logo{{width:28px;padding:3px 5px}}
.sth-name{{text-align:left;max-width:160px;overflow:hidden;text-overflow:ellipsis}}
.sth-pts{{font-weight:700;color:inherit}}
@media(max-width:768px){{
  .sth-name{{display:none}}
  .sth-extra{{display:none}}
  .sth{{font-size:12px}}
  .sth th,.sth td{{padding:4px 5px}}
}}
</style>
<div style="overflow-x:auto">
<table class="sth"><thead>{thead}</thead><tbody>{"".join(rows_html)}</tbody></table>
</div>""", unsafe_allow_html=True)

        st.caption("Data z API-Football · obnoveno 1× denně")
