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


@st.cache_data(ttl=21600, show_spinner="Načítám tabulku...")
def fetch_standings(league_key: str) -> list:
    """Returns list of standings groups from API. TTL=6h."""
    fetcher = get_fetcher()
    cfg = settings.leagues[league_key]
    return fetcher.get_standings(cfg)


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


tab_pred, tab_tracked, tab_results, tab_stats, tab_standings = st.tabs(["📅 Predikce", "📋 Sledované", "🔍 Výsledky", "📊 Statistiky", "🏆 Tabulka"])


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
                render_bet_validation(fx, feats)

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
            "Data z posledních 60 dní (resolved_fixture_predictions). "
            "Šedé pásmo = 90% Clopper-Pearson interval spolehlivosti; číslo v závorce = počet zápasů v binu."
        )

        with get_db() as db:
            cal_rows = db.query(ResolvedFixturePrediction).all()

        if len(cal_rows) < 50:
            st.info(f"Nedostatek dat pro reliability diagram (potřeba ≥ 50 odehraných zápasů v archivu, aktuálně {len(cal_rows)}).")
        else:
            from scipy.stats import beta as beta_dist

            n_bins = 5
            edges = [i / n_bins for i in range(n_bins + 1)]

            for outcome, prob_attr, outcome_label in [
                ("H", "prob_home", "Výhra domácích"),
                ("D", "prob_draw", "Remíza"),
                ("A", "prob_away", "Výhra hostů"),
            ]:
                probs  = [getattr(r, prob_attr) for r in cal_rows]
                labels = [1 if r.actual_outcome == outcome else 0 for r in cal_rows]

                rows = []
                for i in range(n_bins):
                    lo, hi = edges[i], edges[i + 1]
                    idxs = [j for j, p in enumerate(probs) if lo <= p < hi]
                    n = len(idxs)
                    if n < 5:
                        continue
                    k = sum(labels[j] for j in idxs)
                    mean_pred = sum(probs[j] for j in idxs) / n
                    actual_freq = k / n
                    # Clopper-Pearson 90% CI
                    ci_lo, ci_hi = beta_dist.interval(0.90, k + 0.5, n - k + 0.5)
                    rows.append({
                        "Predicted %": round(mean_pred * 100, 1),
                        "Model": round(actual_freq * 100, 1),
                        "CI low": round(ci_lo * 100, 1),
                        "CI high": round(ci_hi * 100, 1),
                        "N": n,
                    })

                st.caption(f"**{outcome_label}** (celkem {sum(r['N'] for r in rows)} zápasů v binech)")
                if not rows:
                    st.caption("Nedostatek dat v binech.")
                    continue
                df_out = pd.DataFrame(rows).set_index("Predicted %").sort_index()
                chart_df = df_out[["Model"]].copy()
                chart_df["Ideál"] = chart_df.index
                st.line_chart(chart_df, use_container_width=True)
                ci_display = df_out[["Model", "CI low", "CI high", "N"]].copy()
                ci_display.index = [f"{x}%" for x in ci_display.index]
                ci_display.columns = ["Actual %", "CI 90% low", "CI 90% high", "N"]
                st.dataframe(ci_display, use_container_width=True)


# ── TAB 5: Tabulka ─────────────────────────────────────────────────────────────

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
        for group_idx, group in enumerate(groups):
            if len(groups) > 1:
                group_name = group[0].get("group", f"Skupina {group_idx + 1}") if group else f"Skupina {group_idx + 1}"
                st.markdown(f"**{group_name}**")

            rows = []
            for entry in group:
                stats = entry.get(view_key, {})
                goals = stats.get("goals", {})
                gf = goals.get("for", 0) or 0
                ga = goals.get("against", 0) or 0
                row = {
                    "#": entry.get("rank", ""),
                    "Logo": entry["team"].get("logo", ""),
                    "Tým": entry["team"]["name"],
                    "Z": stats.get("played", 0),
                    "V": stats.get("win", 0),
                    "R": stats.get("draw", 0),
                    "P": stats.get("lose", 0),
                    "GF": gf,
                    "GA": ga,
                    "+/-": gf - ga,
                }
                if view_key == "all":
                    row["Body"] = entry.get("points", 0)
                    row["Forma"] = entry.get("form", "")
                else:
                    row["Body"] = stats.get("win", 0) * 3 + stats.get("draw", 0)
                rows.append(row)

            if rows:
                df_std = pd.DataFrame(rows)
                st.dataframe(
                    df_std,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Logo": st.column_config.ImageColumn("", width="small"),
                    },
                )

        st.caption("Data z API-Football · obnoveno každých 6 hodin")
