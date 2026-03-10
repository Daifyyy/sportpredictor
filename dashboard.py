import os
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
from db.models import Base, TrackedPrediction
from db.session import engine
from models.poisson import DixonColesPredictor

MODELS_DIR = Path("models/saved")
PREDICTION_TYPES = ["H", "D", "A", "Under2.5", "Over2.5", "Goals1-3", "Goals2-4", "BTTS_Yes", "BTTS_No"]

flags = {"England": "🏴󠁧󠁢󠁥󠁮󠁧󠁿", "Spain": "🇪🇸", "Germany": "🇩🇪", "Italy": "🇮🇹", "France": "🇫🇷"}
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
def get_model(league_key: str) -> DixonColesPredictor | None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / f"dixon_coles_{league_key}.joblib"
    if path.exists():
        try:
            return DixonColesPredictor.load(path)
        except Exception:
            pass
    # Train fresh
    fetcher = get_fetcher()
    cfg = settings.leagues[league_key]
    history = fetcher.get_fixtures(cfg, status="FT")
    completed = [f for f in history if f.result is not None]
    if len(completed) < 50:
        return None
    m = DixonColesPredictor()
    m.train(completed)
    m.save(path)
    return m


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
        row = TrackedPrediction(
            fixture_id=fixture["fixture_id"],
            league=league,
            home_team=fixture["home_team"],
            away_team=fixture["away_team"],
            match_date=datetime.fromisoformat(fixture["date"]),
            prediction_type=prediction_type,
            model_prob=prob_map.get(prediction_type),
        )
        db.add(row)
        db.commit()
    return "ok"


# ── App layout ─────────────────────────────────────────────────────────────────

Base.metadata.create_all(engine)

st.set_page_config(page_title="Football Tracker", page_icon="⚽", layout="wide")
st.title("⚽ Football Prediction Tracker")

tab_pred, tab_tracked, tab_stats = st.tabs(["📅 Predikce", "📋 Sledované predikce", "📊 Statistiky"])


# ── TAB 1: Predikce ────────────────────────────────────────────────────────────

with tab_pred:
    st.subheader("Nadcházející zápasy")

    league = st.selectbox(
        "Liga",
        options=league_keys,
        format_func=lambda k: leagues_display[k],
        key="pred_league",
    )

    if st.button("Načíst predikce", key="load_pred"):
        st.session_state.pop("upcoming_data", None)
        st.session_state.pop("upcoming_league", None)

    if st.session_state.get("upcoming_league") != league:
        st.session_state.pop("upcoming_data", None)

    if "upcoming_data" not in st.session_state:
        with st.spinner("Načítám predikce... (první spuštění může trvat déle)"):
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
                    lam = pred.expected_goals_home or 1.3
                    mu = pred.expected_goals_away or 1.0
                    gp = compute_goal_probs(lam, mu)
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
        prob_cols = ["P(H)%", "P(D)%", "P(A)%", "P(O2.5)%", "P(U2.5)%", "P(BTTS)%", "P(1-3g)%", "P(2-4g)%"]
        df = pd.DataFrame([{
            "Datum": f["date"][:16].replace("T", " "),
            "Domácí": f["home_team"],
            "Hosté": f["away_team"],
            "P(H)%": round(f["prob_home"] * 100, 1),
            "P(D)%": round(f["prob_draw"] * 100, 1),
            "P(A)%": round(f["prob_away"] * 100, 1),
            "P(O2.5)%": round(f["over2_5"] * 100, 1),
            "P(U2.5)%": round(f["under2_5"] * 100, 1),
            "P(BTTS)%": round(f["btts_yes"] * 100, 1),
            "P(1-3g)%": round(f["goals1_3"] * 100, 1),
            "P(2-4g)%": round(f["goals2_4"] * 100, 1),
        } for f in fixtures])

        def highlight_high(val):
            if isinstance(val, float) and val >= 65:
                return "background-color: #1a6e3c; color: white; font-weight: bold"
            return ""

        styled = df.style.applymap(highlight_high, subset=prob_cols).format(
            {col: "{:.1f}" for col in prob_cols}
        )
        st.dataframe(styled, use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("Přidat ke sledování")

        match_options = {
            f"{f['home_team']} vs {f['away_team']} ({f['date'][:10]})": f
            for f in fixtures
        }
        selected_match_label = st.selectbox("Vyber zápas", options=list(match_options.keys()), key="sel_match")
        selected_pred_type = st.selectbox("Typ predikce", options=PREDICTION_TYPES, key="sel_type")

        if st.button("Přidat ke sledování", key="add_track"):
            chosen = match_options[selected_match_label]
            result = save_tracking(chosen, league, selected_pred_type, None)
            if result == "ok":
                st.success("Predikce přidána ke sledování.")
            elif result == "duplicate":
                st.warning("Tato predikce je již sledována.")
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
            table_data.append({
                "Datum": r.match_date.strftime("%d.%m.%Y %H:%M"),
                "Liga": leagues_display.get(r.league, r.league),
                "Zápas": f"{r.home_team} vs {r.away_team}",
                "Typ predikce": r.prediction_type,
                "Pravděpodobnost": prob_str,
                "Skóre": score,
                "Správně?": status_icon,
            })
        st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)


# ── TAB 3: Statistiky ──────────────────────────────────────────────────────────

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
