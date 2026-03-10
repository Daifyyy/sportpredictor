import os
from datetime import datetime, timezone

import pandas as pd
import requests
import streamlit as st

for _k in ["DATABASE_URL", "API_URL", "API_FOOTBALL_KEY"]:
    if _k in st.secrets and not os.getenv(_k):
        os.environ[_k] = st.secrets[_k]

API = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")

from config.settings import settings
from db.models import TrackedPrediction
from db.session import engine
from sqlalchemy.orm import Session as SASession

st.set_page_config(page_title="Football Tracker", page_icon="⚽", layout="wide")
st.title("⚽ Football Prediction Tracker")

flags = {"England": "🏴󠁧󠁢󠁥󠁮󠁧󠁿", "Spain": "🇪🇸", "Germany": "🇩🇪", "Italy": "🇮🇹", "France": "🇫🇷"}
leagues_display = {
    k: f"{flags.get(cfg.country, '')} {cfg.name}"
    for k, cfg in settings.leagues.items()
}
league_keys = list(leagues_display.keys())

PREDICTION_TYPES = ["H", "D", "A", "Under2.5", "Over2.5", "Goals1-3", "Goals2-4", "BTTS_Yes", "BTTS_No"]


def get_db() -> SASession:
    return SASession(engine)


tab_pred, tab_tracked, tab_stats = st.tabs(["📅 Predikce", "📋 Sledované predikce", "📊 Statistiky"])


# ── TAB 1: Predikce ──────────────────────────────────────────────────────────

with tab_pred:
    st.subheader("Nadcházející zápasy")

    league = st.selectbox(
        "Liga",
        options=league_keys,
        format_func=lambda k: leagues_display[k],
        key="pred_league",
    )

    if st.button("Načíst predikce", key="load_pred"):
        st.session_state["upcoming_data"] = None
        st.session_state["upcoming_league"] = None

    if st.session_state.get("upcoming_league") != league:
        st.session_state["upcoming_data"] = None

    if st.session_state.get("upcoming_data") is None:
        with st.spinner("Načítám..."):
            try:
                r = requests.get(f"{API}/upcoming/{league}", timeout=60)
                if r.ok:
                    st.session_state["upcoming_data"] = r.json()
                    st.session_state["upcoming_league"] = league
                else:
                    st.error(f"API error: {r.text}")
                    st.session_state["upcoming_data"] = []
            except Exception as e:
                st.error(f"Nelze se připojit k API: {e}")
                st.session_state["upcoming_data"] = []

    fixtures = st.session_state.get("upcoming_data") or []

    if not fixtures:
        st.info("Žádné nadcházející zápasy nebo API není dostupné.")
    else:
        df = pd.DataFrame([{
            "Datum": f["date"][:16].replace("T", " "),
            "Domácí": f["home_team"],
            "Hosté": f["away_team"],
            "P(H)%": f"{f['prob_home']*100:.1f}",
            "P(D)%": f"{f['prob_draw']*100:.1f}",
            "P(A)%": f"{f['prob_away']*100:.1f}",
            "P(O2.5)%": f"{f['over2_5']*100:.1f}",
            "P(U2.5)%": f"{f['under2_5']*100:.1f}",
            "P(BTTS)%": f"{f['btts_yes']*100:.1f}",
            "P(1-3g)%": f"{f['goals1_3']*100:.1f}",
            "P(2-4g)%": f"{f['goals2_4']*100:.1f}",
        } for f in fixtures])

        st.dataframe(df, use_container_width=True, hide_index=True)

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
            prob_map = {
                "H": chosen["prob_home"],
                "D": chosen["prob_draw"],
                "A": chosen["prob_away"],
                "Under2.5": chosen["under2_5"],
                "Over2.5": chosen["over2_5"],
                "Goals1-3": chosen["goals1_3"],
                "Goals2-4": chosen["goals2_4"],
                "BTTS_Yes": chosen["btts_yes"],
                "BTTS_No": chosen["btts_no"],
            }
            payload = {
                "fixture_id": chosen["fixture_id"],
                "league": league,
                "home_team": chosen["home_team"],
                "away_team": chosen["away_team"],
                "match_date": chosen["date"],
                "prediction_type": selected_pred_type,
                "model_prob": prob_map.get(selected_pred_type),
            }
            try:
                resp = requests.post(f"{API}/track", json=payload, timeout=10)
                if resp.status_code == 200:
                    st.success("Predikce přidána ke sledování.")
                elif resp.status_code == 409:
                    st.warning("Tato predikce je již sledována.")
                else:
                    st.error(f"Chyba: {resp.text}")
            except Exception as e:
                st.error(f"Chyba připojení: {e}")


# ── TAB 2: Sledované predikce ─────────────────────────────────────────────────

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
            try:
                r = requests.post(f"{API}/resolve", timeout=120)
                if r.ok:
                    d = r.json()
                    st.success(f"Vyřešeno: {d['resolved']} predikcí")
                else:
                    st.error(f"Chyba: {r.text}")
            except Exception as e:
                st.error(f"Chyba připojení: {e}")

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


# ── TAB 3: Statistiky ─────────────────────────────────────────────────────────

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
            if r.prediction_type not in by_type:
                by_type[r.prediction_type] = {"count": 0, "correct": 0}
            by_type[r.prediction_type]["count"] += 1
            if r.correct:
                by_type[r.prediction_type]["correct"] += 1

            if r.league not in by_league:
                by_league[r.league] = {"count": 0, "correct": 0}
            by_league[r.league]["count"] += 1
            if r.correct:
                by_league[r.league]["correct"] += 1

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Podle typu predikce")
            type_rows = [
                {
                    "Typ": k,
                    "Počet": v["count"],
                    "Správně": v["correct"],
                    "Úspěšnost %": round(v["correct"] / v["count"] * 100, 1) if v["count"] else 0,
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
                    "Úspěšnost %": round(v["correct"] / v["count"] * 100, 1) if v["count"] else 0,
                }
                for k, v in by_league.items()
            ]
            df_league = pd.DataFrame(league_rows).sort_values("Úspěšnost %", ascending=False)
            st.dataframe(df_league, use_container_width=True, hide_index=True)

            if not df_league.empty:
                st.bar_chart(df_league.set_index("Liga")["Úspěšnost %"])
