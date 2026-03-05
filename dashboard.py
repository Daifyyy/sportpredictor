"""
Football Predictor — Streamlit Dashboard
Reads directly from Supabase. No local API needed for display.
Actions (retrain, resolve) require local API: uvicorn api.app:app --reload

Deploy: Streamlit Community Cloud
Secrets needed: DATABASE_URL
"""
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

from config.settings import settings
from db.models import BacktestRunRow, BankrollRow, PredictionRow
from db.session import SessionLocal

API = "http://127.0.0.1:8000"

st.set_page_config(page_title="Football Predictor", page_icon="⚽", layout="wide")
st.title("⚽ Football Predictor Dashboard")


# ── DB connection ─────────────────────────────────────────────────────────────

@st.cache_resource
def get_db():
    return SessionLocal()


db = get_db()


# ── API availability check ────────────────────────────────────────────────────

@st.cache_data(ttl=10)
def api_alive() -> bool:
    try:
        requests.get(f"{API}/config/leagues", timeout=2)
        return True
    except Exception:
        return False


# ── Config from settings ──────────────────────────────────────────────────────

flags = {"England": "🏴󠁧󠁢󠁥󠁮󠁧󠁿", "Spain": "🇪🇸", "Germany": "🇩🇪", "Italy": "🇮🇹", "France": "🇫🇷"}
leagues_display = {
    k: f"{flags.get(cfg.country, '')} {cfg.name}"
    for k, cfg in settings.leagues.items()
}

# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.header("Nastavení")
league = st.sidebar.selectbox(
    "Liga",
    options=list(leagues_display.keys()),
    format_func=lambda k: leagues_display[k],
)

st.sidebar.divider()
st.sidebar.subheader("Akce")

local_api = api_alive()
if not local_api:
    st.sidebar.info("💡 Akce vyžadují lokální API:\n`uvicorn api.app:app --reload`")

_btn = dict(use_container_width=True, disabled=not local_api)

if st.sidebar.button("🧠 Přetrénovat modely", type="primary", **_btn):
    r = requests.post(f"{API}/retrain", timeout=10)
    if r.status_code == 202:
        st.sidebar.info("Trénink spuštěn ↓")
        st.session_state["show_retrain_log"] = True
    elif r.status_code == 409:
        st.sidebar.warning("Trénink už běží.")
    else:
        st.sidebar.error(r.text)

if st.sidebar.button("🔄 Načíst predikce", **_btn):
    with st.spinner("Načítám..."):
        r = requests.get(f"{API}/predictions/{league}", timeout=60)
        st.sidebar.success(f"{len(r.json())} zápasů") if r.ok else st.sidebar.error(r.text)
        st.cache_data.clear()

if st.sidebar.button("✅ Resolve výsledků", **_btn):
    with st.spinner("Resolvuji..."):
        r = requests.post(f"{API}/resolve/{league}", timeout=60)
        if r.ok:
            d = r.json()
            st.sidebar.success(f"Nově: {d['resolved']} | Celkem: {d['resolved'] + d['already_resolved']}")
            st.cache_data.clear()
        else:
            st.sidebar.error(r.text)

if st.sidebar.button("🎯 Kalibrovat modely", **_btn):
    with st.spinner("Kalibrace..."):
        r = requests.post(f"{API}/calibrate/{league}", timeout=120)
        if r.ok:
            for model, res in r.json().items():
                if res["status"] == "ok":
                    st.sidebar.success(f"{model}: {res['samples']} vzorků")
                else:
                    st.sidebar.warning(f"{model}: {res.get('reason')}")
        else:
            st.sidebar.error(r.text)

if st.sidebar.button("💹 Update bankroll", **_btn):
    with st.spinner("Počítám..."):
        r = requests.post(f"{API}/bankroll/update/{league}", timeout=60)
        if r.ok:
            d = r.json()
            st.sidebar.success(f"Bankroll: {d['bankroll']:.2f} | +{d['added']} betů")
            st.cache_data.clear()
        else:
            st.sidebar.error(r.text)

st.sidebar.divider()
st.sidebar.caption("🕘 Auto: predikce 09:00, resolve 23:00 (jen lokálně)")
if local_api:
    st.sidebar.caption("🟢 Lokální API aktivní")
else:
    st.sidebar.caption("🔴 Lokální API offline")

# ── Live retrain log ──────────────────────────────────────────────────────────

if st.session_state.get("show_retrain_log"):
    try:
        rs = requests.get(f"{API}/retrain/status", timeout=10).json()
        is_running = rs.get("running", False)
        steps = rs.get("steps", [])
        error = rs.get("error")
        icons = {"info": "ℹ️", "success": "✅", "warning": "⚠️", "error": "❌"}
        label = "🧠 Trénink probíhá..." if is_running else ("❌ Chyba" if error else "✅ Hotovo")
        state = "running" if is_running else ("error" if error else "complete")
        with st.status(label, expanded=is_running, state=state):
            for step in steps:
                st.write(f"`{step['time']}` {icons.get(step['level'], 'ℹ️')} {step['msg']}")
        if is_running:
            import time; time.sleep(2); st.rerun()
        else:
            st.session_state["show_retrain_log"] = False
    except Exception:
        pass

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_pred, tab_perf, tab_vb, tab_kelly, tab_bankroll, tab_bt, tab_cal = st.tabs([
    "📅 Predikce", "📊 Výkonnost", "💰 Value Bety",
    "📐 Kelly", "💹 Bankroll", "🧪 Backtest", "🎯 Kalibrace",
])


# ── Helpers ───────────────────────────────────────────────────────────────────

def model_label(name: str) -> str:
    return name.replace("_", " ").title()


@st.cache_data(ttl=300)
def load_predictions(league_key: str):
    return db.query(PredictionRow).filter(PredictionRow.league_key == league_key).all()


@st.cache_data(ttl=300)
def load_bankroll(league_key: str):
    return db.query(BankrollRow).filter(BankrollRow.league_key == league_key)\
        .order_by(BankrollRow.match_date).all()


@st.cache_data(ttl=300)
def load_backtest(league_key: str):
    return db.query(BacktestRunRow).filter(BacktestRunRow.league_key == league_key)\
        .order_by(BacktestRunRow.run_at.desc()).all()


# ── TAB 1: Predikce ───────────────────────────────────────────────────────────

with tab_pred:
    st.subheader(f"Nadcházející zápasy — {leagues_display.get(league)}")

    rows = load_predictions(league)
    now = datetime.utcnow()
    upcoming = [r for r in rows if r.actual_outcome is None and r.match_date > now]

    if not upcoming:
        st.info("Žádné nadcházející predikce. Klikni 'Načíst predikce'.")
    else:
        # Group by fixture_id → one row per match, columns per model
        by_fixture: dict[int, list] = {}
        for r in upcoming:
            by_fixture.setdefault(r.fixture_id, []).append(r)

        pivot_rows = []
        for fixture_id, preds in sorted(by_fixture.items(), key=lambda x: x[1][0].match_date):
            p0 = preds[0]
            row = {"Datum": p0.match_date.strftime("%d.%m"), "Domácí": p0.home_team, "Hosté": p0.away_team}
            vb_all = []
            for p in preds:
                ml = model_label(p.model_name)
                row[f"{ml} tip"] = p.predicted_outcome or "—"
                row[f"{ml} P(H)"] = f"{p.prob_home:.0%}"
                row[f"{ml} P(D)"] = f"{p.prob_draw:.0%}"
                row[f"{ml} P(A)"] = f"{p.prob_away:.0%}"
                vb_all.extend(p.value_bets or [])
            if vb_all:
                row["💰"] = ", ".join(sorted(set(vb_all)))
            pivot_rows.append(row)

        st.dataframe(pd.DataFrame(pivot_rows), use_container_width=True, hide_index=True)

        # Value bets highlight
        vb_rows = []
        for fixture_id, preds in by_fixture.items():
            for p in preds:
                if p.value_bets:
                    vb_rows.append({
                        "Datum": p.match_date.strftime("%d.%m"),
                        "Zápas": f"{p.home_team} vs {p.away_team}",
                        "Model": model_label(p.model_name),
                        "Tip": ", ".join(p.value_bets),
                        "P(H)": f"{p.prob_home:.0%}",
                        "P(D)": f"{p.prob_draw:.0%}",
                        "P(A)": f"{p.prob_away:.0%}",
                    })
        if vb_rows:
            st.subheader("💰 Value bety")
            st.dataframe(pd.DataFrame(vb_rows), use_container_width=True, hide_index=True)


# ── TAB 2: Výkonnost ──────────────────────────────────────────────────────────

with tab_perf:
    st.subheader(f"Reálná úspěšnost — {leagues_display.get(league)}")

    rows = load_predictions(league)
    resolved = [r for r in rows if r.actual_outcome is not None]

    if not resolved:
        st.info("Žádné vyřešené predikce. Po odehrání zápasů klikni 'Resolve výsledků'.")
    else:
        by_model: dict[str, list] = {}
        for r in resolved:
            by_model.setdefault(r.model_name, []).append(r)

        perf_rows = []
        for mname, mrows in by_model.items():
            correct = [r for r in mrows if r.correct]
            vb = [r for r in mrows if r.value_bets]
            vb_ok = [r for r in vb if r.correct]
            perf_rows.append({
                "Model": model_label(mname),
                "Vyřešeno": len(mrows),
                "Accuracy": len(correct) / len(mrows) if mrows else None,
                "Value bety": len(vb),
                "VB správně": len(vb_ok),
                "VB accuracy": len(vb_ok) / len(vb) if vb else None,
                "_model": mname,
            })

        # Metriky
        cols = st.columns(len(perf_rows))
        for col, p in zip(cols, perf_rows):
            acc = f"{p['Accuracy']:.1%}" if p["Accuracy"] is not None else "N/A"
            col.metric(p["Model"], acc, help=f"Vyřešeno: {p['Vyřešeno']}")

        # Graf
        df_perf = pd.DataFrame(perf_rows)
        df_ok = df_perf[df_perf["Accuracy"].notna()]
        if not df_ok.empty:
            fig = px.bar(df_ok, x="Model", y="Accuracy", text=df_ok["Accuracy"].apply(lambda x: f"{x:.1%}"),
                         color="Model", title="Accuracy modelů", labels={"Accuracy": "Přesnost"})
            fig.update_layout(showlegend=False, yaxis_tickformat=".0%", yaxis_range=[0, 1])
            st.plotly_chart(fig, use_container_width=True)

        # VB úspěšnost
        vb_data = [p for p in perf_rows if p["Value bety"] > 0]
        if vb_data:
            st.subheader("💰 Úspěšnost value betů")
            c2 = st.columns(len(vb_data))
            for col, p in zip(c2, vb_data):
                vba = f"{p['VB accuracy']:.1%}" if p["VB accuracy"] is not None else "N/A"
                col.metric(p["Model"], vba, help=f"{p['VB správně']}/{p['Value bety']}")

        with st.expander("Detailní tabulka"):
            disp = pd.DataFrame(perf_rows).drop(columns=["_model"])
            disp["Accuracy"] = disp["Accuracy"].apply(lambda x: f"{x:.1%}" if x else "N/A")
            disp["VB accuracy"] = disp["VB accuracy"].apply(lambda x: f"{x:.1%}" if x else "N/A")
            st.dataframe(disp, use_container_width=True, hide_index=True)


# ── TAB 3: Value Bety ─────────────────────────────────────────────────────────

with tab_vb:
    st.subheader(f"Value bety — {leagues_display.get(league)}")

    rows = load_predictions(league)
    now = datetime.utcnow()
    vb_rows = [r for r in rows if r.value_bets and r.actual_outcome is None and r.match_date > now]

    if not vb_rows:
        st.info("Žádné nadcházející value bety.")
    else:
        data = [{
            "Datum": r.match_date.strftime("%d.%m"),
            "Zápas": f"{r.home_team} vs {r.away_team}",
            "Model": model_label(r.model_name),
            "Tip": ", ".join(r.value_bets),
            "P(H)": f"{r.prob_home:.0%}",
            "P(D)": f"{r.prob_draw:.0%}",
            "P(A)": f"{r.prob_away:.0%}",
        } for r in sorted(vb_rows, key=lambda x: x.match_date)]
        st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)


# ── TAB 4: Kelly ──────────────────────────────────────────────────────────────

with tab_kelly:
    st.subheader(f"Kelly stake suggestions — {leagues_display.get(league)}")
    st.caption("Fractional Kelly 25% | Stake z bankrollu 1000 jednotek")

    rows = load_predictions(league)
    now = datetime.utcnow()
    vb_rows = [r for r in rows if r.value_bets and r.actual_outcome is None and r.match_date > now]

    if not vb_rows:
        st.info("Žádné nadcházející value bety pro Kelly výpočet.")
    else:
        KELLY_FRACTION = 0.25
        BANKROLL = 1000.0
        outcome_names = {"H": "Domácí", "D": "Remíza", "A": "Hosté"}
        prob_map_keys = {"H": "prob_home", "D": "prob_draw", "A": "prob_away"}
        odds_map_keys = {"H": "odds_home", "D": "odds_draw", "A": "odds_away"}

        kelly_rows = []
        for r in sorted(vb_rows, key=lambda x: x.match_date):
            for vb in r.value_bets:
                o = vb[0]
                prob = getattr(r, prob_map_keys[o], None)
                odds = getattr(r, odds_map_keys[o], None)
                if not prob or not odds or odds <= 1:
                    continue
                b = odds - 1
                kelly = max(0.0, (prob * b - (1 - prob)) / b) * KELLY_FRACTION
                if kelly <= 0:
                    continue
                kelly_rows.append({
                    "Datum": r.match_date.strftime("%d.%m"),
                    "Zápas": f"{r.home_team} vs {r.away_team}",
                    "Model": model_label(r.model_name),
                    "Sázka na": outcome_names.get(o, o),
                    "P(model)": f"{prob:.1%}",
                    "Kurz": f"{odds:.2f}",
                    "Kelly %": f"{kelly*100:.2f}%",
                    "Stake z 1000": f"{kelly*BANKROLL:.1f}",
                })

        if kelly_rows:
            st.dataframe(pd.DataFrame(kelly_rows), use_container_width=True, hide_index=True)
        else:
            st.info("Nedostatek dat pro Kelly výpočet (chybí kurzy).")


# ── TAB 5: Bankroll ───────────────────────────────────────────────────────────

with tab_bankroll:
    st.subheader(f"Bankroll tracking — {leagues_display.get(league)}")

    bk_rows = load_bankroll(league)

    if not bk_rows:
        st.info("Zatím žádná data. Klikni 'Update bankroll' po Resolve.")
    else:
        df_bk = pd.DataFrame([{
            "match_date": r.match_date.strftime("%Y-%m-%d"),
            "Zápas": f"{r.home_team} vs {r.away_team}",
            "Model": model_label(r.model_name),
            "Sázka": r.bet_on,
            "Kurz": r.odds,
            "Stake %": r.stake_pct,
            "Výsledek": r.outcome,
            "P&L %": r.pnl_pct,
            "Bankroll": r.bankroll_after,
        } for r in bk_rows])

        final_bk = df_bk["Bankroll"].iloc[-1]
        roi = (final_bk - 1000) / 1000 * 100
        wins = (df_bk["Výsledek"] == "win").sum()
        total = len(df_bk)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Bankroll", f"{final_bk:.2f}", f"{roi:+.1f}%")
        m2.metric("Celkem betů", total)
        m3.metric("Výhry", f"{wins} / {total}")
        m4.metric("Hit rate", f"{wins/total:.1%}" if total else "N/A")

        fig = px.line(df_bk, x="match_date", y="Bankroll", color="Model",
                      markers=True, title="Vývoj bankrollu",
                      labels={"match_date": "Datum"})
        fig.add_hline(y=1000, line_dash="dash", line_color="gray", annotation_text="Start")
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Všechny bety"):
            st.dataframe(df_bk, use_container_width=True, hide_index=True)


# ── TAB 6: Backtest ───────────────────────────────────────────────────────────

with tab_bt:
    st.subheader(f"Backtest history — {leagues_display.get(league)}")

    bt_rows = load_backtest(league)

    if not bt_rows:
        st.info("Žádná data. Zavolej GET /track-record/{league}.")
    else:
        df_bt = pd.DataFrame([{
            "run_at": r.run_at.strftime("%Y-%m-%d %H:%M"),
            "Model": model_label(r.model_name),
            "Train": r.n_train,
            "Test": r.n_test,
            "Accuracy": r.accuracy,
            "Brier": r.brier_score,
            "Log Loss": r.log_loss,
        } for r in bt_rows])

        latest = df_bt.groupby("Model").first().reset_index()
        cols = st.columns(len(latest))
        for col, (_, row) in zip(cols, latest.iterrows()):
            col.metric(row["Model"], f"{row['Accuracy']:.1%}",
                       help=f"Brier: {row['Brier']} | LogLoss: {row['Log Loss']} | Test: {row['Test']}")

        if len(df_bt) > len(latest):
            fig = px.line(df_bt.sort_values("run_at"), x="run_at", y="Accuracy",
                          color="Model", markers=True, title="Accuracy v čase",
                          labels={"run_at": "Čas"})
            fig.update_layout(yaxis_tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)

        with st.expander("Všechny runs"):
            st.dataframe(df_bt, use_container_width=True, hide_index=True)


# ── TAB 7: Kalibrace ──────────────────────────────────────────────────────────

with tab_cal:
    st.subheader(f"Kalibrace modelů — {leagues_display.get(league)}")

    if not local_api:
        st.info("Kalibrace vyžaduje lokální API (spusť uvicorn) — zobrazuje live backtest.")
    else:
        # Calibration status from DB
        rows = load_predictions(league)
        resolved = [r for r in rows if r.actual_outcome is not None]
        by_model: dict[str, list] = {}
        for r in resolved:
            by_model.setdefault(r.model_name, []).append(r)

        if by_model:
            st.subheader("Status kalibrační vrstvy")
            scols = st.columns(max(len(by_model), 1))
            for col, (mname, mrows) in zip(scols, by_model.items()):
                needed = max(0, 50 - len(mrows))
                if needed == 0:
                    col.success(f"**{model_label(mname)}**\n\n✅ Připraveno ({len(mrows)} vzorků)")
                else:
                    col.warning(f"**{model_label(mname)}**\n\n⏳ Chybí {needed} predikcí")
            st.divider()

        with st.spinner("Načítám kalibraci (backtest, 1–2 min)..."):
            try:
                r = requests.get(f"{API}/calibration/{league}", timeout=300)
                cal_data = r.json() if r.ok else []
            except Exception:
                cal_data = []

        if not cal_data:
            st.warning("Kalibrace nedostupná.")
        else:
            outcome_labels = {"H": "Výhra domácích", "D": "Remíza", "A": "Výhra hostů"}

            ece_cols = st.columns(len(cal_data))
            for col, mc in zip(ece_cols, cal_data):
                col.metric(model_label(mc["model"]), f"ECE {mc['overall_ece']:.4f}",
                           help=f"Vzorků: {mc['n_samples']}")

            for mc in cal_data:
                st.subheader(f"{model_label(mc['model'])} — ECE {mc['overall_ece']:.4f}")
                ocols = st.columns(3)
                for col, oc in zip(ocols, mc["outcomes"]):
                    bins = oc["bins"]
                    if not bins:
                        continue
                    df_b = pd.DataFrame(bins)
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=df_b["predicted_prob"], y=df_b["actual_freq"],
                        mode="lines+markers", name="Model",
                        marker=dict(size=df_b["count"] / df_b["count"].max() * 14 + 5),
                        line=dict(color="#2196F3"),
                    ))
                    fig.add_trace(go.Scatter(
                        x=[0, 1], y=[0, 1], mode="lines", name="Ideální",
                        line=dict(dash="dash", color="gray"),
                    ))
                    fig.update_layout(
                        title=f"{outcome_labels.get(oc['outcome'])} (ECE {oc['ece']:.4f})",
                        xaxis=dict(range=[0, 1], title="Predikce"),
                        yaxis=dict(range=[0, 1], title="Skutečnost"),
                        height=300, showlegend=False,
                        margin=dict(t=50, b=40, l=40, r=10),
                    )
                    col.plotly_chart(fig, use_container_width=True)
