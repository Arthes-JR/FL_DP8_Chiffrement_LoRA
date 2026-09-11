"""
Dashboard Streamlit complet — Federated Learning (theme18).
Lance :  python -m streamlit run dashboard.py
"""
from __future__ import annotations
import re, time
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Federated Learning — Dashboard", page_icon="🔒", layout="wide")

ROOT = Path(__file__).resolve().parent.parent
CSV_GLOBAL = ROOT / "src" / "results" / "metrics_history.csv"
LOGS_DIRS = [ROOT / "src" / "logs", ROOT / "logs"]
FIGURES_DIR = ROOT / "figures"

# --------------------------------------------------------------------------
# Dossiers des 6 runs de comparaison (results_A ... results_F)
# --------------------------------------------------------------------------
COMPARE_DIRS = {
    "A : FedAvg secure (sans decrochage)": ROOT / "src" / "results_A",
    "B : FedAvg secure (avec decrochage)": ROOT / "src" / "results_B",
    "C : Mediane       (sans decrochage)": ROOT / "src" / "results_C",
    "D : Mediane       (avec decrochage)": ROOT / "src" / "results_D",
    "E : Trimmed       (sans decrochage)": ROOT / "src" / "results_E",
    "F : Trimmed       (avec decrochage)": ROOT / "src" / "results_F",
}
COMPARE_CSV = FIGURES_DIR / "compare_strategies.csv"

# Palette stable pour les 6 runs
RUN_COLORS = {
    "A": "#2E86DE", "B": "#54A0FF",
    "C": "#EE5A24", "D": "#F79F1F",
    "E": "#A29BFE", "F": "#6C5CE7",
}


def pick_logs_dir():
    ex = [d for d in LOGS_DIRS if d.exists() and list(d.glob("*.log"))]
    return max(ex, key=lambda d: max(f.stat().st_mtime for f in d.glob("*.log"))) if ex else None


def _load_compare_csvs() -> dict:
    """Charge les metrics_history.csv des 6 runs disponibles."""
    out = {}
    for label, d in COMPARE_DIRS.items():
        csv = d / "metrics_history.csv"
        if csv.exists():
            try:
                out[label] = pd.read_csv(csv)
            except Exception as e:
                st.warning(f"Impossible de lire {csv} : {e}")
    return out


st.title("🔒 Federated Learning — Dashboard")
st.caption("Prototype fédéré — DP-SGD + LoRA + Shamir (theme18).")

with st.sidebar:
    st.header("⚙️ Options")
    auto = st.checkbox("🔄 Auto-rafraîchir (5 s)", value=False)
    logs_dir = pick_logs_dir()
    st.markdown("---")
    st.markdown(f"**CSV** : `{CSV_GLOBAL.relative_to(ROOT)}`")
    st.markdown(f"**Logs** : `{logs_dir.relative_to(ROOT) if logs_dir else '—'}`")
    st.markdown(f"**Figures** : `{FIGURES_DIR.relative_to(ROOT)}`")
    st.markdown("---")
    n_runs = sum(1 for d in COMPARE_DIRS.values() if (d / "metrics_history.csv").exists())
    st.markdown(f"**Comparaison** : {n_runs}/6 runs disponibles")

if auto:
    time.sleep(5); st.rerun()

tab_global, tab_clients, tab_dp, tab_model, tab_events, tab_figures, tab_compare, tab_raw = st.tabs(
    ["📊 Vue globale", "👥 Par client", "🔐 DP (ε)", "🧠 Modèle / LoRA",
     "🛡️ Événements", "🖼️ Figures", "🔬 Comparaison stratégies", "📄 Données brutes"]
)

# ============================================================ TAB 1
with tab_global:
    if not CSV_GLOBAL.exists():
        st.error(f"CSV introuvable : {CSV_GLOBAL}"); st.stop()
    df = pd.read_csv(CSV_GLOBAL)
    last = df.iloc[-1]
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Round", int(last["round"]))
    c2.metric("Accuracy", f"{last['accuracy']:.4f}")
    c3.metric("F1", f"{last['f1']:.4f}")
    c4.metric("Loss", f"{last['loss']:.4f}")
    c5.metric("Clients actifs", int(last["n_active_clients"]))
    if "epsilon" in df.columns:
        c6.metric("ε cumulé", f"{last['epsilon']:.3f}")
    st.markdown("---")
    cA, cB = st.columns(2)
    with cA:
        st.plotly_chart(px.line(df, x="round", y="accuracy", markers=True,
                                title="Accuracy par round"), use_container_width=True)
    with cB:
        st.plotly_chart(px.line(df, x="round", y="loss", markers=True,
                                title="Loss par round"), use_container_width=True)
    st.plotly_chart(px.line(df, x="round", y=["precision","recall","f1"],
                            markers=True, title="Précision / Rappel / F1"), use_container_width=True)
    st.plotly_chart(px.bar(df, x="round", y="n_active_clients",
                           title="Clients actifs par round"), use_container_width=True)
    st.plotly_chart(px.line(df, x="round", y="duration_s", markers=True,
                            title="Durée par round (s)"), use_container_width=True)

# ============================================================ TAB 2
with tab_clients:
    st.subheader("👥 Métriques par client")
    if not logs_dir:
        st.warning("Aucun dossier de logs."); st.stop()

    client_csvs = sorted(logs_dir.glob("client_*_metrics.csv"))
    if client_csvs:
        dfs = []
        for f in client_csvs:
            d = pd.read_csv(f); d["client"] = int(re.search(r"client_(\d+)", f.name).group(1))
            dfs.append(d)
        dfc = pd.concat(dfs, ignore_index=True)

        st.markdown("### ⏱️ Durée locale par client et par round (s)")
        piv = dfc.pivot_table(index="round", columns="client", values="duration_s")
        st.plotly_chart(px.line(piv, markers=True, labels={"value":"durée (s)"}),
                        use_container_width=True)

        if "grad_norm" in dfc.columns:
            st.markdown("### 📉 Norme L2 du pseudo-gradient (par client)")
            piv2 = dfc.pivot_table(index="round", columns="client", values="grad_norm")
            st.plotly_chart(px.line(piv2, markers=True, labels={"value":"||Δ||₂"}), 
                            use_container_width=True)

        st.markdown("### 🔥 Heatmap présence client × round")
        pres = dfc.pivot_table(index="client", columns="round", values="duration_s",
                               aggfunc="count").fillna(0)
        st.plotly_chart(px.imshow(pres, aspect="auto", color_continuous_scale="Greens",
                                  title="Client actif (≥1) / absent (0)"),
                        use_container_width=True)

        st.markdown("### 📋 Tableau détaillé")
        st.dataframe(dfc.sort_values(["client","round"]), use_container_width=True, height=400)

    client_logs = sorted(logs_dir.glob("client_*.log"))
    if client_logs:
        st.markdown("### 📄 Logs bruts")
        for lg in client_logs:
            with st.expander(lg.name):
                st.code(lg.read_text(encoding="utf-8", errors="replace"), language="text")

# ============================================================ TAB 3 — DP
with tab_dp:
    st.subheader("🔐 Confidentialité différentielle (DP-SGD)")
    st.markdown("### ⚙️ Paramètres DP (config)")
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("C (clip_norm)", "1.0")
    p2.metric("σ (noise_multiplier)", "0.6")
    p3.metric("δ (delta)", "1e-5")
    p4.metric("q (sampling_rate)", "1.0")

    if CSV_GLOBAL.exists():
        df = pd.read_csv(CSV_GLOBAL)
        if "epsilon" in df.columns:
            st.markdown("### 📈 Budget ε cumulé par round")
            fig = px.line(df, x="round", y="epsilon", markers=True,
                          title="ε cumulé (estimation Eq. 5.6)")
            fig.update_traces(line_color="#8E44AD")
            st.plotly_chart(fig, use_container_width=True)
            eps_last = df["epsilon"].iloc[-1]
            st.info(f"**ε final après {len(df)} rounds = {eps_last:.3f}** "
                    f"(pour δ = 1e-5). Plus ε est petit, plus la garantie DP est forte.")
        else:
            st.warning("⚠️ La colonne `epsilon` n'est pas encore dans le CSV. "
                       "Applique le patch sur `server.py` puis relance la simulation.")

# ============================================================ TAB 4 — Modèle
with tab_model:
    st.subheader("🧠 Modèle fédéré — LoRA + tête de sortie")
    st.markdown(r"""
**Équation centrale (Eq. 2.2 du mémoire)** :

$$h = W_0 x + BAx$$

| Composant | Dimensions | Statut |
|---|---|---|
| `W0` (poids de base) | (32, 20) | ❄️ **gelé**, seed=42, partagé partout |
| `A` (LoRA down)      | ( 8, 20) | 🔥 **entraîné**, transmis au serveur |
| `B` (LoRA up)        | (32,  8) | 🔥 **entraîné**, transmis au serveur |
| `w_out` (tête)       | (32,)    | 🔥 **entraîné** |
| `b_out` (biais)      | (1,)     | 🔥 **entraîné** |

**Rang LoRA `r = 8`** — appliqué sur la projection d'entrée (20 → 32).
""")
    d_in, d_h, r = 20, 32, 8
    n_lora = r*d_in + d_h*r
    n_head = d_h + 1
    n_base = d_h*d_in
    total = n_lora + n_head
    st.markdown("### 📊 Paramètres")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("LoRA (A+B)", f"{n_lora:,}")
    k2.metric("Tête de sortie", f"{n_head:,}")
    k3.metric("Base gelée W0", f"{n_base:,}")
    k4.metric("Total transmis / round", f"{total:,}")

# ============================================================ TAB 5
with tab_events:
    st.subheader("🛡️ Événements de sécurité")
    if logs_dir:
        srv = logs_dir / "server.log"
        if srv.exists():
            for line in srv.read_text(encoding="utf-8", errors="replace").splitlines():
                if any(k in line for k in ["decrochage","Shamir","reconstruite","connecte","configuration"]):
                    st.markdown(f"- `{line.strip()}`")
            with st.expander("Voir tout server.log"):
                st.code(srv.read_text(encoding="utf-8", errors="replace"), language="text")

# ============================================================ TAB 6
with tab_figures:
    st.subheader("🖼️ Figures générées par `visualize.py`")
    if FIGURES_DIR.exists():
        for png in sorted(FIGURES_DIR.glob("*.png")):
            st.markdown(f"### {png.stem}")
            st.image(str(png), use_container_width=True)

# ============================================================ TAB 7 — NOUVEAU : Comparaison
with tab_compare:
    st.subheader("🔬 Comparaison des 6 stratégies d'agrégation")
    st.caption("Lecture des dossiers `src/results_A` ... `src/results_F` et du CSV "
               "`figures/compare_strategies.csv`.")

    runs = _load_compare_csvs()

    if not runs:
        st.error("❌ Aucun résultat de comparaison trouvé. "
                 "Lance d'abord `python src/compare_strategies.py`.")
        st.stop()

    # ----- Bandeau de métriques finales -----
    st.markdown("### 📌 Métriques finales (dernier round)")
    cols = st.columns(len(runs))
    for col, (label, df_r) in zip(cols, runs.items()):
        suffix = label.split(":")[0].strip()
        last = df_r.iloc[-1]
        col.markdown(f"**{label}**")
        col.metric("F1",     f"{last['f1']:.4f}")
        col.metric("Acc",    f"{last['accuracy']:.4f}")
        col.metric("Rappel", f"{last['recall']:.4f}")
        col.metric("Rounds", int(last["round"]))

    st.markdown("---")

    # ----- 1. Courbes F1 superposées -----
    st.markdown("### 📈 Courbes F1 superposées (6 runs)")
    fig_f1 = go.Figure()
    for label, df_r in runs.items():
        suffix = label.split(":")[0].strip()
        fig_f1.add_trace(go.Scatter(
            x=df_r["round"], y=df_r["f1"],
            mode="lines",
            name=label,
            line=dict(color=RUN_COLORS.get(suffix, "#888"), width=2,
                      dash="solid" if "avec" in label else "dot"),
        ))
    fig_f1.update_layout(
        xaxis_title="Round", yaxis_title="F1",
        hovermode="x unified", height=500,
        legend=dict(orientation="h", yanchor="bottom", y=-0.35),
    )
    st.plotly_chart(fig_f1, use_container_width=True)

    # ----- 2. Courbes Accuracy / Loss superposées -----
    st.markdown("### 📉 Accuracy et Loss superposées")
    m1, m2 = st.columns(2)
    for col, metric in zip((m1, m2), ("accuracy", "loss")):
        fig = go.Figure()
        for label, df_r in runs.items():
            suffix = label.split(":")[0].strip()
            fig.add_trace(go.Scatter(
                x=df_r["round"], y=df_r[metric],
                mode="lines", name=label,
                line=dict(color=RUN_COLORS.get(suffix, "#888"),
                          dash="solid" if "avec" in label else "dot"),
            ))
        fig.update_layout(title=metric.capitalize(),
                          xaxis_title="Round", yaxis_title=metric,
                          height=420, showlegend=(metric == "accuracy"),
                          legend=dict(orientation="h", y=-0.3))
        col.plotly_chart(fig, use_container_width=True)

    # ----- 3. Barres groupées (métriques finales) -----
    st.markdown("### 📊 Barres comparatives des métriques finales")
    rows = []
    for label, df_r in runs.items():
        last = df_r.iloc[-1]
        rows.append({
            "run": label,
            "F1": last["f1"],
            "Accuracy": last["accuracy"],
            "Precision": last["precision"],
            "Rappel": last["recall"],
        })
    df_bar = pd.DataFrame(rows).set_index("run")
    fig_bar = px.bar(
        df_bar.reset_index().melt(id_vars="run", var_name="métrique", value_name="valeur"),
        x="run", y="valeur", color="métrique", barmode="group",
        title="Métriques finales par run",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig_bar.update_layout(xaxis_tickangle=-20, height=500)
    st.plotly_chart(fig_bar, use_container_width=True)

    # ----- 4. Impact du décrochage -----
    st.markdown("### 🔀 Impact du décrochage du client 3 (Δ F1)")
    pairs = [("A", "B", "FedAvg secure"),
             ("C", "D", "Médiane"),
             ("E", "F", "Trimmed")]
    impact_rows = []
    for a, b, name in pairs:
        la = next((l for l in runs if l.startswith(a)), None)
        lb = next((l for l in runs if l.startswith(b)), None)
        if la and lb:
            f1_no  = runs[la].iloc[-1]["f1"]
            f1_yes = runs[lb].iloc[-1]["f1"]
            impact_rows.append({
                "Stratégie": name,
                "Sans décrochage": f1_no,
                "Avec décrochage": f1_yes,
                "Δ F1": f1_yes - f1_no,
            })
    if impact_rows:
        df_imp = pd.DataFrame(impact_rows)
        fig_imp = go.Figure()
        fig_imp.add_trace(go.Bar(name="Sans décrochage",
                                 x=df_imp["Stratégie"], y=df_imp["Sans décrochage"],
                                 marker_color="#2E86DE",
                                 text=[f"{v:.3f}" for v in df_imp["Sans décrochage"]],
                                 textposition="outside"))
        fig_imp.add_trace(go.Bar(name="Avec décrochage",
                                 x=df_imp["Stratégie"], y=df_imp["Avec décrochage"],
                                 marker_color="#EE5A24",
                                 text=[f"{v:.3f}" for v in df_imp["Avec décrochage"]],
                                 textposition="outside"))
        fig_imp.update_layout(barmode="group", title="F1 final : avec vs sans décrochage",
                              yaxis_title="F1", height=450)
        st.plotly_chart(fig_imp, use_container_width=True)

        st.markdown("**Tableau récapitulatif de l'impact :**")
        st.dataframe(df_imp.style.format({
            "Sans décrochage": "{:.4f}",
            "Avec décrochage": "{:.4f}",
            "Δ F1": "{:+.4f}",
        }), use_container_width=True)

    # ----- 5. Radar des 3 stratégies -----
    st.markdown("### 🕸️ Radar des 3 stratégies (moyenne des 2 scénarios)")
    groups = {
        "FedAvg secure": [l for l in runs if l.startswith(("A", "B"))],
        "Médiane":       [l for l in runs if l.startswith(("C", "D"))],
        "Trimmed":       [l for l in runs if l.startswith(("E", "F"))],
    }
    radar_labels = ["F1", "Accuracy", "Precision", "Rappel", "1 - Loss"]
    fig_radar = go.Figure()
    for name, ls in groups.items():
        if not ls: continue
        agg = pd.concat([runs[l] for l in ls]).groupby("round").mean(numeric_only=True)
        vals = [agg["f1"].mean(), agg["accuracy"].mean(),
                agg["precision"].mean(), agg["recall"].mean(),
                1 - agg["loss"].mean()]
        vals += vals[:1]
        fig_radar.add_trace(go.Scatterpolar(
            r=vals, theta=radar_labels + radar_labels[:1],
            fill="toself", name=name,
        ))
    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                            height=500, showlegend=True)
    st.plotly_chart(fig_radar, use_container_width=True)

    # ----- 6. Tableau récapitulatif + téléchargement -----
    st.markdown("### 📋 Tableau récapitulatif (depuis `figures/compare_strategies.csv`)")
    if COMPARE_CSV.exists():
        df_summary = pd.read_csv(COMPARE_CSV)
        st.dataframe(df_summary, use_container_width=True)
        st.download_button(
            "⬇️ Télécharger compare_strategies.csv",
            df_summary.to_csv(index=False).encode("utf-8"),
            file_name="compare_strategies.csv", mime="text/csv",
        )
    else:
        st.info("`figures/compare_strategies.csv` introuvable — relance `compare_strategies.py`.")

    # ----- 7. Figures PNG déjà générées -----
    st.markdown("### 🖼️ Figures PNG de comparaison")
    for png_name in ["compare_strategies.png",
                     "compare_metrics_grouped.png",
                     "compare_dropout_impact.png",
                     "compare_radar.png"]:
        p = FIGURES_DIR / png_name
        if p.exists():
            st.markdown(f"**{png_name}**")
            st.image(str(p), use_container_width=True)

# ============================================================ TAB 8
with tab_raw:
    st.subheader("📄 metrics_history.csv")
    if CSV_GLOBAL.exists():
        df = pd.read_csv(CSV_GLOBAL)
        st.dataframe(df, use_container_width=True, height=600)
        st.download_button("⬇️ Télécharger le CSV",
                           df.to_csv(index=False).encode("utf-8"),
                           file_name="metrics_history.csv", mime="text/csv")