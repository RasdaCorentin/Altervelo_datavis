"""Page Monitoring : qualité du modèle dans le temps + détail par station."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from data import (
    get_mae_timeseries, get_monitoring_global, get_monitoring_rounded,
    get_monitoring_per_station, get_pipeline_health, get_prediction_traces,
    get_training_export,
)
from components import (
    color_mae, render_station_map, section_header,
    kpi_card, freshness_badge,
)
from glossary import help_badge


def render() -> None:
    if "export_df" not in st.session_state:
        st.session_state.export_df = None

    section_header(
        "Monitoring du modèle",
        "Qualité du modèle XGBoost v3 mesurée sur backtest + live (delayed feedback).",
        term="mae",
    )

    horizon = st.segmented_control(
        "Horizon évalué",
        options=[60, 90, 120, 150],
        format_func=lambda h: f"T + {h} min",
        default=60,
        key="monit_horizon",
    )
    if horizon is None:
        horizon = 60

    kpi = get_monitoring_global(horizon)
    health = get_pipeline_health()

    freshness_badge(health["last_obs"], label=f"Dernière obs {health['last_obs']}")
    st.caption(
        f"Évalué sur **{kpi['n']:,}** prédictions (backtest + live) — "
        f"dernière pred live : {health['last_pred'] or '–'}"
    )
    st.write("")

    if kpi["n"] == 0:
        st.warning("Aucune prédiction évaluée pour cet horizon.")
        return

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("MAE modèle", kpi["mae_model"], term="mae", fmt=".3f",
                 hint="erreur absolue moyenne, en vélos")
    with c2:
        kpi_card("MAE persistance", kpi["mae_pers"], term="persistance", fmt=".3f",
                 hint="baseline « rien ne change »")
    with c3:
        kpi_card("Hit rate ±1 vélo", kpi["hit_rate_1"] * 100, term="hit_rate", fmt=".1f",
                 hint="% de preds à ≤ 1 vélo près")
    with c4:
        kpi_card("Hit rate ±2 vélos", kpi["hit_rate_2"] * 100, term="hit_rate", fmt=".1f",
                 hint="% de preds à ≤ 2 vélos près")

    st.write("")
    lift = kpi["lift_pct"]
    if lift >= 0:
        st.success(
            f"Le modèle **bat** la persistance de **{lift:+.1f}%** "
            f"sur l'historique évalué à {horizon} min."
        )
    else:
        st.error(
            f"La persistance **domine** de **{-lift:.1f}%** à {horizon} min — "
            f"attendu sur séries quasi-stationnaires."
        )

    with st.expander("Lecture honnête : MAE arrondie (entier vs entier)", expanded=False):
        rk = get_monitoring_rounded(horizon)
        st.caption(
            "Les MAE brutes ci-dessus comparent un entier (persistance) à un nombre continu "
            "(modèle), ce qui gonfle artificiellement l'erreur du modèle de ~0.5 vélo par "
            "fraction même quand il vise la bonne classe. La MAE arrondie remplace y_pred par "
            "round(y_pred) pour remettre les deux prédicteurs sur la même échelle entière. "
            "À utiliser comme contrepoint, pas comme métrique de remplacement — l'arrondi "
            "masque la nuance continue (intervalles, tendances) que le modèle apporte."
        )
        rc1, rc2, rc3 = st.columns(3)
        with rc1:
            kpi_card("MAE persistance", rk["mae_pers"], term="persistance", fmt=".3f",
                     hint="déjà entière par construction")
        with rc2:
            kpi_card("MAE modèle (arrondie)", rk["mae_model_round"],
                     term="mae_arrondie", fmt=".3f",
                     hint="round(y_pred) vs y_obs")
        with rc3:
            lift_r = rk["lift_pct"]
            verdict = "modèle bat persistance" if lift_r >= 0 else "persistance domine"
            kpi_card("Lift recalculé", lift_r, term="lift", fmt="+.1f",
                     accent=("#16a34a" if lift_r >= 0 else "#dc2626"),
                     hint=f"{verdict} (en %)")
        st.caption(
            f"Le modèle (arrondi) et la persistance prédisent **le même entier** dans "
            f"**{rk['pct_same_class'] * 100:.1f}%** des cas évalués — "
            f"les désaccords restants concentrent la vraie valeur ajoutée du modèle."
        )

    st.write("")
    section_header(f"MAE rolling — horizon {horizon} min",
                   "Moyenne par heure des erreurs absolues.", term="mae")
    ts = get_mae_timeseries(horizon, freq="h")
    if ts.empty:
        st.info("Pas encore de série temporelle.")
    else:
        st.line_chart(ts, height=280)

    per_st = get_monitoring_per_station(horizon)
    if per_st.empty:
        st.warning("Aucune donnée évaluée à cet horizon.")
        return

    st.write("")
    section_header(
        "Carte des stations",
        "Couleur = MAE moyenne par station (vert = parfait, rouge = stations ratées).",
    )
    MAX_MAE_FIXED = 0.1
    st.caption(
        f"Échelle de couleur fixe : MAE = 0 → vert, MAE ≥ {MAX_MAE_FIXED} → rouge plein. "
        f"Repère : la MAE globale du modèle est de l'ordre de 0.1 vélo par station (1 vélo "
        f"d'erreur tous les 10 prédictions), donc tout ce qui dépasse cette borne est anormal."
    )
    per_st["color"] = per_st["mae_model"].apply(lambda m: color_mae(m, MAX_MAE_FIXED))
    tooltip = ("<b>{station_name}</b><br/>"
               "MAE modèle : {mae_model}<br/>"
               "MAE persistance : {mae_pers}<br/>"
               "n prédictions : {n}")
    render_station_map(per_st, tooltip_html=tooltip, height=420)

    st.write("")
    section_header(
        "Trajectoires prédites vs observées",
        "Pour une station donnée : ce qui a été observé, ce que la persistance "
        "annonçait et ce que le modèle annonçait — alignés sur le même axe temporel.",
        term="trace",
    )
    stations_sorted = per_st.sort_values("station_name")
    tcol1, tcol2 = st.columns([3, 1])
    with tcol1:
        sel_name = st.selectbox(
            "Station", stations_sorted["station_name"].tolist(), key="monit_trace_station",
        )
    with tcol2:
        days = st.selectbox(
            "Fenêtre", [3, 7, 14, 30, None],
            format_func=lambda d: "Tout l'historique" if d is None else f"{d} derniers jours",
            index=1, key="monit_trace_days",
        )
    sel_idx = int(stations_sorted.loc[stations_sorted["station_name"] == sel_name, "station_index"].iloc[0])
    traces = get_prediction_traces(horizon, sel_idx, days=days)
    if traces.empty:
        st.info("Aucune prédiction évaluée pour cette station à cet horizon (peut-être trop récent).")
    else:
        st.line_chart(traces, height=320)
        st.caption(
            f"La **persistance** copie l'observation avec un retard de {horizon} min — "
            f"c'est précisément le défaut que le **modèle** cherche à corriger. "
            f"Sur les stations calmes, les trois courbes se superposent presque ; "
            f"sur un hub, le modèle doit visiblement précéder l'observation."
        )

    cols_show = ["station_name", "n", "mae_model", "mae_pers", "lift_pct", "err_max"]
    rename = {
        "station_name": "Station", "n": "n",
        "mae_model": "MAE", "mae_pers": "MAE pers.",
        "lift_pct": "Lift %", "err_max": "Err. max",
    }
    fmt = {"MAE": "{:.3f}", "MAE pers.": "{:.3f}",
           "Lift %": "{:+.1f}", "Err. max": "{:.1f}"}

    st.write("")
    section_header("Top 5 stations", "Bien prédites vs ratées.")
    tab_ok, tab_ko = st.tabs(["Bien prédites (MAE basse)", "Ratées (MAE haute)"])
    with tab_ok:
        st.dataframe(
            per_st.nsmallest(5, "mae_model")[cols_show].rename(columns=rename)
                  .style.format(fmt),
            use_container_width=True, hide_index=True,
        )
    with tab_ko:
        st.dataframe(
            per_st.nlargest(5, "mae_model")[cols_show].rename(columns=rename)
                  .style.format(fmt),
            use_container_width=True, hide_index=True,
        )

    with st.expander("Détail complet (debug)"):
        st.dataframe(
            per_st[cols_show].rename(columns=rename).style.format(fmt),
            use_container_width=True, hide_index=True,
        )
        st.caption(
            f"Volumétrie : {health['n_obs']:,} obs ({health['n_obs_live']:,} live) · "
            f"{health['n_pred_back']:,} preds backtest · {health['n_pred_live']:,} preds live"
        )

    st.write("")
    cols = st.columns([6, 1])
    with cols[0]:
        st.caption("Le modèle peut perdre contre la persistance à 60 min ?")
    with cols[1]:
        with st.popover("Pourquoi ?"):
            st.markdown(
                "Sur des séries quasi-stationnaires (vélos peu utilisés sur de longues plages), "
                "« ne rien prédire » est une baseline très forte. Le modèle XGBoost ne reprend "
                "l'avantage que sur les hubs (Port, Casabona) où les flux dominent."
            )

    st.write("")
    with st.expander("Export CSV d'entraînement", expanded=False):
        st.caption(
            "Exporte toutes les observations DB (historique + live) jointes aux métadonnées "
            "station — CSV prêt pour `train_xgboostplus_delta.py`."
        )
        if st.button("Préparer l'export", key="btn_export_training"):
            with st.spinner("Extraction en cours..."):
                st.session_state.export_df = get_training_export()
        if st.session_state.export_df is not None:
            df_exp = st.session_state.export_df
            st.success(
                f"{len(df_exp):,} lignes · {df_exp['station_index'].nunique()} stations · "
                f"{str(df_exp['timestamp'].min())[:10]} → {str(df_exp['timestamp'].max())[:10]}"
            )
            st.download_button(
                "⬇ Télécharger observations_export.csv",
                data=df_exp.to_csv(index=False).encode("utf-8"),
                file_name="observations_export.csv",
                mime="text/csv",
                key="dl_training_csv",
            )
