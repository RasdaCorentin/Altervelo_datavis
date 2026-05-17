"""Requêtes SQL utilisées par les 3 pages, toutes cachées 5 min."""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

DASHBOARD_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DASHBOARD_DIR))

from db import DB_PATH


def _conn() -> sqlite3.Connection:
    # check_same_thread=False : Streamlit multi-thread
    return sqlite3.connect(DB_PATH, check_same_thread=False)


@st.cache_data(ttl=300)
def get_stations() -> pd.DataFrame:
    with _conn() as c:
        return pd.read_sql("SELECT * FROM stations ORDER BY station_index", c)


@st.cache_data(ttl=3600)
def get_station_volatility() -> pd.Series:
    """Std historique de num_vehicles_available par station_index.
    Sert de baseline adaptative : un Δ de 3 est énorme pour une petite station
    (std=0.8) mais normal pour un hub (std=4). NaN remplacé par 1 (neutre)."""
    with _conn() as c:
        df = pd.read_sql(
            "SELECT station_index, num_vehicles_available FROM observations",
            c,
        )
    return (df.groupby("station_index")["num_vehicles_available"]
              .std().fillna(1.0))


@st.cache_data(ttl=300)
def get_latest_snapshot() -> tuple[str, pd.DataFrame]:
    """Retourne (timestamp, df enrichi avec lat/lon/capacity/name)."""
    with _conn() as c:
        latest = c.execute(
            "SELECT MAX(timestamp) FROM observations"
        ).fetchone()[0]
        df = pd.read_sql("""
            SELECT o.*, s.station_name, s.lat, s.lon, s.capacity
            FROM observations o
            JOIN stations s USING (station_index)
            WHERE o.timestamp = ?
            ORDER BY station_index
        """, c, params=(latest,))
    return latest, df


@st.cache_data(ttl=300)
def get_latest_predictions(horizon_min: int) -> tuple[str, pd.DataFrame]:
    """Dernières prédictions live pour un horizon donné (jointes aux stations)."""
    with _conn() as c:
        latest = c.execute(
            "SELECT MAX(ts_pred) FROM predictions WHERE source='live' AND horizon_min=?",
            (horizon_min,),
        ).fetchone()[0]
        if latest is None:
            return None, pd.DataFrame()
        df = pd.read_sql("""
            SELECT p.*, s.station_name, s.lat, s.lon, s.capacity
            FROM predictions p
            JOIN stations s USING (station_index)
            WHERE p.ts_pred = ? AND p.horizon_min = ?
            ORDER BY station_index
        """, c, params=(latest, horizon_min))
    return latest, df


@st.cache_data(ttl=300)
def get_monitoring_global(horizon_min: int) -> dict:
    """KPI globaux pour un horizon : MAE rolling, lift, n."""
    with _conn() as c:
        row = c.execute("""
            SELECT
              COUNT(*) AS n,
              AVG(err_model) AS mae_model,
              AVG(err_pers)  AS mae_pers,
              AVG(CASE WHEN err_model <= 1 THEN 1.0 ELSE 0.0 END) AS hit_rate_1,
              AVG(CASE WHEN err_model <= 2 THEN 1.0 ELSE 0.0 END) AS hit_rate_2
            FROM predictions
            WHERE horizon_min = ? AND y_obs IS NOT NULL
        """, (horizon_min,)).fetchone()
    n, mae_m, mae_p, hr1, hr2 = row
    lift = (mae_p - mae_m) / mae_p * 100 if mae_p else 0
    return dict(n=n, mae_model=mae_m, mae_pers=mae_p,
                lift_pct=lift, hit_rate_1=hr1, hit_rate_2=hr2)


@st.cache_data(ttl=300)
def get_monitoring_per_station(horizon_min: int) -> pd.DataFrame:
    """MAE par station pour un horizon — pour identifier les stations difficiles."""
    with _conn() as c:
        df = pd.read_sql("""
            SELECT s.station_name, p.station_index,
                   COUNT(*) AS n,
                   AVG(p.err_model) AS mae_model,
                   AVG(p.err_pers)  AS mae_pers,
                   MAX(p.err_model) AS err_max,
                   s.lat, s.lon, s.capacity
            FROM predictions p
            JOIN stations s USING (station_index)
            WHERE p.horizon_min = ? AND p.y_obs IS NOT NULL
            GROUP BY s.station_name, p.station_index, s.lat, s.lon, s.capacity
            ORDER BY mae_model DESC
        """, c, params=(horizon_min,))
    df["lift_pct"] = (df["mae_pers"] - df["mae_model"]) / df["mae_pers"].replace(0, pd.NA) * 100
    return df


@st.cache_data(ttl=300)
def get_mae_timeseries(horizon_min: int, freq: str = "h") -> pd.DataFrame:
    """MAE rolling agrégée par bucket temporel (heure par défaut)."""
    with _conn() as c:
        df = pd.read_sql("""
            SELECT ts_target, err_model, err_pers
            FROM predictions
            WHERE horizon_min = ? AND y_obs IS NOT NULL
            ORDER BY ts_target
        """, c, params=(horizon_min,), parse_dates=["ts_target"])
    if df.empty:
        return df
    df = df.set_index("ts_target")
    g = df.resample(freq).mean()
    g.columns = ["MAE modèle", "MAE persistance"]
    return g.dropna()


@st.cache_data(ttl=300)
def get_prediction_traces(horizon_min: int, station_index: int,
                          days: int | None = 7) -> pd.DataFrame:
    """Trois courbes alignées sur ts_target pour une station + un horizon.

    Colonnes : Observation, Persistance, Modèle. Index = ts_target (datetime).
    days=None → tout l'historique évalué ; sinon limite aux N derniers jours.
    """
    where_days = ""
    params: tuple = (horizon_min, station_index)
    if days is not None:
        where_days = "AND ts_target >= datetime('now', ?)"
        params = (horizon_min, station_index, f"-{int(days)} days")
    with _conn() as c:
        df = pd.read_sql(f"""
            SELECT ts_target,
                   y_obs     AS Observation,
                   y_current AS Persistance,
                   y_pred    AS "Modèle"
            FROM predictions
            WHERE horizon_min = ?
              AND station_index = ?
              AND y_obs IS NOT NULL
              {where_days}
            ORDER BY ts_target
        """, c, params=params, parse_dates=["ts_target"])
    return df.set_index("ts_target") if not df.empty else df


@st.cache_data(ttl=300)
def get_monitoring_rounded(horizon_min: int) -> dict:
    """MAE comparable entier-vs-entier : round(y_pred) vs y_current.

    Calcul à la volée, pas de stockage. pct_same_class = part des cas où
    le modèle (arrondi) prédit le même entier que la persistance.
    """
    with _conn() as c:
        row = c.execute("""
            SELECT
              COUNT(*) AS n,
              AVG(ABS(y_obs - ROUND(y_pred)))    AS mae_model_round,
              AVG(ABS(y_obs - y_current))        AS mae_pers,
              AVG(CASE WHEN ROUND(y_pred) = y_current THEN 1.0 ELSE 0.0 END) AS pct_same_class
            FROM predictions
            WHERE horizon_min = ? AND y_obs IS NOT NULL
        """, (horizon_min,)).fetchone()
    n, mae_m, mae_p, same = row
    lift = (mae_p - mae_m) / mae_p * 100 if mae_p else 0
    return dict(n=n, mae_model_round=mae_m, mae_pers=mae_p,
                lift_pct=lift, pct_same_class=same)


@st.cache_data(ttl=60)
def get_pipeline_health() -> dict:
    """Fraîcheur des données : combien de temps depuis le dernier snapshot."""
    with _conn() as c:
        last_obs = c.execute("SELECT MAX(timestamp) FROM observations").fetchone()[0]
        last_pred = c.execute(
            "SELECT MAX(ts_pred) FROM predictions WHERE source='live'"
        ).fetchone()[0]
        n_obs = c.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
        n_obs_live = c.execute(
            "SELECT COUNT(*) FROM observations WHERE source='live'"
        ).fetchone()[0]
        n_pred_live = c.execute(
            "SELECT COUNT(*) FROM predictions WHERE source='live'"
        ).fetchone()[0]
        n_pred_back = c.execute(
            "SELECT COUNT(*) FROM predictions WHERE source='backtest'"
        ).fetchone()[0]
    return dict(
        last_obs=last_obs, last_pred=last_pred,
        n_obs=n_obs, n_obs_live=n_obs_live,
        n_pred_live=n_pred_live, n_pred_back=n_pred_back,
    )


@st.cache_data(ttl=300)
def get_training_export() -> pd.DataFrame:
    """Export complet observations JOIN stations — CSV prêt pour ré-entraîner le modèle."""
    with _conn() as c:
        return pd.read_sql("""
            SELECT o.timestamp, o.station_index,
                   s.station_name, s.lat, s.lon, s.capacity,
                   o.num_vehicles_available, o.num_docks_available,
                   o.num_docks_disabled, o.num_vehicles_disabled,
                   o.is_imputed, o.n_vehicles_actifs, o.n_vehicles_disabled_obs,
                   o.mean_battery, o.min_battery, o.max_battery, o.pct_low_battery,
                   o.n_transit_0_150m, o.n_transit_150_300m, o.n_transit_300_450m,
                   o.n_transit_450_600m, o.n_transit_600_750m, o.n_transit_750_900m,
                   o.dist_nearest_transit_m, o.is_obs_missing, o.source
            FROM observations o
            JOIN stations s USING (station_index)
            ORDER BY o.timestamp, o.station_index
        """, c)
