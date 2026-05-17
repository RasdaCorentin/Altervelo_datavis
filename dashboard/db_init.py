#!/usr/bin/env python3
"""One-shot : CSV → SQLite + pré-remplissage des prédictions backtest.

À lancer une fois pour bootstrap velos.db. Idempotent : DROP+CREATE des tables
à chaque appel, donc rejouer écrase tout (pas de migration incrémentale).

Sortie attendue :
  - 31 stations
  - ~38 999 observations historiques
  - ~31 000 prédictions backtest (4 horizons × ~7 800 lignes)
"""
from __future__ import annotations

import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

DASHBOARD_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DASHBOARD_DIR.parent
sys.path.insert(0, str(DASHBOARD_DIR))

from features import (
    HORIZON_STEPS, build_features, feature_columns,
)

DB_PATH = DASHBOARD_DIR / "velos.db"
STATIONS_CSV = PROJECT_ROOT / "data" / "stations_clean.csv"
ENRICHED_CSV = PROJECT_ROOT / "data" / "stations_enriched.csv"
FLOW_CSV     = PROJECT_ROOT / "data" / "vehicle_flow.csv"

def model_path(h_min: int) -> Path:
    return PROJECT_ROOT / f"xgb_velos_v4_h{h_min}min.json"

TEST_RATIO = 0.2
TS_FMT = "%Y-%m-%dT%H:%M:%S%z"  # ISO 8601 avec timezone — clé de JOIN cohérente

SCHEMA = """
DROP TABLE IF EXISTS stations;
DROP TABLE IF EXISTS observations;
DROP TABLE IF EXISTS predictions;
DROP TABLE IF EXISTS pipeline_runs;

CREATE TABLE stations (
    station_index INTEGER PRIMARY KEY,
    station_name  TEXT NOT NULL,
    lat REAL, lon REAL,
    capacity INTEGER
);

CREATE TABLE observations (
    timestamp TEXT NOT NULL,
    station_index INTEGER NOT NULL,
    num_vehicles_available INTEGER,
    num_docks_available INTEGER,
    num_docks_disabled INTEGER,
    num_vehicles_disabled INTEGER,
    is_imputed INTEGER,
    n_vehicles_actifs INTEGER,
    n_vehicles_disabled_obs INTEGER,
    mean_battery REAL, min_battery REAL, max_battery REAL, pct_low_battery REAL,
    n_transit_0_150m INTEGER, n_transit_150_300m INTEGER, n_transit_300_450m INTEGER,
    n_transit_450_600m INTEGER, n_transit_600_750m INTEGER, n_transit_750_900m INTEGER,
    dist_nearest_transit_m REAL,
    is_obs_missing INTEGER,
    n_arrivees_0_150m_30min REAL, n_departs_0_150m_30min REAL, net_flow_0_150m_30min REAL,
    n_arrivees_0_150m_60min REAL, n_departs_0_150m_60min REAL, net_flow_0_150m_60min REAL,
    n_arrivees_0_150m_120min REAL, n_departs_0_150m_120min REAL, net_flow_0_150m_120min REAL,
    n_arrivees_150_300m_30min REAL, n_departs_150_300m_30min REAL, net_flow_150_300m_30min REAL,
    n_arrivees_150_300m_60min REAL, n_departs_150_300m_60min REAL, net_flow_150_300m_60min REAL,
    n_arrivees_150_300m_120min REAL, n_departs_150_300m_120min REAL, net_flow_150_300m_120min REAL,
    n_arrivees_300_450m_60min REAL, n_departs_300_450m_60min REAL,
    n_arrivees_450_600m_60min REAL, n_departs_450_600m_60min REAL,
    source TEXT NOT NULL DEFAULT 'historical',
    PRIMARY KEY (timestamp, station_index)
);
CREATE INDEX idx_obs_ts ON observations(timestamp);

CREATE TABLE predictions (
    ts_pred TEXT NOT NULL,
    ts_target TEXT NOT NULL,
    horizon_min INTEGER NOT NULL,
    station_index INTEGER NOT NULL,
    y_pred REAL NOT NULL,
    y_current REAL NOT NULL,
    y_obs REAL,
    err_model REAL,
    err_pers REAL,
    source TEXT NOT NULL DEFAULT 'live',
    PRIMARY KEY (ts_pred, horizon_min, station_index)
);
CREATE INDEX idx_pred_target ON predictions(ts_target, station_index);
CREATE INDEX idx_pred_horizon ON predictions(horizon_min);

CREATE TABLE pipeline_runs (
    ts_run TEXT PRIMARY KEY,
    duration_ms INTEGER,
    n_obs_inserted INTEGER,
    n_pred_inserted INTEGER,
    status TEXT,
    error_msg TEXT
);
"""

OBS_COLS = [
    "timestamp", "station_index",
    "num_vehicles_available", "num_docks_available",
    "num_docks_disabled", "num_vehicles_disabled",
    "is_imputed", "n_vehicles_actifs", "n_vehicles_disabled_obs",
    "mean_battery", "min_battery", "max_battery", "pct_low_battery",
    "n_transit_0_150m", "n_transit_150_300m", "n_transit_300_450m",
    "n_transit_450_600m", "n_transit_600_750m", "n_transit_750_900m",
    "dist_nearest_transit_m", "is_obs_missing",
    "n_arrivees_0_150m_30min", "n_departs_0_150m_30min", "net_flow_0_150m_30min",
    "n_arrivees_0_150m_60min", "n_departs_0_150m_60min", "net_flow_0_150m_60min",
    "n_arrivees_0_150m_120min", "n_departs_0_150m_120min", "net_flow_0_150m_120min",
    "n_arrivees_150_300m_30min", "n_departs_150_300m_30min", "net_flow_150_300m_30min",
    "n_arrivees_150_300m_60min", "n_departs_150_300m_60min", "net_flow_150_300m_60min",
    "n_arrivees_150_300m_120min", "n_departs_150_300m_120min", "net_flow_150_300m_120min",
    "n_arrivees_300_450m_60min", "n_departs_300_450m_60min",
    "n_arrivees_450_600m_60min", "n_departs_450_600m_60min",
]


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


def load_enriched(parse_dates: bool = True) -> pd.DataFrame:
    """Charge stations_enriched.csv et merge vehicle_flow.csv si disponible."""
    df = pd.read_csv(ENRICHED_CSV, parse_dates=["timestamp"] if parse_dates else False)
    if FLOW_CSV.exists():
        flow = pd.read_csv(FLOW_CSV, parse_dates=["timestamp"] if parse_dates else False)
        before = df.shape[1]
        df = df.merge(flow, on=["timestamp", "station_index"], how="left")
        print(f"   vehicle_flow.csv mergé : +{df.shape[1] - before} colonnes flux")
    else:
        print(f"   ⚠  {FLOW_CSV.name} absent — colonnes flux seront NULL")
    return df


def load_stations(conn: sqlite3.Connection) -> int:
    df = pd.read_csv(STATIONS_CSV, parse_dates=["timestamp"])
    stations = (df[["station_index", "station_name", "lat", "lon", "capacity"]]
                .drop_duplicates("station_index")
                .sort_values("station_index"))
    stations.to_sql("stations", conn, if_exists="append", index=False)
    return len(stations)


def load_observations(conn: sqlite3.Connection) -> int:
    df = load_enriched()
    # Colonnes flux absentes (si vehicle_flow.csv manquant) → NULL
    for c in OBS_COLS:
        if c not in df.columns:
            df[c] = np.nan
    obs = df[OBS_COLS].copy()
    # Normalise en UTC pour aligner avec les données live (merge.py stocke en UTC)
    obs["timestamp"] = obs["timestamp"].dt.tz_convert("UTC").dt.strftime(TS_FMT)
    obs["source"] = "historical"
    obs.to_sql("observations", conn, if_exists="append", index=False, chunksize=5000)
    return len(obs)


def temporal_split(df: pd.DataFrame, ratio: float = TEST_RATIO) -> pd.DataFrame:
    cutoff = df["timestamp"].quantile(1 - ratio)
    return df[df["timestamp"] >= cutoff].copy()


def backtest_predictions(conn: sqlite3.Connection) -> int:
    df = load_enriched()
    df = df.sort_values(["station_index", "timestamp"]).reset_index(drop=True)

    total = 0
    for h_min, h_steps in HORIZON_STEPS.items():
        path = model_path(h_min)
        if not path.exists():
            print(f"   horizon {h_min}min : modèle absent ({path.name}), skip")
            continue

        feats = build_features(df, horizon_steps=h_steps)
        test = temporal_split(feats)
        if test.empty:
            print(f"   horizon {h_min}min : test set vide, skip")
            continue

        cols = feature_columns(test)
        X = test[cols]

        model = xgb.Booster()
        model.load_model(str(path))
        X_ordered = X[model.feature_names]
        delta_pred = model.predict(xgb.DMatrix(X_ordered, enable_categorical=True))

        ts_pred = test["timestamp"].dt.tz_convert("UTC")
        ts_target = ts_pred + pd.Timedelta(minutes=h_min)
        y_current = test["current_value"].astype(float).values
        y_pred = np.clip(y_current + delta_pred, 0, None)

        out = pd.DataFrame({
            "ts_pred": ts_pred.dt.strftime(TS_FMT).values,
            "ts_target": ts_target.dt.strftime(TS_FMT).values,
            "horizon_min": h_min,
            "station_index": test["station_index"].astype(int).values,
            "y_pred": y_pred,
            "y_current": y_current,
            "y_obs": np.nan,
            "err_model": np.nan,
            "err_pers": np.nan,
            "source": "backtest",
        })
        out.to_sql("predictions", conn, if_exists="append", index=False, chunksize=5000)
        print(f"   horizon {h_min}min : {len(out):,} prédictions backtest")
        total += len(out)
    return total


def backfill_predictions(conn: sqlite3.Connection) -> int:
    """JOIN predictions ↔ observations (ts_target == observation.timestamp)."""
    cur = conn.cursor()
    cur.execute("""
        UPDATE predictions
        SET y_obs = (
            SELECT o.num_vehicles_available
            FROM observations o
            WHERE o.timestamp = predictions.ts_target
              AND o.station_index = predictions.station_index
        )
        WHERE y_obs IS NULL
    """)
    cur.execute("""
        UPDATE predictions
        SET err_model = ABS(y_pred - y_obs),
            err_pers  = ABS(y_current - y_obs)
        WHERE y_obs IS NOT NULL AND err_model IS NULL
    """)
    conn.commit()
    n = cur.execute("SELECT COUNT(*) FROM predictions WHERE y_obs IS NOT NULL").fetchone()[0]
    return n


def main():
    t0 = time.time()
    if DB_PATH.exists():
        print(f"⚠  {DB_PATH.name} existe → réinitialisation complète")
    print(f"1/5 schéma SQL → {DB_PATH.name}")
    conn = sqlite3.connect(DB_PATH)
    init_schema(conn)

    print("2/5 load stations…")
    n_stations = load_stations(conn)
    print(f"   {n_stations} stations")

    print("3/5 load observations historiques…")
    n_obs = load_observations(conn)
    print(f"   {n_obs:,} observations")

    print("4/5 backtest sur test set (4 horizons)…")
    n_pred = backtest_predictions(conn)
    print(f"   total : {n_pred:,} prédictions backtest")

    print("5/5 backfill y_obs / err_model / err_pers…")
    n_obs_filled = backfill_predictions(conn)
    coverage = (n_obs_filled / n_pred * 100) if n_pred else 0
    print(f"   {n_obs_filled:,} / {n_pred:,} predictions ont y_obs ({coverage:.1f}%)")

    conn.close()
    print(f"OK en {time.time() - t0:.1f}s — {DB_PATH}")


if __name__ == "__main__":
    main()
