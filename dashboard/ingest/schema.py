#!/usr/bin/env python3
"""Crée toutes les tables SQLite nécessaires au dashboard (idempotent).

Couvre les tables d'ingestion (raw_* + clean_*) ET les tables dashboard
(stations, observations avec flux véhicules, predictions, pipeline_runs).
Sûr de relancer plusieurs fois : CREATE TABLE IF NOT EXISTS partout.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from db import connect

SCHEMA = """
CREATE TABLE IF NOT EXISTS raw_station_status (
    ts_collect TEXT NOT NULL,
    station_id TEXT NOT NULL,
    is_installed INTEGER,
    is_renting INTEGER,
    is_returning INTEGER,
    num_docks_available INTEGER,
    num_docks_disabled INTEGER,
    num_vehicles_available INTEGER,
    num_vehicles_disabled INTEGER,
    last_reported INTEGER,
    vehicle_types_available TEXT,
    PRIMARY KEY (ts_collect, station_id)
);
CREATE INDEX IF NOT EXISTS idx_rss_ts ON raw_station_status(ts_collect);

CREATE TABLE IF NOT EXISTS raw_vehicle_status (
    ts_collect TEXT NOT NULL,
    vehicle_id TEXT NOT NULL,
    vehicle_type_id TEXT,
    lat REAL,
    lon REAL,
    current_fuel_percent REAL,
    current_range_meters REAL,
    is_disabled INTEGER,
    is_reserved INTEGER,
    last_reported INTEGER,
    station_id TEXT,
    PRIMARY KEY (ts_collect, vehicle_id)
);
CREATE INDEX IF NOT EXISTS idx_rvs_ts ON raw_vehicle_status(ts_collect);

CREATE TABLE IF NOT EXISTS stations_clean (
    timestamp TEXT NOT NULL,
    station_index INTEGER NOT NULL,
    station_name TEXT,
    lat REAL,
    lon REAL,
    capacity INTEGER,
    num_docks_available INTEGER,
    num_docks_disabled INTEGER,
    num_vehicles_available INTEGER,
    num_vehicles_disabled INTEGER,
    count_x2 INTEGER,
    is_imputed INTEGER,
    PRIMARY KEY (timestamp, station_index)
);
CREATE INDEX IF NOT EXISTS idx_sc_ts ON stations_clean(timestamp);

CREATE TABLE IF NOT EXISTS vehicles_clean (
    timestamp TEXT NOT NULL,
    vehicle_id TEXT NOT NULL,
    station_index INTEGER NOT NULL,
    lat REAL,
    lon REAL,
    current_fuel_percent REAL,
    current_range_meters REAL,
    is_disabled INTEGER,
    PRIMARY KEY (timestamp, vehicle_id)
);
CREATE INDEX IF NOT EXISTS idx_vc_ts ON vehicles_clean(timestamp);

CREATE TABLE IF NOT EXISTS stations (
    station_index INTEGER PRIMARY KEY,
    station_name  TEXT NOT NULL,
    lat REAL, lon REAL,
    capacity INTEGER
);

CREATE TABLE IF NOT EXISTS observations (
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
    source TEXT NOT NULL DEFAULT 'live',
    PRIMARY KEY (timestamp, station_index)
);
CREATE INDEX IF NOT EXISTS idx_obs_ts ON observations(timestamp);

CREATE TABLE IF NOT EXISTS predictions (
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
CREATE INDEX IF NOT EXISTS idx_pred_target ON predictions(ts_target, station_index);
CREATE INDEX IF NOT EXISTS idx_pred_horizon ON predictions(horizon_min);

CREATE TABLE IF NOT EXISTS pipeline_runs (
    ts_run TEXT PRIMARY KEY,
    duration_ms INTEGER,
    n_obs_inserted INTEGER,
    n_pred_inserted INTEGER,
    status TEXT,
    error_msg TEXT
);
"""


def main():
    conn = connect()
    conn.executescript(SCHEMA)
    conn.commit()
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()]
    conn.close()
    print(f"OK — tables présentes : {', '.join(tables)}")


if __name__ == "__main__":
    main()
