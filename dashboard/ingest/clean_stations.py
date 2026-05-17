#!/usr/bin/env python3
"""Étape 2 — raw_station_status → stations_clean.

Pour chaque ligne raw : map station_id -> station_index/name, quantize au pas 30 min
(closest-to-center si plusieurs raw dans le même slot), localize UTC+4, parse
vehicle_types JSON -> count_x2, drop colonnes constantes.

Sortie : stations_clean (PK timestamp, station_index), is_imputed=0 pour les vraies
observations. Le comblement des trous est fait par fill_gaps.py (étape 3).

Idempotent : INSERT OR REPLACE, mais ne touche que les lignes is_imputed=0 (pour
éviter d'écraser une imputation par une autre imputation entre deux runs).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from db import connect  # noqa: E402

from _common import GRID_MIN, TZ_REUNION, fetch_station_info, pipeline_step  # noqa: E402

NUMERIC_COLS = [
    "num_docks_available", "num_docks_disabled",
    "num_vehicles_available", "num_vehicles_disabled",
]

INSERT_SQL = """INSERT OR REPLACE INTO stations_clean
    (timestamp, station_index, station_name, lat, lon, capacity,
     num_docks_available, num_docks_disabled,
     num_vehicles_available, num_vehicles_disabled,
     count_x2, is_imputed)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)"""

TS_FMT = "%Y-%m-%dT%H:%M:%S%z"


def extract_x2(json_str) -> int:
    if not isinstance(json_str, str):
        return 0
    try:
        for vt in json.loads(json_str):
            if vt.get("vehicle_type_id") == "x2":
                return int(vt.get("count", 0))
    except Exception:
        pass
    return 0


def clean(conn) -> int:
    raw = pd.read_sql(
        "SELECT * FROM raw_station_status",
        conn, parse_dates=["ts_collect"],
    )
    if raw.empty:
        return 0

    # Mapping station_id -> info
    info = fetch_station_info()
    info_map = {r.station_id: r for r in info.itertuples()}
    next_idx = info["station_index"].max() + 1
    extra = []
    for sid in raw["station_id"].unique():
        if sid not in info_map:
            extra.append({
                "station_id": sid, "station_index": next_idx,
                "station_name": "", "lat": None, "lon": None, "capacity": 0,
            })
            next_idx += 1
    if extra:
        info = pd.concat([info, pd.DataFrame(extra)], ignore_index=True)
    df = raw.merge(
        info[["station_id", "station_index", "station_name", "lat", "lon", "capacity"]],
        on="station_id", how="left",
    ).drop(columns=["station_id"])

    # Quantize 30-min : closest-to-center par (station_index, slot)
    df["ts_collect"] = pd.to_datetime(df["ts_collect"], utc=True)
    df["slot"] = df["ts_collect"].dt.round(f"{GRID_MIN}min")
    df["dist"] = (df["ts_collect"] - df["slot"]).abs()
    df = (df.sort_values("dist")
            .drop_duplicates(["station_index", "slot"], keep="first")
            .drop(columns=["dist", "ts_collect"])
            .rename(columns={"slot": "timestamp"}))

    # Localize UTC -> UTC+4
    df["timestamp"] = df["timestamp"].dt.tz_convert(TZ_REUNION)

    # Parse vehicle_types_available -> count_x2
    df["count_x2"] = df["vehicle_types_available"].apply(extract_x2)

    # Casts
    for c in NUMERIC_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").round().astype("Int64")
    df["station_index"] = df["station_index"].astype(int)
    df["capacity"] = df["capacity"].fillna(0).astype(int)

    # Format timestamp ISO 8601 + tz pour SQLite
    df["timestamp"] = df["timestamp"].dt.strftime(TS_FMT)

    rows = [
        (
            r.timestamp, int(r.station_index), r.station_name,
            float(r.lat) if pd.notna(r.lat) else None,
            float(r.lon) if pd.notna(r.lon) else None,
            int(r.capacity),
            None if pd.isna(r.num_docks_available) else int(r.num_docks_available),
            None if pd.isna(r.num_docks_disabled) else int(r.num_docks_disabled),
            None if pd.isna(r.num_vehicles_available) else int(r.num_vehicles_available),
            None if pd.isna(r.num_vehicles_disabled) else int(r.num_vehicles_disabled),
            int(r.count_x2),
        )
        for r in df.itertuples(index=False)
    ]
    cur = conn.cursor()
    cur.executemany(INSERT_SQL, rows)
    return len(rows)


def upsert_stations(conn) -> None:
    """Synchronise la table stations (métadonnées) depuis l'API GBFS."""
    info = fetch_station_info()
    conn.executemany(
        "INSERT OR IGNORE INTO stations (station_index, station_name, lat, lon, capacity) "
        "VALUES (?, ?, ?, ?, ?)",
        [(int(r.station_index), r.station_name,
          float(r.lat) if pd.notna(r.lat) else None,
          float(r.lon) if pd.notna(r.lon) else None,
          int(r.capacity)) for r in info.itertuples()],
    )


def main():
    with pipeline_step("clean_stations"):
        conn = connect()
        try:
            upsert_stations(conn)
            n = clean(conn)
            conn.commit()
        finally:
            conn.close()
        print(f"OK clean_stations — {n} lignes (INSERT OR REPLACE, is_imputed=0)")


if __name__ == "__main__":
    main()
