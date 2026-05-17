#!/usr/bin/env python3
"""Inférence live : prend le dernier snapshot, prédit les 4 horizons, INSERT.

Idempotent via INSERT OR IGNORE (PK = ts_pred + horizon_min + station_index).
Rejouer sur le même snapshot = no-op.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

DASHBOARD_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DASHBOARD_DIR.parent
sys.path.insert(0, str(DASHBOARD_DIR))

from db import connect
from features import HORIZON_STEPS, build_features, feature_columns

TS_FMT = "%Y-%m-%dT%H:%M:%S%z"


def model_path(h_min: int) -> Path:
    return PROJECT_ROOT / f"xgb_velos_v4_h{h_min}min.json"


INSERT_SQL = """INSERT OR IGNORE INTO predictions
    (ts_pred, ts_target, horizon_min, station_index,
     y_pred, y_current, y_obs, err_model, err_pers, source)
    VALUES (?, ?, ?, ?, ?, ?, NULL, NULL, NULL, 'live')"""


def predict_latest(conn) -> dict[int, int]:
    """Retourne {horizon_min: nb prédictions insérées}."""
    df = pd.read_sql("""
        SELECT o.*, s.capacity
        FROM observations o
        JOIN stations s USING (station_index)
        ORDER BY station_index, timestamp
    """, conn)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    if "source" in df.columns:
        df = df.drop(columns=["source"])

    feats = build_features(df)
    if feats.empty:
        print("⚠  features vides — pas assez d'historique pour les lags")
        return {}

    latest_ts = feats["timestamp"].max()
    snapshot = feats[feats["timestamp"] == latest_ts]
    print(f"snapshot @ {latest_ts}  ({len(snapshot)} stations)")

    cols = feature_columns(snapshot)
    X = snapshot[cols]
    y_current = snapshot["current_value"].astype(float).values
    station_idx = snapshot["station_index"].astype(int).values
    ts_pred_str = latest_ts.strftime(TS_FMT)

    inserted = {}
    for h_min in HORIZON_STEPS:
        path = model_path(h_min)
        if not path.exists():
            print(f"   {h_min}min : {path.name} absent, skip")
            continue
        model = xgb.Booster()
        model.load_model(str(path))
        # Aligne strictement l'ordre des colonnes sur celui vu à l'entraînement
        X_ordered = X[model.feature_names]
        delta_pred = model.predict(xgb.DMatrix(X_ordered, enable_categorical=True))
        y_pred = np.clip(y_current + delta_pred, 0, None)

        ts_target_str = (latest_ts + pd.Timedelta(minutes=h_min)).strftime(TS_FMT)
        rows = [
            (ts_pred_str, ts_target_str, h_min, int(station_idx[i]),
             float(y_pred[i]), float(y_current[i]))
            for i in range(len(snapshot))
        ]
        cur = conn.cursor()
        cur.executemany(INSERT_SQL, rows)
        conn.commit()
        inserted[h_min] = cur.rowcount
        print(f"   {h_min}min : +{cur.rowcount} lignes (sur {len(rows)})")
    return inserted


def main():
    t0 = time.time()
    conn = connect()
    inserted = predict_latest(conn)
    total = sum(inserted.values())
    conn.close()
    print(f"OK : {total} prédictions insérées en {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
