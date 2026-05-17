"""Construction des features XGBoost pour inférence et backtest.

Doit rester aligné avec train_xgboostplus_delta.py — toute divergence ici crée
un train/serve skew (le modèle reçoit des features différentes de celles vues
à l'entraînement et prédit n'importe quoi).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LAGS = [1, 2, 4, 6, 12, 48]
ROLL_WINDOWS = [4, 12, 48]
DIFF_LAGS = [1, 4, 48]
TARGET_COL = "num_vehicles_available"

# Doit rester strictement aligné avec script/train_xgboostplus_delta.py.
LAG_MODE = "full"  # "full" | "cyclic_only"
CYCLIC_DROP_FEATURES = {
    "current_value",
    "lag_1", "lag_2", "lag_4", "lag_6",
    "roll_mean_4", "roll_mean_12",
    "roll_std_4", "roll_std_12",
    "diff_1", "diff_4",
}

HORIZONS_MIN = [60, 90, 120, 150]
STEP_MIN = 30
HORIZON_STEPS = {h: h // STEP_MIN for h in HORIZONS_MIN}  # 60->2, 90->3, ...


def time_regime(hour: int) -> int:
    if 1 <= hour < 7:   return 0
    if 7 <= hour < 12:  return 1
    if 12 <= hour < 17: return 2
    if 17 <= hour < 21: return 3
    return 4


def build_features(df: pd.DataFrame, horizon_steps: int | None = None) -> pd.DataFrame:
    """Construit features lag/roll/diff/cyclical. Si horizon_steps, ajoute la cible Δy."""
    df = df.copy()
    ts = df["timestamp"]
    df["hour"] = ts.dt.hour
    df["dow"] = ts.dt.dayofweek
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)
    df["time_regime"] = df["hour"].apply(time_regime)

    g = df.groupby("station_index")[TARGET_COL]
    for lag in LAGS:
        df[f"lag_{lag}"] = g.shift(lag)
    for w in ROLL_WINDOWS:
        df[f"roll_mean_{w}"] = g.shift(1).rolling(w).mean().reset_index(0, drop=True)
        df[f"roll_std_{w}"] = g.shift(1).rolling(w).std().reset_index(0, drop=True)

    df["current_value"] = df[TARGET_COL]
    for d in DIFF_LAGS:
        df[f"diff_{d}"] = df[TARGET_COL] - g.shift(d)

    # flow_density : arrivées/départs 60min normalisés par densité locale
    local_density_cols = [c for c in ["n_transit_0_150m", "n_transit_150_300m"] if c in df.columns]
    if local_density_cols:
        local_density = df[local_density_cols].sum(axis=1)
        arr_60 = [c for c in df.columns if c.startswith("n_arrivees_") and c.endswith("_60min")]
        dep_60 = [c for c in df.columns if c.startswith("n_departs_") and c.endswith("_60min")]
        if arr_60:
            df["flow_density_arrivees_60min"] = df[arr_60].sum(axis=1) / (1.0 + local_density)
        if dep_60:
            df["flow_density_departs_60min"] = df[dep_60].sum(axis=1) / (1.0 + local_density)

    # station_index doit être catégoriel (aligné avec l'entraînement v4).
    df["station_index"] = df["station_index"].astype("category")

    needed = [f"lag_{l}" for l in LAGS] + [f"diff_{d}" for d in DIFF_LAGS]

    if horizon_steps is not None:
        y_future = g.shift(-horizon_steps)
        df["y_abs"] = y_future
        df["y"] = y_future - df[TARGET_COL]
        return df.dropna(subset=["y"] + needed).reset_index(drop=True)

    return df.dropna(subset=needed).reset_index(drop=True)


FEATURE_DROP_COLS = {
    "timestamp", "station_name", "y", "y_abs", TARGET_COL,
    "count_x2", "lat", "lon", "source",
}
if LAG_MODE == "cyclic_only":
    FEATURE_DROP_COLS = FEATURE_DROP_COLS | CYCLIC_DROP_FEATURES


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in FEATURE_DROP_COLS]
