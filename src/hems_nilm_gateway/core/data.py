from __future__ import annotations

# Zweck: Datenaufbereitung für Training/Evaluation (DEDDIAG-DB bzw. CSV)

from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# ---------- Normalisierung ----------
@dataclass
class ZScore:
    # Z-Score: Parameter für z-Normalisierung
    mean: float
    std: float

    @classmethod
    def fit(cls, x: np.ndarray) -> "ZScore":
        # Mittelwert und Standardabweichung
        m = float(np.mean(x))
        s = float(np.std(x))
        s = s + 1e-9  # Division durch 0 vermeiden
        return cls(m, s)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def to_json(self) -> dict:
        # Export: Parameter als JSON-kompatibles Dict
        return asdict(self)


# ---------- DEDDIAG laden als DataFrame ----------
def _to_df(
    item_id: int,
    t0: datetime,
    t1: datetime,
    db_cfg: Dict[str, Any],
    schema: str,
) -> pd.DataFrame:
    # DB-Loader: liest eine Item-Serie als DataFrame ['time','value']
    from hems_nilm_gateway.training.datasets.deddiag_pg import (
        DeddiagPgConfig,
        DeddiagPgDataset,
        DbConfigLike,
    )

    ds = DeddiagPgDataset(DeddiagPgConfig(db=DbConfigLike(**db_cfg)), schema=schema)
    try:
        return ds.to_df(item_id=item_id, start=t0, end=t1)
    finally:
        ds.close()


def load_series_from_deddiag_multi(
    mains_item_id: int,
    target_item_ids: Sequence[int],
    t0: datetime,
    t1: datetime,
    db_cfg: Dict[str, Any],
    schema: str = "public",
) -> pd.DataFrame:
    # Multi-Loader: Mains + Geräte auf gemeinsamer Zeitachse zusammenführen
    df_m = _to_df(mains_item_id, t0, t1, db_cfg, schema)
    if df_m.empty:
        return pd.DataFrame(columns=["time", "mains"])

    df_m = df_m.rename(columns={"value": "mains"})
    df = df_m

    for tid in target_item_ids:
        dfi = _to_df(int(tid), t0, t1, db_cfg, schema)
        if dfi.empty:
            # Fehlende Serie: als Null-Spalte auffüllen
            df[f"dev_{tid}"] = 0.0
            continue
        dfi = dfi.rename(columns={"value": f"dev_{tid}"})

        # Schnittmenge der Timestamps
        df = pd.merge(df, dfi, on="time", how="inner")

    return df.sort_values("time").reset_index(drop=True)


# ---------- CSV: Laden als DataFrame ----------
def load_series_from_csv_multi(
    csv_path: Union[str, Path],
    timestamp_col: str,
    mains_col: str,
    target_cols: Sequence[str],
    sep: str = ";",
    decimal: str = ",",
) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV-Datei nicht gefunden: {path}")

    df = pd.read_csv(path, sep=sep, decimal=decimal)

    # Spalten prüfen
    if timestamp_col not in df.columns:
        raise KeyError(f"Spalte '{timestamp_col}' nicht in CSV enthalten.")
    if mains_col not in df.columns:
        raise KeyError(f"Spalte '{mains_col}' nicht in CSV enthalten.")

    # Zeitstempel -> datetime (ungültige Werte -> NaT)
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")

    # Numerische Spalten -> float (ungültige Werte -> NaN)
    num_cols = [mains_col] + list(target_cols)
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = np.nan

    # Ungültige Zeilen verwerfen (Zeit oder Mains fehlt)
    df = df.dropna(subset=[timestamp_col, mains_col]).copy()

    df = df.rename(columns={timestamp_col: "time", mains_col: "mains"})

    # Zielspalten: NaN -> 0.0
    for col in target_cols:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = df[col].fillna(0.0)

    df = df.sort_values("time").reset_index(drop=True)
    cols = ["time", "mains"] + list(target_cols)
    return df[cols]


# ---------- Arrays ----------
def build_arrays_multi(
    df: pd.DataFrame,
    target_item_ids: Sequence[int],
    on_w: float,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    # Konvertiert DataFrame -> Arrays (p_total und binäre states je Gerät)
    if df.empty:
        return np.array([], dtype=np.float32), np.empty((0, 0), dtype=np.uint8), []

    p_total = df["mains"].astype(float).to_numpy(dtype=np.float32)

    # Geräte-Power sammeln
    dev_cols: List[np.ndarray] = []
    device_ids: List[int] = []
    for tid in target_item_ids:
        col = f"dev_{int(tid)}"
        if col in df.columns:
            s = df[col].astype(float).to_numpy(dtype=np.float32)
        else:
            s = np.zeros(len(df), dtype=np.float32)
        dev_cols.append(s)
        device_ids.append(int(tid))

    dev_pw = (
        np.stack(dev_cols, axis=-1)
        if dev_cols
        else np.zeros((len(df), 0), dtype=np.float32)
    )

    # Binarisierung: Leistung >= on_w -> ON
    states = (dev_pw >= float(on_w)).astype(np.uint8)
    return p_total, states, device_ids


def build_arrays_from_csv_multi(
    df: pd.DataFrame,
    target_cols: Sequence[str],
    on_w: float,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    # Konvertiert CSV-DataFrame -> Arrays
    if df.empty:
        return np.array([], dtype=np.float32), np.empty((0, 0), dtype=np.uint8), []

    p_total = df["mains"].astype(float).to_numpy(dtype=np.float32)

    dev_cols: List[np.ndarray] = []
    labels: List[str] = []
    for col in target_cols:
        if col in df.columns:
            s = df[col].astype(float).to_numpy(dtype=np.float32)
        else:
            s = np.zeros(len(df), dtype=np.float32)
        dev_cols.append(s)
        labels.append(str(col))

    dev_pw = (
        np.stack(dev_cols, axis=-1)
        if dev_cols
        else np.zeros((len(df), 0), dtype=np.float32)
    )

    # Binarisierung: Leistung >= on_w -> ON
    states = (dev_pw >= float(on_w)).astype(np.uint8)
    return p_total, states, labels


# ---------- Dataset ----------
class WindowDatasetSeq2Seq(Dataset):
    # Window-Dataset liefert (feats(T,2), targets(T,D)) für das Training
    def __init__(
        self,
        p_total: np.ndarray,      # (N,)
        states: np.ndarray,       # (N, D) uint8
        window: int,
        stride: int,
        normalizer: ZScore,
    ):
        # Validierung (Längen und Dimensionen)
        assert p_total.ndim == 1 and states.ndim == 2 and len(p_total) == len(states)

        self.p_total = p_total
        self.states = states.astype(np.uint8)
        self.window = int(window)
        self.stride = int(stride)
        self.norm = normalizer

        # Startindizes aller gültigen Fenster
        idxs: List[int] = []
        i, N = 0, len(self.p_total)
        while i + self.window <= N:
            idxs.append(i)
            i += self.stride
        self.idxs = np.asarray(idxs, dtype=np.int64)

    def __len__(self) -> int:
        # Anzahl Fenster
        return len(self.idxs)

    def __getitem__(self, j: int):
        # Fenster j -> Feature-Sequenz + Target-Sequenz
        i0 = int(self.idxs[j])
        i1 = i0 + self.window

        x = self.p_total[i0:i1]

        # Feature 0: z-Normalisierung
        x_norm = self.norm.transform(x)

        # Feature 1: dP/dt
        prev = self.p_total[i0 - 1] if i0 > 0 else self.p_total[i0]
        dp = np.diff(np.concatenate(([prev], x))).astype(np.float32)

        feats = np.stack([x_norm, dp], axis=-1).astype(np.float32)   # (T, 2)
        y_seq = self.states[i0:i1, :].astype(np.float32)             # (T, D)

        return torch.from_numpy(feats), torch.from_numpy(y_seq)
