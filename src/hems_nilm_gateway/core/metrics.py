from __future__ import annotations

# Zweck: Metriken/KPIs für NILM (Sigmoid, Tau-Schwellen, Konfusion, Precision/Recall/F1)

from typing import Any, Dict, Sequence, Tuple, Union

import numpy as np

# Typ: Tau pro Head bzw. fest
Taus = Union[float, Sequence[float]]


def sigmoid(x: np.ndarray) -> np.ndarray:
    # Elementweise Sigmoid für Logits -> Wahrscheinlichkeiten
    return 1.0 / (1.0 + np.exp(-x))


def _taus_to_row_vector(taus: Taus, D: int) -> np.ndarray:
    # Normalisierung: Tau als (1, D) Matrix (je Head)
    if isinstance(taus, (float, int)):
        return np.full((1, D), float(taus), dtype=np.float64)
    arr = np.asarray(list(taus), dtype=np.float64).reshape(1, -1)
    if arr.shape[1] != D:
        raise ValueError(f"taus length {arr.shape[1]} != D {D}")
    return arr


def confusion_at_tau_multi(
    logits: np.ndarray,  # (N, D)
    y_true: np.ndarray,  # (N, D)
    taus: Taus,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Konfusion pro Head bei festen Tau: tp, fp, fn, tn
    p = sigmoid(logits)
    D = p.shape[1]
    tau_mat = _taus_to_row_vector(taus, D)
    y_hat = (p >= tau_mat).astype(np.uint8)

    y = y_true.astype(np.uint8)
    tp = np.sum((y_hat == 1) & (y == 1), axis=0)
    fp = np.sum((y_hat == 1) & (y == 0), axis=0)
    fn = np.sum((y_hat == 0) & (y == 1), axis=0)
    tn = np.sum((y_hat == 0) & (y == 0), axis=0)
    return tp.astype(int), fp.astype(int), fn.astype(int), tn.astype(int)


def prec_rec_f1(
    tp: np.ndarray, fp: np.ndarray, fn: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Precision/Recall/F1 pro Head
    tp = tp.astype(float)
    fp = fp.astype(float)
    fn = fn.astype(float)

    precision = tp / np.maximum(1.0, tp + fp)
    recall = tp / np.maximum(1.0, tp + fn)
    f1 = np.where(
        (precision + recall) == 0.0,
        0.0,
        2.0 * precision * recall / (precision + recall),
    )
    return precision, recall, f1


def best_f1_per_head(
    logits: np.ndarray,  # (N, D)
    y_true: np.ndarray,  # (N, D)
    num: int = 201,
) -> Dict[str, Any]:
    # Bestes Tau pro Head
    D = logits.shape[1]
    taus = np.linspace(0.0, 1.0, num=num, dtype=np.float64)

    best_tau = np.zeros(D, dtype=np.float64)
    best_f1 = np.zeros(D, dtype=np.float64)
    best_p = np.zeros(D, dtype=np.float64)
    best_r = np.zeros(D, dtype=np.float64)

    sig = sigmoid(logits)

    for d in range(D):
        yd = y_true[:, d].astype(np.uint8)
        sd = sig[:, d]

        # Optimum pro Head
        f1_best = -1.0
        tau_best = 0.5
        p_best = 0.0
        r_best = 0.0

        for t in taus:
            yhat = (sd >= t).astype(np.uint8)
            tp = int(np.sum((yhat == 1) & (yd == 1)))
            fp = int(np.sum((yhat == 1) & (yd == 0)))
            fn = int(np.sum((yhat == 0) & (yd == 1)))

            precision = tp / max(1, (tp + fp))
            recall = tp / max(1, (tp + fn))
            f1 = 0.0 if (precision + recall) == 0.0 else (2.0 * precision * recall / (precision + recall))

            if f1 > f1_best:
                f1_best, tau_best, p_best, r_best = f1, float(t), precision, recall

        best_tau[d], best_f1[d], best_p[d], best_r[d] = tau_best, f1_best, p_best, r_best

    macro_f1 = float(np.mean(best_f1)) if D > 0 else 0.0
    return {
        "per_device": {
            "tau": best_tau.tolist(),
            "precision": best_p.tolist(),
            "recall": best_r.tolist(),
            "f1": best_f1.tolist(),
        },
        "macro_f1": macro_f1,
    }


def compute_kpis_multi(
    logits_te: np.ndarray,  # (N, D)
    y_te: np.ndarray,       # (N, D)
    taus: Taus,
) -> Dict[str, Any]:
    # KPI bei festen Tau: Konfusion + P,R,F1
    tp, fp, fn, tn = confusion_at_tau_multi(logits_te, y_te, taus)
    p, r, f1 = prec_rec_f1(tp, fp, fn)
    macro = float(np.mean(f1)) if f1.size else 0.0

    # Tau als Liste ausgeben (für JSON)
    if isinstance(taus, (float, int)):
        tau_list = [float(taus)] * (logits_te.shape[1] if logits_te.ndim == 2 else 1)
    else:
        tau_list = [float(t) for t in taus]

    return {
        "thresholds_tau": tau_list,
        "classification": {
            "tp": tp.tolist(),
            "fp": fp.tolist(),
            "fn": fn.tolist(),
            "tn": tn.tolist(),
            "precision": p.tolist(),
            "recall": r.tolist(),
            "f1": f1.tolist(),
            "macro_f1": macro,
        },
    }


def f1_curve_per_head(
    logits: np.ndarray,   # (N, D)
    y_true: np.ndarray,   # (N, D)
    num: int = 201,
) -> Dict[str, Any]:
    # Kurven: F1(Tau) pro Head für Plotting
    D = logits.shape[1]
    taus = np.linspace(0.0, 1.0, num=num, dtype=np.float64)
    curves = {"taus": taus.tolist(), "f1_per_head": []}

    sig = sigmoid(logits)

    for d in range(D):
        yd = y_true[:, d].astype(np.uint8)
        sd = sig[:, d]
        f1s = []

        for t in taus:
            yhat = (sd >= t).astype(np.uint8)
            tp = int(np.sum((yhat == 1) & (yd == 1)))
            fp = int(np.sum((yhat == 1) & (yd == 0)))
            fn = int(np.sum((yhat == 0) & (yd == 1)))

            p = tp / max(1, (tp + fp))
            r = tp / max(1, (tp + fn))
            f1 = 0.0 if (p + r) == 0.0 else (2.0 * p * r / (p + r))
            f1s.append(float(f1))

        curves["f1_per_head"].append(f1s)

    return curves
