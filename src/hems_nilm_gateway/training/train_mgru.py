from __future__ import annotations

# Zweck: Training/Evaluation des M-GRU Seq2Seq-Modells inkl. KPI-Export und Plots

import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

from hems_nilm_gateway.core.model_mgru import MGRUNetMultiSeq2Seq
from hems_nilm_gateway.core.data import (
    ZScore,
    WindowDatasetSeq2Seq,
    load_series_from_deddiag_multi,
    build_arrays_multi,
    load_series_from_csv_multi,
    build_arrays_from_csv_multi,
)
from hems_nilm_gateway.core import metrics as k

# Plot-Backend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# ---------- Utils ----------
def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_outdir(base: Path, prefix: str) -> Path:
    # Artefakt-Ordner: Zeitstempel für Unterordner
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out = base / f"{prefix}_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def amp_components():
    # AMP: torch.amp und torch.cuda.amp (je nach Torch-Version)
    autocast_ctx = getattr(torch.amp, "autocast", None)
    GradScalerCtor = getattr(torch.amp, "GradScaler", None)
    if autocast_ctx is None:
        from torch.cuda.amp import autocast as autocast_ctx
    if GradScalerCtor is None:
        from torch.cuda.amp import GradScaler as GradScalerCtor
    return autocast_ctx, GradScalerCtor


def _assert_finite_np(name: str, arr: np.ndarray) -> None:
    # Abbruch bei NaN in Eingangsarrays
    if not np.isfinite(arr).all():
        n_nan = int(np.isnan(arr).sum())
        n_pos = int(np.isposinf(arr).sum())
        n_neg = int(np.isneginf(arr).sum())
        raise RuntimeError(
            f"{name} enthält nicht-endliche Werte: NaN={n_nan}, +inf={n_pos}, -inf={n_neg}. "
            "Daten (CSV/DB) auf ungültige Zahlen prüfen."
        )


@torch.no_grad()
def run_inference_collect_seq2seq(net: nn.Module, loader: DataLoader, device: str):
    # Modell Seq2Seq-Logits ((B,T,D) -> (B*T,D))
    net.eval()
    logits_all, y_all = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logit_btD = net(x).detach().cpu().numpy()   # (B, T, D)
        y_btD = y.numpy()                           # (B, T, D)
        B, T, D = logit_btD.shape
        logits_all.append(logit_btD.reshape(B * T, D))
        y_all.append(y_btD.reshape(B * T, D))
    logit = np.concatenate(logits_all) if logits_all else np.empty((0, 0))
    y = np.concatenate(y_all) if y_all else np.empty((0, 0))
    return logit, y


# ---------- Terminal KPIs ----------
def print_epoch_summary(
    epoch: int,
    train_loss_avg: float,
    best_macro_f1_so_far: float,
    per_head: Dict[str, Any],
    device_labels: List[str],
    improved: bool,
) -> None:
    # Konsolen-Output: Loss + (Macro)-F1 + Tau_Stern/P/R/F1 pro Gerät
    taus = per_head["tau"]
    f1s = per_head["f1"]
    prec = per_head["precision"]
    rec = per_head["recall"]

    flag = " **" if improved else ""
    print(f"[{epoch:02d}] loss={train_loss_avg:.4f} valMacroF1={best_macro_f1_so_far:.3f}{flag}")
    for i, name in enumerate(device_labels):
        print(f"   - {name:<18} τ={taus[i]:.2f}  F1={f1s[i]:.3f}  P={prec[i]:.3f}  R={rec[i]:.3f}")


def print_final_test_summary(
    kpis: Dict[str, Any],
    device_labels: List[str]
) -> None:
    # Konsolen-Output: finale Test-KPIs pro Gerät + Konfusions-Zahlen
    cl = kpis["classification"]
    taus = kpis["thresholds_tau"]
    f1 = cl["f1"]; p = cl["precision"]; r = cl["recall"]
    tp, fp, fn, tn = cl["tp"], cl["fp"], cl["fn"], cl["tn"]
    macro = cl["macro_f1"]

    print("\n=== TEST SUMMARY ===")
    print(f"Macro-F1: {macro:.3f}")
    for i, name in enumerate(device_labels):
        support_pos = tp[i] + fn[i]
        print(
            f"- {name:<18} τ={taus[i]:.2f}  F1={f1[i]:.3f}  P={p[i]:.3f}  R={r[i]:.3f} "
            f" | TP={tp[i]} FP={fp[i]} FN={fn[i]} TN={tn[i]}  (pos={support_pos})"
        )


# ---------- Plots ----------
def _confmat(ax, tp: int, fp: int, fn: int, tn: int, title: str):
    # Plot: 2x2 Konfusions-Matrix
    cm = np.array([[tp, fn], [fp, tn]], dtype=float)
    cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1.0)
    im = ax.imshow(cmn, vmin=0, vmax=1, cmap="viridis")
    ax.set_title(title)
    ax.set_xticks([0, 1], labels=["Pos", "Neg"])
    ax.set_yticks([0, 1], labels=["Pos", "Neg"])
    labels = [["TP", "FN"], ["FP", "TN"]]
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i,
                f"{labels[i][j]}\n{cmn[i, j]:.2f}\n({int(cm[i, j])})",
                ha="center", va="center",
                color="white" if cmn[i, j] > 0.5 else "black", fontsize=9
            )
    return im


def _timeline_indices(y_true: np.ndarray, slice_len: int = 800) -> Tuple[int, int]:
    # Plot: Segment mit vielen Positiven Zuständen zur Visualisierung der Sigmoidfunktion
    n = len(y_true)
    if n <= slice_len or y_true.sum() == 0:
        return 0, min(n, slice_len)
    step = max(1, slice_len // 5)
    best_s, best_cnt = 0, -1
    for s in range(0, n - slice_len + 1, step):
        cnt = int(y_true[s:s + slice_len].sum())
        if cnt > best_cnt:
            best_cnt, best_s = cnt, s
    return best_s, best_s + slice_len


def plot_report(
    outdir: Path,
    logits_te: np.ndarray,   # (N, D)
    y_te: np.ndarray,        # (N, D)
    best_taus: List[float],  # (D,)
    device_labels: List[str],
    curves_va: Dict[str, Any],
) -> Tuple[Path, Path]:
    # Report-PNGs (Zusammenfassung + Timelines) für Dokumentation
    D = logits_te.shape[1]
    probs_te = 1.0 / (1.0 + np.exp(-logits_te))

    # ---- Abb 1: Zusammenfassung ----
    h_conf = 2 if D <= 2 else 2
    w_conf = 2 if D <= 2 else min(3, D)
    conf_cells = min(D, h_conf * w_conf)

    fig = plt.figure(figsize=(16, 9))
    gs = GridSpec(3, 6, figure=fig, height_ratios=[2, 2, 2])

    # Konfusion je Gerät
    for i in range(conf_cells):
        ax = fig.add_subplot(gs[0, i % 6] if D <= 6 else gs[0, i])
        if i >= D:
            break
        tau = best_taus[i]
        yhat = (probs_te[:, i] >= tau).astype(np.uint8)
        yt = y_te[:, i].astype(np.uint8)
        tp = int(np.sum((yhat == 1) & (yt == 1)))
        fp = int(np.sum((yhat == 1) & (yt == 0)))
        fn = int(np.sum((yhat == 0) & (yt == 1)))
        tn = int(np.sum((yhat == 0) & (yt == 0)))
        _confmat(ax, tp, fp, fn, tn, title=device_labels[i])

    # Colorbar
    if D > 0 and fig.axes and fig.axes[0].images:
        cb = fig.colorbar(fig.axes[0].images[0], ax=fig.axes[:conf_cells], fraction=0.02, pad=0.02)
        cb.ax.set_title("norm.")

    # Balken: Precision/Recall/F1
    ax_bar = fig.add_subplot(gs[1, :3])
    taus_arr = np.asarray(best_taus, dtype=np.float64).reshape(1, -1)
    yhat = (probs_te >= taus_arr).astype(np.uint8)
    tp = np.sum((yhat == 1) & (y_te == 1), axis=0)
    fp = np.sum((yhat == 1) & (y_te == 0), axis=0)
    fn = np.sum((yhat == 0) & (y_te == 1), axis=0)
    p = tp / np.maximum(1, tp + fp)
    r = tp / np.maximum(1, tp + fn)
    f1 = np.where((p + r) == 0.0, 0.0, 2 * p * r / (p + r))
    x = np.arange(D)
    width = 0.28
    ax_bar.bar(x - width, p, width, label="Precision")
    ax_bar.bar(x,         r, width, label="Recall")
    ax_bar.bar(x + width, f1, width, label="F1")
    ax_bar.set_ylim(0, 1)
    ax_bar.set_xticks(x, device_labels, rotation=15)
    ax_bar.set_title("Test: Precision/Recall/F1 je Gerät")
    ax_bar.grid(True, axis="y", alpha=0.3)
    ax_bar.legend(loc="lower right")

    # Kurven: F1(Tau) aus Validierung
    ax_curves = fig.add_subplot(gs[1:, 3:])
    taus = np.asarray(curves_va["taus"], dtype=np.float64)
    for i in range(D):
        f1s = np.asarray(curves_va["f1_per_head"][i], dtype=np.float64)
        ax_curves.plot(taus, f1s, label=device_labels[i], linewidth=2)

        # Markierung von Tau_Stern
        tstar = best_taus[i]
        idx = int(np.argmin(np.abs(taus - tstar)))
        ax_curves.scatter([taus[idx]], [f1s[idx]], s=25)

    ax_curves.set_xlabel("τ")
    ax_curves.set_ylabel("F1")
    ax_curves.set_ylim(0, 1)
    ax_curves.grid(True, alpha=0.3)
    ax_curves.set_title("Val: F1(τ) je Gerät")
    ax_curves.legend(loc="lower right")

    fig.suptitle("NILM Mehrgeräte Zusammenfassung (Seq2Seq)", fontsize=14)
    out_summary = outdir / "summary.png"
    fig.savefig(out_summary, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- Abb 2: Timelines ----
    fig2 = plt.figure(figsize=(16, 8))
    gs2 = GridSpec(D, 1, figure=fig2)
    sig = probs_te

    for i in range(D):
        ax = fig2.add_subplot(gs2[i, 0])
        yt = y_te[:, i].astype(np.uint8)
        s, e = _timeline_indices(yt, slice_len=800)
        idx = np.arange(s, e)

        # Wahrheitswerte (step), Score (line), Pred (step)
        ax.step(idx, yt[s:e], where="post", label="GT", linewidth=1.2)
        ax.plot(idx, sig[s:e, i], label="Score", linewidth=1.2)
        ax.axhline(best_taus[i], linestyle="--", linewidth=1.0, label="τ*")
        yhat = (sig[s:e, i] >= best_taus[i]).astype(np.uint8)
        ax.step(idx, yhat, where="post", label="Pred", alpha=0.8)

        ax.set_ylim(-0.1, 1.1)
        ax.set_ylabel(device_labels[i])
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc="upper right")

    if fig2.axes:
        fig2.axes[-1].set_xlabel("Zeitindex (über Fenster)")
    fig2.suptitle("Zeitverlauf: Wahrheitszustaende vs. Score vs. Pred", fontsize=14)
    out_timeline = outdir / "timelines.png"
    fig2.savefig(out_timeline, dpi=150, bbox_inches="tight")
    plt.close(fig2)

    return out_summary, out_timeline


# ---------- Training ----------
def main() -> None:
    # Hauptprogramm: Daten laden -> Train/Val/Test -> Artefakte/Reports schreiben
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg: Dict[str, Any] = yaml.safe_load(Path(args.config).read_text())

    # Reproduzierbarkeit + Geraete-Auswahl
    set_seed(int(cfg.get("seed", 42)))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if cfg.get("train", {}).get("require_cuda", False) and device != "cuda":
        raise RuntimeError("CUDA nicht verfügbar.")

    # Torch Performance
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    # --- Daten laden ---
    ds_cfg = cfg["dataset"]
    on_w = float(ds_cfg.get("on_w", 5.0))
    source = ds_cfg.get("source")  # "deddiag" / "csv" / None

    device_labels: List[str]
    #Alte CSV
    if source is None:
        if "csv_path" in ds_cfg:
            csv_path = Path(ds_cfg["csv_path"])
            ts_col = ds_cfg.get("csv_timestamp_col", "timestamp")
            mains_col = ds_cfg.get("csv_mains_col", "phase_sum")
            target_cols = ds_cfg.get("csv_device_cols")
            if not target_cols:
                raise RuntimeError("Alte CSV-Config erkannt, aber 'csv_device_cols' fehlt.")

            df = load_series_from_csv_multi(
                csv_path=csv_path,
                timestamp_col=ts_col,
                mains_col=mains_col,
                target_cols=target_cols,
                sep=ds_cfg.get("csv_sep", ";"),
                decimal=ds_cfg.get("csv_decimal", ","),
            )
            if df.empty:
                raise RuntimeError("Keine Daten aus CSV geladen.")

            # Optional: Zeitfenster schneiden
            csv_start = ds_cfg.get("csv_start")
            csv_end = ds_cfg.get("csv_end")
            if csv_start is not None:
                t0 = datetime.fromisoformat(csv_start)
                df = df[df["time"] >= t0]
            if csv_end is not None:
                t1 = datetime.fromisoformat(csv_end)
                df = df[df["time"] < t1]
            df = df.sort_values("time").reset_index(drop=True)
            if df.empty:
                raise RuntimeError("Keine CSV-Daten im angegebenen Zeitraum.")

            p_total, states, device_labels = build_arrays_from_csv_multi(
                df,
                target_cols=target_cols,
                on_w=on_w,
            )

        else:
            # DEDDIAG-Config
            t0 = datetime.fromisoformat(ds_cfg["start"])
            t1 = datetime.fromisoformat(ds_cfg["end"])

            target_ids: List[int] = [int(x) for x in ds_cfg["target_item_ids"]]
            df = load_series_from_deddiag_multi(
                mains_item_id=int(ds_cfg["mains_item_id"]),
                target_item_ids=target_ids,
                t0=t0,
                t1=t1,
                db_cfg=ds_cfg["db"],
                schema=ds_cfg.get("schema", "public"),
            )
            if df.empty:
                raise RuntimeError("Keine Daten geladen.")

            p_total, states, device_ids = build_arrays_multi(df, target_ids, on_w=on_w)

            # Namen für Output/Plots
            label_map: Dict[int, str] = {
                24: "Washing Machine (24)",
                26: "Dish Washer (26)",
                35: "Refrigerator (35)",
            }
            device_labels = [label_map.get(d, f"Device {d}") for d in device_ids]

    else:
        # CSV-Config
        if source == "csv":
            csv_cfg = ds_cfg["csv"]
            csv_path = Path(csv_cfg["path"])
            target_cols = csv_cfg["device_cols"]

            df = load_series_from_csv_multi(
                csv_path=csv_path,
                timestamp_col=csv_cfg.get("timestamp_col", "timestamp"),
                mains_col=csv_cfg.get("mains_col", "phase_sum"),
                target_cols=target_cols,
                sep=csv_cfg.get("sep", ";"),
                decimal=csv_cfg.get("decimal", ","),
            )
            if df.empty:
                raise RuntimeError("Keine Daten aus CSV geladen.")

            # Optional: Zeitfenster schneiden
            csv_start = csv_cfg.get("start")
            csv_end = csv_cfg.get("end")
            if csv_start is not None:
                t0 = datetime.fromisoformat(csv_start)
                df = df[df["time"] >= t0]
            if csv_end is not None:
                t1 = datetime.fromisoformat(csv_end)
                df = df[df["time"] < t1]
            df = df.sort_values("time").reset_index(drop=True)
            if df.empty:
                raise RuntimeError("Keine CSV-Daten im angegebenen Zeitraum.")

            p_total, states, device_labels = build_arrays_from_csv_multi(
                df,
                target_cols=target_cols,
                on_w=on_w,
            )

        elif source == "deddiag":
            dd_cfg = ds_cfg["deddiag"]
            t0 = datetime.fromisoformat(dd_cfg["start"])
            t1 = datetime.fromisoformat(dd_cfg["end"])

            target_ids: List[int] = [int(x) for x in dd_cfg["target_item_ids"]]
            df = load_series_from_deddiag_multi(
                mains_item_id=int(dd_cfg["mains_item_id"]),
                target_item_ids=target_ids,
                t0=t0,
                t1=t1,
                db_cfg=dd_cfg["db"],
                schema=dd_cfg.get("schema", "public"),
            )
            if df.empty:
                raise RuntimeError("Keine Daten geladen.")

            p_total, states, device_ids = build_arrays_multi(df, target_ids, on_w=on_w)

            # Namen für Output/Plots
            label_map: Dict[int, str] = {
                24: "Washing Machine (24)",
                26: "Dish Washer (26)",
                35: "Refrigerator (35)",
            }
            device_labels = [label_map.get(d, f"Device {d}") for d in device_ids]

        else:
            raise ValueError(f"dataset.source muss 'deddiag' oder 'csv' sein, nicht '{source}'.")

    # --- Datencheck ---
    _assert_finite_np("p_total", p_total)
    _assert_finite_np("states", states.astype(np.float32))

    D = states.shape[1]
    if D == 0:
        raise RuntimeError("Keine Target-Geräte gefunden (D=0).")

    # --- Zeitlicher Split der Daten: 70/15/15 ---
    n = len(p_total)
    n_te = max(int(0.15 * n), 1)
    n_va = max(int(0.15 * n), 1)
    n_tr = max(n - n_va - n_te, 1)

    p_tr, s_tr = p_total[:n_tr], states[:n_tr]
    p_va, s_va = p_total[n_tr:n_tr + n_va], states[n_tr:n_tr + n_va]
    p_te, s_te = p_total[-n_te:], states[-n_te:]

    # --- Datasets & Loader ---
    scaler = ZScore.fit(p_tr)
    window, stride = int(cfg["window"]), int(cfg["stride"])

    ds_tr = WindowDatasetSeq2Seq(p_tr, s_tr, window, stride, scaler)
    ds_va = WindowDatasetSeq2Seq(p_va, s_va, window, stride, scaler)
    ds_te = WindowDatasetSeq2Seq(p_te, s_te, window, stride, scaler)

    dl_tr = DataLoader(
        ds_tr,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        num_workers=0,
        pin_memory=(device == "cuda"),
    )
    dl_va = DataLoader(
        ds_va,
        batch_size=64,
        shuffle=False,
        num_workers=0,
        pin_memory=(device == "cuda"),
    )
    dl_te = DataLoader(
        ds_te,
        batch_size=64,
        shuffle=False,
        num_workers=0,
        pin_memory=(device == "cuda"),
    )

    # --- Modell/Optimierer ---
    net: nn.Module = MGRUNetMultiSeq2Seq(
        num_devices=D,
        hidden=int(cfg["model"]["hidden"]),
        layers=int(cfg["model"]["layers"]),
        dropout=float(cfg["model"].get("dropout", 0.0)),
    ).to(device)

    # Inbalance: "pos_weight" je Gerät (BCEWithLogitsLoss)
    pos = s_tr.sum(axis=0).astype(float) + 1e-6
    neg = (len(s_tr) - s_tr.sum(axis=0)).astype(float) + 1e-6
    pos_weight = torch.tensor(neg / pos, dtype=torch.float32, device=device)
    loss_cls = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    opt = torch.optim.AdamW(net.parameters(), lr=float(cfg["train"]["lr"]))

    # AMP (CUDA)
    use_amp = bool(cfg["train"].get("amp", device == "cuda"))
    autocast_ctx, GradScalerCtor = amp_components()
    scaler_amp = GradScalerCtor(enabled=(use_amp and device == "cuda"))

    # Vorherige Abbruch: Bestes Macro-F1 auf Val
    patience = int(cfg["train"].get("patience", 5))
    best_state = {k: v.detach().cpu() for k, v in net.state_dict().items()}
    best_macro_f1 = 0.0
    no_improve = 0
    best_per_head: Optional[Dict[str, Any]] = None

    for ep in range(1, int(cfg["train"]["epochs"]) + 1):
        net.train()
        total = 0.0

        # --- Training Loop ---
        for x, y in dl_tr:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            if use_amp and device == "cuda":
                # AMP
                with autocast_ctx(device_type="cuda", enabled=True):  # type: ignore[misc]
                    logits = net(x)
                    loss = loss_cls(logits, y)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Loss ist nicht endlich (NaN/Inf) in Epoche {ep}.")
                scaler_amp.scale(loss).backward()
                scaler_amp.step(opt)
                scaler_amp.update()
            else:
                logits = net(x)
                loss = loss_cls(logits, y)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Loss ist nicht endlich (NaN/Inf) in Epoche {ep}.")
                loss.backward()
                opt.step()

            total += float(loss.item()) * x.size(0)

        # --- Validierung: Metriken ---
        logits_va, y_va = run_inference_collect_seq2seq(net, dl_va, device)
        per_head = k.best_f1_per_head(logits_va, y_va)
        macro_f1 = float(per_head["macro_f1"])

        # Bestes Modell "merken"
        improved = macro_f1 > best_macro_f1
        if improved:
            best_state = {k2: v.detach().cpu() for k2, v in net.state_dict().items()}
            best_macro_f1 = macro_f1
            best_per_head = per_head
            no_improve = 0
        else:
            no_improve += 1

        # Fortschritt ausgeben
        print_epoch_summary(
            ep,
            train_loss_avg=total / max(1, len(ds_tr)),
            best_macro_f1_so_far=macro_f1,
            per_head=per_head["per_device"],
            device_labels=device_labels,
            improved=improved,
        )

        if no_improve >= patience:
            print(f"Früher Abbruch (patience={patience})")
            break

    # -------- Bestes Modell laden --------
    net.load_state_dict(best_state)
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    # Val: Tau_Stern und F1(Tau)-Kurven mit besten Gewichten
    logits_va_best, y_va_best = run_inference_collect_seq2seq(net, dl_va, device)
    best_per_head = k.best_f1_per_head(logits_va_best, y_va_best)
    curves_va_best = k.f1_curve_per_head(logits_va_best, y_va_best, num=201)
    taus = best_per_head["per_device"]["tau"]

    # Test: KPIs mit Val-Tau_Stern
    logits_te, y_te = run_inference_collect_seq2seq(net, dl_te, device)
    kpis = k.compute_kpis_multi(logits_te, y_te, taus)

    # --- Artefakte schreiben ---
    outdir = make_outdir(Path(cfg["artifacts"]["out_dir"]), cfg["artifacts"]["artifact_prefix"])
    torch.save(net.state_dict(), outdir / "model.pt")
    (outdir / "config.yaml").write_text(Path(args.config).read_text(), encoding="utf-8")

    # Normalizer (ZScore) speichern
    sc = ZScore.fit(p_tr)
    (outdir / "normalizer.json").write_text(
        json.dumps({"mean": float(sc.mean), "std": float(sc.std)}, indent=2),
        encoding="utf-8",
    )

    # KPIs speichern (inkl. Tau-Liste)
    (outdir / "kpis.json").write_text(json.dumps(kpis, indent=2), encoding="utf-8")

    # CSV: Geraete-Metriken
    cl = kpis["classification"]
    rows = []
    for i, name in enumerate(device_labels):
        rows.append(
            {
                "device": name,
                "tau": float(taus[i]),
                "precision": float(cl["precision"][i]),
                "recall": float(cl["recall"][i]),
                "f1": float(cl["f1"][i]),
                "tp": int(cl["tp"][i]),
                "fp": int(cl["fp"][i]),
                "fn": int(cl["fn"][i]),
                "tn": int(cl["tn"][i]),
            }
        )
    import csv
    with (outdir / "per_device_metrics.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Plots: Zusammenfassung + Timelines
    sum_png, tln_png = plot_report(
        outdir=outdir,
        logits_te=logits_te,
        y_te=y_te,
        best_taus=[float(t) for t in taus],
        device_labels=device_labels,
        curves_va=curves_va_best,
    )

    # Finaler Report (Konsole)
    print_final_test_summary(kpis, device_labels)
    print(f"\nArtefakte gespeichert in: {outdir}")
    print(f"Report:   {sum_png}")
    print(f"Timeline: {tln_png}")


if __name__ == "__main__":
    # Entry-Point: python train_seq2seq.py --config <yaml>
    main()
