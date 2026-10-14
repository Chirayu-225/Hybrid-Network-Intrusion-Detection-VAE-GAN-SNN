"""Standard evaluation on the held-out TEST split: metrics.json, summary.md and figures.

Everything is computed on the natural-prevalence test set, with thresholds fixed on validation.
(The full ablation study - baseline vs +GAN, unknown-attack hold-out, SNN vs ANN - is the next milestone.)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)

from nids.config import Config, report_dir
from nids.data.io import load_meta, load_split
from nids.fusion import alert_funnel
from nids.inference import HybridNIDS
from nids.models.snn import synaptic_op_estimate
from nids.utils import get_logger, save_json

log = get_logger("nids.evaluation")

# ---- figure styling (validated reference palette, light surface)
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"
BLUE_RAMP = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
ACTION_COLORS = {"allow": "#86b6ef", "monitor": "#2a78d6", "block": "#104281"}  # ordinal: light -> dark


def _style(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _new_fig(w: float = 7.2, h: float = 4.4) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(w, h), dpi=150, facecolor=SURFACE)
    return fig, ax


def _title(ax: plt.Axes, title: str, subtitle: str | None = None) -> None:
    ax.set_title(title, loc="left", fontsize=11.5, color=INK, fontweight="bold", pad=22 if subtitle else 10)
    if subtitle:
        ax.text(0, 1.03, subtitle, transform=ax.transAxes, fontsize=9, color=INK_2, va="bottom")


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------------------------------- figures
def fig_confusion(cm: np.ndarray, classes: list[str], path: Path) -> None:
    row = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, row, out=np.zeros_like(cm, dtype=float), where=row > 0)
    fig, ax = _new_fig(7.4, 6.0)
    cmap = LinearSegmentedColormap.from_list("blue_seq", BLUE_RAMP)
    ax.imshow(norm, cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), classes, rotation=35, ha="right")
    ax.set_yticks(range(len(classes)), [f"{c} (n={int(r)})" for c, r in zip(classes, row[:, 0])])
    ax.set_xlabel("Predicted class", color=INK_2, fontsize=9)
    ax.set_ylabel("True class", color=INK_2, fontsize=9)
    for i in range(len(classes)):
        for j in range(len(classes)):
            v = norm[i, j]
            if v >= 0.005:
                ax.text(j, i, f"{v * 100:.0f}", ha="center", va="center", fontsize=8,
                        color="#ffffff" if v > 0.5 else INK)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    _title(ax, "SNN confusion matrix (% of each true class)", "Test split, natural prevalence")
    _save(fig, path)


def fig_roc(fpr: np.ndarray, tpr: np.ndarray, auroc: float, op: tuple[float, float], path: Path) -> None:
    fig, ax = _new_fig(5.8, 4.6)
    _style(ax, "both")
    ax.plot([0, 1], [0, 1], color=GRID, linewidth=1.5, linestyle="--")
    ax.plot(fpr, tpr, color=BLUE, linewidth=2)
    ax.scatter([op[0]], [op[1]], s=45, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=3)
    ax.annotate(f"benign p99 threshold\nTPR {op[1]:.2f} at FPR {op[0]:.3f}", (op[0], op[1]),
                xytext=(op[0] + 0.12, max(op[1] - 0.22, 0.05)), fontsize=8.5, color=INK_2,
                arrowprops={"arrowstyle": "-", "color": GRID})
    ax.set_xlabel("False positive rate (benign flagged)", color=INK_2, fontsize=9)
    ax.set_ylabel("True positive rate (attacks flagged)", color=INK_2, fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    _title(ax, f"VAE anomaly detector, AUROC {auroc:.3f}", "Benign vs any attack, unsupervised (trained on benign only)")
    _save(fig, path)


def fig_risk(risk: np.ndarray, is_attack: np.ndarray, thresholds: dict[str, float], path: Path) -> None:
    fig, ax = _new_fig(7.2, 4.8)
    _style(ax, "y")
    bins = np.linspace(0, 1, 41)
    handles = []
    for mask, color, name in ((~is_attack, BLUE, "Benign"), (is_attack, ORANGE, "Attack")):
        if mask.sum() == 0:
            continue
        hist, _ = np.histogram(risk[mask], bins=bins)
        share = np.where(hist > 0, hist / hist.sum(), np.nan)  # empty bins leave a gap instead of a drop to zero
        ax.stairs(share, bins, color=color, linewidth=2)
        handles.append(plt.Line2D([0], [0], color=color, linewidth=2, label=f"{name} (n={int(mask.sum()):,})"))
    for key, ls in (("monitor", ":"), ("block", "--")):
        ax.axvline(thresholds[key], color=INK_2, linewidth=1.2, linestyle=ls)
        ax.text(thresholds[key], 1.02, key, transform=ax.get_xaxis_transform(), fontsize=8.5, color=INK_2,
                ha="center", va="bottom")
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 1.5)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Combined risk score", color=INK_2, fontsize=9)
    ax.set_ylabel("Share of flows in class (log)", color=INK_2, fontsize=9)
    ax.legend(handles=handles, ncol=2, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16),
              fontsize=9, labelcolor=INK_2)
    _title(ax, "Risk score separates benign from attack traffic", "Dashed lines: calibrated monitor / block thresholds")
    _save(fig, path)


def fig_f1(names: list[str], f1: list[float], support: list[int], rare: set[str], path: Path) -> None:
    fig, ax = _new_fig(7.2, 4.4)
    _style(ax, "x")
    y = np.arange(len(names))[::-1]
    ax.barh(y, f1, color=BLUE, height=0.55)
    ax.set_yticks(y, [f"{n}{' *' if n in rare else ''}  (n={s:,})" for n, s in zip(names, support)])
    for yi, v in zip(y, f1):
        ax.text(min(v + 0.015, 1.0), yi, f"{v:.2f}", va="center", fontsize=9, color=INK)
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("F1 score", color=INK_2, fontsize=9)
    _title(ax, "Per-class F1 of the SNN", "* = rare class (GAN augmentation target)")
    _save(fig, path)


def fig_actions(classes: list[str], pred_actions: dict[str, np.ndarray], path: Path) -> None:
    fig, ax = _new_fig(7.2, 4.6)
    _style(ax, "x")
    y = np.arange(len(classes))[::-1]
    left = np.zeros(len(classes))
    for action in ("allow", "monitor", "block"):
        share = np.array([pred_actions[c][action] for c in classes])
        ax.barh(y, share, left=left, color=ACTION_COLORS[action], height=0.6, edgecolor=SURFACE, linewidth=2, label=action)
        for yi, base, s in zip(y, left, share):
            if s >= 0.07:
                ax.text(base + s / 2, yi, f"{s * 100:.0f}%", ha="center", va="center", fontsize=8.5,
                        color="#ffffff" if action != "allow" else INK)
        left += share
    ax.set_yticks(y, classes)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
    ax.legend(ncol=3, frameon=False, loc="lower left", bbox_to_anchor=(-0.01, 1.0), fontsize=9, labelcolor=INK_2)
    ax.set_title("Recommended action by true class", loc="left", fontsize=11.5, color=INK, fontweight="bold", pad=34)
    _save(fig, path)


# ---------------------------------------------------------------------------------------- metrics
def _alert_stats(is_attack: np.ndarray, alerts: np.ndarray) -> dict[str, float]:
    n = len(alerts)
    return {
        "alerts_pct_of_flows": float(alerts.mean() * 100),
        "attack_recall": float(alerts[is_attack].mean()) if is_attack.any() else float("nan"),
        "precision": float(is_attack[alerts].mean()) if alerts.any() else float("nan"),
        "false_alerts_per_1k_flows": float((alerts & ~is_attack).sum() / n * 1000),
    }


def evaluate(cfg: Config, snn_tag: str = "snn", split: str = "test") -> dict[str, Any]:
    meta = load_meta(cfg)
    classes, rare = meta["classes"], set(meta["rare_classes"])
    X, y = load_split(cfg, split)
    nids = HybridNIDS.load(cfg, snn_tag)
    out = nids.predict_scaled(X)
    is_attack = y != meta["benign_index"]
    C = len(classes)
    out_dir = report_dir(cfg) / snn_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- supervised: SNN classification
    pred = out["pred_idx"]
    present = [i for i in range(C) if (y == i).any()]
    rep = classification_report(y, pred, labels=list(range(C)), target_names=classes, output_dict=True, zero_division=0)
    cm = confusion_matrix(y, pred, labels=list(range(C)))
    rare_f1 = [rep[classes[i]]["f1-score"] for i in present if classes[i] in rare]
    classification = {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=present, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, pred, labels=present, average="weighted", zero_division=0)),
        "rare_class_macro_f1": float(np.mean(rare_f1)) if rare_f1 else None,
        "per_class": {c: {k: float(rep[c][k]) for k in ("precision", "recall", "f1-score")} | {"support": int(rep[c]["support"])}
                      for c in classes},
        "confusion_matrix": cm.tolist(),
    }

    # ---- unsupervised: VAE anomaly detector (benign vs any attack)
    raw = out["anomaly_raw"]
    vae_metrics: dict[str, Any] = {}
    fpr = tpr = None
    if is_attack.any() and (~is_attack).any():
        fpr, tpr, _ = roc_curve(is_attack, raw)
        thr99 = nids.vae.scorer.raw_threshold(0.99)
        vae_metrics = {
            "auroc": float(roc_auc_score(is_attack, raw)),
            "auprc": float(average_precision_score(is_attack, raw)),
            "tpr_at_1pct_fpr": float(np.interp(0.01, fpr, tpr)),
            "fpr_at_95pct_tpr": float(np.interp(0.95, tpr, fpr)),
            "detection_rate_at_benign_p99": {
                classes[i]: float((raw[y == i] >= thr99).mean()) for i in present
            },
        }

    # ---- fused system
    fused_auroc = float(roc_auc_score(is_attack, out["risk"])) if is_attack.any() and (~is_attack).any() else None
    funnel = alert_funnel(is_attack, out["action"])
    thr_p95 = nids.vae.scorer.raw_threshold(0.95)
    comparators = {
        "snn_only (predicted != Benign)": _alert_stats(is_attack, pred != meta["benign_index"]),
        "vae_only (benign p95 threshold)": _alert_stats(is_attack, raw >= thr_p95),
        "fused (monitor + block)": _alert_stats(is_attack, np.isin(out["action"], ["monitor", "block"])),
        "fused (block only)": _alert_stats(is_attack, out["action"] == "block"),
    }
    by_class_actions = {
        classes[i]: {a: float((out["action"][y == i] == a).mean()) for a in ("allow", "monitor", "block")}
        for i in present
    }

    # ---- SNN activity / energy proxy on a sample
    rng = np.random.default_rng(cfg.seed)
    sample_idx = rng.choice(len(X), size=min(len(X), 5000), replace=False)
    with torch.no_grad():
        xb = torch.from_numpy(X[sample_idx]).to(nids.snn.device)
        gen = torch.Generator(device=nids.snn.device).manual_seed(cfg.snn.eval_seed)
        _, stats = nids.snn.model.eval()(xb, generator=gen, return_stats=True)
    spike = {**stats, **synaptic_op_estimate(nids.snn.model, stats)}

    metrics = {
        "split": split,
        "n_flows": int(len(y)),
        "thresholds": {"monitor": nids.engine.thresholds.monitor, "block": nids.engine.thresholds.block},
        "classification": classification,
        "vae": vae_metrics,
        "fused": {"risk_auroc": fused_auroc, "alert_funnel": funnel, "comparators": comparators,
                  "actions_by_true_class": by_class_actions},
        "snn_activity": spike,
    }
    save_json(metrics, out_dir / "metrics.json")

    # ---- figures
    fig_confusion(cm, classes, out_dir / "confusion_matrix.png")
    if fpr is not None:
        thr_fpr = float(np.interp(0.01, fpr, fpr))
        fig_roc(fpr, tpr, vae_metrics["auroc"], (thr_fpr, vae_metrics["tpr_at_1pct_fpr"]), out_dir / "vae_roc.png")
        fig_risk(out["risk"], is_attack, metrics["thresholds"], out_dir / "risk_distribution.png")
    fig_f1(
        [classes[i] for i in present], [classification["per_class"][classes[i]]["f1-score"] for i in present],
        [classification["per_class"][classes[i]]["support"] for i in present], rare, out_dir / "per_class_f1.png",
    )
    fig_actions([classes[i] for i in present], {classes[i]: by_class_actions[classes[i]] for i in present},
                out_dir / "action_by_class.png")
    _write_summary(metrics, classes, out_dir / "summary.md")
    log.info("Evaluation written to %s | macro-F1 %.3f | VAE AUROC %s | fused risk AUROC %s", out_dir,
             classification["macro_f1"], vae_metrics.get("auroc"), fused_auroc)
    return metrics


def _write_summary(m: dict[str, Any], classes: list[str], path: Path) -> None:
    c, v, f = m["classification"], m["vae"], m["fused"]
    lines = [f"# Evaluation summary ({m['split']} split, {m['n_flows']:,} flows, natural prevalence)", ""]
    lines += ["## SNN classification", "", "| Metric | Value |", "|---|---|"]
    lines += [f"| Accuracy | {c['accuracy']:.4f} |", f"| Balanced accuracy | {c['balanced_accuracy']:.4f} |",
              f"| Macro F1 | {c['macro_f1']:.4f} |", f"| Weighted F1 | {c['weighted_f1']:.4f} |"]
    if c["rare_class_macro_f1"] is not None:
        lines.append(f"| Rare-class macro F1 | {c['rare_class_macro_f1']:.4f} |")
    lines += ["", "| Class | Precision | Recall | F1 | Support |", "|---|---|---|---|---|"]
    for name in classes:
        p = c["per_class"][name]
        lines.append(f"| {name} | {p['precision']:.3f} | {p['recall']:.3f} | {p['f1-score']:.3f} | {p['support']:,} |")
    if v:
        lines += ["", "## VAE anomaly detection (benign vs any attack)", "", "| Metric | Value |", "|---|---|",
                  f"| AUROC | {v['auroc']:.4f} |", f"| AUPRC | {v['auprc']:.4f} |",
                  f"| TPR at 1% FPR | {v['tpr_at_1pct_fpr']:.4f} |", f"| FPR at 95% TPR | {v['fpr_at_95pct_tpr']:.4f} |"]
    lines += ["", "## Alert volume (what an analyst has to look at)", "",
              "| Strategy | Alerts (% of flows) | Attack recall | Precision | False alerts / 1k flows |", "|---|---|---|---|---|"]
    for name, s in f["comparators"].items():
        lines.append(f"| {name} | {s['alerts_pct_of_flows']:.2f} | {s['attack_recall']:.3f} | {s['precision']:.3f} | "
                     f"{s['false_alerts_per_1k_flows']:.2f} |")
    a = m["snn_activity"]
    lines += ["", "## SNN activity (energy proxy, not a measurement)", "",
              f"- Mean hidden-layer firing rates: {', '.join(f'{r:.3f}' for r in a['hidden_rates'])}",
              f"- Per flow: {a['snn_accumulates_per_flow']:,.0f} accumulates + {a['snn_macs_per_flow']:,.0f} MACs (SNN) "
              f"vs {a['ann_macs_per_flow']:,.0f} MACs (same-shape ANN)",
              f"- Estimated energy ratio SNN / ANN: {a['energy_ratio_snn_over_ann']:.2f} "
              "(45 nm figures: MAC 4.6 pJ, AC 0.9 pJ; ignores memory traffic and neuron updates)", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
