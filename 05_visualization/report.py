"""Result tables (CSV / Markdown / LaTeX) and figures. Everything is computed from files written by the evaluation
cells; no number in any table is typed by hand. Cells that have not been measured stay empty (NaN)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from . import metrics as M
from .evalrun import COND_LABEL, COND_ORDER

MAIN_COLS = [("dice_pos", "Dice (lesion imgs)", True), ("dice_all", "Dice (all imgs)", True), ("iou_pos", "IoU (lesion imgs)", True),
             ("sensitivity", "Sensitivity", False), ("precision", "Precision", False), ("specificity", "Specificity*", False),
             ("mcc", "MCC", False), ("hd95", "HD95 (px)", False), ("hd95_pen", "HD95 penalised (px)", False),
             ("bf1", "Boundary F1", False), ("lesion_dr", "Lesion detection rate", False),
             ("dice_small", "Small-lesion Dice", False), ("img_roc_auc", "Image ROC-AUC", False), ("img_ap", "Image PR-AUC", False),
             ("pix_ap", "Pixel PR-AUC", False), ("accuracy", "Accuracy*", False)]


def fmt(x, nd=3):
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def fmt_ci(m, lo, hi, nd=3):
    return "" if np.isnan(m) else f"{m:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"


def load_summaries(res_dir: Path, run: str, conds=COND_ORDER) -> Dict[str, dict]:
    out = {}
    for c in conds:
        p = Path(res_dir) / run / f"{c}_summary.json"
        if p.exists():
            out[c] = json.loads(p.read_text())
    return out


def per_image(res_dir: Path, run: str, cond: str) -> Optional[pd.DataFrame]:
    p = Path(res_dir) / run / f"{cond}_per_image.csv"
    return pd.read_csv(p) if p.exists() else None


# ----------------------------------------------------------------------------------------------
def empty_tables(runs: Sequence[str]) -> Dict[str, pd.DataFrame]:
    """Tables with EMPTY cells, created before any training so that nothing can be filled in by hand."""
    main = pd.DataFrame(index=list(runs), columns=[c[1] for c in MAIN_COLS], dtype=object)
    rob = pd.DataFrame(index=list(runs), columns=[COND_LABEL[c] for c in COND_ORDER] + ["Drop sev 2 (abs)", "Drop sev 2 (%)"], dtype=object)
    small = pd.DataFrame(index=list(runs), columns=[f"{m} {s}" for s in ("small", "medium", "large") for m in ("Dice", "Lesion DR")], dtype=object)
    eff = pd.DataFrame(index=list(runs), columns=["Params (M)", "GMACs", "Latency fp32 (ms)", "Latency fp16 (ms)", "Peak mem fp16 (GB)", "Throughput (img/s)"], dtype=object)
    return dict(table1_main_clean=main, table2_robustness=rob, table3_lesion_size=small, table4_efficiency=eff)


def save_table(df: pd.DataFrame, name: str, out_dir: Path, caption: str = ""):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{name}.csv")
    try:
        (out_dir / f"{name}.md").write_text(df.to_markdown())
    except ImportError:
        (out_dir / f"{name}.md").write_text(df.to_string())
    (out_dir / f"{name}.tex").write_text(df.to_latex(escape=True, caption=caption or name.replace("_", " "), label=f"tab:{name}"))


# ----------------------------------------------------------------------------------------------
def table_main(res_dir: Path, runs: Sequence[str], cond: str = "clean", n_boot: int = 2000) -> pd.DataFrame:
    rows = {}
    for r in runs:
        s = load_summaries(res_dir, r, [cond]).get(cond)
        if s is None:
            continue
        df = per_image(res_dir, r, cond)
        pos = df[df.has_gt == 1]
        row = {}
        for key, label, ci in MAIN_COLS:
            v = s.get(key, np.nan)
            if ci:
                src = pos.dice if key == "dice_pos" else (df.dice if key == "dice_all" else pos.iou)
                m, lo, hi = M.bootstrap_ci(src.values, n_boot=n_boot)
                row[label] = fmt_ci(m, lo, hi)
            else:
                row[label] = fmt(v, 2 if key.startswith("hd95") else 3)
        row["Missed lesions (n)"] = int(s["n_missed"])
        rows[r] = row
    return pd.DataFrame(rows).T


def table_robustness(res_dir: Path, runs: Sequence[str], metric: str = "dice_pos") -> pd.DataFrame:
    rows = {}
    for r in runs:
        s = load_summaries(res_dir, r)
        if not s:
            continue
        row = {COND_LABEL[c]: fmt(s[c][metric]) for c in COND_ORDER if c in s}
        if "clean" in s and "sev2" in s:
            d = s["clean"][metric] - s["sev2"][metric]
            row["Drop sev 2 (abs)"] = fmt(d)
            row["Drop sev 2 (%)"] = fmt(100 * d / s["clean"][metric], 1) if s["clean"][metric] else ""
        if "clean" in s and "unseen_sev2" in s:
            d = s["clean"][metric] - s["unseen_sev2"][metric]
            row["Drop unseen sev 2 (abs)"] = fmt(d)
            row["Drop unseen sev 2 (%)"] = fmt(100 * d / s["clean"][metric], 1) if s["clean"][metric] else ""
        rows[r] = row
    return pd.DataFrame(rows).T


def table_lesion_size(res_dir: Path, runs: Sequence[str], cond: str = "clean") -> pd.DataFrame:
    rows = {}
    for r in runs:
        s = load_summaries(res_dir, r, [cond]).get(cond)
        if s is None:
            continue
        row = {}
        for sz in ("small", "medium", "large"):
            row[f"Dice {sz} (n={s.get('n_' + sz, '')})"] = fmt(s.get(f"dice_{sz}", np.nan))
            row[f"Lesion DR {sz}"] = fmt(s.get(f"lesion_dr_{sz}", np.nan))
        rows[r] = row
    return pd.DataFrame(rows).T


def table_stats(res_dir: Path, pairs: Sequence[tuple], conds: Sequence[str] = ("clean", "sev2", "unseen_sev2"),
                n_boot: int = 5000) -> pd.DataFrame:
    """Paired comparison B - A of per-image Dice (same images, same corruption) with bootstrap CI and Wilcoxon p."""
    rows = []
    for a, b in pairs:
        for c in conds:
            da, db = per_image(res_dir, a, c), per_image(res_dir, b, c)
            if da is None or db is None:
                continue
            for label, mask in (("lesion images", da.has_gt == 1), ("all images", da.has_gt >= 0)):
                r = M.paired_compare(da.dice[mask].values, db.dice[mask].values, n_boot=n_boot)
                rows.append({"A (reference)": a, "B": b, "condition": COND_LABEL[c], "images": label, "n": r["n"],
                             "mean diff B-A": fmt(r["mean_diff"], 4), "95% CI": f"[{fmt(r.get('ci_lo', np.nan), 4)}, {fmt(r.get('ci_hi', np.nan), 4)}]",
                             "B better (%)": fmt(100 * r.get("frac_better", np.nan), 1), "Wilcoxon p": f"{r.get('wilcoxon_p', np.nan):.3g}"})
    return pd.DataFrame(rows)


def table_efficiency(eff: Dict[str, dict]) -> pd.DataFrame:
    rows = {}
    for r, e in eff.items():
        rows[r] = {"Params (M)": fmt(e["params_M"], 2), "GMACs": fmt(e["gmacs"], 1),
                   "Latency fp32 (ms)": fmt(e.get("latency_fp32_ms", e.get("latency_cpu_ms", np.nan)), 1),
                   "Latency fp16 (ms)": fmt(e.get("latency_fp16_ms", np.nan), 1),
                   "Peak mem fp16 (GB)": fmt(e.get("peak_mem_fp16_gb", np.nan), 2),
                   "Throughput (img/s)": fmt(e.get("throughput_fp16_img_s", np.nan), 1), "Device": e.get("gpu", "")}
    return pd.DataFrame(rows).T


# ----------------------------------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------------------------------
def _plt():
    import matplotlib
    import matplotlib.pyplot as plt
    return plt


def plot_training_curves(runs_dir: Path, runs: Sequence[str], out: Path):
    plt = _plt()
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    for r in runs:
        p = Path(runs_dir) / r / "history.csv"
        if not p.exists():
            continue
        h = pd.read_csv(p)
        ax[0].plot(h.epoch, h.train_total, label=r); ax[1].plot(h.epoch, h.val_loss, label=r)
        ax[2].plot(h.epoch, h.val_dice_all, label=r)
    for a, t in zip(ax, ("train loss (total, incl. auxiliary terms)", "validation segmentation loss (BCE+Dice)", "validation Dice (all images, empty convention)")):
        a.set_title(t, fontsize=10); a.set_xlabel("epoch"); a.grid(alpha=.3)
    ax[2].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out, dpi=130); return fig


def plot_robustness(res_dir: Path, runs: Sequence[str], out: Path, metric: str = "dice_pos"):
    plt = _plt()
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
    for r in runs:
        s = load_summaries(res_dir, r)
        if "clean" not in s:
            continue
        for a, pre, ttl in ((ax[0], "sev", "Seen device families"), (ax[1], "unseen_sev", "UNSEEN device family")):
            ys = [s["clean"][metric]] + [s[f"{pre}{k}"][metric] if f"{pre}{k}" in s else np.nan for k in (1, 2, 3)]
            a.plot([0, 1, 2, 3], ys, marker="o", label=r); a.set_title(ttl, fontsize=10)
    for a in ax:
        a.set_xticks([0, 1, 2, 3]); a.set_xticklabels(["clean", "sev 1", "sev 2", "sev 3"]); a.grid(alpha=.3)
    ax[0].set_ylabel(f"{metric} (lesion images)"); ax[1].legend(fontsize=7)
    fig.suptitle("Robustness to SYNTHETIC artifacts (not real-world clinical validation)", fontsize=10)
    fig.tight_layout(); fig.savefig(out, dpi=130); return fig


def plot_curves(res_dir: Path, runs: Sequence[str], out: Path, cond: str = "clean"):
    from sklearn.metrics import precision_recall_curve, roc_curve
    plt = _plt()
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    for r in runs:
        df = per_image(res_dir, r, cond)
        hp = Path(res_dir) / r / f"{cond}_hist.npz"
        if df is None:
            continue
        fpr, tpr, _ = roc_curve(df.has_gt, df.max_prob); ax[0].plot(fpr, tpr, label=r)
        pr, rc, _ = precision_recall_curve(df.has_gt, df.max_prob); ax[1].plot(rc, pr, label=r)
        if hp.exists():
            h = np.load(hp)
            tp = np.cumsum(h["pos"][::-1]); fp = np.cumsum(h["neg"][::-1])
            ax[2].plot(tp / max(1, tp[-1]), tp / np.maximum(tp + fp, 1), label=r)
    ax[0].set_title("Image-level ROC (score = max pixel prob.)", fontsize=10); ax[0].set_xlabel("FPR"); ax[0].set_ylabel("TPR")
    ax[1].set_title("Image-level precision-recall", fontsize=10); ax[1].set_xlabel("recall"); ax[1].set_ylabel("precision")
    ax[2].set_title("Pixel-level precision-recall", fontsize=10); ax[2].set_xlabel("recall"); ax[2].set_ylabel("precision")
    for a in ax:
        a.grid(alpha=.3)
    ax[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(out, dpi=130); return fig


def plot_qualitative(keep: Dict[str, dict], gts: np.ndarray, ops: Dict[str, dict], out: Path, cond: str = "clean"):
    """keep: {run: {'probs','arts','imgs'}} for the same chosen images; gts: [K,S,S]."""
    plt = _plt()
    runs = list(keep)
    K = len(gts)
    fig, ax = plt.subplots(K, 2 + len(runs), figsize=(2.6 * (2 + len(runs)), 2.6 * K), squeeze=False)
    for k in range(K):
        img = keep[runs[0]]["imgs"][k]
        ax[k, 0].imshow(img, cmap="gray", vmin=0, vmax=1); ax[k, 0].set_title("image" if k == 0 else "")
        ax[k, 1].imshow(img, cmap="gray", vmin=0, vmax=1); ax[k, 1].contour(gts[k], [0.5], colors="lime", linewidths=1.0)
        ax[k, 1].set_title("ground truth" if k == 0 else "")
        for j, r in enumerate(runs):
            p = keep[r]["probs"][k]
            pred = M.postprocess(p, **ops[r])
            a = ax[k, 2 + j]; a.imshow(img, cmap="gray", vmin=0, vmax=1)
            a.contour(gts[k], [0.5], colors="lime", linewidths=0.8); a.contour(pred, [0.5], colors="red", linewidths=1.0)
            a.set_title(r if k == 0 else "", fontsize=8)
    for a in ax.ravel():
        a.axis("off")
    fig.suptitle(f"condition: {COND_LABEL.get(cond, cond)}   green = ground truth, red = prediction", fontsize=9)
    fig.tight_layout(); fig.savefig(out, dpi=120); return fig


def plot_lesion_size(res_dir: Path, runs: Sequence[str], out: Path):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 4))
    w = 0.8 / max(1, len(runs))
    for i, r in enumerate(runs):
        s = load_summaries(res_dir, r, ["clean"]).get("clean")
        if s is None:
            continue
        ax.bar(np.arange(3) + i * w, [s.get(f"dice_{k}", np.nan) for k in ("small", "medium", "large")], w, label=r)
    ax.set_xticks(np.arange(3) + 0.4 - w / 2); ax.set_xticklabels(["small", "medium", "large"]); ax.set_ylabel("Dice (lesion images)")
    ax.set_title("Dice by lesion size (tertiles of training-set lesion area)", fontsize=10); ax.legend(fontsize=7); ax.grid(alpha=.3, axis="y")
    fig.tight_layout(); fig.savefig(out, dpi=130); return fig
