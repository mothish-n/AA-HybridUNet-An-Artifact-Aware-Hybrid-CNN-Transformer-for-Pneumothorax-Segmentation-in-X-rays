"""Evaluation metrics. CPU/numpy, one image at a time, so every definition is explicit and testable.

Conventions (state these in the paper)
    * Everything is computed at the cache resolution (512x512) on binary masks. Distances are in PIXELS of that grid.
    * Post-processing ("operating point"):  binarise at `thr`, drop connected components smaller than `min_comp` pixels,
      and call the whole image empty if the remaining area is smaller than `min_img` pixels. 8-connectivity.
    * Per-image Dice:  2TP / (2TP+FP+FN);  if GT and prediction are BOTH empty the image scores 1 ("empty convention",
      as in the SIIM-ACR challenge).  If only one of them is empty it scores 0.
    * Dice_pos / IoU_pos / Sens_pos : mean over images that HAVE a lesion (an empty prediction scores 0). This is the
      primary segmentation-quality number because negatives cannot inflate it.
    * Dice_all : mean over ALL images with the empty convention (penalises false alarms on negatives).
    * Pooled ("micro") metrics: TP/FP/FN/TN summed over every pixel of every image, then Sens, Prec, Spec, MCC, Acc.
      Specificity and accuracy are dominated by the huge number of true-negative background pixels (>99%), so they sit
      near 1.0 for any sensible model and carry little information; they are reported for completeness only.
    * HD95: 95th percentile of the symmetric surface distances, only for images where GT AND prediction are non-empty.
      Missed lesions (GT>0, prediction empty) are counted separately (`n_missed`); `hd95_pen` replaces their (undefined)
      distance by the image diagonal so that misses are penalised instead of silently dropped.
      False alarms on negatives (GT empty) have no distance and are reported through specificity / image-level metrics.
    * Boundary F1: boundary pixels = mask minus its 1-px erosion; a boundary pixel is a hit if the other boundary has a
      pixel within `tau` px (default 2 px at 512 ~ 4 px at the original 1024). BF1=0 when GT>0 and prediction is empty.
    * Lesion detection rate: a GT connected component is DETECTED if >= `lesion_overlap` (10%) of its area is covered by
      the prediction (a deliberately lenient, documented criterion). Lesion precision: fraction of predicted components
      that touch the GT.
    * Image-level detection: score = max probability; ROC-AUC and PR-AUC (average precision) with sklearn; plus
      sensitivity/specificity of "prediction non-empty" at the operating point.
    * Pixel-level ROC/PR curves are built from 1000-bin histograms of the probabilities (exact up to bin width).
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import ndimage

STRUCT8 = np.ones((3, 3), dtype=bool)
DEFAULT_OP = dict(thr=0.5, min_comp=0, min_img=0)


# ----------------------------------------------------------------------------------------------
# post-processing
# ----------------------------------------------------------------------------------------------
def postprocess(prob: np.ndarray, thr: float = 0.5, min_comp: int = 0, min_img: int = 0) -> np.ndarray:
    pred = prob >= thr
    if (min_comp > 0 or min_img > 0) and pred.any():
        lab, n = ndimage.label(pred, structure=STRUCT8)
        areas = np.bincount(lab.ravel(), minlength=n + 1)
        keep = areas >= max(1, min_comp)
        keep[0] = False
        pred = keep[lab]
        if pred.sum() < min_img:
            pred = np.zeros_like(pred)
    return pred


def _surface(m: np.ndarray) -> np.ndarray:
    return m & ~ndimage.binary_erosion(m, structure=ndimage.generate_binary_structure(2, 1), border_value=0)


def boundary_distances(pred: np.ndarray, gt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Distances (pixels) from each predicted-surface pixel to the GT surface, and vice versa."""
    sp, sg = _surface(pred), _surface(gt)
    d_to_gt = ndimage.distance_transform_edt(~sg)
    d_to_pred = ndimage.distance_transform_edt(~sp)
    return d_to_gt[sp], d_to_pred[sg]


def hd95(pred: np.ndarray, gt: np.ndarray) -> float:
    if not pred.any() or not gt.any():
        return float("nan")
    a, b = boundary_distances(pred, gt)
    return float(np.percentile(np.concatenate([a, b]), 95))


def boundary_f1(pred: np.ndarray, gt: np.ndarray, tau: float = 2.0) -> float:
    if not gt.any():
        return float("nan")
    if not pred.any():
        return 0.0
    a, b = boundary_distances(pred, gt)          # a: pred->gt, b: gt->pred
    prec, rec = float((a <= tau).mean()), float((b <= tau).mean())
    return 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)


# ----------------------------------------------------------------------------------------------
# per-image statistics
# ----------------------------------------------------------------------------------------------
def image_row(prob: np.ndarray, gt: np.ndarray, op: dict, tau: float = 2.0, lesion_overlap: float = 0.10,
              boundary: bool = True) -> Tuple[dict, np.ndarray, np.ndarray]:
    gt = gt.astype(bool)
    pred = postprocess(prob, **op)
    tp = int((pred & gt).sum()); pa = int(pred.sum()); ga = int(gt.sum())
    fp, fn = pa - tp, ga - tp
    tn = gt.size - tp - fp - fn
    if ga == 0 and pa == 0:
        dice = 1.0
    else:
        dice = 2.0 * tp / (2.0 * tp + fp + fn)
    iou = 1.0 if (ga == 0 and pa == 0) else tp / max(1, tp + fp + fn)
    row = dict(tp=tp, fp=fp, fn=fn, tn=tn, gt_area=ga, pred_area=pa, has_gt=int(ga > 0), has_pred=int(pa > 0),
               dice=dice, iou=iou, max_prob=float(prob.max()))
    row["sens"] = tp / ga if ga > 0 else float("nan")
    # boundary metrics (positives only)
    diag = float(math.hypot(*gt.shape))
    if ga > 0 and boundary:
        h = hd95(pred, gt)
        row["hd95"] = h
        row["hd95_pen"] = h if not math.isnan(h) else diag
        row["bf1"] = boundary_f1(pred, gt, tau)
    else:
        row["hd95"] = row["hd95_pen"] = row["bf1"] = float("nan")
    # lesion level
    if ga > 0:
        lab_g, ng = ndimage.label(gt, structure=STRUCT8)
        sizes = np.bincount(lab_g.ravel(), minlength=ng + 1)[1:]
        cov = np.bincount(lab_g[pred], minlength=ng + 1)[1:] if pa > 0 else np.zeros(ng, int)
        row["n_gt_lesions"] = int(ng)
        row["n_gt_detected"] = int((cov / np.maximum(sizes, 1) >= lesion_overlap).sum())
    else:
        row["n_gt_lesions"] = row["n_gt_detected"] = 0
    if pa > 0:
        lab_p, npc = ndimage.label(pred, structure=STRUCT8)
        hit = np.bincount(lab_p[gt], minlength=npc + 1)[1:] > 0 if ga > 0 else np.zeros(npc, bool)
        row["n_pred_comps"], row["n_pred_hit"] = int(npc), int(hit.sum())
    else:
        row["n_pred_comps"] = row["n_pred_hit"] = 0
    return row, pred, gt


# ----------------------------------------------------------------------------------------------
# whole-set evaluation
# ----------------------------------------------------------------------------------------------
def curves_from_hist(pos_hist: np.ndarray, neg_hist: np.ndarray) -> Dict[str, float]:
    P, Nn = float(pos_hist.sum()), float(neg_hist.sum())
    if P == 0 or Nn == 0:
        return dict(pix_roc_auc=float("nan"), pix_ap=float("nan"))
    tp = np.cumsum(pos_hist[::-1]).astype(np.float64)
    fp = np.cumsum(neg_hist[::-1]).astype(np.float64)
    tpr, fpr = tp / P, fp / Nn
    prec = tp / np.maximum(tp + fp, 1)
    trap = getattr(np, "trapezoid", None) or np.trapz
    roc = float(trap(np.concatenate([[0], tpr]), np.concatenate([[0], fpr])))
    ap = float(np.sum(np.diff(np.concatenate([[0], tpr])) * prec))
    return dict(pix_roc_auc=roc, pix_ap=ap)


def evaluate_predictions(probs: np.ndarray, gts: np.ndarray, op: Optional[dict] = None, tau: float = 2.0,
                         boundary: bool = True, bins: int = 1000) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """probs [N,S,S] float (any), gts [N,S,S] {0,1}. Returns (per-image DataFrame, pixel histograms)."""
    op = dict(DEFAULT_OP if op is None else op)
    rows = []
    pos_h, neg_h = np.zeros(bins, np.int64), np.zeros(bins, np.int64)
    for i in range(len(probs)):
        p = np.asarray(probs[i], dtype=np.float32)
        g = np.asarray(gts[i]) > 0
        b = np.minimum((p * bins).astype(np.int64), bins - 1)
        pos_h += np.bincount(b[g], minlength=bins)
        neg_h += np.bincount(b[~g], minlength=bins)
        row, _, _ = image_row(p, g, op, tau=tau, boundary=boundary)
        rows.append(row)
    return pd.DataFrame(rows), dict(pos=pos_h, neg=neg_h)


def _mcc(tp, fp, fn, tn) -> float:
    den = math.sqrt(float(tp + fp) * float(tp + fn) * float(tn + fp) * float(tn + fn))
    return float((tp * tn - fp * fn) / den) if den > 0 else float("nan")


def summarize(df: pd.DataFrame, hists: Optional[Dict[str, np.ndarray]] = None,
              small_edges: Optional[Tuple[float, float]] = None) -> Dict[str, float]:
    pos = df[df.has_gt == 1]
    tp, fp, fn, tn = (int(df[c].sum()) for c in ("tp", "fp", "fn", "tn"))
    s: Dict[str, float] = {}
    s["n_images"], s["n_pos_images"] = len(df), len(pos)
    s["dice_pos"] = float(pos.dice.mean()) if len(pos) else float("nan")
    s["dice_all"] = float(df.dice.mean())
    s["iou_pos"] = float(pos.iou.mean()) if len(pos) else float("nan")
    s["iou_all"] = float(df.iou.mean())
    s["sens_pos_img"] = float(pos.sens.mean()) if len(pos) else float("nan")
    s["dice_pooled"] = 2 * tp / max(1, 2 * tp + fp + fn)
    s["iou_pooled"] = tp / max(1, tp + fp + fn)
    s["sensitivity"] = tp / max(1, tp + fn)
    s["precision"] = tp / max(1, tp + fp)
    s["specificity"] = tn / max(1, tn + fp)
    s["mcc"] = _mcc(tp, fp, fn, tn)
    s["accuracy"] = (tp + tn) / max(1, tp + tn + fp + fn)
    s["hd95"] = float(np.nanmean(pos.hd95)) if len(pos) and pos.hd95.notna().any() else float("nan")
    s["hd95_pen"] = float(pos.hd95_pen.mean()) if len(pos) else float("nan")
    s["n_missed"] = int(((df.has_gt == 1) & (df.has_pred == 0)).sum())
    s["bf1"] = float(pos.bf1.mean()) if len(pos) else float("nan")
    ng, nd = int(df.n_gt_lesions.sum()), int(df.n_gt_detected.sum())
    s["lesion_dr"] = nd / ng if ng else float("nan")
    npc, nph = int(df.n_pred_comps.sum()), int(df.n_pred_hit.sum())
    s["lesion_precision"] = nph / npc if npc else float("nan")
    # image-level detection
    y, sc = df.has_gt.values, df.max_prob.values
    s["img_sens"] = float(((df.has_pred == 1) & (df.has_gt == 1)).sum() / max(1, (df.has_gt == 1).sum()))
    s["img_spec"] = float(((df.has_pred == 0) & (df.has_gt == 0)).sum() / max(1, (df.has_gt == 0).sum()))
    s["img_fp_rate"] = 1.0 - s["img_spec"]
    if 0 < y.sum() < len(y):
        from sklearn.metrics import average_precision_score, roc_auc_score
        s["img_roc_auc"] = float(roc_auc_score(y, sc))
        s["img_ap"] = float(average_precision_score(y, sc))
    else:
        s["img_roc_auc"] = s["img_ap"] = float("nan")
    if hists is not None:
        s.update(curves_from_hist(hists["pos"], hists["neg"]))
    # lesion-size strata (edges measured on TRAIN positives)
    if small_edges is not None and len(pos):
        e1, e2 = small_edges
        for name, m in (("small", pos.gt_area <= e1), ("medium", (pos.gt_area > e1) & (pos.gt_area <= e2)), ("large", pos.gt_area > e2)):
            sub = pos[m]
            s[f"dice_{name}"] = float(sub.dice.mean()) if len(sub) else float("nan")
            s[f"n_{name}"] = int(len(sub))
            ngs = int(sub.n_gt_lesions.sum())
            s[f"lesion_dr_{name}"] = float(sub.n_gt_detected.sum() / ngs) if ngs else float("nan")
    return s


# ----------------------------------------------------------------------------------------------
# operating-point tuning on the VALIDATION set
# ----------------------------------------------------------------------------------------------
def tune_postprocess(probs: np.ndarray, gts: np.ndarray, thrs: Sequence[float], min_comps: Sequence[int],
                     min_imgs: Sequence[int]) -> Tuple[dict, pd.DataFrame]:
    """Grid search over (thr, min_comp, min_img). Objective = Dice_all (empty convention); ties -> higher Dice_pos."""
    T, C, A = len(thrs), len(min_comps), len(min_imgs)
    s_all, s_pos = np.zeros((T, C, A)), np.zeros((T, C, A))
    n_pos = 0
    mc = np.asarray(min_comps)[:, None]
    mi = np.asarray(min_imgs)[None, :]
    for i in range(len(probs)):
        p = np.asarray(probs[i], dtype=np.float32)
        g = np.asarray(gts[i]) > 0
        ga = int(g.sum())
        n_pos += int(ga > 0)
        gf = g.ravel().astype(np.float64)
        for ti, t in enumerate(thrs):
            pb = p >= t
            if pb.any():
                lab, n = ndimage.label(pb, structure=STRUCT8)
                lf = lab.ravel()
                m = lf > 0
                areas = np.bincount(lf[m], minlength=n + 1)[1:].astype(np.float64)
                inter = np.bincount(lf[m], weights=gf[m], minlength=n + 1)[1:]
            else:
                areas = inter = np.zeros(0)
            keep = areas[None, :] >= mc                                   # [C, ncomp]
            pa_c = (keep * areas[None, :]).sum(1)[:, None]                # [C,1]
            tp_c = (keep * inter[None, :]).sum(1)[:, None]
            ok = pa_c >= mi                                               # [C,A]
            pa = np.where(ok, pa_c, 0.0)
            tp = np.where(ok, tp_c, 0.0)
            fp, fn = pa - tp, ga - tp
            den = 2 * tp + fp + fn
            d = np.where((ga == 0) & (pa == 0), 1.0, 2 * tp / np.maximum(den, 1e-9))
            s_all[ti] += d
            if ga > 0:
                s_pos[ti] += d
    N = len(probs)
    s_all /= N
    s_pos /= max(1, n_pos)
    rows = [dict(thr=thrs[a], min_comp=min_comps[b], min_img=min_imgs[c], dice_all=s_all[a, b, c], dice_pos=s_pos[a, b, c])
            for a in range(T) for b in range(C) for c in range(A)]
    grid = pd.DataFrame(rows).sort_values(["dice_all", "dice_pos"], ascending=False).reset_index(drop=True)
    best = grid.iloc[0]
    return dict(thr=float(best.thr), min_comp=int(best.min_comp), min_img=int(best.min_img)), grid


# ----------------------------------------------------------------------------------------------
# statistics
# ----------------------------------------------------------------------------------------------
def bootstrap_ci(x: Sequence[float], n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> Tuple[float, float, float]:
    x = np.asarray(x, dtype=np.float64)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), (n_boot, len(x)))
    means = x[idx].mean(1)
    return float(x.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_compare(a: Sequence[float], b: Sequence[float], n_boot: int = 5000, seed: int = 0) -> Dict[str, float]:
    """Paired comparison of per-image scores of model B vs model A (same images, same order)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    keep = ~(np.isnan(a) | np.isnan(b))
    d = (b - a)[keep]
    out = dict(n=int(len(d)), mean_diff=float(d.mean()) if len(d) else float("nan"))
    if len(d) == 0:
        return out
    _, lo, hi = bootstrap_ci(d, n_boot=n_boot, seed=seed)
    out.update(ci_lo=lo, ci_hi=hi, frac_better=float((d > 0).mean()), frac_worse=float((d < 0).mean()))
    if np.any(d != 0):
        from scipy.stats import wilcoxon
        try:
            out["wilcoxon_p"] = float(wilcoxon(d, zero_method="wilcox").pvalue)
        except ValueError:
            out["wilcoxon_p"] = float("nan")
    else:
        out["wilcoxon_p"] = 1.0
    return out
