"""Synthetic ICU artifact generator (ECG wires, chest drainage tubes, monitoring lines, + an UNSEEN catheter family).

IMPORTANT - what this is and is not
    These are hand-designed, procedurally drawn overlays. They are a controlled stress test for robustness, NOT a
    faithful simulation of real devices, and results on them are NOT clinical validation.

Design
    * Four families. TRAIN_FAMILIES are used when training with artifacts; UNSEEN_FAMILY is never used in training and
      is only used at test time to probe generalisation to a device shape the model has not seen.
    * Three severities (1 = mild, 2 = moderate, 3 = severe) control the number of devices, their width, opacity.
    * Everything is drawn in the *image* only. The pneumothorax mask is never touched, so the label is preserved even
      where a device crosses the lesion.
    * The function also returns a binary ARTIFACT mask (where a synthetic device is visibly present). That mask is the
      only artifact label that exists; real devices that are already in the X-ray are NOT labelled (see engine/losses).
    * Fully deterministic given the numpy Generator, so validation/test conditions are identical for every model.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np

TRAIN_FAMILIES: Tuple[str, ...] = ("ecg_wire", "chest_tube", "monitor_line")
UNSEEN_FAMILY: str = "catheter_loop"
ALL_FAMILIES = TRAIN_FAMILIES + (UNSEEN_FAMILY,)

SEVERITY: Dict[int, dict] = {
    1: dict(n=(1, 2), width=0.85, opacity=(0.35, 0.55)),
    2: dict(n=(2, 4), width=1.00, opacity=(0.50, 0.80)),
    3: dict(n=(4, 6), width=1.25, opacity=(0.75, 1.00)),
}
LABEL_THRESHOLD = 0.10   # a pixel counts as "artifact" when opacity*coverage exceeds this


# ----------------------------------------------------------------------------------------------
# geometry helpers (coordinates are in pixels; S = image side)
# ----------------------------------------------------------------------------------------------
def _bezier(p0, p1, p2, p3, n=80) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n)[:, None]
    p0, p1, p2, p3 = (np.asarray(p, dtype=np.float32) for p in (p0, p1, p2, p3))
    return ((1 - t) ** 3) * p0 + 3 * ((1 - t) ** 2) * t * p1 + 3 * (1 - t) * (t ** 2) * p2 + (t ** 3) * p3


def _edge_point(rng: np.random.Generator, S: int, edge: str) -> np.ndarray:
    u = rng.uniform(0.15, 0.85) * S
    return {"top": np.array([u, 0.0]), "left": np.array([0.0, rng.uniform(0.2, 0.7) * S]),
            "right": np.array([S - 1.0, rng.uniform(0.2, 0.7) * S]), "bottom": np.array([u, S - 1.0])}[edge]


def _draw_line(pts: np.ndarray, S: int, width: float) -> np.ndarray:
    """Anti-aliased polyline -> float mask in [0,1]."""
    m = np.zeros((S, S), np.uint8)
    cv2.polylines(m, [np.round(pts * 16).astype(np.int32)], False, 255, thickness=max(1, int(round(width))),
                  lineType=cv2.LINE_AA, shift=4)
    out = m.astype(np.float32) / 255.0
    sigma = max(0.4, 0.35 * width)
    return cv2.GaussianBlur(out, (0, 0), sigma) if sigma > 0 else out


def _disc(S: int, centre: np.ndarray, r: float) -> np.ndarray:
    m = np.zeros((S, S), np.uint8)
    cv2.circle(m, (int(centre[0]), int(centre[1])), max(1, int(round(r))), 255, -1, lineType=cv2.LINE_AA)
    return m.astype(np.float32) / 255.0


# ----------------------------------------------------------------------------------------------
# families: each returns (list of (mask, intensity) layers, label_mask)
# ----------------------------------------------------------------------------------------------
def _ecg_wire(rng, S, wscale):
    sc = S / 512.0
    start = _edge_point(rng, S, rng.choice(["top", "left", "right"]))
    end = np.array([rng.uniform(0.3, 0.7) * S, rng.uniform(0.25, 0.6) * S])
    c1 = start + (end - start) * 0.33 + rng.normal(0, 0.08 * S, 2)
    c2 = start + (end - start) * 0.66 + rng.normal(0, 0.08 * S, 2)
    pts = np.clip(_bezier(start, c1, c2, end), 0, S - 1)
    w = rng.uniform(1.5, 3.2) * sc * wscale
    body = _draw_line(pts, S, w)
    pad = _disc(S, end, rng.uniform(5, 8) * sc * wscale)
    ring = np.clip(pad - cv2.erode(pad, np.ones((3, 3), np.uint8)) * 0.6, 0, 1)
    inten = rng.uniform(0.85, 1.0)
    return [(body, inten), (np.maximum(pad * 0.5, ring), inten)], np.maximum(body, pad)


def _chest_tube(rng, S, wscale):
    sc = S / 512.0
    side = rng.choice(["left", "right"])
    start = _edge_point(rng, S, side)
    end = np.array([rng.uniform(0.25, 0.75) * S, rng.uniform(0.2, 0.75) * S])
    c1 = start + (end - start) * 0.4 + rng.normal(0, 0.06 * S, 2)
    c2 = start + (end - start) * 0.75 + rng.normal(0, 0.06 * S, 2)
    pts = np.clip(_bezier(start, c1, c2, end), 0, S - 1)
    w = rng.uniform(7, 13) * sc * wscale
    body = _draw_line(pts, S, w)                                   # translucent tube
    edge = np.clip(body - cv2.erode(body, np.ones((3, 3), np.uint8), iterations=max(1, int(round(w / 5)))), 0, 1)
    stripe = _draw_line(pts, S, max(1.0, w * 0.18))                # radiopaque line
    return [(body, rng.uniform(0.55, 0.72)), (edge, rng.uniform(0.8, 0.95)), (stripe, 1.0)], body


def _monitor_line(rng, S, wscale):
    sc = S / 512.0
    start = _edge_point(rng, S, rng.choice(["top", "left", "right"]))
    end = _edge_point(rng, S, rng.choice(["bottom", "left", "right"]))
    c1 = start + (end - start) * 0.3 + rng.normal(0, 0.12 * S, 2)
    c2 = start + (end - start) * 0.7 + rng.normal(0, 0.12 * S, 2)
    pts = np.clip(_bezier(start, c1, c2, end, n=120), 0, S - 1)
    w = rng.uniform(2.0, 4.5) * sc * wscale
    body = _draw_line(pts, S, w)
    beads = np.zeros((S, S), np.float32)
    for k in range(10, len(pts) - 5, int(rng.integers(14, 24))):
        beads = np.maximum(beads, _disc(S, pts[k], w * 1.1))
    inten = rng.uniform(0.7, 0.95)
    return [(body, inten), (beads, min(1.0, inten + 0.05))], np.maximum(body, beads)


def _catheter_loop(rng, S, wscale):
    """UNSEEN family: central-line style catheter entering from the neck, running down the mediastinum and ending
    in a pigtail loop with a bright tip marker and periodic dash markers. Never used in training."""
    sc = S / 512.0
    x0 = rng.uniform(0.40, 0.60) * S
    start = np.array([x0, 0.0])
    end = np.array([x0 + rng.normal(0, 0.05 * S), rng.uniform(0.35, 0.55) * S])
    mid = _bezier(start, start + [rng.normal(0, 0.04 * S), 0.15 * S], end - [rng.normal(0, 0.04 * S), 0.15 * S], end, n=70)
    r = rng.uniform(0.025, 0.045) * S
    th = np.linspace(0, 2.4 * np.pi, 50)
    loop = np.stack([end[0] + r * (th / 7.5) * np.cos(th) * 2.2, end[1] + r * (th / 7.5) * np.sin(th) * 2.2 + r], 1)
    pts = np.clip(np.concatenate([mid, loop], 0), 0, S - 1)
    w = rng.uniform(2.5, 4.5) * sc * wscale
    body = _draw_line(pts, S, w)
    tip = _disc(S, pts[-1], w * 1.8)
    dashes = np.zeros((S, S), np.float32)
    for k in range(5, len(mid), 9):
        dashes = np.maximum(dashes, _disc(S, mid[k], w * 0.9))
    return [(body, rng.uniform(0.75, 0.95)), (tip, 1.0), (dashes, 1.0)], np.maximum(np.maximum(body, tip), dashes)


_FAMILY_FN = {"ecg_wire": _ecg_wire, "chest_tube": _chest_tube, "monitor_line": _monitor_line, "catheter_loop": _catheter_loop}


# ----------------------------------------------------------------------------------------------
# public API
# ----------------------------------------------------------------------------------------------
def add_artifacts(img: np.ndarray, rng: np.random.Generator, families: Sequence[str], severity: int
                  ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Overlay synthetic devices on a float32 image in [0,1] (H==W).

    Returns (corrupted image float32 in [0,1], artifact mask uint8 {0,1}, list of families drawn).
    The input array is not modified.
    """
    if severity not in SEVERITY:
        raise ValueError(f"severity must be one of {sorted(SEVERITY)}")
    for f in families:
        if f not in _FAMILY_FN:
            raise ValueError(f"unknown family {f!r}")
    S = img.shape[0]
    cfg = SEVERITY[severity]
    out = img.astype(np.float32).copy()
    label = np.zeros((S, S), np.float32)
    drawn: List[str] = []
    n = int(rng.integers(cfg["n"][0], cfg["n"][1] + 1))
    for _ in range(n):
        fam = str(families[int(rng.integers(0, len(families)))])
        opacity = float(rng.uniform(*cfg["opacity"]))
        layers, lab = _FAMILY_FN[fam](rng, S, cfg["width"])
        for m, inten in layers:
            a = np.clip(m * opacity, 0.0, 1.0)
            out = out * (1.0 - a) + a * float(inten)
        label = np.maximum(label, np.clip(lab * opacity, 0, 1))
        drawn.append(fam)
    return np.clip(out, 0.0, 1.0), (label > LABEL_THRESHOLD).astype(np.uint8), drawn
