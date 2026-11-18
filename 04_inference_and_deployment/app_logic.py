"""Pure-Python helpers for the Streamlit app (no Streamlit import -> unit-testable)."""
from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

MAX_BYTES = 60 * 1024 * 1024
MIN_SIDE = 128
ALLOWED = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".dcm"}


class InputError(ValueError):
    """Raised with a user-readable message when the uploaded file cannot be used."""


def discover_models(models_dir: Path) -> Dict[str, dict]:
    """Every sub-folder with weights.pt + model_config.json is a model. Returns {folder: info}."""
    out: Dict[str, dict] = {}
    if not Path(models_dir).is_dir():
        return out
    for d in sorted(Path(models_dir).iterdir()):
        if (d / "weights.pt").exists() and (d / "model_config.json").exists():
            cfg = json.loads((d / "model_config.json").read_text())
            tm = d / "test_metrics_clean.json"
            out[d.name] = dict(path=d, kind=cfg.get("kind"), run=cfg.get("run", d.name), img_size=cfg.get("img_size", 512),
                               test=json.loads(tm.read_text()) if tm.exists() else {})
    return out


def decode_upload(name: str, data: bytes) -> np.ndarray:
    """Uploaded bytes -> gray uint8 [H,W]. Raises InputError with a clear message for anything unusable."""
    ext = Path(name).suffix.lower()
    if ext not in ALLOWED:
        raise InputError(f"Unsupported file type '{ext}'. Use PNG, JPG, BMP, TIFF or DICOM (.dcm).")
    if len(data) == 0:
        raise InputError("The file is empty.")
    if len(data) > MAX_BYTES:
        raise InputError(f"File is too large ({len(data) / 1e6:.0f} MB; limit {MAX_BYTES // 1_000_000} MB).")
    if ext == ".dcm":
        try:
            import pydicom
        except ImportError as e:
            raise InputError("DICOM support needs the 'pydicom' package (pip install pydicom).") from e
        try:
            ds = pydicom.dcmread(io.BytesIO(data))
            a = ds.pixel_array.astype(np.float32)
        except Exception as e:  # noqa: BLE001
            raise InputError(f"Could not read the DICOM file: {e}") from e
        if a.ndim == 3:
            a = a[..., 0] if a.shape[-1] in (1, 3, 4) else a[0]
        lo, hi = float(a.min()), float(a.max())
        a = (a - lo) / max(hi - lo, 1e-6)
        if str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
            a = 1.0 - a
        g = (a * 255).astype(np.uint8)
    else:
        arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise InputError("The file could not be decoded as an image (is it corrupted?).")
        if arr.dtype == np.uint16:
            arr = (arr / 257.0).astype(np.uint8)
        elif arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        if arr.ndim == 3:
            arr = cv2.cvtColor(arr[..., :3], cv2.COLOR_BGR2GRAY)
        g = arr
    if min(g.shape) < MIN_SIDE:
        raise InputError(f"Image is too small ({g.shape[1]}x{g.shape[0]} px; minimum side {MIN_SIDE}px).")
    if int(g.max()) == int(g.min()):
        raise InputError("The image is a single flat colour - nothing to analyse.")
    return g


def overlay(gray: np.ndarray, mask: np.ndarray, alpha: float = 0.45, colour=(255, 40, 40), outline: bool = True) -> np.ndarray:
    """RGB overlay of a binary mask on a gray image."""
    rgb = np.stack([gray] * 3, -1).astype(np.float32)
    m = mask.astype(bool)
    for c in range(3):
        rgb[..., c] = np.where(m, (1 - alpha) * rgb[..., c] + alpha * colour[c], rgb[..., c])
    out = rgb.astype(np.uint8)
    if outline and m.any():
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, (255, 255, 0), max(1, gray.shape[0] // 400))
    return out


def heatmap(gray: np.ndarray, prob: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    h = cv2.applyColorMap((np.clip(prob, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)[..., ::-1]
    base = np.stack([gray] * 3, -1)
    return ((1 - alpha) * base + alpha * h).astype(np.uint8)


def png_bytes(arr: np.ndarray) -> bytes:
    a = arr if arr.ndim == 2 else cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", a)
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return buf.tobytes()


def mask_stats(mask_full: np.ndarray, prob_full: np.ndarray) -> Dict[str, float]:
    area = int(mask_full.astype(bool).sum())
    comps = int(cv2.connectedComponents(mask_full.astype(np.uint8), connectivity=8)[0] - 1) if area else 0
    return dict(area_px=area, area_pct=100.0 * area / mask_full.size, n_regions=comps, max_prob=float(prob_full.max()),
                mean_prob_in_mask=float(prob_full[mask_full.astype(bool)].mean()) if area else 0.0)
