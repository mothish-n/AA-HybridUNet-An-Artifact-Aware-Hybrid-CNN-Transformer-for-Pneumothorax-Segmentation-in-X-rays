"""Inference helpers shared by the notebook and the Streamlit app (no training code, no Kaggle paths).

A "bundle" is a folder:
    bundle/
      model_config.json   {"kind": "aa"|"baseline", "model_cfg": {...}, "img_size": 512, "name": "..."}
      weights.pt          {"model": state_dict}
      operating_point.json {"thr":..., "min_comp":..., "min_img":...}   (tuned on the validation split)
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
import torch

from .data.dataset import to_model_input
from .metrics import postprocess
from .models import build_model


def load_bundle(folder, device: Optional[str] = None):
    folder = Path(folder)
    meta = json.loads((folder / "model_config.json").read_text())
    cfg = dict(meta["model_cfg"]); cfg["pretrained"] = False          # weights come from weights.pt, no download
    model = build_model(meta["kind"], cfg)
    ck = torch.load(folder / "weights.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"], strict=True)                   # strict: refuse an incompatible checkpoint
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(dev).eval()
    op_p = folder / "operating_point.json"
    op = json.loads(op_p.read_text()) if op_p.exists() else dict(thr=0.5, min_comp=0, min_img=0)
    return model, meta, op, dev


def to_gray_u8(arr: np.ndarray) -> np.ndarray:
    """RGB/RGBA/gray uint8 or uint16 array -> gray uint8 (same convention as the training cache)."""
    a = arr
    if a.dtype == np.uint16:
        a = (a / 257.0).astype(np.uint8)
    elif a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)
    if a.ndim == 3:
        a = cv2.cvtColor(a[..., :3], cv2.COLOR_RGB2GRAY)
    return a


@torch.no_grad()
def predict_image(model, meta: dict, op: dict, device, gray_u8: np.ndarray, amp: bool = True, thr: Optional[float] = None
                  ) -> Dict[str, object]:
    S = int(meta.get("img_size", 512))
    h0, w0 = gray_u8.shape
    small = cv2.resize(gray_u8, (S, S), interpolation=cv2.INTER_AREA)
    x = torch.from_numpy(small.astype(np.float32) / 255.0)[None, None].to(device)
    t0 = time.perf_counter()
    with torch.autocast("cuda", dtype=torch.float16, enabled=(amp and device.type == "cuda")):
        out = model(to_model_input(x))
    prob = torch.sigmoid(out["seg"].float())[0, 0].cpu().numpy()
    art = torch.sigmoid(out["artifact"].float())[0, 0].cpu().numpy() if "artifact" in out else None
    if device.type == "cuda":
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1e3
    o = dict(op)
    if thr is not None:
        o["thr"] = float(thr)
    mask = postprocess(prob, **o)
    return dict(prob=prob, mask=mask.astype(np.uint8), artifact_prob=art, inference_ms=ms, size=(h0, w0), img_small=small,
                prob_full=cv2.resize(prob, (w0, h0), interpolation=cv2.INTER_LINEAR),
                mask_full=cv2.resize(mask.astype(np.uint8), (w0, h0), interpolation=cv2.INTER_NEAREST))
