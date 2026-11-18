"""Dataset discovery, integrity audit, preprocessing cache and train/val split.

Dataset layout (verified on the real data by the audit cell):
    <root>/png_images/<id>_<train|test>_<0|1>_.png     RGB 1024x1024
    <root>/png_masks/<same file name>.png               grayscale 1024x1024, values {0, 255}
The trailing 0/1 in the file name is the image-level pneumothorax label; the official split is in the name too.

The cache stores every image once, resized, as uint8 (N,S,S) in .npy memmaps so that training never decodes 1024px PNGs:
    images.npy  - grayscale (the audit checks that R==G==B; the 3 channels are re-created on the fly)
    masks.npy   - binary {0,1}; resized with area interpolation + threshold 0.5 so thin lesions are not lost
    meta.csv    - one row per file with every statistic measured during caching (the audit is a by-product)
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

NAME_RE = re.compile(r"^(?P<id>.+?)_(?P<split>train|test)_(?P<label>[01])_?$")


# ----------------------------------------------------------------------------------------------
def find_dataset_root(search: Tuple[str, ...] = ("/kaggle/input",), must_contain: str = "png_images") -> Path:
    for base in search:
        b = Path(base)
        if not b.exists():
            continue
        for p in sorted(b.rglob(must_contain)):
            if p.is_dir() and (p.parent / "png_masks").is_dir():
                return p.parent
    raise FileNotFoundError(f"Could not find a folder with png_images/ and png_masks/ under {search}")


def list_files(root: Path) -> pd.DataFrame:
    root = Path(root)
    img_dir, msk_dir = root / "png_images", root / "png_masks"
    rows, unparsed = [], []
    for p in sorted(img_dir.glob("*.png")):
        m = NAME_RE.match(p.stem)
        if not m:
            unparsed.append(p.name)
            continue
        rows.append(dict(file=p.name, id=m["id"], official_split=m["split"], name_label=int(m["label"]),
                         has_mask_file=(msk_dir / p.name).exists()))
    if unparsed:
        raise ValueError(f"{len(unparsed)} image names do not match <id>_<train|test>_<0|1>_.png, e.g. {unparsed[:3]}")
    df = pd.DataFrame(rows)
    if df.empty:
        raise FileNotFoundError(f"no PNG images in {img_dir}")
    masks_only = sorted(set(p.name for p in msk_dir.glob("*.png")) - set(df.file))
    df.attrs["masks_without_image"] = masks_only
    return df


# ----------------------------------------------------------------------------------------------
def _process(args):
    i, img_path, msk_path, S = args
    out = dict(i=i)
    try:
        raw = open(img_path, "rb").read()
        out["md5"] = hashlib.md5(raw).hexdigest()
        a = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
        if a is None:
            raise ValueError("image decode failed")
        out["img_dtype"] = str(a.dtype)
        if a.dtype == np.uint16:
            a = (a / 257.0).astype(np.uint8)
        out["img_h"], out["img_w"] = int(a.shape[0]), int(a.shape[1])
        if a.ndim == 3:
            c = a[..., :3].astype(np.int16)
            diff = int(max(np.abs(c[..., 0] - c[..., 1]).max(), np.abs(c[..., 1] - c[..., 2]).max()))
            out["rgb_max_diff"] = diff
            gray = a[..., 0] if diff == 0 else cv2.cvtColor(a[..., :3], cv2.COLOR_BGR2GRAY)
        else:
            out["rgb_max_diff"] = 0
            gray = a
        out["img_min"], out["img_max"] = int(gray.min()), int(gray.max())
        img_s = cv2.resize(gray, (S, S), interpolation=cv2.INTER_AREA)

        m = cv2.imdecode(np.fromfile(msk_path, np.uint8), cv2.IMREAD_UNCHANGED)
        if m is None:
            raise ValueError("mask decode failed")
        if m.ndim == 3:
            m = m[..., 0]
        out["mask_h"], out["mask_w"] = int(m.shape[0]), int(m.shape[1])
        vals = np.unique(m)
        out["mask_values"] = ",".join(str(int(v)) for v in vals[:6])
        out["mask_n_values"] = int(len(vals))
        fg = (m > 0)
        out["fg_orig"] = int(fg.sum())
        msk_s = (cv2.resize(fg.astype(np.uint8) * 255, (S, S), interpolation=cv2.INTER_AREA) >= 128).astype(np.uint8)
        out["fg_resized"] = int(msk_s.sum())
        out["ok"] = True
        return out, img_s, msk_s
    except Exception as e:  # noqa: BLE001
        out["ok"] = False
        out["error"] = f"{type(e).__name__}: {e}"
        return out, np.zeros((S, S), np.uint8), np.zeros((S, S), np.uint8)


def build_cache(root: Path, out_dir: Path, size: int = 512, workers: int = 4, force: bool = False,
                limit: Optional[int] = None, log=print) -> Path:
    root, out_dir = Path(root), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = list_files(root)
    if limit:
        df = df.iloc[:limit].reset_index(drop=True)
    sig = hashlib.md5(("|".join(df.file) + f"|{size}").encode()).hexdigest()
    info_p = out_dir / "cache_info.json"
    if not force and info_p.exists() and (out_dir / "meta.csv").exists():
        info = json.loads(info_p.read_text())
        if info.get("signature") == sig and (out_dir / "images.npy").exists() and (out_dir / "masks.npy").exists():
            log(f"[cache] reusing existing cache ({info['n']} files, size {info['size']}) in {out_dir}")
            return out_dir
    N = len(df)
    log(f"[cache] building {N} files at {size}x{size} with {workers} workers ...")
    images = np.lib.format.open_memmap(out_dir / "images.npy", mode="w+", dtype=np.uint8, shape=(N, size, size))
    masks = np.lib.format.open_memmap(out_dir / "masks.npy", mode="w+", dtype=np.uint8, shape=(N, size, size))
    jobs = [(i, str(root / "png_images" / f), str(root / "png_masks" / f), size) for i, f in enumerate(df.file)]
    stats: List[dict] = [None] * N  # type: ignore
    t0 = time.time()
    with Pool(workers) as pool:
        for k, (st, im, mk) in enumerate(pool.imap_unordered(_process, jobs, chunksize=8), 1):
            images[st["i"]], masks[st["i"]] = im, mk
            stats[st["i"]] = st
            if k % 2000 == 0 or k == N:
                log(f"[cache] {k}/{N}  ({k / (time.time() - t0):.0f} files/s)")
    images.flush(); masks.flush()
    del images, masks
    meta = pd.concat([df.reset_index(drop=True), pd.DataFrame(stats).drop(columns=["i"])], axis=1)
    meta["has_gt"] = (meta["fg_orig"].fillna(0) > 0).astype(int)
    meta.to_csv(out_dir / "meta.csv", index=False)
    info_p.write_text(json.dumps(dict(signature=sig, n=N, size=size, created=time.strftime("%Y-%m-%d %H:%M:%S"))))
    log(f"[cache] done in {time.time() - t0:.0f}s -> {out_dir}")
    return out_dir


def open_cache(cache_dir: Path) -> Dict[str, object]:
    cache_dir = Path(cache_dir)
    meta = pd.read_csv(cache_dir / "meta.csv")
    return dict(images=np.load(cache_dir / "images.npy", mmap_mode="r"),
                masks=np.load(cache_dir / "masks.npy", mmap_mode="r"), meta=meta, dir=cache_dir,
                size=int(np.load(cache_dir / "images.npy", mmap_mode="r").shape[-1]))


# ----------------------------------------------------------------------------------------------
def audit_report(meta: pd.DataFrame, masks_without_image: Optional[List[str]] = None, log=print) -> pd.DataFrame:
    """Print and return a PASS/WARN table of integrity checks measured on EVERY file."""
    checks = []

    def add(name, ok, detail):
        checks.append(dict(check=name, status="PASS" if ok else "WARN", detail=detail))

    add("all image files decoded", bool(meta["ok"].all()), f"{int(meta['ok'].sum())}/{len(meta)} ok")
    add("every image has a mask file", bool(meta["has_mask_file"].all()), f"{int(meta['has_mask_file'].sum())}/{len(meta)}")
    if masks_without_image:
        add("no orphan masks", False, f"{len(masks_without_image)} masks without image")
    else:
        add("no orphan masks", True, "0")
    same_size = bool(((meta.img_h == meta.mask_h) & (meta.img_w == meta.mask_w)).all())
    add("image and mask sizes match", same_size, f"image sizes {meta.groupby(['img_h', 'img_w']).size().to_dict()}")
    vals = meta["mask_values"].astype(str).value_counts().to_dict()
    add("masks are binary {0,255} (or {0})", bool(meta["mask_n_values"].max() <= 2), f"value sets: {vals}")
    n_rgb = int((meta.rgb_max_diff > 0).sum())
    add("RGB channels identical (grayscale cache is lossless)", n_rgb == 0,
        f"{n_rgb} images with channel differences (max diff {int(meta.rgb_max_diff.max())})")
    dis = int((meta.name_label != meta.has_gt).sum())
    add("file-name label == mask content", dis == 0, f"{dis} disagreements")
    vanish = int(((meta.fg_orig > 0) & (meta.fg_resized == 0)).sum())
    add("no lesion vanishes when resizing", vanish == 0, f"{vanish} positive masks become empty after resizing")
    dup = meta[meta.duplicated("md5", keep=False)]
    add("no exact duplicate images", len(dup) == 0, f"{len(dup)} files in {dup.md5.nunique()} duplicate groups")
    if len(dup):
        cross = dup.groupby("md5").official_split.nunique()
        add("no duplicates across official train/test", int((cross > 1).sum()) == 0,
            f"{int((cross > 1).sum())} duplicate groups span train and test")
    out = pd.DataFrame(checks)
    for _, r in out.iterrows():
        log(f"  [{r.status}] {r.check}: {r.detail}")
    return out


def make_split(meta: pd.DataFrame, val_frac: float = 0.10, seed: int = 42) -> pd.DataFrame:
    """Official test rows stay 'test'. Official train rows are split into train/val, stratified on the mask-based
    label, with byte-identical images kept on the same side (grouped by MD5). Returns meta with a 'split' column."""
    from sklearn.model_selection import train_test_split

    meta = meta.copy()
    meta["split"] = "test"
    tr = meta[meta.official_split == "train"]
    first = tr.drop_duplicates("md5")                       # one representative per identical-image group
    # .to_numpy(dtype=object): pandas 3 / pyarrow-backed string columns are not indexable by sklearn
    groups_tr, groups_val = train_test_split(first.md5.to_numpy(dtype=object), test_size=val_frac, random_state=seed,
                                             stratify=first.has_gt.to_numpy())
    val_md5 = set(str(x) for x in groups_val)
    meta.loc[(meta.official_split == "train") & (~meta.md5.isin(val_md5)), "split"] = "train"
    meta.loc[(meta.official_split == "train") & (meta.md5.isin(val_md5)), "split"] = "val"
    assert set(meta[meta.split == "train"].md5).isdisjoint(set(meta[meta.split == "val"].md5))
    assert (meta[meta.official_split == "test"].split == "test").all()
    return meta


def small_lesion_edges(meta: pd.DataFrame) -> Tuple[float, float]:
    """Tertile edges of lesion area (pixels at cache resolution) measured on TRAIN positives only."""
    a = meta[(meta.split == "train") & (meta.fg_resized > 0)].fg_resized.values
    return float(np.quantile(a, 1 / 3)), float(np.quantile(a, 2 / 3))
