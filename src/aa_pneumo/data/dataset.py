"""Training / evaluation dataset: cached uint8 arrays -> augmentation -> optional synthetic artifacts -> tensors.

Outputs (per sample)
    image        float32 [1,S,S] in [0,1]      (1 channel; replicated to 3 and ImageNet-normalised on the GPU)
    mask         float32 [1,S,S] {0,1}         pneumothorax target (never altered by artifacts)
    art          float32 [1,S,S] {0,1}         synthetic-artifact mask (all zeros when no artifact was added)
    has_art      float32 scalar                1 when `art` is a reliable label (synthetic artifact was drawn), else 0
    idx          int64                         row in the cache
    image_clean  float32 [1,S,S]  (only when pair=True)  the same augmented image WITHOUT artifacts

Augmentation policy (TRAIN only; identical for every model): horizontal flip, small rotation/scale/translation,
brightness+contrast, gamma, mild Gaussian noise. Geometry is applied identically to image and mask (nearest for the
mask). No elastic warps, no vertical flips, no large rotations (they would distort anatomy).
"""
from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from .artifacts import TRAIN_FAMILIES, add_artifacts


@dataclass
class AugConfig:
    hflip_p: float = 0.5
    affine_p: float = 0.7
    rotate_deg: float = 10.0
    scale: Tuple[float, float] = (0.92, 1.08)
    translate: float = 0.05
    bc_p: float = 0.6
    brightness: float = 0.08
    contrast: float = 0.15
    gamma_p: float = 0.4
    gamma: Tuple[float, float] = (0.8, 1.25)
    noise_p: float = 0.3
    noise_sigma: Tuple[float, float] = (0.005, 0.02)

    def to_dict(self):
        return dict(self.__dict__)


def augment(img_u8: np.ndarray, msk_u8: np.ndarray, rng: np.random.Generator, c: AugConfig):
    S = img_u8.shape[0]
    img, msk = img_u8, msk_u8
    if rng.random() < c.hflip_p:
        img, msk = img[:, ::-1], msk[:, ::-1]
    img = np.ascontiguousarray(img)
    msk = np.ascontiguousarray(msk)
    if rng.random() < c.affine_p:
        ang = rng.uniform(-c.rotate_deg, c.rotate_deg)
        sc = rng.uniform(*c.scale)
        M = cv2.getRotationMatrix2D((S / 2.0, S / 2.0), ang, sc)
        M[:, 2] += rng.uniform(-c.translate, c.translate, 2) * S
        img = cv2.warpAffine(img, M, (S, S), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        msk = cv2.warpAffine(msk, M, (S, S), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    x = img.astype(np.float32) / 255.0
    if rng.random() < c.bc_p:
        x = (x - 0.5) * rng.uniform(1 - c.contrast, 1 + c.contrast) + 0.5 + rng.uniform(-c.brightness, c.brightness)
    x = np.clip(x, 0.0, 1.0)
    if rng.random() < c.gamma_p:
        x = np.power(x, rng.uniform(*c.gamma))
    if rng.random() < c.noise_p:
        x = np.clip(x + rng.normal(0.0, rng.uniform(*c.noise_sigma), x.shape).astype(np.float32), 0.0, 1.0)
    return x.astype(np.float32), msk.astype(np.uint8)


class PneumoDataset(Dataset):
    def __init__(self, cache_dir, indices: Sequence[int], train: bool, aug: Optional[AugConfig] = None,
                 art_mix: float = 0.0, art_families: Sequence[str] = TRAIN_FAMILIES,
                 eval_art: Optional[Tuple[Sequence[str], int]] = None, pair: bool = False, seed: int = 42):
        """train=True  : random augmentation; each sample is corrupted with probability art_mix (random severity 1-3).
        train=False : no augmentation; if eval_art=(families, severity) every sample is corrupted deterministically
                      (seeded by seed+idx+severity so all models see exactly the same pixels).
        pair=True   : also return `image_clean` and ALWAYS corrupt `image` (used by the consistency loss)."""
        self.cache_dir = str(cache_dir)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.train, self.aug = train, aug or AugConfig()
        self.art_mix, self.art_families = float(art_mix), tuple(art_families)
        self.eval_art, self.pair, self.seed = eval_art, pair, int(seed)
        self._epoch = mp.Value("i", 0)
        self._img = self._msk = None
        self.labels = None  # filled by caller if needed

    def set_epoch(self, e: int):
        self._epoch.value = int(e)

    def _open(self):
        if self._img is None:
            self._img = np.load(f"{self.cache_dir}/images.npy", mmap_mode="r")
            self._msk = np.load(f"{self.cache_dir}/masks.npy", mmap_mode="r")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, k: int):
        self._open()
        idx = int(self.indices[k])
        img_u8, msk_u8 = np.asarray(self._img[idx]), np.asarray(self._msk[idx])
        S = img_u8.shape[0]
        art = np.zeros((S, S), np.uint8)
        has = 0.0
        if self.train:
            rng = np.random.default_rng([self.seed, int(self._epoch.value), idx])
            x, m = augment(img_u8, msk_u8, rng, self.aug)
            x_clean = x
            if self.pair or (self.art_mix > 0 and rng.random() < self.art_mix):
                x, art, _ = add_artifacts(x, rng, self.art_families, int(rng.integers(1, 4)))
                has = 1.0
        else:
            x, m = img_u8.astype(np.float32) / 255.0, msk_u8
            x_clean = x
            if self.eval_art is not None:
                fams, sev = self.eval_art
                rng = np.random.default_rng([self.seed, 7919, idx, int(sev)])
                x, art, _ = add_artifacts(x, rng, fams, int(sev))
                has = 1.0
        out = dict(image=torch.from_numpy(x)[None], mask=torch.from_numpy(m.astype(np.float32))[None],
                   art=torch.from_numpy(art.astype(np.float32))[None], has_art=torch.tensor(has, dtype=torch.float32),
                   idx=torch.tensor(idx, dtype=torch.int64))
        if self.pair:
            out["image_clean"] = torch.from_numpy(x_clean)[None]
        return out


class EpochSampler(Sampler):
    """Every epoch: ALL positive images + a fresh random `neg_frac` of the negatives (so all negatives are still seen
    over the course of training). Deterministic given (seed, epoch), which makes resuming exact."""

    def __init__(self, labels: np.ndarray, neg_frac: float = 1.0, seed: int = 42, max_samples: Optional[int] = None):
        self.labels = np.asarray(labels).astype(int)
        self.neg_frac, self.seed, self.max_samples, self.epoch = float(neg_frac), int(seed), max_samples, 0

    def set_epoch(self, e: int):
        self.epoch = int(e)

    def _n(self):
        pos = int((self.labels == 1).sum())
        neg = int(round(self.neg_frac * int((self.labels == 0).sum())))
        n = pos + neg
        return min(n, self.max_samples) if self.max_samples else n

    def __len__(self):
        return self._n()

    def __iter__(self):
        rng = np.random.default_rng([self.seed, self.epoch])
        pos = np.flatnonzero(self.labels == 1)
        neg = np.flatnonzero(self.labels == 0)
        neg = rng.permutation(neg)[: int(round(self.neg_frac * len(neg)))]
        allidx = rng.permutation(np.concatenate([pos, neg]))
        return iter(allidx[: self._n()].tolist())


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def to_model_input(x1: torch.Tensor) -> torch.Tensor:
    """[B,1,S,S] in [0,1] -> [B,3,S,S] ImageNet-normalised (same function for train/val/test/inference)."""
    x = x1.expand(-1, 3, -1, -1)
    mean = torch.tensor(IMAGENET_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean) / std
