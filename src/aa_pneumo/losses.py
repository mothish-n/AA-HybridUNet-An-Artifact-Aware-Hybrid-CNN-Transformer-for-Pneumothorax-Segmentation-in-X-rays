"""Loss functions. All maths in float32. Every term is documented here so it can be copied into the paper.

Segmentation loss (BOTH models):   L_seg = w_bce * BCEwithLogits(z, y)  +  w_dice * (1 - SoftDice_batch(sigmoid(z), y))
    BCE         : per-pixel cross-entropy, mean over pixels.
    SoftDice    : computed over the WHOLE BATCH (sum over all images and pixels):  2*sum(p*y)+s / (sum(p)+sum(y)+s), s=1.
                  A per-image Dice is a bad idea here: ~78% of images are negative (y all zero) so their per-image Dice
                  is ~0 whatever the model does and it swamps the gradient. The batch-level version gives negatives a
                  sensible "predict nothing" signal and lets positives dominate where they exist.

Deep supervision (AA-HybridUNet only):  L_ds = mean_k L_seg(z_k, avgpool(y -> size of z_k))   (soft target, so a small
    lesion does not vanish at low resolution), weight w_ds.
Artifact loss (AA-HybridUNet with artifact head, only on samples with has_art=1):
    L_art = BCEwithLogits(a, t, pos_weight) + (1 - SoftDice(sigmoid(a), t))   over the corrupted samples only.
    Clean samples are NOT supervised, because real devices in them are unlabelled and must not be taught as "no artifact".
Consistency (optional): L_cons = BCEwithLogits(z_corrupted, stopgrad(sigmoid(z_clean)))  (soft-label cross-entropy)
    - the prediction on the corrupted copy is pulled towards the prediction on the clean copy of the same image.
Total: L = L_seg + w_ds*L_ds + w_art*L_art + w_cons*L_cons
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class LossConfig:
    w_bce: float = 1.0
    w_dice: float = 1.0
    w_ds: float = 0.3
    w_art: float = 0.5
    w_cons: float = 0.5
    art_pos_weight: float = 10.0
    smooth: float = 1.0

    def to_dict(self):
        return dict(self.__dict__)


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    p = torch.sigmoid(logits.float())
    t = target.float()
    inter = (p * t).sum()
    return 1.0 - (2.0 * inter + smooth) / (p.sum() + t.sum() + smooth)


def seg_loss(logits: torch.Tensor, target: torch.Tensor, cfg: LossConfig) -> torch.Tensor:
    logits, target = logits.float(), target.float()
    return cfg.w_bce * F.binary_cross_entropy_with_logits(logits, target) + cfg.w_dice * soft_dice_loss(logits, target, cfg.smooth)


def compute_loss(out: Dict[str, object], y: torch.Tensor, art_t: Optional[torch.Tensor], has_art: Optional[torch.Tensor],
                 cfg: LossConfig, cons_pair: bool = False) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """y/art_t: [B,1,S,S]; has_art: [B]. If cons_pair, the batch is [clean ; corrupted] (each B/2)."""
    parts: Dict[str, torch.Tensor] = {}
    seg = out["seg"]
    parts["seg"] = seg_loss(seg, y, cfg)
    total = parts["seg"]
    aux = out.get("aux")
    if aux and cfg.w_ds > 0:
        ds = torch.stack([seg_loss(z, F.adaptive_avg_pool2d(y.float(), z.shape[-2:]), cfg) for z in aux]).mean()
        parts["ds"] = ds
        total = total + cfg.w_ds * ds
    art = out.get("artifact")
    if art is not None and cfg.w_art > 0 and has_art is not None and bool((has_art > 0).any()):
        sel = has_art > 0
        a, t = art[sel].float(), art_t[sel].float()
        bce = F.binary_cross_entropy_with_logits(a, t, pos_weight=torch.tensor(cfg.art_pos_weight, device=a.device))
        la = bce + soft_dice_loss(a, t, cfg.smooth)
        parts["art"] = la
        total = total + cfg.w_art * la
    if cons_pair and cfg.w_cons > 0:
        n = seg.shape[0] // 2
        lc = F.binary_cross_entropy_with_logits(seg[n:].float(), torch.sigmoid(seg[:n].detach().float()))
        parts["cons"] = lc
        total = total + cfg.w_cons * lc
    parts["total"] = total
    return total, parts
