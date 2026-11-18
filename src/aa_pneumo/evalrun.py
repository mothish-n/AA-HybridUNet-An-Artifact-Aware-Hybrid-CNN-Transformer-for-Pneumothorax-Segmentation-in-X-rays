"""Evaluate one trained model on one split under several conditions (clean, artifact severities, unseen family)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from . import metrics as M
from .data.artifacts import TRAIN_FAMILIES, UNSEEN_FAMILY
from .data.dataset import PneumoDataset
from .engine import gather_gts, predict_probs

COND_ORDER = ["clean", "sev1", "sev2", "sev3", "unseen_sev1", "unseen_sev2", "unseen_sev3"]
COND_LABEL = {"clean": "Clean", "sev1": "Seen, sev 1", "sev2": "Seen, sev 2", "sev3": "Seen, sev 3",
              "unseen_sev1": "Unseen, sev 1", "unseen_sev2": "Unseen, sev 2", "unseen_sev3": "Unseen, sev 3"}


def condition_spec(name: str) -> Optional[Tuple[Sequence[str], int]]:
    if name == "clean":
        return None
    fam = (UNSEEN_FAMILY,) if name.startswith("unseen") else TRAIN_FAMILIES
    return fam, int(name[-1])


def evaluate_conditions(model, cache_dir, indices, op: dict, small_edges, conds: Sequence[str], out_dir: Path, device,
                        workers: int = 4, amp: bool = True, bs: int = 16, seed: int = 42, tau: float = 2.0,
                        force: bool = False, keep_idx: Sequence[int] = (), keep_conds: Sequence[str] = ("clean", "sev2"), log=print) -> Dict[str, dict]:
    """Saves <cond>_summary.json, <cond>_per_image.csv, <cond>_hist.npz in out_dir. Skips conditions already on disk
    (unless force). Returns {cond: summary}. keep_idx: positions (within `indices`) whose probability maps are returned
    under summary['_keep'] for qualitative figures (not saved to JSON)."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    gts = gather_gts(cache_dir, indices)
    res: Dict[str, dict] = {}
    for cond in conds:
        sp = out_dir / f"{cond}_summary.json"
        want_keep = bool(keep_idx) and cond in keep_conds
        if sp.exists() and not force and not want_keep:
            res[cond] = json.loads(sp.read_text()); continue
        ds = PneumoDataset(cache_dir, indices, train=False, eval_art=condition_spec(cond), seed=seed)
        probs, arts = predict_probs(model, ds, device, bs=bs, workers=workers, amp=amp, return_art=True)
        df, hists = M.evaluate_predictions(probs, gts, op, tau=tau)
        s = M.summarize(df, hists, small_edges)
        s.update(op=dict(op), condition=cond)
        df.insert(0, "idx", np.asarray(indices))
        df.to_csv(out_dir / f"{cond}_per_image.csv", index=False)
        np.savez_compressed(out_dir / f"{cond}_hist.npz", pos=hists["pos"], neg=hists["neg"])
        sp.write_text(json.dumps(s, indent=1))
        if want_keep:
            s["_keep"] = dict(probs=probs[list(keep_idx)].astype(np.float32), arts=arts[list(keep_idx)].astype(np.float32),
                              imgs=np.stack([np.asarray(ds[int(k)]["image"][0]) for k in keep_idx]))
        res[cond] = s
        log(f"    {cond:12s} Dice_pos {s['dice_pos']:.4f}  Dice_all {s['dice_all']:.4f}  sens {s['sensitivity']:.3f}  prec {s['precision']:.3f}  "
            f"HD95 {s['hd95']:.1f}  img-AUC {s['img_roc_auc']:.3f}")
        del probs, arts
    return res
