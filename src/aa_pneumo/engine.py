"""Training / validation / prediction engine.

* Resumable: every epoch writes last.pt (model, optimizer, scheduler, scaler, epoch, history, config, best metrics).
  Every write is atomic (temp file + rename) and the previous file is kept as <name>.prev, so an interrupted save can
  never destroy the only copy of the weights.
* Time guard: training stops cleanly before a deadline (Kaggle's 12 h session limit) and can be resumed later.
* Model selection: best epoch = highest validation Dice_all (empty convention) over a small (threshold x min-area)
  grid, evaluated on the clean validation split. The test split is never touched here.
"""
from __future__ import annotations

import copy
import csv
import json
import math
import os
import random
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .data.dataset import AugConfig, EpochSampler, PneumoDataset, to_model_input
from .losses import LossConfig, compute_loss
from .models import build_model, count_parameters


# ----------------------------------------------------------------------------------------------
@dataclass
class RunConfig:
    name: str
    kind: str                         # 'baseline' | 'aa'
    model_cfg: dict
    art_mix: float = 0.0              # probability that a TRAIN sample gets synthetic artifacts
    consistency: bool = False         # clean/corrupted pairs + consistency loss
    epochs: int = 20
    description: str = ""


@dataclass
class TrainConfig:
    img_size: int = 512
    effective_bs: int = 8
    lr_enc: float = 2e-4
    lr_new: float = 6e-4
    weight_decay: float = 1e-4
    warmup_epochs: float = 1.0
    min_lr_ratio: float = 0.01
    grad_clip: float = 1.0
    neg_frac: float = 0.5
    patience: int = 6
    min_epochs: int = 8
    seed: int = 42
    workers: int = 4
    amp: bool = True
    log_every: int = 100
    sel_thrs: Tuple[float, ...] = (0.3, 0.4, 0.5, 0.6, 0.7)
    sel_min_imgs: Tuple[int, ...] = (0, 250, 1000)    # image-level "ignore tiny predictions" areas (pixels @ img_size)
    aug: dict = field(default_factory=lambda: AugConfig().to_dict())
    loss: dict = field(default_factory=lambda: LossConfig().to_dict())


# ----------------------------------------------------------------------------------------------
def seed_everything(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def worker_init(_):
    cv2 = __import__("cv2")
    cv2.setNumThreads(0)


def atomic_save(obj, path: Path, keep_backup: bool = True):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    if keep_backup and path.exists():
        os.replace(path, path.with_name(path.name + ".prev"))
    os.replace(tmp, path)


def load_ckpt(path: Path, device="cpu"):
    return torch.load(path, map_location=device, weights_only=False)


def make_loader(ds, bs, shuffle=False, sampler=None, workers=4, drop_last=False):
    return DataLoader(ds, batch_size=bs, shuffle=(shuffle and sampler is None), sampler=sampler, num_workers=workers,
                      pin_memory=torch.cuda.is_available(), drop_last=drop_last, worker_init_fn=worker_init,
                      persistent_workers=False, prefetch_factor=(4 if workers > 0 else None))


def make_scheduler(opt, total_steps: int, warmup_steps: int, min_ratio: float):
    def f(step):
        if step < warmup_steps:
            return 0.05 + 0.95 * step / max(1, warmup_steps)
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))
    return torch.optim.lr_scheduler.LambdaLR(opt, f)


# ----------------------------------------------------------------------------------------------
# validation (GPU, streaming): grid of (threshold, min_img) -> Dice_all / Dice_pos, plus image-level AUC
# ----------------------------------------------------------------------------------------------
@torch.no_grad()
def validate(model, loader, device, tc: TrainConfig, loss_cfg: LossConfig, amp: bool) -> Dict[str, float]:
    model.eval()
    thrs = torch.tensor(tc.sel_thrs, device=device).view(-1, 1, 1, 1, 1)   # [T,1,1,1,1]
    mins = tc.sel_min_imgs
    d_all: List[torch.Tensor] = []     # per image: [B, T, A]
    has_l, maxp_l = [], []
    loss_sum, n = 0.0, 0
    from .losses import seg_loss
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["mask"].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=(amp and device.type == "cuda")):
            out = model(to_model_input(x))
        z = out["seg"].float()
        loss_sum += float(seg_loss(z, y, loss_cfg)) * x.size(0); n += x.size(0)
        p = torch.sigmoid(z)                                              # [B,1,S,S]
        pred = (p.unsqueeze(0) >= thrs)                                   # [T,B,1,S,S]
        gt = (y > 0.5).unsqueeze(0)
        tp = (pred & gt).flatten(2).sum(2).float()                        # [T,B]
        pa = pred.flatten(2).sum(2).float()
        ga = gt.flatten(2).sum(2).float().expand_as(tp)
        cols = []
        for m in mins:
            keep = (pa >= max(1, m)) if m > 0 else torch.ones_like(pa, dtype=torch.bool)
            pa_m, tp_m = pa * keep, tp * keep
            d = torch.where((ga == 0) & (pa_m == 0), torch.ones_like(tp_m), 2 * tp_m / (pa_m + ga).clamp_min(1.0))
            cols.append(d)
        d_all.append(torch.stack(cols, 2).permute(1, 0, 2))               # [B,T,A]
        has_l.append((y.flatten(1).max(1).values > 0.5).cpu())
        maxp_l.append(p.flatten(1).max(1).values.cpu())
    D = torch.cat(d_all).cpu()                                            # [N,T,A]
    has = torch.cat(has_l).numpy().astype(bool)
    dice_all = D.mean(0)                                                  # [T,A]
    dice_pos = D[torch.from_numpy(has)].mean(0) if has.any() else torch.full_like(dice_all, float("nan"))
    ti, ai = np.unravel_index(int(torch.argmax(dice_all)), tuple(dice_all.shape))
    res = dict(val_loss=loss_sum / max(1, n), val_dice_all=float(dice_all[ti, ai]), val_dice_pos=float(dice_pos[ti, ai]),
               best_thr=float(tc.sel_thrs[ti]), best_min_img=int(mins[ai]),
               val_dice_pos_at_0p5=float(dice_pos[list(tc.sel_thrs).index(0.5), 0]) if 0.5 in tc.sel_thrs else float("nan"))
    if has.any() and (~has).any():
        from sklearn.metrics import roc_auc_score
        res["val_img_auc"] = float(roc_auc_score(has, torch.cat(maxp_l).numpy()))
    else:
        res["val_img_auc"] = float("nan")
    return res


# ----------------------------------------------------------------------------------------------
@torch.no_grad()
def predict_probs(model, ds, device, bs: int = 16, workers: int = 4, amp: bool = True, return_art: bool = False,
                  desc: str = "") -> np.ndarray:
    """Sigmoid probabilities for every sample of `ds` (shuffle=False), float16 [N,S,S]. Optionally the artifact map."""
    model.eval()
    loader = make_loader(ds, bs, workers=workers)
    S = int(np.load(f"{ds.cache_dir}/images.npy", mmap_mode="r").shape[-1])
    probs = np.zeros((len(ds), S, S), np.float16)
    arts = np.zeros((len(ds), S, S), np.float16) if return_art else None
    k = 0
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=(amp and device.type == "cuda")):
            out = model(to_model_input(x))
        b = x.size(0)
        probs[k:k + b] = torch.sigmoid(out["seg"].float())[:, 0].cpu().numpy().astype(np.float16)
        if return_art and "artifact" in out:
            arts[k:k + b] = torch.sigmoid(out["artifact"].float())[:, 0].cpu().numpy().astype(np.float16)
        k += b
    return (probs, arts) if return_art else probs


def gather_gts(cache_dir, indices) -> np.ndarray:
    m = np.load(f"{cache_dir}/masks.npy", mmap_mode="r")
    return np.stack([np.asarray(m[int(i)]) for i in indices])


# ----------------------------------------------------------------------------------------------
def probe_micro_batch(run: RunConfig, tc: TrainConfig, device, candidates=(8, 6, 4, 3, 2, 1), pair: bool = False,
                      mem_fraction: float = 0.88, log=print) -> int:
    """Largest micro-batch whose full training step (fwd+bwd+AdamW) fits in GPU memory. Uses random data."""
    if device.type != "cuda":
        return 2
    total = torch.cuda.get_device_properties(0).total_memory
    cfg = dict(run.model_cfg); cfg["pretrained"] = False
    for mb in candidates:
        try:
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            model = build_model(run.kind, cfg).to(device).train()
            opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
            scaler = torch.amp.GradScaler("cuda")
            n = mb * (2 if pair else 1)
            x = torch.rand(n, 1, tc.img_size, tc.img_size, device=device)
            y = (torch.rand(n, 1, tc.img_size, tc.img_size, device=device) > 0.99).float()
            for _ in range(2):
                with torch.autocast("cuda", dtype=torch.float16):
                    out = model(to_model_input(x))
                loss, _ = compute_loss(out, y, y, torch.ones(n, device=device), LossConfig(**tc.loss), cons_pair=pair)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            peak = torch.cuda.max_memory_allocated()
            del model, opt, x, y, out, loss
            torch.cuda.empty_cache()
            log(f"  [probe] micro-batch {mb}{' (x2 pair)' if pair else ''}: peak {peak / 1e9:.1f} GB of {total / 1e9:.1f} GB")
            if peak < mem_fraction * total:
                return mb
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            log(f"  [probe] micro-batch {mb}: out of memory")
    return 1


# ----------------------------------------------------------------------------------------------
def train_run(run: RunConfig, tc: TrainConfig, cache_dir: Path, train_idx, val_idx, train_labels, run_dir: Path,
              micro_bs: int, deadline: Optional[float] = None, max_train_samples: Optional[int] = None,
              log: Callable = print) -> Dict[str, object]:
    """Train (or resume) one run. Returns a status dict. Never touches the test split."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = tc.amp and device.type == "cuda"
    run_dir = Path(run_dir); run_dir.mkdir(parents=True, exist_ok=True)
    done_p, last_p, best_p, hist_p = run_dir / "done.json", run_dir / "last.pt", run_dir / "best.pt", run_dir / "history.csv"
    if done_p.exists():
        st = json.loads(done_p.read_text())
        log(f"[{run.name}] already finished ({st['status']}, {st['epochs_run']} epochs) - skipping")
        return st
    seed_everything(tc.seed)
    loss_cfg = LossConfig(**tc.loss)
    pair = run.consistency
    mb = max(1, micro_bs // 2) if pair else micro_bs
    accum = max(1, tc.effective_bs // mb)

    train_ds = PneumoDataset(cache_dir, train_idx, train=True, aug=AugConfig(**tc.aug), art_mix=run.art_mix, pair=pair, seed=tc.seed)
    val_ds = PneumoDataset(cache_dir, val_idx, train=False)
    sampler = EpochSampler(train_labels, tc.neg_frac, tc.seed, max_train_samples)
    train_loader = make_loader(train_ds, mb, sampler=sampler, workers=tc.workers, drop_last=True)
    val_loader = make_loader(val_ds, max(2, mb * 2), workers=tc.workers)
    steps_per_epoch = max(1, len(train_loader) // accum)
    total_steps = steps_per_epoch * run.epochs

    model = build_model(run.kind, run.model_cfg).to(device)
    groups = [dict(params=list(model.encoder_parameters()), lr=tc.lr_enc),
              dict(params=list(model.non_encoder_parameters()), lr=tc.lr_new)]
    opt = torch.optim.AdamW(groups, weight_decay=tc.weight_decay)
    sched = make_scheduler(opt, total_steps, int(tc.warmup_epochs * steps_per_epoch), tc.min_lr_ratio)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    start_epoch, best, bad, history, secs = 0, -1.0, 0, [], []
    if last_p.exists():
        ck = load_ckpt(last_p, device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"]); sched.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch, best, bad, history, secs = ck["epoch"] + 1, ck["best"], ck["bad"], ck["history"], ck.get("secs", [])
        log(f"[{run.name}] RESUMING from epoch {start_epoch}/{run.epochs} (best val Dice_all so far {best:.4f})")
        del ck
    else:
        log(f"[{run.name}] new run: {count_parameters(model)['total'] / 1e6:.2f} M params, micro-batch {mb} x accum {accum} "
            f"= effective {mb * accum}{' images (x2 clean+corrupted pairs)' if pair else ''}, {len(train_loader)} micro-steps/epoch, "
            f"{run.epochs} epochs")
    (run_dir / "run_config.json").write_text(json.dumps(dict(run=asdict(run), train=asdict(tc), micro_bs=mb, accum=accum), indent=1, default=str))

    status, nonfinite = "done", 0
    for epoch in range(start_epoch, run.epochs):
        if deadline is not None and secs:
            est = float(np.mean(secs[-3:])) * 1.15
            if time.time() + est > deadline:
                status = "paused"
                log(f"[{run.name}] time budget reached - pausing before epoch {epoch + 1}; re-run the notebook to resume.")
                break
        t0 = time.time()
        sampler.set_epoch(epoch); train_ds.set_epoch(epoch)
        model.train()
        acc = {}
        cnt = 0
        n_micro = (len(train_loader) // accum) * accum
        opt.zero_grad(set_to_none=True)
        for it, batch in enumerate(train_loader):
            if it >= n_micro:
                break
            x, y = batch["image"].to(device, non_blocking=True), batch["mask"].to(device, non_blocking=True)
            art_t, has = batch["art"].to(device, non_blocking=True), batch["has_art"].to(device, non_blocking=True)
            if pair:
                xc = batch["image_clean"].to(device, non_blocking=True)
                x = torch.cat([xc, x]); y = torch.cat([y, y])
                art_t = torch.cat([torch.zeros_like(art_t), art_t]); has = torch.cat([torch.zeros_like(has), has])
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                out = model(to_model_input(x))
            loss, parts = compute_loss(out, y, art_t, has, loss_cfg, cons_pair=pair)
            if not torch.isfinite(loss):
                nonfinite += 1
                opt.zero_grad(set_to_none=True)
                if nonfinite > 20:
                    raise RuntimeError(f"[{run.name}] too many non-finite losses (epoch {epoch + 1}, step {it})")
                continue
            scaler.scale(loss / accum).backward()
            for k_, v in parts.items():
                acc[k_] = acc.get(k_, 0.0) + v.detach()
            cnt += 1
            if (it + 1) % accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
                scaler.step(opt); scaler.update(); sched.step()
                opt.zero_grad(set_to_none=True)
                step = (it + 1) // accum
                if step % tc.log_every == 0:
                    msg = " ".join(f"{k_}={float(v) / max(1, cnt):.3f}" for k_, v in acc.items())
                    log(f"  [{run.name}] ep {epoch + 1}/{run.epochs} step {step}/{steps_per_epoch} {msg} "
                        f"lr={sched.get_last_lr()[1]:.2e} {time.time() - t0:.0f}s")
        train_secs = time.time() - t0
        val = validate(model, val_loader, device, tc, loss_cfg, amp)
        ep_secs = time.time() - t0
        secs.append(ep_secs)
        row = dict(epoch=epoch + 1, secs=round(ep_secs, 1), lr=sched.get_last_lr()[1],
                   **{f"train_{k_}": float(v) / max(1, cnt) for k_, v in acc.items()}, **val)
        history.append(row)
        pd.DataFrame(history).to_csv(hist_p, index=False)
        improved = val["val_dice_all"] > best
        log(f"[{run.name}] epoch {epoch + 1}/{run.epochs} | train_loss {row['train_total']:.4f} | val_loss {val['val_loss']:.4f} | "
            f"val Dice_all {val['val_dice_all']:.4f} (Dice_pos {val['val_dice_pos']:.4f} @thr {val['best_thr']}, min_img {val['best_min_img']}) | "
            f"img AUC {val['val_img_auc']:.3f} | {ep_secs:.0f}s (train {train_secs:.0f}s)" + ("  * best" if improved else ""))
        if improved:
            best, bad = val["val_dice_all"], 0
            atomic_save(dict(model=model.state_dict(), epoch=epoch, val=val, run=asdict(run), model_cfg=run.model_cfg,
                             kind=run.kind), best_p)
        else:
            bad += 1
        atomic_save(dict(model=model.state_dict(), optimizer=opt.state_dict(), scheduler=sched.state_dict(),
                         scaler=scaler.state_dict(), epoch=epoch, best=best, bad=bad, history=history, secs=secs,
                         run=asdict(run), train=asdict(tc)), last_p)
        if epoch + 1 >= tc.min_epochs and bad >= tc.patience:
            status = "early_stopped"
            log(f"[{run.name}] early stopping: no improvement for {tc.patience} epochs")
            break
    epochs_run = len(history)
    if status != "paused":
        final = dict(status=status, epochs_run=epochs_run, best_val_dice_all=best,
                     best_epoch=int(pd.DataFrame(history).val_dice_all.idxmax() + 1) if history else None,
                     mean_epoch_secs=float(np.mean(secs)) if secs else None, finished=time.strftime("%Y-%m-%d %H:%M:%S"))
        done_p.write_text(json.dumps(final, indent=1))
        return final
    return dict(status="paused", epochs_run=epochs_run, best_val_dice_all=best, mean_epoch_secs=float(np.mean(secs)) if secs else None)
