"""Computational-efficiency measurements: parameters, GMACs, latency, peak GPU memory (inference, batch 1)."""
from __future__ import annotations

import time
from typing import Dict

import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

from .data.dataset import to_model_input


def measure_efficiency(model, img_size: int = 512, device=None, n_iter: int = 60, warmup: int = 10) -> Dict[str, float]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    out: Dict[str, float] = {"params_M": sum(p.numel() for p in model.parameters()) / 1e6, "img_size": img_size}
    x = torch.rand(1, 1, img_size, img_size, device=device)
    xin = to_model_input(x)
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        model(xin)
    out["gmacs"] = fc.get_total_flops() / 2 / 1e9          # 1 MAC = 2 FLOPs (matmul/conv/attention counted)

    def timed(amp: bool):
        ts = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            for i in range(warmup + n_iter):
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t = time.perf_counter()
                model(xin)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                if i >= warmup:
                    ts.append((time.perf_counter() - t) * 1e3)
        return float(np.mean(ts)), float(np.std(ts))

    if device.type == "cuda":
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        out["latency_fp32_ms"], out["latency_fp32_std"] = timed(False)
        out["peak_mem_fp32_gb"] = torch.cuda.max_memory_allocated() / 1e9
        torch.cuda.reset_peak_memory_stats()
        out["latency_fp16_ms"], out["latency_fp16_std"] = timed(True)
        out["peak_mem_fp16_gb"] = torch.cuda.max_memory_allocated() / 1e9
        out["throughput_fp16_img_s"] = 1000.0 / out["latency_fp16_ms"]
        out["gpu"] = torch.cuda.get_device_name(0)
    else:
        out["latency_cpu_ms"], out["latency_cpu_std"] = timed(False)
        out["gpu"] = "cpu"
    return out
