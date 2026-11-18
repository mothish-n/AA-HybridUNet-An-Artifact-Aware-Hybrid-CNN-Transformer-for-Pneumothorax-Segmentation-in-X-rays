from .aa_hybrid_unet import (  # noqa: F401
    AAHybridUNet, AAHybridUNetConfig, ArtifactGatedSkip, ViTBottleneck, count_parameters,
)
from .baseline_unet import BaselineConfig, BaselineUNet  # noqa: F401


def build_model(kind: str, cfg: dict):
    """kind: 'baseline' | 'aa'. cfg: dict of config fields (extra keys ignored)."""
    if kind == "baseline":
        return BaselineUNet(BaselineConfig.from_dict(cfg))
    if kind == "aa":
        return AAHybridUNet(AAHybridUNetConfig.from_dict(cfg))
    raise ValueError(f"unknown model kind {kind!r}")
