"""Baseline: standard U-Net with an ImageNet-pretrained EfficientNet-B4 encoder (binary output).

Standard decoder only: bilinear upsample -> concat skip -> 2x (conv3x3-BN-ReLU). No transformer, no detail stem,
no artifact head, no gates, no deep supervision. The last block has no skip (upsample to full resolution), exactly
like the usual segmentation_models_pytorch U-Net layout.

  f5[448@/32] -> D0(+f4[160@/16]) -> D1(+f3[56@/8]) -> D2(+f2[32@/4]) -> D3(+f1[24@/2]) -> D4(no skip, @/1) -> 1x1 conv
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Sequence

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from .aa_hybrid_unet import conv_bn_relu


@dataclass
class BaselineConfig:
    encoder_name: str = "efficientnet_b4"
    pretrained: bool = True
    drop_path_rate: float = 0.1
    in_chans: int = 3
    decoder_channels: Sequence[int] = (256, 128, 64, 32, 16)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["decoder_channels"] = list(self.decoder_channels)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "BaselineConfig":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        if "decoder_channels" in known:
            known["decoder_channels"] = tuple(known["decoder_channels"])
        return cls(**known)


class _Block(nn.Module):
    def __init__(self, cin: int, cskip: int, cout: int):
        super().__init__()
        self.c1 = conv_bn_relu(cin + cskip, cout)
        self.c2 = conv_bn_relu(cout, cout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor = None) -> torch.Tensor:
        if skip is None:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        else:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], 1)
        return self.c2(self.c1(x))


class BaselineUNet(nn.Module):
    def __init__(self, cfg: BaselineConfig = None):
        super().__init__()
        cfg = cfg or BaselineConfig()
        self.cfg = cfg
        self.encoder = timm.create_model(
            cfg.encoder_name, pretrained=cfg.pretrained, features_only=True, out_indices=(0, 1, 2, 3, 4),
            in_chans=cfg.in_chans, drop_path_rate=cfg.drop_path_rate,
        )
        ch = list(self.encoder.feature_info.channels())
        if list(self.encoder.feature_info.reduction()) != [2, 4, 8, 16, 32]:
            raise RuntimeError("Unexpected encoder strides")
        dc = list(cfg.decoder_channels)
        ins = [ch[4]] + dc[:-1]
        skips = [ch[3], ch[2], ch[1], ch[0], 0]
        self.blocks = nn.ModuleList([_Block(i, s, o) for i, s, o in zip(ins, skips, dc)])
        self.head = nn.Conv2d(dc[-1], 1, 1)
        for root in (self.blocks, self.head):
            for m in root.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Dict[str, object]:
        if x.shape[-1] % 32 or x.shape[-2] % 32:
            raise ValueError("Input height/width must be divisible by 32")
        f1, f2, f3, f4, f5 = self.encoder(x)
        d = f5
        for blk, sk in zip(self.blocks, [f4, f3, f2, f1, None]):
            d = blk(d, sk)
        return {"seg": self.head(d)}

    def encoder_parameters(self):
        return self.encoder.parameters()

    def non_encoder_parameters(self):
        enc = {id(p) for p in self.encoder.parameters()}
        return (p for p in self.parameters() if id(p) not in enc)
