"""AA-HybridUNet: Artifact-Aware Hybrid U-Net for pneumothorax segmentation.

Data flow for a 512x512 input (channels in brackets, spatial size after '@'):

    image [3@512]
      |-- DetailStem ----------------------------------------> stem [32@512]
      |-- EfficientNet-B4 encoder -> f1[24@256] f2[32@128] f3[56@64] f4[160@32] f5[448@16]
      |                                                          |
      |        ViTBottleneck(f5) -> b[448@16]  (residual, starts as identity)
      |
      |-- ArtifactHead(f1..f4, stem) -> artifact logits [1@512]  (only if enabled)
      |                                   |
      |                                   v  (probability, optionally detached)
      '-- Decoder: 5 blocks, each = upsample -> ArtifactGatedSkip(skip, dec, artifact) -> concat -> 2x conv
             D4: b + f4   -> [128@32]  --aux head
             D3: D4 + f3  -> [ 96@64]  --aux head
             D2: D3 + f2  -> [ 64@128] --aux head
             D1: D2 + f1  -> [ 48@256] --aux head
             D0: D1 + stem-> [ 32@512] --main seg head -> pneumothorax logits [1@512]

Outputs are raw LOGITS (no sigmoid) so BCE-with-logits and AMP are safe.

Gated skip (the "artifact-conditioned" part):

    g      = min_gate + (1 - min_gate) * sigmoid(Conv(skip, decoder, artifact))      # per pixel and channel
    skip'  = skip * ((1 - a) + a * g)        # a = artifact presence in [0, 1]

Where no artifact is predicted (a ~ 0) the skip passes through untouched, exactly like a plain U-Net.
Where an artifact is predicted (a ~ 1) the network *learns* how much of each channel to keep, based on the
skip features and decoder context, and it can never go below ``min_gate`` -- so regions that contain a
device are not blindly suppressed (the anatomy underneath a wire is still visible to the decoder).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------------------------
#Configuration
# ----------------------------------------------------------------------------------------------
@dataclass
class AAHybridUNetConfig:
    # encoder
    encoder_name: str = "efficientnet_b4"
    pretrained: bool = True          # needs internet (Hugging Face hub) the first time
    drop_path_rate: float = 0.1
    in_chans: int = 3
    img_size: int = 512              # only used to size the ViT positional embedding; other sizes are interpolated
    grad_checkpointing: bool = False  # trade compute for GPU memory in the encoder
    # ViT bottleneck
    use_vit: bool = True
    vit_dim: int = 256
    vit_depth: int = 2
    vit_heads: int = 8
    vit_mlp_ratio: float = 4.0
    vit_dropout: float = 0.1
    # detail stem + decoder
    stem_channels: int = 32
    decoder_channels: Sequence[int] = (128, 96, 64, 48, 32)
    deep_supervision: bool = True
    # artifact branch
    use_artifact_head: bool = True
    artifact_channels: int = 32
    use_gated_skips: bool = True
    detach_artifact_for_gates: bool = True   # gates read the artifact map but do not push segmentation gradients into it
    min_gate: float = 0.2

    def validate(self) -> None:
        if self.use_gated_skips and not self.use_artifact_head:
            raise ValueError("use_gated_skips=True requires use_artifact_head=True (the gates are conditioned on it).")
        if len(self.decoder_channels) != 5:
            raise ValueError("decoder_channels must have exactly 5 entries (one per decoder block).")
        if not 0.0 <= self.min_gate < 1.0:
            raise ValueError("min_gate must be in [0, 1).")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["decoder_channels"] = list(self.decoder_channels)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "AAHybridUNetConfig":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        if "decoder_channels" in known:
            known["decoder_channels"] = tuple(known["decoder_channels"])
        return cls(**known)


# ----------------------------------------------------------------------------------------------
# Building blocks
# ----------------------------------------------------------------------------------------------
def conv_bn_relu(cin: int, cout: int, k: int = 3) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, padding=k // 2, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class DetailStem(nn.Module):
    """Two 3x3 convs at FULL resolution -> fine edge/texture features for small pneumothorax boundaries."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.body = nn.Sequential(conv_bn_relu(in_ch, out_ch // 2), conv_bn_relu(out_ch // 2, out_ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class ViTBottleneck(nn.Module):
    """Small transformer on the deepest feature map for global context.

    1x1 conv -> + positional embedding -> N pre-norm transformer layers -> 1x1 conv -> residual add.
    The output projection is zero-initialised, so at the start of training the block is an identity and
    the pretrained CNN features are not disturbed; it learns to add global context gradually.
    """

    def __init__(self, in_ch: int, dim: int, depth: int, heads: int, mlp_ratio: float, dropout: float,
                 grid: tuple):
        super().__init__()
        self.proj_in = nn.Conv2d(in_ch, dim, 1)
        self.pos = nn.Parameter(torch.zeros(1, dim, grid[0], grid[1]))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=int(dim * mlp_ratio), dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.proj_out = nn.Conv2d(dim, in_ch, 1)

    def init_identity(self) -> None:
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        t = self.proj_in(x)
        pos = self.pos
        if pos.shape[-2:] != (h, w):  # inference at another resolution
            pos = F.interpolate(pos.float(), size=(h, w), mode="bicubic", align_corners=False)
        t = t + pos.to(t.dtype)
        t = t.flatten(2).transpose(1, 2)            # B, N, C
        t = self.norm(self.encoder(t))
        t = t.transpose(1, 2).reshape(b, -1, h, w)  # B, C, H, W
        return x + self.proj_out(t)


class ArtifactHead(nn.Module):
    """Predicts a full-resolution map of synthetic wires/tubes/lines.

    Uses the four shallowest encoder levels (thin lines need high resolution, long lines need context)
    plus the full-resolution detail stem.
    """

    def __init__(self, enc_channels: Sequence[int], stem_ch: int, mid: int):
        super().__init__()
        self.proj = nn.ModuleList([conv_bn_relu(c, mid, 1) for c in enc_channels])  # f1..f4
        self.fuse = conv_bn_relu(mid * len(enc_channels), mid, 3)
        self.refine = nn.Sequential(conv_bn_relu(mid + stem_ch, mid, 3), nn.Conv2d(mid, 1, 1))

    def forward(self, feats: Sequence[torch.Tensor], stem: torch.Tensor) -> torch.Tensor:
        size0 = feats[0].shape[-2:]
        xs = [self.proj[0](feats[0])]
        for p, f in zip(self.proj[1:], feats[1:]):
            xs.append(F.interpolate(p(f), size=size0, mode="bilinear", align_corners=False))
        x = self.fuse(torch.cat(xs, 1))
        x = F.interpolate(x, size=stem.shape[-2:], mode="bilinear", align_corners=False)
        return self.refine(torch.cat([x, stem], 1))  # logits [B,1,H,W]


class ArtifactGatedSkip(nn.Module):
    """skip' = skip * ((1 - a) + a * g), g in [min_gate, 1]. See module docstring."""

    def __init__(self, skip_ch: int, dec_ch: int, min_gate: float, init_bias: float = 4.0):
        super().__init__()
        mid = max(8, skip_ch // 2)
        self.min_gate = min_gate
        self.init_bias = init_bias
        self.skip_proj = nn.Conv2d(skip_ch, mid, 1, bias=False)
        self.dec_proj = nn.Conv2d(dec_ch, mid, 1, bias=False)
        self.art_proj = nn.Conv2d(2, mid, 1)  # [max-pooled, avg-pooled] artifact presence
        self.norm = nn.BatchNorm2d(mid)
        self.out = nn.Conv2d(mid, skip_ch, 1)

    def init_identity(self) -> None:
        """Start close to a plain skip connection (gate ~ 1)."""
        nn.init.normal_(self.out.weight, std=1e-3)
        nn.init.constant_(self.out.bias, self.init_bias)

    def forward(self, skip: torch.Tensor, dec_up: torch.Tensor, art_prob: torch.Tensor) -> torch.Tensor:
        size = skip.shape[-2:]
        # Max-pool keeps thin wires visible at coarse levels; avg-pool tells how much of the cell is covered.
        a_max = F.adaptive_max_pool2d(art_prob, size) if art_prob.shape[-2:] != size else art_prob
        a_avg = F.adaptive_avg_pool2d(art_prob, size) if art_prob.shape[-2:] != size else art_prob
        z = self.skip_proj(skip) + self.dec_proj(dec_up) + self.art_proj(torch.cat([a_max, a_avg], 1).to(skip.dtype))
        g = torch.sigmoid(self.out(F.relu(self.norm(z))))
        g = self.min_gate + (1.0 - self.min_gate) * g
        a = a_max.to(skip.dtype)
        return skip * ((1.0 - a) + a * g)


class DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, gated: bool, min_gate: float):
        super().__init__()
        self.gate = ArtifactGatedSkip(skip_ch, in_ch, min_gate) if gated else None
        self.conv1 = conv_bn_relu(in_ch + skip_ch, out_ch)
        self.conv2 = conv_bn_relu(out_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, art_prob: Optional[torch.Tensor]) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        if self.gate is not None:
            skip = self.gate(skip, x, art_prob)
        return self.conv2(self.conv1(torch.cat([x, skip], 1)))


# ----------------------------------------------------------------------------------------------
# Full model
# ----------------------------------------------------------------------------------------------
class AAHybridUNet(nn.Module):
    """Returns a dict:
        'seg'      : [B,1,H,W] pneumothorax logits
        'aux'      : list of 4 logit maps at native decoder resolution (H/16, H/8, H/4, H/2) -- training only
        'artifact' : [B,1,H,W] artifact logits (only if use_artifact_head)
    """

    def __init__(self, cfg: Optional[AAHybridUNetConfig] = None):
        super().__init__()
        cfg = cfg or AAHybridUNetConfig()
        cfg.validate()
        self.cfg = cfg

        self.encoder = timm.create_model(
            cfg.encoder_name, pretrained=cfg.pretrained, features_only=True, out_indices=(0, 1, 2, 3, 4),
            in_chans=cfg.in_chans, drop_path_rate=cfg.drop_path_rate,
        )
        if cfg.grad_checkpointing:
            self.encoder.set_grad_checkpointing(True)
        ch = list(self.encoder.feature_info.channels())
        red = list(self.encoder.feature_info.reduction())
        if red != [2, 4, 8, 16, 32]:
            raise RuntimeError(f"Unexpected encoder strides {red}; the decoder assumes [2,4,8,16,32].")
        self.enc_channels = ch

        self.stem = DetailStem(cfg.in_chans, cfg.stem_channels)
        grid = (max(1, cfg.img_size // 32),) * 2
        self.vit = (ViTBottleneck(ch[4], cfg.vit_dim, cfg.vit_depth, cfg.vit_heads, cfg.vit_mlp_ratio,
                                  cfg.vit_dropout, grid) if cfg.use_vit else None)
        self.artifact_head = (ArtifactHead(ch[:4], cfg.stem_channels, cfg.artifact_channels)
                              if cfg.use_artifact_head else None)

        dc = list(cfg.decoder_channels)
        in_chs = [ch[4]] + dc[:-1]
        skip_chs = [ch[3], ch[2], ch[1], ch[0], cfg.stem_channels]
        self.decoder = nn.ModuleList([
            DecoderBlock(i, s, o, cfg.use_gated_skips, cfg.min_gate) for i, s, o in zip(in_chs, skip_chs, dc)
        ])
        self.seg_head = nn.Conv2d(dc[-1], 1, 1)
        self.aux_heads = nn.ModuleList([nn.Conv2d(c, 1, 1) for c in dc[:4]]) if cfg.deep_supervision else None

        self._init_new_modules()

    # -- init --------------------------------------------------------------------------------
    def _init_new_modules(self) -> None:
        new = [self.stem, self.decoder, self.seg_head]
        if self.artifact_head is not None:
            new.append(self.artifact_head)
        if self.aux_heads is not None:
            new.append(self.aux_heads)
        if self.vit is not None:
            new += [self.vit.proj_in, self.vit.proj_out]
        for root in new:
            for m in root.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        for m in self.modules():
            if isinstance(m, (ArtifactGatedSkip, ViTBottleneck)):
                m.init_identity()

    # -- helpers -----------------------------------------------------------------------------
    def encoder_parameters(self):
        return self.encoder.parameters()

    def non_encoder_parameters(self):
        enc = {id(p) for p in self.encoder.parameters()}
        return (p for p in self.parameters() if id(p) not in enc)

    # -- forward -----------------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> Dict[str, object]:
        h, w = x.shape[-2:]
        if h % 32 or w % 32:
            raise ValueError(f"Input height/width must be divisible by 32, got {h}x{w}.")

        stem = self.stem(x)
        f1, f2, f3, f4, f5 = self.encoder(x)
        b = self.vit(f5) if self.vit is not None else f5

        out: Dict[str, object] = {}
        art_prob = None
        if self.artifact_head is not None:
            art_logits = self.artifact_head([f1, f2, f3, f4], stem)
            out["artifact"] = art_logits
            if self.cfg.use_gated_skips:
                art_prob = torch.sigmoid(art_logits.float())
                if self.cfg.detach_artifact_for_gates:
                    art_prob = art_prob.detach()

        skips = [f4, f3, f2, f1, stem]
        feats: List[torch.Tensor] = []
        d = b
        for blk, sk in zip(self.decoder, skips):
            d = blk(d, sk, art_prob)
            feats.append(d)

        out["seg"] = self.seg_head(d)
        if self.aux_heads is not None and self.training:
            out["aux"] = [head(f) for head, f in zip(self.aux_heads, feats[:4])]
        return out


def count_parameters(model: nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}
