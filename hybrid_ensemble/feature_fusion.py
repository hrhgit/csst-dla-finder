"""Trainable, feature-level fusion of the dilated and WZX DLA backbones."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from csst_dla.fits_utils import read_image
from data import HybridTestDataset, HybridTrainDataset, build_wzx_style_channels, build_wzx_style_channels_rows


def build_wzx_views(flux: np.ndarray, clean: np.ndarray) -> np.ndarray:
    return build_wzx_style_channels(flux, clean)


class DualFusionTrainDataset(HybridTrainDataset):
    """Hybrid targets with both the eight-channel and WZX three-channel views."""

    def __init__(
        self,
        targets_npz: str | Path,
        train_fits: str | Path,
        split: str,
        max_samples: int | None = None,
    ):
        super().__init__(
            targets_npz,
            train_fits,
            split,
            input_mode="all",
            max_samples=max_samples,
            cache_channels=True,
        )
        raw_flux = read_image(train_fits, "FLUX").astype(np.float32)
        raw_clean = read_image(train_fits, "FLUX_CLEAN").astype(np.float32)
        self.wzx_channels = build_wzx_style_channels_rows(raw_flux[self.indices], raw_clean[self.indices])

    def __getitem__(self, row: int):
        hybrid, center, region, lognhi, mask, offset, offset_weight, count, index = super().__getitem__(row)
        return (
            hybrid,
            torch.from_numpy(self.wzx_channels[row].copy()),
            torch.tensor(float(self.zq[row]), dtype=torch.float32),
            center,
            region,
            lognhi,
            mask,
            offset,
            offset_weight,
            count,
            index,
        )


class DualFusionTestDataset(HybridTestDataset):
    def __init__(self, test_fits: str | Path):
        super().__init__(test_fits, input_mode="all")
        raw_flux = read_image(test_fits, "FLUX").astype(np.float32)
        raw_clean = read_image(test_fits, "FLUX_CLEAN").astype(np.float32)
        self.wzx_channels = build_wzx_style_channels_rows(raw_flux, raw_clean)

    def __getitem__(self, row: int):
        hybrid, index = super().__getitem__(row)
        return hybrid, torch.from_numpy(self.wzx_channels[row].copy()), torch.tensor(float(self.zq[row]), dtype=torch.float32), index


class FusionResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=5, padding=2),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, channels),
        )
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class DualTowerFusionNet(nn.Module):
    """Join the frozen feature maps from both trained backbones.

    ``plain`` learns all heads from concatenated features.  The two residual
    modes instead begin from one source model's native heads and learn only a
    correction informed by the other backbone.
    """

    def __init__(
        self,
        dilated_backbone: nn.Module,
        wzx_backbone: nn.Module,
        merge_mode: Literal["plain", "residual_dilated", "residual_wzx"] = "plain",
        width: int = 128,
        depth: int = 3,
        freeze_backbones: bool = True,
    ):
        super().__init__()
        if merge_mode not in {"plain", "residual_dilated", "residual_wzx"}:
            raise ValueError(f"unknown merge_mode={merge_mode!r}")
        self.dilated_backbone = dilated_backbone
        self.wzx_backbone = wzx_backbone
        self.merge_mode = merge_mode
        self.freeze_backbones = bool(freeze_backbones)

        d_channels = int(self.dilated_backbone.refine[0].out_channels)
        w_channels = int(self.wzx_backbone.stem[0].out_channels)
        self.fuse = nn.Sequential(
            nn.Conv1d(d_channels + w_channels, width, kernel_size=5, padding=2),
            nn.GroupNorm(8, width),
            nn.SiLU(),
            *[FusionResidualBlock(width) for _ in range(depth)],
        )
        self.center_delta = nn.Conv1d(width, 1, kernel_size=3, padding=1)
        self.region_delta = nn.Conv1d(width, 1, kernel_size=3, padding=1)
        self.lognhi_delta = nn.Conv1d(width, 1, kernel_size=1)
        self.offset_delta = nn.Conv1d(width, 1, kernel_size=3, padding=1)
        self.count_delta = nn.Sequential(
            nn.Linear(width * 2 + 1, width),
            nn.SiLU(),
            nn.Linear(width, 3),
        )
        if self.freeze_backbones:
            for module in (self.dilated_backbone, self.wzx_backbone):
                for parameter in module.parameters():
                    parameter.requires_grad_(False)
        if merge_mode != "plain":
            for head in (self.center_delta, self.region_delta, self.lognhi_delta, self.offset_delta):
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)
            nn.init.zeros_(self.count_delta[-1].weight)
            nn.init.zeros_(self.count_delta[-1].bias)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbones:
            self.dilated_backbone.eval()
            self.wzx_backbone.eval()
        return self

    def _dilated_features(self, x: torch.Tensor) -> torch.Tensor:
        backbone = self.dilated_backbone
        original_length = x.shape[-1]
        x = backbone.stem(x)
        skips = []
        for idx, stage in enumerate(backbone.stages):
            x = stage(x)
            if backbone.use_skip and idx < len(backbone.stages) - 1:
                skips.append(backbone.skip_projections[idx](x))
        x = backbone.upsample1(x)
        if backbone.use_skip and skips:
            x = torch.cat([x] + skips, dim=1)
        if x.shape[-1] != original_length:
            x = F.interpolate(x, size=original_length, mode="linear", align_corners=False)
        return backbone.refine(x)

    def _wzx_features(self, spectrum: torch.Tensor, z_qso: torch.Tensor) -> torch.Tensor:
        backbone = self.wzx_backbone
        inputs = backbone._append_context_channels(spectrum, z_qso)
        return backbone.backbone(backbone.stem(inputs))

    def _dilated_base(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        backbone = self.dilated_backbone
        return {
            "center_logits": backbone.heat_head(features).squeeze(1),
            "region_logits": backbone.region_head(features).squeeze(1),
            "lognhi_raw": 0.2 + 1.5 * backbone.lognhi_head(features).squeeze(1),
            "offset_raw": torch.tanh(backbone.offset_head(features).squeeze(1)),
            "count_logits": backbone.count_head(features.mean(dim=-1)),
        }

    def _wzx_base(self, features: torch.Tensor, z_qso: torch.Tensor) -> dict[str, torch.Tensor]:
        backbone = self.wzx_backbone
        if z_qso.ndim == 1:
            z_qso = z_qso.unsqueeze(1)
        average = F.adaptive_avg_pool1d(features, 1).flatten(1)
        maximum = F.adaptive_max_pool1d(features, 1).flatten(1)
        lognhi = 19.0 + 4.0 * torch.sigmoid(backbone.lognhi_head(features).squeeze(1))
        return {
            "center_logits": backbone.heatmap_head(features).squeeze(1),
            "region_logits": backbone.region_head(features).squeeze(1),
            "lognhi_raw": lognhi - 20.3,
            "offset_raw": 0.5 * torch.tanh(backbone.offset_head(features).squeeze(1)),
            "count_logits": backbone.count_head(torch.cat([average, maximum, z_qso], dim=1)),
        }

    def forward(self, hybrid: torch.Tensor, wzx: torch.Tensor, z_qso: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.freeze_backbones:
            with torch.no_grad():
                d_features = self._dilated_features(hybrid)
                w_features = self._wzx_features(wzx, z_qso)
        else:
            d_features = self._dilated_features(hybrid)
            w_features = self._wzx_features(wzx, z_qso)
        shared = self.fuse(torch.cat([d_features, w_features], dim=1))
        center_delta = self.center_delta(shared).squeeze(1)
        region_delta = self.region_delta(shared).squeeze(1)
        lognhi_delta = self.lognhi_delta(shared).squeeze(1)
        offset_delta = self.offset_delta(shared).squeeze(1)
        pooled = torch.cat(
            [F.adaptive_avg_pool1d(shared, 1).flatten(1), F.adaptive_max_pool1d(shared, 1).flatten(1), z_qso.view(-1, 1)],
            dim=1,
        )
        count_delta = self.count_delta(pooled)
        if self.merge_mode == "plain":
            return {
                "center_logits": center_delta,
                "region_logits": region_delta,
                "lognhi_raw": lognhi_delta,
                "offset_raw": torch.tanh(offset_delta),
                "count_logits": count_delta,
            }
        base = self._dilated_base(d_features) if self.merge_mode == "residual_dilated" else self._wzx_base(w_features, z_qso)
        return {
            "center_logits": base["center_logits"] + center_delta,
            "region_logits": base["region_logits"] + region_delta,
            "lognhi_raw": base["lognhi_raw"] + lognhi_delta,
            "offset_raw": torch.clamp(base["offset_raw"] + 0.5 * torch.tanh(offset_delta), -1.0, 1.0),
            "count_logits": base["count_logits"] + count_delta,
        }
