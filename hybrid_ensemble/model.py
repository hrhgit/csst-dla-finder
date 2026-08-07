from __future__ import annotations

import torch
from torch import nn


class ResidualBlock1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 7, dilation: int = 1):
        super().__init__()
        padding = dilation * (kernel_size // 2)
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels, 1),
            nn.BatchNorm1d(channels),
        )
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class HybridDlaNet(nn.Module):
    """Shared 1D CNN with heatmap, LOGNHI, broad-region, and count heads."""

    def __init__(self, in_channels: int, hidden: int = 32, num_blocks: int = 6, with_offset: bool = True):
        super().__init__()
        self.with_offset = with_offset
        dilations = [1, 2, 4, 8, 16, 32][:num_blocks]
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=9, padding=4),
            nn.BatchNorm1d(hidden),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList([ResidualBlock1d(hidden, dilation=d) for d in dilations])
        self.pixel_head = nn.Sequential(
            nn.Conv1d(hidden, hidden, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden),
            nn.SiLU(),
            nn.Conv1d(hidden, 4 if with_offset else 3, kernel_size=1),
        )
        self.count_head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 3),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.stem(x)
        for block in self.blocks:
            feat = block(feat)
        pixel = self.pixel_head(feat)
        pooled = torch.cat([feat.mean(dim=-1), feat.amax(dim=-1)], dim=1)
        out = {
            "center_logits": pixel[:, 0],
            "region_logits": pixel[:, 1],
            "lognhi_raw": pixel[:, 2],
            "count_logits": self.count_head(pooled),
        }
        if self.with_offset:
            out["offset_raw"] = pixel[:, 3]
        return out


def input_channels(input_mode: str) -> int:
    if input_mode == "flux":
        return 6
    if input_mode == "residual":
        return 7
    if input_mode == "all":
        return 8
    raise ValueError(f"unknown input_mode: {input_mode}")
