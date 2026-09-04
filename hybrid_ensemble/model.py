"""Compatibility adapter for the local dilated five-head DLA architecture.

The surrounding hybrid pipeline intentionally keeps its established data,
training, decoding, evaluation, and checkpoint conventions.  This module only
maps the local model's output semantics to that pipeline's model contract.
"""
from __future__ import annotations

from models.dilated_resnet_5head import _build_dilated_resnet_5head


def input_channels(input_mode: str) -> int:
    if input_mode == "raw":
        return 1
    if input_mode == "flux":
        return 6
    if input_mode == "residual":
        return 7
    if input_mode == "all":
        return 8
    if input_mode == "all_wzx":
        return 11
    raise ValueError(f"unknown input_mode: {input_mode}")


def _stages_for(num_blocks: int) -> list[tuple[int, int]]:
    """Use the local architecture's layouts for the 681-pixel data grid."""
    layouts = {
        2: [(1, 1), (2, 1)],
        3: [(1, 1), (2, 1), (4, 2)],
        4: [(1, 1), (2, 1), (4, 2), (4, 2)],
        5: [(1, 1), (2, 1), (4, 2), (4, 2), (4, 2)],
    }
    if num_blocks not in layouts:
        raise ValueError("num_blocks must be one of 2, 3, 4, or 5 for dilated-resnet-5head")
    return layouts[num_blocks]


class HybridDlaNet:
    """Expose the local DilatedResNet5Head through the hybrid pipeline contract."""

    def __init__(
        self,
        in_channels: int,
        hidden: int = 96,
        num_blocks: int = 4,
        with_offset: bool = True,
        norm_type: str = "layer",
        head_layers: int = 1,
    ):
        self.with_offset = with_offset
        self.model = _build_dilated_resnet_5head(
            n_bins=0,
            in_channels=in_channels,
            width=hidden,
            stages=_stages_for(num_blocks),
            use_skip=False,
            use_se=False,
            norm_type=norm_type,
            head_layers=head_layers,
        )

    def to(self, *args, **kwargs):
        self.model.to(*args, **kwargs)
        return self

    def __getattr__(self, name):
        if name in {"model", "with_offset"}:
            return object.__getattribute__(self, name)
        return getattr(self.model, name)

    def __call__(self, x):
        local = self.model(x)
        output = {
            "center_logits": local["heat_logits"],
            "region_logits": local["region_logits"],
            # Local: 20.5 + 1.5 * raw. Pipeline: 20.3 + raw.
            "lognhi_raw": 0.2 + 1.5 * local["lognhi"],
            "count_logits": local["count_prob"],
        }
        if self.with_offset:
            output["offset_raw"] = local["offset"]
        return output
