"""Dilated Residual CNN Backbone with 5 Heads for DLA Detection.

按照流程图实现：
  A[输入光谱特征] --> B[1D Conv Stem]
  B --> C[Dilated Residual CNN Backbone]
  C --> D[共享时序特征]

  D --> E[Heatmap Head]      --> 每个 wavelength bin 的 DLA 中心概率
  D --> F[Count Head]        --> 整条光谱预测 N_DLA = 0 / 1 / 2
  D --> G[Region Head]       --> 宽 DLA 区域监督
  D --> H[LOGNHI Head]       --> 每个 bin 的 LOGNHI 预测（经校准）
  D --> I[Offset Head]       --> 亚像素中心偏移: peak_pix + offset

Architecture (runI 配置，无下采样 + gridding 修正)：
  - Stem: Conv1d(k=7, s=1) + BN + SiLU（不 下采样，保留亚像素信息）
  - Backbone: 4 stages of Dilated Residual Blocks
    Stage 1: dilation=1, channels=width
    Stage 2: dilation=2, channels=width*2
    Stage 3: dilation=4, channels=width*4
    Stage 4: dilation=4 (gridding 修正，避免 dilation=8 的稀疏采样)
  - Upsample: 1x1 Conv 降通道（stem 不下采样，无需上采样）
  - 5 Heads: heat/region/offset 用 3x3 Conv，lognhi 用 1x1，count = GlobalAvgPool + Linear
  - 激活: SiLU
"""
from __future__ import annotations

from typing import Any


# logNHI 标签归一化：log_abs_target = (logNHI - BIAS) / SCALE
# 数据 logNHI ∈ [19.5, 22.5]，中位 ~20.5
# BIAS=20.5, SCALE=1.5 → target ∈ [-0.67, +1.33]，overflow 仅高端 ~18%（弱 DLA 完全在线性区）
LOGNHI_BIAS = 20.5
LOGNHI_SCALE = 1.5

# lognhi_head 在非 DLA 位置无监督信号，输出 ~0（对应 logNHI=20.5），
# 导致 FP 候选的 lognhi 预测偏高（~20.5），通过 lognhi_fp_threshold=19.5 过滤。
BG_LOGNHI = 18.0  # 非 DLA 位置的背景 logNHI target（低于 FP 阈值 19.5）
BG_LOGNHI_TARGET = (BG_LOGNHI - LOGNHI_BIAS) / LOGNHI_SCALE  # 归一化: -1.667


def import_torch():
    import torch
    return torch


# FPN 跳跃连接中每个 skip 投影的目标通道数（width // 4）
SKIP_CH_DIV = 4


def _build_dilated_resnet_5head(
    n_bins: int,
    in_channels: int = 1,
    width: int = 96,
    n_count_classes: int = 3,
    dilations: tuple[int, int, int, int] = (1, 2, 4, 4),
    stages: list[tuple[int, int]] | None = None,
    use_skip: bool = False,
    use_se: bool = False,
    norm_type: str = "batch",
    head_layers: int = 1,
) -> Any:
    """构建 Dilated ResNet + 5 Heads 模型（runI 配置）。

    参数 width: backbone 通道宽度，默认 96（2.0M params，适配 16K 样本规模）。
    参数 dilations: 4 个 stage 的 dilation 配置（向后兼容，仅当 stages=None 时使用）。
      默认 (1,2,4,4) 是旧数据 (2774 pix, 2 Å/pix, trough~20 pix) 的最优配置。
      新数据 (681 pix, 8 Å/pix, trough~5 pix) 应减小 dilation，例如 (1,1,2,2)，
      避免 stage3/4 在 trough 内采样不足（dilation=4 时 5 pix trough 仅 1.2 个采样点）。
    参数 stages: 显式指定 backbone 的 stage 列表，每个元素为 (out_channels_mult, dilation)。
      - 默认 None → 由 dilations 推导：[(1,d1),(2,d2),(4,d3),(4,d4)]（4 stage，向后兼容）
      - 3 stage 例子（减层，约 1.1M params）：[(1,1),(2,1),(4,2)]
      - 5 stage 例子（加层，约 2.9M params）：[(1,1),(2,1),(4,2),(4,2),(4,2)]
      最后一个 stage 的 out_channels_mult 决定 bottleneck 通道数（c_bneck = width * mult）。
    参数 use_skip: 是否启用 FPN 跳跃连接（默认 False）。当 True 时，每个 stage（除最后
      一个）的输出经 1x1 conv 投影到 width//SKIP_CH_DIV 通道后，与 upsample1 的输出拼接，
      再送入 refine 模块。保留早期 stage 的高分辨率空间信息，弥补深层 dilation 导致的
      细节丢失。参数量增加约 ~5%。
    参数 use_se: 是否启用 SE 通道注意力（默认 False）。在每个 DilatedResBlock 的残差
      输出后加 SE 模块（全局平均池化 → FC → SiLU → FC → Sigmoid → 通道重标定），
      帮助模型自适应地重新校准通道响应。reduction=16，参数量增加约 1–2%。
    参数 norm_type: 归一化层类型，"batch"（默认，向后兼容）或 "layer"。
      - "batch": BatchNorm1d，在 (B, L) 维度归一化每个通道，跨样本统计。
        train/eval 模式不一致（训练用 batch stats，推理用 running stats），
        当 running stats 未收敛到真实分布时会导致推理特征尺度漂移。
      - "layer": GroupNorm(1, C)（等价于对通道维度 C 做 LayerNorm），
        每条光谱独立归一化，train/eval 完全一致，无 running stats。
        语义正确：光谱是 per-sample 数据，不应跨样本归一化。
    参数 head_layers: 每个输出 head 的层数（默认 1，向后兼容）。
      - 1: 单层（原始配置：conv 3x3/1x1 或 Linear）
      - 2: 一层隐藏层 (w→w) + 输出层 (w→1/3)，隐藏层含 Norm + SiLU
      - 3: 两层隐藏层 (w→w→w) + 输出层 (w→1/3)
      增加头深度可让 head 在共享特征上做非线性变换，可能改善 head 间特征解耦。

    forward(features) 返回 dict:
      heat_prob:    [B, n_bins]   sigmoid, 中心 bin 概率
      region_prob:  [B, n_bins]   sigmoid, DLA 区域覆盖概率
      lognhi:       [B, n_bins]   raw, logNHI = BIAS + SCALE * lognhi
      offset:       [B, n_bins]   tanh, 亚像素偏移 [-1, 1]
      count_prob:   [B, 3]        logits, DLA 数目（0/1/2）
    """
    torch = import_torch()
    nn = torch.nn
    F = torch.nn.functional

    act_fn = F.silu

    def make_norm(num_channels: int):
        """归一化层工厂：根据 norm_type 返回 BN 或 LN。

        - "batch": BatchNorm1d，跨样本统计（train/eval 不一致）
        - "layer": GroupNorm(1, C)，每样本独立归一化 C 维度（train/eval 一致）
          GroupNorm(num_groups=1) 等价于 LayerNorm on channel dim，
          对 [B, C, L] 在 C 维度归一化，保持 L 空间独立性。
        """
        if norm_type == "layer":
            return nn.GroupNorm(num_groups=1, num_channels=num_channels)
        return nn.BatchNorm1d(num_channels)

    # 由 dilations 推导默认 stages（向后兼容 4 stage 配置）
    if stages is None:
        d1, d2, d3, d4 = dilations
        stages = [(1, d1), (2, d2), (4, d3), (4, d4)]

    # --- Head 构建器：支持可配置层数 ---
    def _make_conv_head(in_ch: int, out_ch: int, kernel_size: int, n_layers: int):
        """构建 Conv1d head，n_layers=1 为单层（向后兼容），>=2 含隐藏层。"""
        pad = kernel_size // 2
        if n_layers <= 1:
            return nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=pad)
        layers: list[nn.Module] = []
        for _ in range(n_layers - 1):
            layers.append(nn.Conv1d(in_ch, in_ch, kernel_size=kernel_size, padding=pad))
            layers.append(make_norm(in_ch))
            layers.append(nn.SiLU())
        layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=pad))
        return nn.Sequential(*layers)

    def _make_linear_head(in_dim: int, out_dim: int, n_layers: int):
        """构建 Linear head（count head），n_layers=1 为单层（向后兼容），>=2 含隐藏层。"""
        if n_layers <= 1:
            return nn.Linear(in_dim, out_dim)
        layers: list[nn.Module] = []
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(in_dim, in_dim))
            layers.append(make_norm(in_dim))
            layers.append(nn.SiLU())
        layers.append(nn.Linear(in_dim, out_dim))
        return nn.Sequential(*layers)

    class SEBlock(nn.Module):
        """Squeeze-and-Excitation 通道注意力块。

        GlobalAvgPool → FC(↓r) → SiLU → FC(↑r) → Sigmoid → 通道重标定。
        reduction=16，参数量 ~ 2 * C^2 / r。
        """

        def __init__(self, channels: int, reduction: int = 16):
            super().__init__()
            reduced = max(1, channels // reduction)
            self.fc = nn.Sequential(
                nn.Linear(channels, reduced),
                nn.SiLU(),
                nn.Linear(reduced, channels),
                nn.Sigmoid(),
            )

        def forward(self, x):
            # x: [B, C, L]
            w = self.fc(x.mean(dim=-1))  # [B, C]
            return x * w.unsqueeze(-1)  # [B, C, L]

    class DilatedResBlock(nn.Module):
        def __init__(self, c_in: int, c_out: int, dilation: int = 1):
            super().__init__()
            k = 3
            pad = dilation * (k // 2)
            self.conv1 = nn.Conv1d(c_in, c_out, kernel_size=k, padding=pad, dilation=dilation)
            self.bn1 = make_norm(c_out)
            self.conv2 = nn.Conv1d(c_out, c_out, kernel_size=k, padding=pad, dilation=dilation)
            self.bn2 = make_norm(c_out)
            self.shortcut = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()
            self.se = SEBlock(c_out) if use_se else nn.Identity()

        def forward(self, x):
            identity = self.shortcut(x)
            out = act_fn(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out = act_fn(out + identity)
            return self.se(out)

    class ConvStem(nn.Module):
        """runI: Conv(s=1) 无 MaxPool，不下采样，保留亚像素信息"""

        def __init__(self, c_in: int, c_out: int):
            super().__init__()
            self.conv = nn.Conv1d(c_in, c_out, kernel_size=7, stride=1, padding=3)
            self.bn = make_norm(c_out)

        def forward(self, x):
            return act_fn(self.bn(self.conv(x)))

    class DilatedResNet5Head(nn.Module):
        def __init__(self):
            super().__init__()
            w = width

            self.stem = ConvStem(in_channels, w)

            # stages 由参数控制：每个 stage = (out_channels_mult, dilation)
            # 默认 [(1,d1),(2,d2),(4,d3),(4,d4)] 兼容旧 ckpt（4 stage）
            # 新数据建议减小 dilation 并可调整 stage 数量匹配 5 pix trough
            self.stages = nn.ModuleList()
            c_in = w
            for mult, dil in stages:
                c_out = w * mult
                self.stages.append(DilatedResBlock(c_in, c_out, dilation=dil))
                c_in = c_out

            c_bneck = c_in  # 最后一个 stage 的输出通道数
            # runI: stem 不下采样，无需 upsample，只 1x1 conv 降通道
            self.upsample1 = nn.Conv1d(c_bneck, w, kernel_size=1)

            # FPN 跳跃连接：每个非最后 stage 的输出投影到 w//SKIP_CH_DIV 通道
            self.use_skip = use_skip
            self.skip_projections = nn.ModuleList()
            self.skip_channels = 0
            if use_skip:
                skip_ch = max(1, w // SKIP_CH_DIV)
                c_skip_in = w
                for idx, (mult, _) in enumerate(stages):
                    c_skip_out = w * mult
                    if idx < len(stages) - 1:  # 非最后 stage
                        self.skip_projections.append(nn.Conv1d(c_skip_out, skip_ch, kernel_size=1))
                        self.skip_channels += skip_ch
                    c_skip_in = c_skip_out

            refine_in = w + self.skip_channels
            self.refine = nn.Sequential(
                nn.Conv1d(refine_in, w, kernel_size=5, padding=2),
                make_norm(w),
                nn.SiLU(),
                nn.Conv1d(w, w, kernel_size=3, padding=1),
                make_norm(w),
                nn.SiLU(),
            )

            # runI: heat/region/offset head 用 3x3 kernel，lognhi 用 1x1
            # head_layers 控制头深度，=1 时为单层（向后兼容）
            self.heat_head = _make_conv_head(w, 1, kernel_size=3, n_layers=head_layers)
            self.region_head = _make_conv_head(w, 1, kernel_size=3, n_layers=head_layers)
            self.lognhi_head = _make_conv_head(w, 1, kernel_size=1, n_layers=head_layers)
            self.offset_head = _make_conv_head(w, 1, kernel_size=3, n_layers=head_layers)
            self.count_head = _make_linear_head(w, n_count_classes, n_layers=head_layers)

        def forward(self, x):
            if x.ndim == 3 and x.shape[1] != in_channels and x.shape[2] == in_channels:
                x = x.transpose(1, 2)
            orig_len = x.shape[-1]

            x = self.stem(x)

            # 收集 skip features（FPN 跳跃连接）
            skip_feats = []
            for idx, stage in enumerate(self.stages):
                x = stage(x)
                if self.use_skip and idx < len(self.stages) - 1:
                    skip_feats.append(self.skip_projections[idx](x))

            x = self.upsample1(x)
            if self.use_skip and skip_feats:
                x = torch.cat([x] + skip_feats, dim=1)
            if x.shape[-1] != orig_len:
                x = F.interpolate(x, size=orig_len, mode="linear", align_corners=False)
            shared = self.refine(x)

            heat_logits = self.heat_head(shared).squeeze(1)
            heat = torch.sigmoid(heat_logits)
            region_logits = self.region_head(shared).squeeze(1)
            region = torch.sigmoid(region_logits)
            lognhi = self.lognhi_head(shared).squeeze(1)
            offset = torch.tanh(self.offset_head(shared)).squeeze(1)
            pooled = shared.mean(dim=-1)
            count_logits = self.count_head(pooled)

            return {
                "heat_prob": heat,
                "heat_logits": heat_logits,
                "region_prob": region,
                "region_logits": region_logits,
                "lognhi": lognhi,
                "offset": offset,
                "count_prob": count_logits,
            }

    return DilatedResNet5Head()


def predict_raw_outputs_5head(
    model: Any,
    features: Any,
    indices: Any,
    batch_size: int = 96,
) -> dict[str, Any]:
    """批量推理，返回 5 个 head 的 numpy 输出。"""
    import numpy as np
    torch = import_torch()
    device = next(model.parameters()).device
    n = features.shape[0]
    heat_list, region_list, lognhi_list, off_list, count_list = [], [], [], [], []

    model.eval()
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch = features[start:end]
            t = torch.from_numpy(np.ascontiguousarray(batch)).float().to(device)
            out = model(t)
            heat_list.append(out["heat_prob"].cpu().numpy())
            region_list.append(out["region_prob"].cpu().numpy())
            lognhi_list.append(out["lognhi"].cpu().numpy())
            off_list.append(out["offset"].cpu().numpy())
            count_logits = out["count_prob"]
            count_prob = torch.softmax(count_logits, dim=-1).cpu().numpy()
            count_list.append(count_prob)

    return {
        "heat_prob": np.concatenate(heat_list, axis=0),
        "region_prob": np.concatenate(region_list, axis=0),
        "lognhi": np.concatenate(lognhi_list, axis=0),
        "offset": np.concatenate(off_list, axis=0),
        "count_prob": np.concatenate(count_list, axis=0),
    }


def main() -> None:
    """Smoke test: 验证 forward 通过，打印不同配置的参数量。"""
    torch = import_torch()
    n_bins_test = 681
    x = torch.zeros(2, 1, n_bins_test)

    # 对比不同配置（d1122 新数据 dilation + c0 单通道）
    configs = [
        ("3 stage", [(1, 1), (2, 1), (4, 2)], False, False),
        ("4 stage (default d1122)", [(1, 1), (2, 1), (4, 2), (4, 2)], False, False),
        ("4 stage + SE", [(1, 1), (2, 1), (4, 2), (4, 2)], False, True),
        ("5 stage", [(1, 1), (2, 1), (4, 2), (4, 2), (4, 2)], False, False),
        ("5 stage + SE", [(1, 1), (2, 1), (4, 2), (4, 2), (4, 2)], False, True),
    ]
    for name, stages, skip, se in configs:
        model = _build_dilated_resnet_5head(
            n_bins=n_bins_test, in_channels=1, width=96, stages=stages,
            use_skip=skip, use_se=se,
        )
        with torch.no_grad():
            out = model(x)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"[{name}] stages={stages} skip={skip} se={se} params={total_params:,}")
        for k, val in out.items():
            print(f"    {k}: {tuple(val.shape)}  range=[{val.min():.3f}, {val.max():.3f}]")


if __name__ == "__main__":
    main()
