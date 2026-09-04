#!/usr/bin/env python3
"""Generate a submission from a trained dual-tower feature fusion checkpoint."""
from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "hybrid_ensemble"))
sys.path.insert(0, "/home/heruihua")

from evaluate_hybrid import load_checkpoint, resolve_device
from feature_fusion import DualFusionTestDataset, DualTowerFusionNet
from predict_hybrid import write_submission
from csst_dla_wzx_pkg.inference import load_model_from_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test-fits", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu", "mps"], default="auto")
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--min-distance", type=int, default=10)
    parser.add_argument("--min-z-dla", type=float, default=1.10)
    return parser.parse_args()


def load_fusion(path: str | Path, device: torch.device):
    checkpoint = torch.load(path, map_location=device)
    config = checkpoint["config"]
    dilated, _ = load_checkpoint(config["dilated_checkpoint"], device)
    wzx, _ = load_model_from_checkpoint(config["wzx_checkpoint"], device)
    model = DualTowerFusionNet(
        dilated.model,
        wzx,
        merge_mode=config["merge_mode"],
        width=int(config["fusion_width"]),
        depth=int(config["fusion_depth"]),
        freeze_backbones=bool(config["freeze_backbones"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


@torch.inference_mode()
def predict(model, dataset, device: torch.device, batch_size: int) -> dict[str, np.ndarray]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=device.type == "cuda")
    heat, lognhi, offset, count_logits, rows = [], [], [], [], []
    for hybrid, wzx, zq, row in loader:
        output = model(hybrid.to(device), wzx.to(device), zq.to(device))
        heat.append(torch.sigmoid(output["center_logits"]).cpu().numpy())
        lognhi.append((20.3 + output["lognhi_raw"]).cpu().numpy())
        offset.append(output["offset_raw"].cpu().numpy())
        count_logits.append(output["count_logits"].cpu().numpy())
        rows.append(np.asarray(row, dtype=np.int64))
    return {
        "heatmap": np.concatenate(heat),
        "lognhi": np.concatenate(lognhi),
        "offset": np.concatenate(offset),
        "count_logits": np.concatenate(count_logits),
        "rows": np.concatenate(rows),
    }


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    model, checkpoint = load_fusion(args.checkpoint, device)
    dataset = DualFusionTestDataset(args.test_fits)
    prediction = predict(model, dataset, device, args.batch_size)
    config = {
        "threshold": float(args.threshold if args.threshold is not None else checkpoint.get("threshold", 0.45)),
        "min_distance": args.min_distance,
        "min_z_dla": args.min_z_dla,
        "use_offset": True,
        "lognhi_clip_min": 20.3,
        "lognhi_clip_max": 22.5,
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_submission(output, dataset, prediction, config)
    output.with_suffix(".config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
