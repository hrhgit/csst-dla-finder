#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import csv
import json
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import HybridTestDataset
from decode import average_predictions, pick_peaks, pixel_to_wavelength, soft_peak_wavelength, softmax, predict_member
from evaluate_hybrid import load_checkpoint, resolve_device


def write_submission(out_path: str | Path, dataset: HybridTestDataset, pred: dict, config: dict) -> None:
    threshold = float(config.get("threshold", 0.45))
    min_distance = int(config.get("min_distance", 10))
    min_z_dla = float(config.get("min_z_dla", 1.55))
    count_bias = np.asarray(config.get("count_bias", [0.0, 0.0, 0.0]), dtype=np.float32)
    count_min_prob = float(config.get("count_min_prob", 0.0))
    velocity_offset_kms = float(config.get("velocity_offset_kms", 0.0))
    soft_radius = int(config.get("soft_radius", 0))
    soft_power = float(config.get("soft_power", 3.0))
    use_offset = bool(config.get("use_offset", True))
    low_lognhi_threshold = config.get("low_lognhi_threshold")
    low_lognhi_max = float(config.get("low_lognhi_max", 20.5))
    low_lognhi_snr_max = float(config.get("low_lognhi_snr_max", 3.0))
    cal = config.get("lognhi_calibration", {"slope": 1.0, "intercept": 0.0})
    slope = float(cal.get("slope", 1.0))
    intercept = float(cal.get("intercept", 0.0))
    lognhi_clip_min = float(config.get("lognhi_clip_min", 20.3))
    lognhi_clip_max = float(config.get("lognhi_clip_max", 22.5))
    count_prob = softmax(pred['count_logits'] + count_bias[None,:])
    # if "count_prob" in pred and np.allclose(count_bias, 0.0):
    #     count_prob = pred["count_prob"]
    # else:
    #     count_prob = softmax(pred["count_logits"] + count_bias[None, :])
    row_to_pred = {int(row): i for i, row in enumerate(pred["rows"])}

    with Path(out_path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "Z_DLA1", "LOGNHI1", "Z_DLA2", "LOGNHI2"])
        for row in range(len(dataset)):
            i = row_to_pred[row]
            n_pick = int(np.argmax(count_prob[i]))
            if n_pick > 0 and float(count_prob[i, n_pick]) < count_min_prob:
                n_pick = 0
            peaks = pick_peaks(
                pred["heatmap"][i],
                dataset.wavelength,
                float(dataset.zq[row]),
                n_pick=n_pick,
                threshold=threshold,
                min_distance=min_distance,
                min_z_dla=min_z_dla,
            )
            values: list[float] = []
            for pix in peaks[:2]:
                if use_offset and "offset" in pred:
                    lambda_dla = pixel_to_wavelength(pix + float(pred["offset"][i, pix]), dataset.wavelength)
                else:
                    lambda_dla = soft_peak_wavelength(
                        pred["heatmap"][i],
                        dataset.wavelength,
                        pix,
                        radius=soft_radius,
                        power=soft_power,
                    )
                z_dla = float(lambda_dla / 1215.67 - 1.0)
                if velocity_offset_kms:
                    z_dla = z_dla + velocity_offset_kms / 299792.458 * (1.0 + z_dla)
                lognhi = float(
                    np.clip(
                        slope * float(pred["lognhi"][i, pix]) + intercept,
                        lognhi_clip_min,
                        lognhi_clip_max,
                    )
                )
                confidence = float(pred["heatmap"][i, pix] * count_prob[i, n_pick])
                if (
                    low_lognhi_threshold is not None
                    and lognhi < low_lognhi_max
                    and float(dataset.snr[row]) < low_lognhi_snr_max
                    and confidence < float(low_lognhi_threshold)
                ):
                    continue
                values.extend([z_dla, lognhi])
            while len(values) < 4:
                values.extend([-1.0, 0.0])
            writer.writerow(
                [
                    int(dataset.targetid[row]),
                    f"{values[0]:.6f}",
                    f"{values[1]:.6f}",
                    f"{values[2]:.6f}",
                    f"{values[3]:.6f}",
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate test CSV with a Hybrid Ensemble config.")
    parser.add_argument("--ensemble-config", required=True)
    parser.add_argument("--test-fits", default="test.fits")
    parser.add_argument("--out", default="hybrid_ensemble/runs/submission_hybrid.csv")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    args = parser.parse_args()

    config = json.loads(Path(args.ensemble_config).read_text(encoding="utf-8"))
    device = resolve_device(args.device)
    preds = []
    weights = []
    ref_dataset = None
    for member in config["models"]:
        model, cfg = load_checkpoint(member["path"], device)
        dataset = HybridTestDataset(args.test_fits, cfg["input_mode"])
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        preds.append(predict_member(model, loader, device))
        weights.append(float(member.get("weight", 1.0)))
        ref_dataset = dataset
    assert ref_dataset is not None
    avg = average_predictions(preds, weights)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_submission(out, ref_dataset, avg, config)
    print(f"wrote {out}")
    print(f"rows: {len(ref_dataset)}")


if __name__ == "__main__":
    main()
