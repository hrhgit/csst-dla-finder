#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "hybrid_ensemble"))

from csst_dla.scoring import labels_to_truth, score_catalog
from data import HybridTrainDataset, make_scored_count_targets
from decode import average_predictions, decode_validation_catalog, predict_member, softmax
from evaluate_hybrid import load_checkpoint, resolve_device


def calibrate_count_logits(logits: np.ndarray, target_rates: np.ndarray) -> np.ndarray:
    """Fit an additive count-head bias to the validation class rates."""
    bias = np.zeros(3, dtype=np.float64)
    best = bias.copy()
    best_error = float("inf")
    for _ in range(6):
        improved = False
        for class_id in range(3):
            candidates = np.linspace(-8.0, 12.0, 401, dtype=np.float64)
            errors = []
            for value in candidates:
                trial = bias.copy()
                trial[class_id] = value
                rates = softmax(logits + trial[None, :]).mean(axis=0)
                errors.append(float(np.sum((rates - target_rates) ** 2)))
            choice = int(np.argmin(errors))
            if errors[choice] + 1e-12 < best_error:
                best_error = errors[choice]
                bias[class_id] = float(candidates[choice])
                improved = True
        if not improved:
            break
    best = bias.copy()
    return best.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune decoding only on the validation split.")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--targets", default="outputs/cnn_targets_seed42.npz")
    parser.add_argument("--train-fits", default="data/new/train_5e5.fits")
    parser.add_argument("--out", required=True)
    parser.add_argument("--min-z-dla", type=float, default=1.55)
    parser.add_argument("--truth-min-lognhi", type=float, default=19.5)
    parser.add_argument("--lognhi-max", type=float, default=22.5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu", "mps"], default="cuda")
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[0.30, 0.35, 0.40, 0.45, 0.50, 0.55],
        help="解码置信度阈值候选列表。",
    )
    parser.add_argument(
        "--count-bias-scales",
        nargs="+",
        type=float,
        default=[0.25, 0.50, 0.75, 1.00, 1.25, 1.50],
        help="count 校正偏置相对标定值的缩放系数列表。",
    )
    parser.add_argument(
        "--low-lognhi-thresholds",
        nargs="+",
        type=float,
        default=[-1.0],
        help="低 LOGNHI 区间的 confidence 额外过滤阈值；-1 表示关闭该过滤。",
    )
    parser.add_argument("--low-lognhi-max", type=float, default=20.5)
    parser.add_argument("--low-lognhi-snr-max", type=float, default=3.0)
    args = parser.parse_args()

    device = resolve_device(args.device)
    preds = []
    weights = []
    ref_ds = None
    for model_path in args.models:
        model, cfg = load_checkpoint(model_path, device)
        ds = HybridTrainDataset(args.targets, args.train_fits, "val", cfg["input_mode"])
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
        preds.append(predict_member(model, loader, device))
        weights.append(1.0)
        ref_ds = ds
    assert ref_ds is not None
    pred = average_predictions(preds, weights)
    truth = labels_to_truth(ref_ds.labels, ref_ds.indices, min_lognhi=args.truth_min_lognhi)

    true_counts = make_scored_count_targets(
        ref_ds.labels,
        ref_ds.indices,
        min_lognhi=args.truth_min_lognhi,
    )
    target_rates = np.bincount(true_counts, minlength=3).astype(np.float64) / len(true_counts)
    raw_rates = np.bincount(np.argmax(pred["count_logits"], axis=1), minlength=3) / len(true_counts)
    calibrated_bias = calibrate_count_logits(pred["count_logits"], target_rates)
    calibrated_rates = softmax(pred["count_logits"] + calibrated_bias[None, :]).mean(axis=0)
    print("target count rates:", target_rates.tolist(), flush=True)
    print("raw predicted rates:", raw_rates.tolist(), flush=True)
    print("calibrated bias:", calibrated_bias.tolist(), flush=True)
    print("calibrated rates:", calibrated_rates.tolist(), flush=True)

    results = []
    thresholds = args.thresholds
    scales = args.count_bias_scales
    low_lognhi_filters = [
        None if value < 0 else float(value)
        for value in args.low_lognhi_thresholds
    ]
    for threshold in thresholds:
        for scale in scales:
            bias = calibrated_bias * scale
            for low_filter in low_lognhi_filters:
                catalog = decode_validation_catalog(
                    pred,
                    ref_ds.indices,
                    ref_ds.labels,
                    ref_ds.wavelength,
                    threshold=threshold,
                    min_distance=10,
                    min_z_dla=args.min_z_dla,
                    count_bias=bias.tolist(),
                    lognhi_min=args.truth_min_lognhi,
                    lognhi_max=args.lognhi_max,
                    low_lognhi_threshold=low_filter,
                    low_lognhi_max=args.low_lognhi_max,
                    low_lognhi_snr_max=args.low_lognhi_snr_max,
                )
                score = score_catalog(truth, catalog)
                row = {
                    "threshold": threshold,
                    "count_bias_scale": scale,
                    "count_bias": [float(v) for v in bias],
                    "low_lognhi_threshold": low_filter,
                    "low_lognhi_max": args.low_lognhi_max,
                    "low_lognhi_snr_max": args.low_lognhi_snr_max,
                    "final_score": score.final_score,
                    "detection_score": score.detection_score,
                    "parameter_score": score.parameter_score,
                    "completeness": score.completeness,
                    "purity": score.purity,
                    "n_truth": score.n_truth,
                    "n_pred": score.n_pred,
                    "n_match": score.n_match,
                    "mean_dv": score.mean_dv,
                    "std_dv": score.std_dv,
                    "mean_dlognhi": score.mean_dlognhi,
                    "std_dlognhi": score.std_dlognhi,
                }
                results.append(row)
                print(json.dumps(row, ensure_ascii=False), flush=True)

    results.sort(key=lambda item: item["final_score"], reverse=True)
    output = {
        "models": args.models,
        "min_z_dla": args.min_z_dla,
        "truth_min_lognhi": args.truth_min_lognhi,
        "lognhi_min": args.truth_min_lognhi,
        "lognhi_max": args.lognhi_max,
        "search_space": {
            "thresholds": thresholds,
            "count_bias_scales": scales,
            "low_lognhi_thresholds": low_lognhi_filters,
            "min_distance": 10,
        },
        "calibration": {
            "method": "validation_count_rate_matching",
            "target_rates": target_rates.tolist(),
            "raw_rates": raw_rates.tolist(),
            "bias": calibrated_bias.tolist(),
            "calibrated_rates": calibrated_rates.tolist(),
        },
        "results": results,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
