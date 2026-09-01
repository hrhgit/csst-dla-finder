#!/usr/bin/env python3
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from csst_dla.scoring import labels_to_truth, score_catalog
from data import HybridTrainDataset
from decode import average_predictions, decode_validation_catalog, fit_lognhi_calibration, predict_member
from model import HybridDlaNet, input_channels


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device(name)


def load_checkpoint(path: str | Path, device: torch.device):
    ckpt = torch.load(path, map_location=device)
    cfg = ckpt["config"]
    model = HybridDlaNet(
        input_channels(cfg["input_mode"]),
        hidden=int(cfg.get("hidden", 32)),
        num_blocks=int(cfg.get("num_blocks", 6)),
        with_offset=bool(cfg.get("with_offset", False)),
        norm_type=str(cfg.get("norm_type", "layer")),
        head_layers=int(cfg.get("head_layers", 1)),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    return model, cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate weighted Hybrid Ensemble on validation data.")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--weights", nargs="+", type=float)
    parser.add_argument("--targets", default="outputs/cnn_targets_seed42.npz")
    parser.add_argument("--train-fits", default="train.fits")
    parser.add_argument("--out", default="hybrid_ensemble/runs/ensemble_eval.json")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--min-distance", type=int, default=10)
    parser.add_argument("--min-z-dla", type=float, default=1.55)
    parser.add_argument(
        "--truth-min-lognhi",
        type=float,
        default=19.5,
        help="Minimum LOGNHI used to construct validation truth.",
    )
    parser.add_argument("--lognhi-min", type=float, default=19.5)
    parser.add_argument("--lognhi-max", type=float, default=22.5)
    parser.add_argument("--count-bias", nargs=3, type=float, default=[0.0, 0.0, 0.0])
    parser.add_argument(
        "--count-min-prob",
        type=float,
        default=0.0,
        help="Require max count probability to exceed this value before emitting 1/2 DLA slots.",
    )
    parser.add_argument(
        "--velocity-offset-kms",
        type=float,
        default=0.0,
        help="Optional post-decode velocity offset applied to all predicted Z_DLA values.",
    )
    parser.add_argument("--soft-radius", type=int, default=0, help="Use soft-argmax wavelength decoding within this pixel radius.")
    parser.add_argument("--soft-power", type=float, default=3.0, help="Power applied to heatmap probabilities for soft-argmax decoding.")
    parser.add_argument("--no-offset", action="store_true", help="Ignore an available trained offset head during decoding.")
    parser.add_argument("--low-lognhi-threshold", type=float)
    parser.add_argument("--low-lognhi-max", type=float, default=20.5)
    parser.add_argument("--low-lognhi-snr-max", type=float, default=3.0)
    parser.add_argument("--fit-lognhi-calibration", action="store_true")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu", "mps"], default="auto")
    args = parser.parse_args()

    if args.weights is not None and len(args.weights) != len(args.models):
        raise ValueError("--weights must have the same length as --models")
    weights = args.weights or [1.0] * len(args.models)
    device = resolve_device(args.device)

    preds = []
    member_configs = []
    ref_ds = None
    for model_path in args.models:
        model, cfg = load_checkpoint(model_path, device)
        ds = HybridTrainDataset(args.targets, args.train_fits, "val", cfg["input_mode"])
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
        preds.append(predict_member(model, loader, device))
        member_configs.append({"path": str(model_path), "weight": weights[len(member_configs)], **cfg})
        ref_ds = ds

    assert ref_ds is not None
    avg = average_predictions(preds, weights)
    slope, intercept, n_cal = 1.0, 0.0, 0
    if args.fit_lognhi_calibration:
        slope, intercept, n_cal = fit_lognhi_calibration(
            avg,
            ref_ds.indices,
            ref_ds.labels,
            ref_ds.wavelength,
            threshold=args.threshold,
            min_z_dla=args.min_z_dla,
            count_bias=args.count_bias,
            count_min_prob=args.count_min_prob,
            lognhi_min=args.lognhi_min,
            lognhi_max=args.lognhi_max,
            velocity_offset_kms=args.velocity_offset_kms,
            soft_radius=args.soft_radius,
            soft_power=args.soft_power,
            use_offset=not args.no_offset,
            low_lognhi_threshold=args.low_lognhi_threshold,
            low_lognhi_max=args.low_lognhi_max,
            low_lognhi_snr_max=args.low_lognhi_snr_max,
        )
    pred_catalog = decode_validation_catalog(
        avg,
        ref_ds.indices,
        ref_ds.labels,
        ref_ds.wavelength,
        threshold=args.threshold,
        min_distance=args.min_distance,
        min_z_dla=args.min_z_dla,
        count_bias=args.count_bias,
        count_min_prob=args.count_min_prob,
        lognhi_min=args.lognhi_min,
        lognhi_max=args.lognhi_max,
        lognhi_slope=slope,
        lognhi_intercept=intercept,
        velocity_offset_kms=args.velocity_offset_kms,
        soft_radius=args.soft_radius,
        soft_power=args.soft_power,
        use_offset=not args.no_offset,
        low_lognhi_threshold=args.low_lognhi_threshold,
        low_lognhi_max=args.low_lognhi_max,
        low_lognhi_snr_max=args.low_lognhi_snr_max,
    )
    truth = labels_to_truth(ref_ds.labels, ref_ds.indices, min_lognhi=args.truth_min_lognhi)
    score = score_catalog(truth, pred_catalog)
    result = {
        "models": member_configs,
        "threshold": args.threshold,
        "min_distance": args.min_distance,
        "min_z_dla": args.min_z_dla,
        "truth_min_lognhi": args.truth_min_lognhi,
        "count_bias": args.count_bias,
        "count_min_prob": args.count_min_prob,
        "lognhi_min": args.lognhi_min,
        "lognhi_max": args.lognhi_max,
        "velocity_offset_kms": args.velocity_offset_kms,
        "soft_radius": args.soft_radius,
        "soft_power": args.soft_power,
        "use_offset": not args.no_offset,
        "low_lognhi_threshold": args.low_lognhi_threshold,
        "low_lognhi_max": args.low_lognhi_max,
        "low_lognhi_snr_max": args.low_lognhi_snr_max,
        "lognhi_calibration": {
            "slope": slope,
            "intercept": intercept,
            "n_matches": n_cal,
        },
        "score": {
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
            "bin_details": score.bin_details,
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    config_path = out.with_name(out.stem + "_config.json")
    config_path.write_text(
        json.dumps(
            {
                "models": member_configs,
                "threshold": args.threshold,
                "min_distance": args.min_distance,
                "min_z_dla": args.min_z_dla,
        "count_bias": args.count_bias,
        "count_min_prob": args.count_min_prob,
        "lognhi_min": args.lognhi_min,
        "lognhi_max": args.lognhi_max,
                "velocity_offset_kms": args.velocity_offset_kms,
                "soft_radius": args.soft_radius,
                "soft_power": args.soft_power,
                "use_offset": not args.no_offset,
                "low_lognhi_threshold": args.low_lognhi_threshold,
                "low_lognhi_max": args.low_lognhi_max,
                "low_lognhi_snr_max": args.low_lognhi_snr_max,
                "lognhi_calibration": {"slope": slope, "intercept": intercept},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {out}")
    print(f"wrote {config_path}")
    print(json.dumps({k: v for k, v in result["score"].items() if k != "bin_details"}, indent=2))


if __name__ == "__main__":
    main()
