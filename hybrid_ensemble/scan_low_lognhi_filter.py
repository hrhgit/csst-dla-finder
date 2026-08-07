#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from csst_dla.scoring import labels_to_truth, score_catalog
from data import HybridTrainDataset
from decode import average_predictions, decode_validation_catalog, predict_member
from evaluate_hybrid import load_checkpoint, resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan low-LOGNHI confidence thresholds for a saved ensemble config.")
    parser.add_argument("--ensemble-config", default="hybrid_ensemble/runs/run2/ensemble_eval_config.json")
    parser.add_argument("--targets", default="outputs/cnn_targets_seed42.npz")
    parser.add_argument("--train-fits", default="train.fits")
    parser.add_argument("--out", default="hybrid_ensemble/runs/run2/low_lognhi_threshold_scan.json")
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75])
    parser.add_argument("--include-none", action="store_true")
    parser.add_argument("--low-lognhi-max", type=float, default=20.5)
    parser.add_argument("--low-lognhi-snr-max", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    args = parser.parse_args()

    config = json.loads(Path(args.ensemble_config).read_text(encoding="utf-8"))
    device = resolve_device(args.device)
    preds = []
    weights = []
    ref_ds = None
    for member in config["models"]:
        model, cfg = load_checkpoint(member["path"], device)
        ds = HybridTrainDataset(args.targets, args.train_fits, "val", cfg["input_mode"])
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
        preds.append(predict_member(model, loader, device))
        weights.append(float(member.get("weight", 1.0)))
        ref_ds = ds
    assert ref_ds is not None

    avg = average_predictions(preds, weights)
    cal = config.get("lognhi_calibration", {"slope": 1.0, "intercept": 0.0})
    truth = labels_to_truth(ref_ds.labels, ref_ds.indices, min_lognhi=20.3)
    scan_values: list[float | None] = []
    if args.include_none:
        scan_values.append(None)
    scan_values.extend(args.thresholds)

    rows = []
    for low_thr in scan_values:
        pred = decode_validation_catalog(
            avg,
            ref_ds.indices,
            ref_ds.labels,
            ref_ds.wavelength,
            threshold=float(config.get("threshold", 0.45)),
            min_distance=int(config.get("min_distance", 10)),
            count_bias=config.get("count_bias", [0.0, 0.0, 0.0]),
            count_min_prob=float(config.get("count_min_prob", 0.0)),
            lognhi_slope=float(cal.get("slope", 1.0)),
            lognhi_intercept=float(cal.get("intercept", 0.0)),
            velocity_offset_kms=float(config.get("velocity_offset_kms", 0.0)),
            soft_radius=int(config.get("soft_radius", 0)),
            soft_power=float(config.get("soft_power", 3.0)),
            use_offset=bool(config.get("use_offset", True)),
            low_lognhi_threshold=low_thr,
            low_lognhi_max=args.low_lognhi_max,
            low_lognhi_snr_max=args.low_lognhi_snr_max,
        )
        score = score_catalog(truth, pred)
        rows.append(
            {
                "low_lognhi_threshold": low_thr,
                "low_lognhi_max": args.low_lognhi_max,
                "low_lognhi_snr_max": args.low_lognhi_snr_max,
                "final_score": score.final_score,
                "detection_score": score.detection_score,
                "parameter_score": score.parameter_score,
                "completeness": score.completeness,
                "purity": score.purity,
                "n_pred": score.n_pred,
                "n_match": score.n_match,
                "mean_dv": score.mean_dv,
                "std_dv": score.std_dv,
                "mean_dlognhi": score.mean_dlognhi,
                "std_dlognhi": score.std_dlognhi,
            }
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2, allow_nan=False), encoding="utf-8")
    best = max(rows, key=lambda row: row["final_score"])
    print(f"wrote {out}")
    print("best:")
    print(json.dumps(best, indent=2))
    print("all:")
    for row in rows:
        print(
            f"thr={row['low_lognhi_threshold']} final={row['final_score']:.6f} "
            f"det={row['detection_score']:.6f} param={row['parameter_score']:.6f} "
            f"n_pred={row['n_pred']} purity={row['purity']:.4f} comp={row['completeness']:.4f}"
        )


if __name__ == "__main__":
    main()

