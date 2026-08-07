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

from csst_dla.scoring import C_KMS, greedy_match, labels_to_truth
from data import HybridTrainDataset
from decode import average_predictions, decode_validation_catalog, predict_member
from evaluate_hybrid import load_checkpoint, resolve_device


def format_bin(lo: float, hi: float) -> str:
    return f"[{lo:g},{'inf' if np.isinf(hi) else f'{hi:g}'})"


def summarize(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "median": 0.0,
            "p90_abs": 0.0,
            "max_abs": 0.0,
        }
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "median": float(np.median(values)),
        "p90_abs": float(np.percentile(np.abs(values), 90)),
        "max_abs": float(np.max(np.abs(values))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze matched parameter errors by SNR/LOGNHI bins.")
    parser.add_argument("--ensemble-config", default="hybrid_ensemble/runs/ensemble_eval_config.json")
    parser.add_argument("--targets", default="outputs/cnn_targets_seed42.npz")
    parser.add_argument("--train-fits", default="train.fits")
    parser.add_argument("--out-json", default="hybrid_ensemble/runs/parameter_bin_diagnostics.json")
    parser.add_argument("--out-csv", default="hybrid_ensemble/runs/parameter_bin_diagnostics.csv")
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
    )
    truth = labels_to_truth(ref_ds.labels, ref_ds.indices, min_lognhi=20.3)
    matches = greedy_match(truth, pred)

    if matches:
        ti = np.asarray([m[0] for m in matches], dtype=int)
        pi = np.asarray([m[1] for m in matches], dtype=int)
        dv = C_KMS * (pred["Z_DLA"][pi] - truth["Z_DLA"][ti]) / (1.0 + truth["Z_DLA"][ti])
        dlog = pred["LOG_NHI"][pi] - truth["LOG_NHI"][ti]
    else:
        ti = pi = np.asarray([], dtype=int)
        dv = dlog = np.asarray([], dtype=np.float32)

    snr_bins = (0, 1, 2, 3, 4, 5, 6, 7, np.inf)
    lognhi_bins = (20.3, 20.5, 21.0, 21.5, 22.0, np.inf)
    rows = []
    for snr_lo, snr_hi in zip(snr_bins[:-1], snr_bins[1:]):
        for log_lo, log_hi in zip(lognhi_bins[:-1], lognhi_bins[1:]):
            truth_bin = (
                (truth["SNR"] >= snr_lo)
                & (truth["SNR"] < snr_hi)
                & (truth["LOG_NHI"] >= log_lo)
                & (truth["LOG_NHI"] < log_hi)
            )
            matched_bin = truth_bin[ti] if ti.size else np.asarray([], dtype=bool)
            dv_bin = dv[matched_bin]
            dlog_bin = dlog[matched_bin]
            dv_s = summarize(dv_bin)
            dlog_s = summarize(dlog_bin)
            n_truth = int(truth_bin.sum())
            n_match = int(dv_bin.size)
            rows.append(
                {
                    "snr_bin": format_bin(snr_lo, snr_hi),
                    "lognhi_bin": format_bin(log_lo, log_hi),
                    "n_truth": n_truth,
                    "n_match": n_match,
                    "match_frac": float(n_match / n_truth) if n_truth else 0.0,
                    "mean_dv": dv_s["mean"],
                    "std_dv": dv_s["std"],
                    "median_dv": dv_s["median"],
                    "p90_abs_dv": dv_s["p90_abs"],
                    "max_abs_dv": dv_s["max_abs"],
                    "mean_dlognhi": dlog_s["mean"],
                    "std_dlognhi": dlog_s["std"],
                    "p90_abs_dlognhi": dlog_s["p90_abs"],
                }
            )

    worst = sorted(
        [row for row in rows if row["n_match"] >= 5],
        key=lambda row: (row["std_dv"], row["p90_abs_dv"]),
        reverse=True,
    )[:10]
    out = {
        "ensemble_config": str(args.ensemble_config),
        "overall": {
            "n_truth": int(len(truth["TARGETID"])),
            "n_pred": int(len(pred["TARGETID"])),
            "n_match": int(len(matches)),
            "mean_dv": summarize(dv)["mean"],
            "std_dv": summarize(dv)["std"],
            "p90_abs_dv": summarize(dv)["p90_abs"],
            "mean_dlognhi": summarize(dlog)["mean"],
            "std_dlognhi": summarize(dlog)["std"],
        },
        "worst_by_std_dv": worst,
        "bins": rows,
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2, allow_nan=False), encoding="utf-8")
    out_csv = Path(args.out_csv)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {out_json}")
    print(f"wrote {out_csv}")
    print(json.dumps({"overall": out["overall"], "worst_by_std_dv": worst[:5]}, indent=2))


if __name__ == "__main__":
    main()
