#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import csv
import json
import sys

import numpy as np
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from csst_dla.scoring import C_KMS, greedy_match, score_catalog

LYA = 1215.67
BAND_BY_SNR_FIELD = {
    "SNR_GU": (2550.0, 4200.0),
    "SNR_GV": (4000.0, 6500.0),
    "SNR_GI": (6200.0, 10000.0),
}


def format_bin(lo: float, hi: float) -> str:
    return f"[{lo:g},{'inf' if np.isinf(hi) else f'{hi:g}'})"


def in_band(wavelength: float, band: tuple[float, float] | None) -> bool:
    if band is None:
        return True
    lo, hi = band
    return lo <= wavelength < hi


def load_truth_dlas(
    path: str | Path,
    min_lognhi: float,
    snr_field: str,
    band: tuple[float, float] | None,
) -> dict[str, np.ndarray]:
    targetid: list[int] = []
    z_qso: list[float] = []
    z_dla: list[float] = []
    lognhi: list[float] = []
    wavelength: list[float] = []
    snr: list[float] = []
    slot_id: list[int] = []

    with fits.open(path, memmap=True) as hdul:
        truth = hdul["TRUTH"].data
        for row in truth:
            for slot in (1, 2):
                if int(row["N_DLA"]) < slot:
                    continue
                value = float(row[f"LOGNHI{slot}"])
                if not np.isfinite(value) or value < min_lognhi:
                    continue
                z = float(row[f"Z_DLA{slot}"])
                lambda_dla = LYA * (1.0 + z)
                if not in_band(lambda_dla, band):
                    continue
                targetid.append(int(row["TARGETID"]))
                z_qso.append(float(row["Z_QSO"]))
                z_dla.append(z)
                lognhi.append(value)
                wavelength.append(lambda_dla)
                snr.append(float(row[snr_field]))
                slot_id.append(slot)

    return {
        "TARGETID": np.asarray(targetid, dtype=np.int64),
        "Z_QSO": np.asarray(z_qso, dtype=np.float32),
        "Z_DLA": np.asarray(z_dla, dtype=np.float32),
        "LOGNHI": np.asarray(lognhi, dtype=np.float32),
        "WAVELENGTH": np.asarray(wavelength, dtype=np.float32),
        "SNR": np.asarray(snr, dtype=np.float32),
        "SLOT": np.asarray(slot_id, dtype=np.int16),
    }


def load_snr_by_target(path: str | Path, snr_field: str) -> dict[int, float]:
    with fits.open(path, memmap=True) as hdul:
        truth = hdul["TRUTH"].data
        return {int(row["TARGETID"]): float(row[snr_field]) for row in truth}


def count_bins(
    dlas: dict[str, np.ndarray],
    snr_bins: tuple[float, ...],
    lognhi_bins: tuple[float, ...],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for snr_lo, snr_hi in zip(snr_bins[:-1], snr_bins[1:]):
        for log_lo, log_hi in zip(lognhi_bins[:-1], lognhi_bins[1:]):
            mask = (
                (dlas["SNR"] >= snr_lo)
                & (dlas["SNR"] < snr_hi)
                & (dlas["LOGNHI"] >= log_lo)
                & (dlas["LOGNHI"] < log_hi)
            )
            rows.append(
                {
                    "snr_bin": format_bin(snr_lo, snr_hi),
                    "snr_lo": None if np.isinf(snr_lo) else float(snr_lo),
                    "snr_hi": None if np.isinf(snr_hi) else float(snr_hi),
                    "lognhi_bin": format_bin(log_lo, log_hi),
                    "lognhi_lo": None if np.isinf(log_lo) else float(log_lo),
                    "lognhi_hi": None if np.isinf(log_hi) else float(log_hi),
                    "n_dla": int(mask.sum()),
                }
            )
    return rows


def load_prediction_csv(
    path: str | Path,
    snr_by_target: dict[int, float],
    band: tuple[float, float] | None,
    pred_min_lognhi: float,
) -> dict[str, np.ndarray]:
    targetid: list[int] = []
    z_dla: list[float] = []
    lognhi: list[float] = []
    wavelength: list[float] = []
    snr: list[float] = []

    with Path(path).open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            tid = int(row["id"])
            for slot in (1, 2):
                z = float(row[f"Z_DLA{slot}"])
                value = float(row[f"LOGNHI{slot}"])
                if z <= 0.0 or value < pred_min_lognhi:
                    continue
                lambda_dla = LYA * (1.0 + z)
                if not in_band(lambda_dla, band):
                    continue
                targetid.append(tid)
                z_dla.append(z)
                lognhi.append(value)
                wavelength.append(lambda_dla)
                snr.append(snr_by_target.get(tid, np.nan))

    return {
        "TARGETID": np.asarray(targetid, dtype=np.int64),
        "Z_DLA": np.asarray(z_dla, dtype=np.float32),
        "LOG_NHI": np.asarray(lognhi, dtype=np.float32),
        "WAVELENGTH": np.asarray(wavelength, dtype=np.float32),
        "SNR": np.asarray(snr, dtype=np.float32),
    }


def summarize(values: np.ndarray) -> tuple[float, float]:
    if values.size == 0:
        return 0.0, 0.0
    return float(values.mean()), float(values.std())


def add_prediction_stats(
    rows: list[dict[str, object]],
    truth_dlas: dict[str, np.ndarray],
    pred: dict[str, np.ndarray],
) -> dict[str, object]:
    truth = {
        "TARGETID": truth_dlas["TARGETID"],
        "Z_DLA": truth_dlas["Z_DLA"],
        "LOG_NHI": truth_dlas["LOGNHI"],
        "SNR": truth_dlas["SNR"],
    }
    matches = greedy_match(truth, pred)
    truth_matched = np.zeros(len(truth["TARGETID"]), dtype=bool)
    pred_matched = np.zeros(len(pred["TARGETID"]), dtype=bool)

    if matches:
        ti = np.asarray([m[0] for m in matches], dtype=int)
        pi = np.asarray([m[1] for m in matches], dtype=int)
        truth_matched[ti] = True
        pred_matched[pi] = True
        dv_all = C_KMS * (pred["Z_DLA"][pi] - truth["Z_DLA"][ti]) / (1.0 + truth["Z_DLA"][ti])
        dlog_all = pred["LOG_NHI"][pi] - truth["LOG_NHI"][ti]
    else:
        ti = np.asarray([], dtype=int)
        dv_all = np.asarray([], dtype=np.float32)
        dlog_all = np.asarray([], dtype=np.float32)

    for row in rows:
        snr_lo = -np.inf if row["snr_lo"] is None else float(row["snr_lo"])
        snr_hi = np.inf if row["snr_hi"] is None else float(row["snr_hi"])
        log_lo = -np.inf if row["lognhi_lo"] is None else float(row["lognhi_lo"])
        log_hi = np.inf if row["lognhi_hi"] is None else float(row["lognhi_hi"])

        truth_bin = (
            (truth["SNR"] >= snr_lo)
            & (truth["SNR"] < snr_hi)
            & (truth["LOG_NHI"] >= log_lo)
            & (truth["LOG_NHI"] < log_hi)
        )
        pred_bin = (
            (pred["SNR"] >= snr_lo)
            & (pred["SNR"] < snr_hi)
            & (pred["LOG_NHI"] >= log_lo)
            & (pred["LOG_NHI"] < log_hi)
        )
        matched_truth_bin = truth_bin[ti] if ti.size else np.asarray([], dtype=bool)
        dv_bin = dv_all[matched_truth_bin]
        dlog_bin = dlog_all[matched_truth_bin]
        mean_dv, std_dv = summarize(dv_bin)
        mean_dlognhi, std_dlognhi = summarize(dlog_bin)

        n_truth = int(truth_bin.sum())
        n_pred = int(pred_bin.sum())
        n_match_truth = int((truth_bin & truth_matched).sum())
        n_match_pred = int((pred_bin & pred_matched).sum())
        row.update(
            {
                "n_pred": n_pred,
                "n_match": n_match_truth,
                "n_match_pred_bin": n_match_pred,
                "completeness": float(n_match_truth / n_truth) if n_truth else 0.0,
                "purity": float(n_match_pred / n_pred) if n_pred else 0.0,
                "mean_dv": mean_dv,
                "std_dv": std_dv,
                "mean_dlognhi": mean_dlognhi,
                "std_dlognhi": std_dlognhi,
            }
        )

    score = score_catalog(truth, pred)
    return {
        "raw_n_truth": int(len(truth["TARGETID"])),
        "raw_n_pred": int(len(pred["TARGETID"])),
        "raw_n_match": int(len(matches)),
        "raw_purity": float(len(matches) / len(pred["TARGETID"])) if len(pred["TARGETID"]) else 0.0,
        "raw_completeness": float(len(matches) / len(truth["TARGETID"]))
        if len(truth["TARGETID"])
        else 0.0,
        "score": {
            "final_score": score.final_score,
            "detection_score": score.detection_score,
            "parameter_score": score.parameter_score,
            "mean_dv": score.mean_dv,
            "std_dv": score.std_dv,
            "mean_dlognhi": score.mean_dlognhi,
            "std_dlognhi": score.std_dlognhi,
        },
    }


def render_table(rows: list[dict[str, object]], with_pred: bool) -> str:
    lines: list[str] = []
    if with_pred:
        lines.append(
            f"{'SNR bin':>10}  {'LOGNHI bin':>13}  {'N_truth':>7}  {'N_pred':>6}  "
            f"{'N_match_T':>9}  {'N_match_P':>9}  {'purity':>8}  {'complete':>8}  "
            f"{'mean_dv':>10}  {'std_dv':>10}  {'mean_dlog':>10}  {'std_dlog':>9}"
        )
        lines.append("-" * 140)
    else:
        lines.append(f"{'SNR bin':>10}  {'LOGNHI bin':>13}  {'N_DLA':>6}")
        lines.append("-" * 35)
    for row in rows:
        if with_pred:
            lines.append(
                f"{row['snr_bin']:>10}  {row['lognhi_bin']:>13}  {row['n_dla']:7d}  "
                f"{row['n_pred']:6d}  {row['n_match']:9d}  {row['n_match_pred_bin']:9d}  "
                f"{row['purity']:8.4f}  {row['completeness']:8.4f}  "
                f"{row['mean_dv']:10.3f}  {row['std_dv']:10.3f}  "
                f"{row['mean_dlognhi']:10.4f}  {row['std_dlognhi']:9.4f}"
            )
        else:
            lines.append(f"{row['snr_bin']:>10}  {row['lognhi_bin']:>13}  {row['n_dla']:6d}")
    return "\n".join(lines)


def render_report(
    args: argparse.Namespace,
    rows: list[dict[str, object]],
    with_pred: bool,
    pred_summary: dict[str, object] | None = None,
) -> str:
    band = None if args.no_band_filter else BAND_BY_SNR_FIELD[args.snr_field]
    lines = [
        f"truth_fits: {args.truth_fits}",
        f"snr_field: {args.snr_field}",
        f"wavelength_band_A: {'all' if band is None else f'[{band[0]:g},{band[1]:g})'}",
        f"min_lognhi: {args.min_lognhi}",
        f"total counted DLAs: {sum(int(row['n_dla']) for row in rows)}",
    ]
    if with_pred:
        total_pred_binned = sum(int(row["n_pred"]) for row in rows)
        total_match_binned = sum(int(row["n_match"]) for row in rows)
        total_truth_binned = sum(int(row["n_dla"]) for row in rows)
        raw_n_truth = int(pred_summary["raw_n_truth"]) if pred_summary else total_truth_binned
        raw_n_pred = int(pred_summary["raw_n_pred"]) if pred_summary else total_pred_binned
        raw_n_match = int(pred_summary["raw_n_match"]) if pred_summary else total_match_binned
        lines.extend(
            [
                f"pred_csv: {args.pred_csv}",
                f"raw predicted DLAs after filters: {raw_n_pred}",
                f"binned predicted DLAs: {total_pred_binned}",
                f"raw matched DLAs: {raw_n_match}",
                f"overall purity: {raw_n_match / raw_n_pred if raw_n_pred else 0.0:.6f}",
                f"overall completeness: {raw_n_match / raw_n_truth if raw_n_truth else 0.0:.6f}",
                "N_match_T is binned by true LOGNHI/SNR and is used for completeness.",
                "N_match_P is binned by predicted LOGNHI/SNR and is used for purity.",
            ]
        )
        if pred_summary:
            score = pred_summary["score"]
            lines.extend(
                [
                    "score using csst_dla.scoring.score_catalog:",
                    f"  final_score: {score['final_score']:.6f}",
                    f"  detection_score: {score['detection_score']:.6f}",
                    f"  parameter_score: {score['parameter_score']:.6f}",
                    f"  mean_dv/std_dv: {score['mean_dv']:.3f} / {score['std_dv']:.3f}",
                    f"  mean_dlognhi/std_dlognhi: {score['mean_dlognhi']:.4f} / {score['std_dlognhi']:.4f}",
                ]
            )
            if raw_n_truth and raw_n_pred < 0.1 * raw_n_truth:
                lines.append(
                    "WARNING: raw predicted DLAs are less than 10% of truth; check the prediction CSV/config/data split before interpreting bin scores."
                )
    lines.append(render_table(rows, with_pred))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Count truth DLAs by SNR and LOGNHI bins.")
    parser.add_argument("--truth-fits", default="hybrid_ensemble/test_truth.fits")
    parser.add_argument("--pred-csv", help="Optional prediction CSV with id,Z_DLA1,LOGNHI1,Z_DLA2,LOGNHI2 columns.")
    parser.add_argument("--min-lognhi", type=float, default=20.3)
    parser.add_argument(
        "--pred-min-lognhi",
        type=float,
        default=None,
        help="Minimum predicted LOGNHI to include. Defaults to --min-lognhi.",
    )
    parser.add_argument("--snr-field", choices=["SNR_GU", "SNR_GV", "SNR_GI"], default="SNR_GU")
    parser.add_argument(
        "--no-band-filter",
        action="store_true",
        help="Do not filter DLAs by the wavelength band implied by --snr-field.",
    )
    parser.add_argument(
        "--snr-bins",
        nargs="+",
        type=float,
        default=[0, 1, 2, 3, 4, 5, 6, 7, np.inf],
        help="Bin edges. Use inf for the final open-ended bin.",
    )
    parser.add_argument(
        "--lognhi-bins",
        nargs="+",
        type=float,
        default=[20.3, 20.5, 21.0, 21.5, 22.0, np.inf],
        help="Bin edges. Use inf for the final open-ended bin.",
    )
    parser.add_argument("--out-csv")
    parser.add_argument("--out-json")
    parser.add_argument("--out-txt", help="Write the printed report to a text file.")
    args = parser.parse_args()
    if args.pred_min_lognhi is None:
        args.pred_min_lognhi = args.min_lognhi

    snr_bins = tuple(float(x) for x in args.snr_bins)
    lognhi_bins = tuple(float(x) for x in args.lognhi_bins)
    if len(snr_bins) < 2 or len(lognhi_bins) < 2:
        raise ValueError("Need at least two edges for both --snr-bins and --lognhi-bins.")

    band = None if args.no_band_filter else BAND_BY_SNR_FIELD[args.snr_field]
    dlas = load_truth_dlas(args.truth_fits, args.min_lognhi, args.snr_field, band)
    rows = count_bins(dlas, snr_bins, lognhi_bins)
    pred_summary = None
    if args.pred_csv:
        snr_by_target = load_snr_by_target(args.truth_fits, args.snr_field)
        pred = load_prediction_csv(args.pred_csv, snr_by_target, band, args.pred_min_lognhi)
        pred_summary = add_prediction_stats(rows, dlas, pred)

    report = render_report(args, rows, with_pred=bool(args.pred_csv), pred_summary=pred_summary)
    print(report)

    if args.out_txt:
        out_txt = Path(args.out_txt)
        out_txt.parent.mkdir(parents=True, exist_ok=True)
        out_txt.write_text(report + "\n", encoding="utf-8")
        print(f"wrote {out_txt}")

    if args.out_csv:
        out_csv = Path(args.out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {out_csv}")

    if args.out_json:
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(
            json.dumps(
                {
                    "truth_fits": str(args.truth_fits),
                    "snr_field": args.snr_field,
                    "wavelength_band_A": None if band is None else list(band),
                    "min_lognhi": args.min_lognhi,
                    "total_counted_dlas": int(len(dlas["TARGETID"])),
                    "prediction_summary": pred_summary,
                    "bins": rows,
                },
                indent=2,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
        print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
