#!/usr/bin/env python3
"""Decoder-level fusion study for the WZX and dilated-resnet DLA models.

This intentionally does not train a third model.  It replays the two saved
checkpoints on the same spectra, swaps one output head at a time, and applies
fixed proposal/verification rules.  The resulting catalogs are scored with the
shared 20.3 hybrid scorer, making the completeness/purity tradeoffs explicit.
"""
from __future__ import annotations

from dataclasses import dataclass
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
sys.path.insert(0, str(ROOT / "hybrid_ensemble"))
sys.path.insert(0, "/home/heruihua")

from csst_dla.scoring import C_KMS, score_catalog
from data import HybridTestDataset
from decode import pick_peaks, pixel_to_wavelength, softmax
from evaluate_hybrid import load_checkpoint, resolve_device
from score_test import _in_band, load_snr_by_target, load_truth
from csst_dla_wzx_pkg.data import feature_requires_flux_clean
from csst_dla_wzx_pkg.decode import select_candidates
from csst_dla_wzx_pkg.fits_io import load_test_fits
from csst_dla_wzx_pkg.inference import load_model_from_checkpoint, make_test_dataset_from_config


@dataclass(frozen=True)
class Candidate:
    z_dla: float
    lognhi: float
    confidence: float
    peak_index: int
    source: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-fits", required=True)
    parser.add_argument("--truth", required=True)
    parser.add_argument("--dilated-ckpt", required=True)
    parser.add_argument("--wzx-ckpt", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu", "mps"], default="auto")
    parser.add_argument("--dilated-batch-size", type=int, default=128)
    parser.add_argument("--wzx-batch-size", type=int, default=256)
    parser.add_argument("--dilated-threshold", type=float, default=0.45)
    parser.add_argument("--dilated-min-distance", type=int, default=10)
    parser.add_argument("--wzx-min-distance", type=int, default=4)
    parser.add_argument("--verify-heat", type=float, default=0.45)
    parser.add_argument("--verify-region", type=float, default=0.50)
    parser.add_argument("--agreement-dv", type=float, default=600.0)
    parser.add_argument("--min-lognhi", type=float, default=20.3)
    parser.add_argument("--max-lognhi", type=float, default=22.5)
    return parser.parse_args()


def _reorder(rows: list[np.ndarray], **values: list[np.ndarray]) -> dict[str, np.ndarray]:
    order = np.argsort(np.concatenate(rows).astype(np.int64))
    return {key: np.concatenate(parts, axis=0)[order] for key, parts in values.items()}


@torch.inference_mode()
def predict_dilated(model, dataset: HybridTestDataset, device: torch.device, batch_size: int) -> dict[str, np.ndarray]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=device.type == "cuda")
    rows: list[np.ndarray] = []
    heat: list[np.ndarray] = []
    region: list[np.ndarray] = []
    offset: list[np.ndarray] = []
    lognhi: list[np.ndarray] = []
    count: list[np.ndarray] = []
    model.eval()
    for x, row_idx in loader:
        out = model(x.to(device, non_blocking=True))
        rows.append(np.asarray(row_idx, dtype=np.int64))
        heat.append(torch.sigmoid(out["center_logits"]).cpu().numpy())
        region.append(torch.sigmoid(out["region_logits"]).cpu().numpy())
        offset.append(out["offset_raw"].cpu().numpy())
        lognhi.append((20.3 + out["lognhi_raw"]).cpu().numpy())
        count.append(torch.softmax(out["count_logits"], dim=1).cpu().numpy())
    return _reorder(rows, heat=heat, region=region, offset=offset, lognhi=lognhi, count=count)


@torch.inference_mode()
def predict_wzx(model, dataset, device: torch.device, batch_size: int) -> dict[str, np.ndarray]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=device.type == "cuda")
    rows: list[np.ndarray] = []
    heat: list[np.ndarray] = []
    region: list[np.ndarray] = []
    offset: list[np.ndarray] = []
    lognhi: list[np.ndarray] = []
    count: list[np.ndarray] = []
    model.eval()
    for batch in loader:
        out = model(
            batch["spectrum"].to(device, non_blocking=True),
            batch["z_qso"].to(device, non_blocking=True),
        )
        rows.append(batch["index"].cpu().numpy().astype(np.int64))
        heat.append(torch.sigmoid(out["heatmap_logits"]).cpu().numpy())
        region.append(torch.sigmoid(out["region_logits"]).cpu().numpy())
        offset.append(out["offset"].cpu().numpy())
        lognhi.append(out["lognhi"].cpu().numpy())
        count.append(torch.softmax(out["count_logits"], dim=1).cpu().numpy())
    return _reorder(rows, heat=heat, region=region, offset=offset, lognhi=lognhi, count=count)


def make_wzx_candidates(
    pred: dict[str, np.ndarray],
    row: int,
    wavelength: np.ndarray,
    z_qso: float,
    min_distance: int,
    min_lognhi: float,
    max_lognhi: float,
    source: str,
    *,
    heat: np.ndarray | None = None,
    region: np.ndarray | None = None,
    offset: np.ndarray | None = None,
    lognhi: np.ndarray | None = None,
    count: np.ndarray | None = None,
) -> list[Candidate]:
    _, selected = select_candidates(
        pred["heat"][row] if heat is None else heat,
        pred["region"][row] if region is None else region,
        pred["offset"][row] if offset is None else offset,
        pred["lognhi"][row] if lognhi is None else lognhi,
        pred["count"][row] if count is None else count,
        wavelength,
        z_qso,
        max_dlas=2,
        min_z=None,
        min_peak_distance=min_distance,
        confidence_threshold=None,
        lognhi_min=min_lognhi,
        lognhi_max=max_lognhi,
    )
    return [
        Candidate(
            z_dla=float(item.z_dla),
            lognhi=float(item.lognhi),
            confidence=float(item.confidence),
            peak_index=int(item.peak_index),
            source=source,
        )
        for item in selected
    ]


def make_dilated_candidates(
    pred: dict[str, np.ndarray],
    row: int,
    wavelength: np.ndarray,
    z_qso: float,
    threshold: float,
    min_distance: int,
    min_lognhi: float,
    max_lognhi: float,
) -> list[Candidate]:
    count_prob = pred["count"][row]
    n_pick = int(np.argmax(count_prob))
    peaks = pick_peaks(
        pred["heat"][row],
        wavelength,
        z_qso,
        n_pick=n_pick,
        threshold=threshold,
        min_distance=min_distance,
        min_z_dla=1.10,
    )
    return [
        Candidate(
            z_dla=float(pixel_to_wavelength(pix + float(pred["offset"][row, pix]), wavelength) / 1215.67 - 1.0),
            lognhi=float(np.clip(pred["lognhi"][row, pix], min_lognhi, max_lognhi)),
            confidence=float(pred["heat"][row, pix] * count_prob[n_pick]),
            peak_index=int(pix),
            source="dilated",
        )
        for pix in peaks
    ]


def apply_dilated_parameters(
    candidates: list[Candidate], pred: dict[str, np.ndarray], row: int, wavelength: np.ndarray,
    min_lognhi: float, max_lognhi: float,
) -> list[Candidate]:
    adjusted: list[Candidate] = []
    for candidate in candidates:
        pix = candidate.peak_index
        z_dla = pixel_to_wavelength(pix + float(pred["offset"][row, pix]), wavelength) / 1215.67 - 1.0
        adjusted.append(
            Candidate(
                z_dla=float(z_dla),
                lognhi=float(np.clip(pred["lognhi"][row, pix], min_lognhi, max_lognhi)),
                confidence=candidate.confidence,
                peak_index=pix,
                source=f"{candidate.source}+dilated_params",
            )
        )
    return adjusted


def dv_between(left: Candidate, right: Candidate) -> float:
    return C_KMS * abs(left.z_dla - right.z_dla) / (1.0 + 0.5 * (left.z_dla + right.z_dla))


def gate_candidates(candidates: list[Candidate], pred: dict[str, np.ndarray], row: int, heat: float | None, region: float | None) -> list[Candidate]:
    kept = []
    for candidate in candidates:
        pix = candidate.peak_index
        if heat is not None and float(pred["heat"][row, pix]) < heat:
            continue
        if region is not None and float(pred["region"][row, pix]) < region:
            continue
        kept.append(candidate)
    return kept


def supplement_candidates(primary: list[Candidate], secondary: list[Candidate], agreement_dv: float) -> list[Candidate]:
    result = list(primary)
    for candidate in secondary:
        if len(result) >= 2:
            break
        if all(dv_between(candidate, present) >= agreement_dv for present in result):
            result.append(candidate)
    return result


def candidate_stats(catalog: list[list[Candidate]]) -> dict[str, int]:
    return {
        "spectra_with_candidate": int(sum(bool(row) for row in catalog)),
        "candidate_count_before_score_filter": int(sum(len(row) for row in catalog)),
    }


def write_submission(path: Path, targetids: np.ndarray, catalog: list[list[Candidate]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "Z_DLA1", "LOGNHI1", "Z_DLA2", "LOGNHI2"])
        for targetid, candidates in zip(targetids, catalog):
            values: list[float] = []
            for candidate in candidates[:2]:
                values.extend([candidate.z_dla, candidate.lognhi])
            while len(values) < 4:
                values.extend([-1.0, 0.0])
            writer.writerow([int(targetid), *(f"{value:.6f}" for value in values)])


def score_candidates(
    catalog: list[list[Candidate]],
    targetids: np.ndarray,
    truth: dict[str, np.ndarray],
    snr_by_target: dict[int, float],
    min_lognhi: float,
) -> dict:
    prediction = {"TARGETID": [], "Z_DLA": [], "LOG_NHI": [], "CONFIDENCE": [], "SNR": []}
    for targetid, candidates in zip(targetids, catalog):
        targetid_int = int(targetid)
        for candidate in candidates[:2]:
            if candidate.z_dla <= 0 or candidate.lognhi < min_lognhi or not _in_band(candidate.z_dla, (2550.0, 4200.0)):
                continue
            prediction["TARGETID"].append(targetid_int)
            prediction["Z_DLA"].append(float(candidate.z_dla))
            prediction["LOG_NHI"].append(float(candidate.lognhi))
            prediction["CONFIDENCE"].append(float(candidate.confidence))
            prediction["SNR"].append(float(snr_by_target.get(targetid_int, np.nan)))
    for key in prediction:
        dtype = np.int64 if key == "TARGETID" else np.float32
        prediction[key] = np.asarray(prediction[key], dtype=dtype)
    result = score_catalog(truth, prediction)
    keys = [
        "final_score", "detection_score", "parameter_score", "completeness", "purity",
        "mean_dv", "std_dv", "mean_dlognhi", "std_dlognhi", "n_truth", "n_pred", "n_match", "bin_details",
    ]
    return {key: getattr(result, key) for key in keys}


def build_catalogs(
    dilated: dict[str, np.ndarray],
    wzx: dict[str, np.ndarray],
    wavelength: np.ndarray,
    z_qso: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, tuple[str, list[list[Candidate]]]]:
    experiments: dict[str, tuple[str, list[list[Candidate]]]] = {}
    names = {
        "wzx_replay": "WZX full replay; reference high-recall proposal model.",
        "dilated_replay": "Dilated full replay; reference high-purity verifier model.",
        "wzx_sel_dilated_count": "Only replace WZX count head with dilated count head.",
        "wzx_sel_dilated_heat": "Only replace WZX heatmap head with dilated heatmap head.",
        "wzx_sel_dilated_region": "Only replace WZX region head with dilated region head.",
        "wzx_sel_dilated_heat_region": "Replace WZX heatmap and region heads together.",
        "wzx_geom_heat_region": "Geometric fusion of both heatmap and region heads; keep WZX count/parameters.",
        "wzx_dilated_parameters": "Keep WZX proposals; replace only offset and LOGNHI parameters.",
        "wzx_di_heat_gate": "WZX proposals retained only where dilated heat is at its native 0.45 threshold.",
        "wzx_di_heat_gate_dilated_fallback": "Use a native dilated candidate only if no WZX proposal survives the heat gate.",
        "wzx_di_heat_gate_dilated_supplement": "Fill an unused heat-gated WZX slot with a distinct native dilated candidate.",
        "wzx_di_region_gate": "WZX proposals retained only where dilated region is at 0.50.",
        "wzx_di_heat_region_gate": "WZX proposals pass both dilated heat and region gates.",
        "wzx_di_count_gate": "WZX proposals truncated by the dilated count prediction.",
        "wzx_di_all_gate": "Count gate plus dilated heat/region verification.",
        "wzx_dilated_agree": "Keep only WZX proposals independently found by dilated within 600 km/s.",
        "wzx_dilated_fallback": "Use dilated only on spectra where WZX proposed no candidate.",
        "wzx_dilated_supplement": "Keep WZX candidates, then fill unused slot with a distinct dilated candidate.",
    }
    catalogs = {key: [] for key in names}
    for row, z_value in enumerate(z_qso):
        base_wzx = make_wzx_candidates(
            wzx, row, wavelength, float(z_value), args.wzx_min_distance, args.min_lognhi, args.max_lognhi, "wzx"
        )
        base_dilated = make_dilated_candidates(
            dilated, row, wavelength, float(z_value), args.dilated_threshold, args.dilated_min_distance,
            args.min_lognhi, args.max_lognhi,
        )
        catalogs["wzx_replay"].append(base_wzx)
        catalogs["dilated_replay"].append(base_dilated)
        catalogs["wzx_sel_dilated_count"].append(make_wzx_candidates(
            wzx, row, wavelength, float(z_value), args.wzx_min_distance, args.min_lognhi, args.max_lognhi,
            "wzx_sel_dilated_count", count=dilated["count"][row],
        ))
        catalogs["wzx_sel_dilated_heat"].append(make_wzx_candidates(
            wzx, row, wavelength, float(z_value), args.wzx_min_distance, args.min_lognhi, args.max_lognhi,
            "wzx_sel_dilated_heat", heat=dilated["heat"][row],
        ))
        catalogs["wzx_sel_dilated_region"].append(make_wzx_candidates(
            wzx, row, wavelength, float(z_value), args.wzx_min_distance, args.min_lognhi, args.max_lognhi,
            "wzx_sel_dilated_region", region=dilated["region"][row],
        ))
        catalogs["wzx_sel_dilated_heat_region"].append(make_wzx_candidates(
            wzx, row, wavelength, float(z_value), args.wzx_min_distance, args.min_lognhi, args.max_lognhi,
            "wzx_sel_dilated_heat_region", heat=dilated["heat"][row], region=dilated["region"][row],
        ))
        catalogs["wzx_geom_heat_region"].append(make_wzx_candidates(
            wzx, row, wavelength, float(z_value), args.wzx_min_distance, args.min_lognhi, args.max_lognhi,
            "wzx_geom_heat_region",
            heat=np.sqrt(np.maximum(wzx["heat"][row], 0.0) * np.maximum(dilated["heat"][row], 0.0)),
            region=np.sqrt(np.maximum(wzx["region"][row], 0.0) * np.maximum(dilated["region"][row], 0.0)),
        ))
        catalogs["wzx_dilated_parameters"].append(apply_dilated_parameters(
            base_wzx, dilated, row, wavelength, args.min_lognhi, args.max_lognhi
        ))
        heat_gated = gate_candidates(base_wzx, dilated, row, args.verify_heat, None)
        region_gated = gate_candidates(base_wzx, dilated, row, None, args.verify_region)
        both_gated = gate_candidates(base_wzx, dilated, row, args.verify_heat, args.verify_region)
        catalogs["wzx_di_heat_gate"].append(heat_gated)
        catalogs["wzx_di_heat_gate_dilated_fallback"].append(heat_gated if heat_gated else base_dilated)
        catalogs["wzx_di_heat_gate_dilated_supplement"].append(
            supplement_candidates(heat_gated, base_dilated, args.agreement_dv)
        )
        catalogs["wzx_di_region_gate"].append(region_gated)
        catalogs["wzx_di_heat_region_gate"].append(both_gated)
        count_limited = base_wzx[: int(np.argmax(dilated["count"][row]))]
        catalogs["wzx_di_count_gate"].append(count_limited)
        catalogs["wzx_di_all_gate"].append(both_gated[: int(np.argmax(dilated["count"][row]))])
        catalogs["wzx_dilated_agree"].append([
            candidate for candidate in base_wzx
            if any(dv_between(candidate, other) < args.agreement_dv for other in base_dilated)
        ])
        catalogs["wzx_dilated_fallback"].append(base_wzx if base_wzx else base_dilated)
        catalogs["wzx_dilated_supplement"].append(supplement_candidates(base_wzx, base_dilated, args.agreement_dv))
    for key, description in names.items():
        experiments[key] = (description, catalogs[key])
    return experiments


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    print(f"device={device}")
    print("Loading dilated checkpoint and test inputs...")
    dilated_model, dilated_config = load_checkpoint(args.dilated_ckpt, device)
    dilated_ds = HybridTestDataset(args.test_fits, dilated_config["input_mode"])
    dilated = predict_dilated(dilated_model, dilated_ds, device, args.dilated_batch_size)
    print(f"dilated rows={len(dilated_ds)}")
    print("Loading WZX checkpoint and test inputs...")
    wzx_model, wzx_config = load_model_from_checkpoint(args.wzx_ckpt, device)
    wavelength, flux, flux_clean, meta = load_test_fits(
        args.test_fits,
        load_flux_clean=feature_requires_flux_clean(wzx_config.get("feature_mode", "all")),
    )
    wzx_ds = make_test_dataset_from_config(flux, flux_clean, meta, wzx_config)
    wzx = predict_wzx(wzx_model, wzx_ds, device, args.wzx_batch_size)
    if not np.allclose(dilated_ds.wavelength, wavelength):
        raise RuntimeError("The two models do not use the same wavelength grid.")
    if not np.array_equal(dilated_ds.targetid, meta["TARGETID"].to_numpy(dtype=np.int64)):
        raise RuntimeError("The two test readers disagree on TARGETID ordering.")
    print(f"wzx rows={len(wzx_ds)}")
    experiments = build_catalogs(dilated, wzx, wavelength, dilated_ds.zq, args)
    truth = load_truth(args.truth, args.min_lognhi, "SNR_GU")
    snr_by_target = load_snr_by_target(args.truth, "SNR_GU")
    summaries: list[dict] = []
    for name, (description, catalog) in experiments.items():
        csv_path = out_dir / f"{name}_predictions.csv"
        score_path = out_dir / f"{name}_score_20p3.json"
        write_submission(csv_path, dilated_ds.targetid, catalog)
        score = score_candidates(catalog, dilated_ds.targetid, truth, snr_by_target, args.min_lognhi)
        score_path.write_text(json.dumps(score, indent=2, allow_nan=False), encoding="utf-8")
        entry = {"name": name, "description": description, **candidate_stats(catalog), "score": score}
        summaries.append(entry)
        print(
            f"{name:28s} final={score['final_score']:.6f} "
            f"comp={score['completeness']:.4f} purity={score['purity']:.4f} "
            f"pred={score['n_pred']} match={score['n_match']}"
        )
    reference = next(item for item in summaries if item["name"] == "wzx_replay")["score"]
    for item in summaries:
        score = item["score"]
        item["delta_vs_wzx_replay"] = {
            key: float(score[key] - reference[key])
            for key in ("final_score", "detection_score", "parameter_score", "completeness", "purity")
        }
    payload = {
        "purpose": "fixed-rule WZX/dilated decoder-head fusion study; no test-set parameter search",
        "args": vars(args),
        "dilated_config": dilated_config,
        "wzx_config": wzx_config,
        "experiments": summaries,
    }
    (out_dir / "fusion_summary.json").write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    (out_dir / "fusion_rules.md").write_text(
        "# Fixed late-fusion study\n\n"
        "All catalogs use the shared `score_test.py` semantics at LOGNHI >= 20.3 and SNR_GU. "
        "The 0.45 heat gate is the dilated model's native decoder threshold; the 0.50 region gate is fixed before scoring. "
        "No fusion threshold was selected from these test results.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
