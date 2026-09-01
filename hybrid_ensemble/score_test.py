#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits


ROOT = Path(__file__).resolve().parents[1]


LYA_REST = 1215.67
BAND_BY_SNR_FIELD = {
    "SNR_GU": (2550.0, 4200.0),
    "SNR_GV": (4000.0, 6500.0),
    "SNR_GI": (6200.0, 10000.0),
}


def _in_band(z: float, band: tuple[float, float]) -> bool:
    wavelength = LYA_REST * (1.0 + z)
    return band[0] <= wavelength < band[1]


def load_truth(
    path: str | Path,
    min_lognhi: float = 20.3,
    snr_field: str = "SNR_GU",
) -> dict[str, np.ndarray]:
    with fits.open(path, memmap=True) as hdul:
        raw = hdul["TRUTH"].data
        targetid: list[int] = []
        z_qso: list[float] = []
        z_dla: list[float] = []
        log_nhi: list[float] = []
        snr: list[float] = []
        for row in raw:
            for slot in (1, 2):
                if int(row[f"N_DLA"]) < slot:
                    continue
                value_lognhi = float(row[f"LOGNHI{slot}"])
                if value_lognhi < min_lognhi:
                    continue
                value_z_dla = float(row[f"Z_DLA{slot}"])
                if not _in_band(value_z_dla, BAND_BY_SNR_FIELD[snr_field]):
                    continue
                targetid.append(int(row["TARGETID"]))
                z_qso.append(float(row["Z_QSO"]))
                z_dla.append(value_z_dla)
                log_nhi.append(value_lognhi)
                snr.append(float(row[snr_field]))

    return {
        "TARGETID": np.asarray(targetid, dtype=np.int64),
        "Z_QSO": np.asarray(z_qso, dtype=np.float32),
        "Z_DLA": np.asarray(z_dla, dtype=np.float32),
        "LOG_NHI": np.asarray(log_nhi, dtype=np.float32),
        "SNR": np.asarray(snr, dtype=np.float32),
    }


def load_snr_by_target(path: str | Path, snr_field: str = "SNR_GU") -> dict[int, float]:
    with fits.open(path, memmap=True) as hdul:
        raw = hdul["TRUTH"].data
        return {int(row["TARGETID"]): float(row[snr_field]) for row in raw}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--truth", default="data/new/test_truth.fits")
    parser.add_argument("--out", required=True)
    parser.add_argument("--min-lognhi", type=float, default=20.3)
    parser.add_argument("--snr-field", choices=tuple(BAND_BY_SNR_FIELD), default="SNR_GU")
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT / "src"))
    from csst_dla.scoring import score_catalog

    snr_by_target = load_snr_by_target(args.truth, args.snr_field)
    catalog = np.genfromtxt(
        args.predictions,
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8",
    )
    if catalog.ndim == 0:
        catalog = np.asarray([catalog])

    prediction = {
        "TARGETID": [],
        "Z_QSO": [],
        "Z_DLA": [],
        "LOG_NHI": [],
        "CONFIDENCE": [],
        "SNR": [],
    }
    for row in catalog:
        target_id = int(row["id"])
        for slot in (1, 2):
            z = float(row[f"Z_DLA{slot}"])
            lognhi = float(row[f"LOGNHI{slot}"])
            if (
                z <= 0
                or lognhi < args.min_lognhi
                or not _in_band(z, BAND_BY_SNR_FIELD[args.snr_field])
            ):
                continue
            prediction["TARGETID"].append(target_id)
            prediction["Z_DLA"].append(z)
            prediction["LOG_NHI"].append(lognhi)
            prediction["CONFIDENCE"].append(1.0)
            prediction["SNR"].append(snr_by_target.get(target_id, np.nan))
    # The scorer only needs Z_QSO when it is present in the prediction;
    # omitting it lets score_catalog inherit the parent quasar from truth.
    del prediction["Z_QSO"]
    prediction["TARGETID"] = np.asarray(prediction["TARGETID"], dtype=np.int64)
    prediction["Z_DLA"] = np.asarray(prediction["Z_DLA"], dtype=np.float32)
    prediction["LOG_NHI"] = np.asarray(prediction["LOG_NHI"], dtype=np.float32)
    prediction["CONFIDENCE"] = np.asarray(prediction["CONFIDENCE"], dtype=np.float32)
    prediction["SNR"] = np.asarray(prediction["SNR"], dtype=np.float32)
    result = score_catalog(
        load_truth(args.truth, args.min_lognhi, args.snr_field),
        prediction,
    )
    keys = [
        "final_score", "detection_score", "parameter_score",
        "completeness", "purity", "mean_dv", "std_dv",
        "mean_dlognhi", "std_dlognhi", "n_truth", "n_pred", "n_match",
        "bin_details",
    ]
    payload = {key: getattr(result, key) for key in keys}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k != "bin_details"}, indent=2))


if __name__ == "__main__":
    main()
