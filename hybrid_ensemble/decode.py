from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from csst_dla.scoring import greedy_match, labels_to_truth
from csst_dla.targets import LYA


@torch.no_grad()
def predict_member(model, loader: DataLoader, device: torch.device) -> dict[str, np.ndarray]:
    model.eval()
    heatmap = []
    lognhi = []
    offset = []
    count_logits = []
    # count_prob = []
    rows = []
    for batch in loader:
        x = batch[0].to(device)
        row_idx = batch[-1]
        out = model(x)
        heatmap.append(torch.sigmoid(out["center_logits"]).cpu().numpy())
        lognhi.append((20.3 + out["lognhi_raw"]).cpu().numpy())
        if "offset_raw" in out:
            offset.append(out["offset_raw"].cpu().numpy())
        count_logits.append(out["count_logits"].cpu().numpy())
        # logits = out["count_logits"].cpu()
        # count_logits.append(logits.numpy())
        # count_prob.append(torch.softmax(logits, dim=1).numpy())
        if hasattr(row_idx, "numpy"):
            rows.append(row_idx.numpy())
        else:
            rows.append(np.asarray(row_idx))
    result = {
        "heatmap": np.concatenate(heatmap),
        "lognhi": np.concatenate(lognhi),
        "count_logits": np.concatenate(count_logits),
        # "count_prob": np.concatenate(count_prob),
        "rows": np.concatenate(rows).astype(np.int64),
    }
    if offset:
        result["offset"] = np.concatenate(offset)
    return result


def average_predictions(preds: list[dict[str, np.ndarray]], weights: list[float] | None = None) -> dict[str, np.ndarray]:
    if weights is None:
        weights = [1.0] * len(preds)
    w = np.asarray(weights, dtype=np.float32)
    w = w / w.sum()
    out = {
        "heatmap": sum(float(wi) * pred["heatmap"] for wi, pred in zip(w, preds)),
        "lognhi": sum(float(wi) * pred["lognhi"] for wi, pred in zip(w, preds)),
        "count_logits": sum(float(wi) * pred["count_logits"] for wi, pred in zip(w, preds)),
        "rows": preds[0]["rows"],
    }
    # if all("count_prob" in pred for pred in preds):
    #     out["count_prob"] = sum(float(wi) * pred["count_prob"] for wi, pred in zip(w, preds))
    if all("offset" in pred for pred in preds):
        out["offset"] = sum(float(wi) * pred["offset"] for wi, pred in zip(w, preds))
    return out


def softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max(axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / ez.sum(axis=1, keepdims=True)


def pick_peaks(
    prob: np.ndarray,
    wavelength: np.ndarray,
    z_qso: float,
    n_pick: int,
    threshold: float = 0.45,
    min_distance: int = 10,
    min_z_dla: float = 1.55,
) -> list[int]:
    if n_pick <= 0:
        return []
    candidates = np.flatnonzero(
        (prob[1:-1] > threshold) & (prob[1:-1] >= prob[:-2]) & (prob[1:-1] >= prob[2:])
    ) + 1
    if candidates.size == 0:
        return []
    order = candidates[np.argsort(prob[candidates])[::-1]]
    kept: list[int] = []
    for pix in order:
        pix = int(pix)
        z_dla = float(wavelength[pix] / LYA - 1.0)
        if z_dla < min_z_dla or z_dla >= z_qso - 0.02:
            continue
        if all(abs(pix - old) >= min_distance for old in kept):
            kept.append(pix)
        if len(kept) >= n_pick:
            break
    return kept


def soft_peak_wavelength(
    prob: np.ndarray,
    wavelength: np.ndarray,
    pix: int,
    radius: int = 0,
    power: float = 3.0,
) -> float:
    if radius <= 0:
        return float(wavelength[pix])
    lo = max(0, pix - radius)
    hi = min(len(wavelength), pix + radius + 1)
    local_prob = np.maximum(prob[lo:hi].astype(np.float64), 0.0)
    weights = np.power(local_prob, power)
    if not np.isfinite(weights).all() or float(weights.sum()) <= 0.0:
        return float(wavelength[pix])
    return float(np.sum(wavelength[lo:hi].astype(np.float64) * weights) / weights.sum())


def pixel_to_wavelength(pixel: float, wavelength: np.ndarray) -> float:
    pixel_grid = np.arange(len(wavelength), dtype=np.float64)
    pixel = float(np.clip(pixel, 0.0, len(wavelength) - 1.0))
    return float(np.interp(pixel, pixel_grid, wavelength.astype(np.float64)))


def decode_validation_catalog(
    pred: dict[str, np.ndarray],
    indices: np.ndarray,
    labels: dict[str, np.ndarray],
    wavelength: np.ndarray,
    threshold: float = 0.45,
    min_distance: int = 10,
    min_z_dla: float = 1.55,
    count_bias: list[float] | None = None,
    count_min_prob: float = 0.0,
    lognhi_min: float = 19.5,
    lognhi_max: float = 22.5,
    lognhi_slope: float = 1.0,
    lognhi_intercept: float = 0.0,
    velocity_offset_kms: float = 0.0,
    soft_radius: int = 0,
    soft_power: float = 3.0,
    use_offset: bool = True,
    low_lognhi_threshold: float | None = None,
    low_lognhi_max: float = 20.5,
    low_lognhi_snr_max: float = 3.0,
) -> dict[str, np.ndarray]:
    rows = []
    bias = np.asarray(count_bias if count_bias is not None else [0.0, 0.0, 0.0], dtype=np.float32)
    count_prob = softmax(pred["count_logits"] + bias[None, :])
    # if "count_prob" in pred and np.allclose(bias, 0.0):
    #     count_prob = pred["count_prob"]
    # else:
    #     count_prob = softmax(pred["count_logits"] + bias[None, :])
    for row, idx in enumerate(indices.astype(int)):
        n_pick = int(np.argmax(count_prob[row]))
        if n_pick > 0 and float(count_prob[row, n_pick]) < count_min_prob:
            n_pick = 0
        peaks = pick_peaks(
            pred["heatmap"][row],
            wavelength,
            float(labels["Z_QSO"][idx]),
            n_pick=n_pick,
            threshold=threshold,
            min_distance=min_distance,
            min_z_dla=min_z_dla,
        )
        for pix in peaks:
            raw_log = float(pred["lognhi"][row, pix])
            if use_offset and "offset" in pred:
                lambda_dla = pixel_to_wavelength(pix + float(pred["offset"][row, pix]), wavelength)
            else:
                lambda_dla = soft_peak_wavelength(
                    pred["heatmap"][row],
                    wavelength,
                    pix,
                    radius=soft_radius,
                    power=soft_power,
                )
            z_dla = float(lambda_dla / LYA - 1.0)
            if velocity_offset_kms:
                z_dla = z_dla + velocity_offset_kms / 299792.458 * (1.0 + z_dla)
            lognhi_out = float(np.clip(lognhi_slope * raw_log + lognhi_intercept, lognhi_min, lognhi_max))
            confidence = float(pred["heatmap"][row, pix] * count_prob[row, n_pick])
            snr = float(labels["SNR_GU"][idx])
            if (
                low_lognhi_threshold is not None
                and lognhi_out < low_lognhi_max
                and snr < low_lognhi_snr_max
                and confidence < low_lognhi_threshold
            ):
                continue
            rows.append(
                {
                    "TARGETID": int(idx),
                    "Z_QSO": float(labels["Z_QSO"][idx]),
                    "Z_DLA": z_dla,
                    "LOG_NHI": lognhi_out,
                    "CONFIDENCE": confidence,
                    "SNR": snr,
                }
            )
    return catalog_from_rows(rows)


def catalog_from_rows(rows: list[dict]) -> dict[str, np.ndarray]:
    if not rows:
        return {
            "TARGETID": np.asarray([], dtype=np.int64),
            "Z_QSO": np.asarray([], dtype=np.float32),
            "Z_DLA": np.asarray([], dtype=np.float32),
            "LOG_NHI": np.asarray([], dtype=np.float32),
            "CONFIDENCE": np.asarray([], dtype=np.float32),
            "SNR": np.asarray([], dtype=np.float32),
        }
    return {
        key: np.asarray([row[key] for row in rows], dtype=np.int64 if key == "TARGETID" else np.float32)
        for key in ["TARGETID", "Z_QSO", "Z_DLA", "LOG_NHI", "CONFIDENCE", "SNR"]
    }


def fit_lognhi_calibration(
    pred: dict[str, np.ndarray],
    indices: np.ndarray,
    labels: dict[str, np.ndarray],
    wavelength: np.ndarray,
    threshold: float,
    min_z_dla: float = 1.55,
    count_bias: list[float] | None = None,
    count_min_prob: float = 0.0,
    lognhi_min: float = 19.5,
    lognhi_max: float = 22.5,
    velocity_offset_kms: float = 0.0,
    soft_radius: int = 0,
    soft_power: float = 3.0,
    use_offset: bool = True,
    low_lognhi_threshold: float | None = None,
    low_lognhi_max: float = 20.5,
    low_lognhi_snr_max: float = 3.0,
) -> tuple[float, float, int]:
    raw_catalog = decode_validation_catalog(
        pred,
        indices,
        labels,
        wavelength,
        threshold=threshold,
        min_z_dla=min_z_dla,
        count_bias=count_bias,
        count_min_prob=count_min_prob,
        lognhi_min=lognhi_min,
        lognhi_max=lognhi_max,
        velocity_offset_kms=velocity_offset_kms,
        soft_radius=soft_radius,
        soft_power=soft_power,
        use_offset=use_offset,
        low_lognhi_threshold=low_lognhi_threshold,
        low_lognhi_max=low_lognhi_max,
        low_lognhi_snr_max=low_lognhi_snr_max,
    )
    truth = labels_to_truth(labels, indices, min_lognhi=lognhi_min)
    matches = greedy_match(truth, raw_catalog)
    if len(matches) < 3:
        return 1.0, 0.0, len(matches)
    true_log = np.asarray([truth["LOG_NHI"][ti] for ti, _, _ in matches], dtype=np.float32)
    pred_log = np.asarray([raw_catalog["LOG_NHI"][pi] for _, pi, _ in matches], dtype=np.float32)
    slope, intercept = np.polyfit(pred_log, true_log, deg=1)
    return float(slope), float(intercept), len(matches)
