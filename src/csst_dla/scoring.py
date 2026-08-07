from __future__ import annotations

from dataclasses import dataclass

import numpy as np

C_KMS = 299792.458


@dataclass
class ScoreResult:
    detection_score: float
    parameter_score: float
    final_score: float
    completeness: float
    purity: float
    mean_dv: float
    std_dv: float
    mean_dlognhi: float
    std_dlognhi: float
    n_truth: int
    n_pred: int
    n_match: int
    bin_details: list[dict]


def labels_to_truth(
    labels: dict[str, np.ndarray],
    indices: np.ndarray,
    min_lognhi: float = 20.3,
    snr_field: str = "SNR_GU",
) -> dict[str, np.ndarray]:
    targetid: list[int] = []
    z_qso: list[float] = []
    z_dla: list[float] = []
    log_nhi: list[float] = []
    snr: list[float] = []

    for idx in np.asarray(indices, dtype=int):
        for slot in (1, 2):
            if labels["N_DLA"][idx] < slot:
                continue
            z = float(labels[f"Z_DLA{slot}"][idx])
            logn = float(labels[f"LOGNHI{slot}"][idx])
            if logn < min_lognhi:
                continue
            targetid.append(int(idx))
            z_qso.append(float(labels["Z_QSO"][idx]))
            z_dla.append(z)
            log_nhi.append(logn)
            snr.append(float(labels[snr_field][idx]))

    return {
        "TARGETID": np.asarray(targetid, dtype=np.int64),
        "Z_QSO": np.asarray(z_qso, dtype=np.float32),
        "Z_DLA": np.asarray(z_dla, dtype=np.float32),
        "LOG_NHI": np.asarray(log_nhi, dtype=np.float32),
        "SNR": np.asarray(snr, dtype=np.float32),
    }


def greedy_match(
    truth: dict[str, np.ndarray], pred: dict[str, np.ndarray], dv_limit: float = 600.0
) -> list[tuple[int, int, float]]:
    candidates: list[tuple[float, int, int]] = []
    for ti, target in enumerate(truth["TARGETID"]):
        pred_idx = np.flatnonzero(pred["TARGETID"] == target)
        if pred_idx.size == 0:
            continue
        dv = C_KMS * np.abs(pred["Z_DLA"][pred_idx] - truth["Z_DLA"][ti]) / (
            1.0 + truth["Z_DLA"][ti]
        )
        for pi, value in zip(pred_idx, dv):
            if value < dv_limit:
                candidates.append((float(value), ti, int(pi)))

    candidates.sort(key=lambda item: item[0])
    used_truth: set[int] = set()
    used_pred: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for dv, ti, pi in candidates:
        if ti in used_truth or pi in used_pred:
            continue
        used_truth.add(ti)
        used_pred.add(pi)
        matches.append((ti, pi, dv))
    return matches


def score_catalog(
    truth: dict[str, np.ndarray],
    pred: dict[str, np.ndarray],
    snr_bins=(0, 1, 2, 3, 4, 5, 6, 7, np.inf),
    lognhi_bins=(20.3, 20.5, 21.0, 21.5, 22.0, np.inf),
) -> ScoreResult:
    matches = greedy_match(truth, pred)
    truth_matched = np.zeros(len(truth["TARGETID"]), dtype=bool)
    pred_matched = np.zeros(len(pred["TARGETID"]), dtype=bool)
    for ti, pi, _ in matches:
        truth_matched[ti] = True
        pred_matched[pi] = True

    weighted_f1 = 0.0
    total_weight = 0
    bin_details: list[dict] = []
    for snr_lo, snr_hi in zip(snr_bins[:-1], snr_bins[1:]):
        for log_lo, log_hi in zip(lognhi_bins[:-1], lognhi_bins[1:]):
            truth_bin = (
                (truth["SNR"] >= snr_lo)
                & (truth["SNR"] < snr_hi)
                & (truth["LOG_NHI"] >= log_lo)
                & (truth["LOG_NHI"] < log_hi)
            )
            n_truth = int(truth_bin.sum())

            # Predicted DLAs are binned by the S/N of their parent spectrum and
            # by their predicted LOG_NHI. This includes false positives on
            # sightlines with no true DLA, which is essential for purity.
            if "SNR" in pred:
                pred_snr = pred["SNR"]
            else:
                snr_by_target = {
                    int(target): float(snr)
                    for target, snr in zip(truth["TARGETID"], truth["SNR"])
                }
                pred_snr = np.asarray(
                    [snr_by_target.get(int(target), np.nan) for target in pred["TARGETID"]],
                    dtype=np.float32,
                )
            pred_bin = (
                (pred_snr >= snr_lo)
                & (pred_snr < snr_hi)
                & (pred["LOG_NHI"] >= log_lo)
                & (pred["LOG_NHI"] < log_hi)
            )
            n_pred_bin = int(pred_bin.sum())
            n_match_truth_bin = int((truth_bin & truth_matched).sum())
            n_match_pred_bin = int((pred_bin & pred_matched).sum())
            completeness = float(n_match_truth_bin / n_truth) if n_truth else 0.0
            purity = (
                float(n_match_pred_bin / n_pred_bin)
                if n_pred_bin
                else 0.0
            )
            f1 = (
                2.0 * completeness * purity / (completeness + purity)
                if completeness + purity > 0
                else 0.0
            )
            weight = n_truth
            weighted_f1 += f1 * weight
            total_weight += n_truth
            if n_truth > 0:
                bin_details.append(
                    {
                        "snr_bin": _format_bin(snr_lo, snr_hi),
                        "snr_lo": _finite_float(snr_lo),
                        "snr_hi": _finite_float(snr_hi),
                        "lognhi_bin": _format_bin(log_lo, log_hi),
                        "lognhi_lo": _finite_float(log_lo),
                        "lognhi_hi": _finite_float(log_hi),
                        "n_truth": n_truth,
                        "n_pred": n_pred_bin,
                        "n_match_truth": n_match_truth_bin,
                        "n_match_pred": n_match_pred_bin,
                        "n_match": min(n_match_truth_bin, n_match_pred_bin),
                        "completeness": completeness,
                        "purity": purity,
                        "f1": f1,
                        "weight": weight,
                        "weighted_f1": f1 * weight,
                    }
                )

    if matches:
        ti = np.asarray([m[0] for m in matches], dtype=int)
        pi = np.asarray([m[1] for m in matches], dtype=int)
        dv = C_KMS * (pred["Z_DLA"][pi] - truth["Z_DLA"][ti]) / (1.0 + truth["Z_DLA"][ti])
        dlog = pred["LOG_NHI"][pi] - truth["LOG_NHI"][ti]
        mean_dv = float(np.mean(dv))
        std_dv = float(np.std(dv))
        mean_dlog = float(np.mean(dlog))
        std_dlog = float(np.std(dlog))
        score_z = float(np.exp(-std_dv / 300.0) * np.exp(-abs(mean_dv) / 150.0))
        score_nhi = float(np.exp(-std_dlog / 0.25) * np.exp(-abs(mean_dlog) / 0.1))
        parameter_score = 0.5 * (score_z + score_nhi)
    else:
        mean_dv = std_dv = mean_dlog = std_dlog = 0.0
        parameter_score = 0.0

    n_truth_total = len(truth["TARGETID"])
    n_pred_total = len(pred["TARGETID"])
    n_match_total = len(matches)
    detection_score = float(weighted_f1 / total_weight) if total_weight else 0.0
    completeness = float(n_match_total / n_truth_total) if n_truth_total else 0.0
    purity = float(n_match_total / n_pred_total) if n_pred_total else 0.0
    final_score = 0.6 * detection_score + 0.4 * parameter_score

    return ScoreResult(
        detection_score=detection_score,
        parameter_score=parameter_score,
        final_score=final_score,
        completeness=completeness,
        purity=purity,
        mean_dv=mean_dv,
        std_dv=std_dv,
        mean_dlognhi=mean_dlog,
        std_dlognhi=std_dlog,
        n_truth=n_truth_total,
        n_pred=n_pred_total,
        n_match=n_match_total,
        bin_details=bin_details,
    )


def _finite_float(value: float):
    return None if np.isinf(value) else float(value)


def _format_bin(lo: float, hi: float) -> str:
    hi_text = "inf" if np.isinf(hi) else f"{hi:g}"
    return f"[{lo:g},{hi_text})"
