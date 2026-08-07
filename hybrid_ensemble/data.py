from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from csst_dla.fits_utils import read_image, read_labels, read_meta
from csst_dla.snr import estimate_snr_gu_proxy
from csst_dla.targets import LYA


def normalize_with_scale(flux: np.ndarray, scale: np.ndarray, eps: float = 1e-30) -> np.ndarray:
    x = flux / np.maximum(scale, eps)
    x = np.nan_to_num(x, nan=1.0, posinf=1.0, neginf=1.0)
    return np.clip(x, 0.0, 3.0).astype(np.float32)


def smooth_flux(flux: np.ndarray, width: int = 15) -> np.ndarray:
    kernel = np.ones(width, dtype=np.float32) / float(width)
    return np.convolve(flux, kernel, mode="same").astype(np.float32)


def wavelength_norm(wavelength: np.ndarray) -> np.ndarray:
    wave_min = float(wavelength.min())
    wave_max = float(wavelength.max())
    return (2.0 * (wavelength.astype(np.float32) - wave_min) / (wave_max - wave_min) - 1.0).astype(
        np.float32
    )


def make_offset_targets(
    wavelength: np.ndarray,
    labels: dict[str, np.ndarray],
    indices: np.ndarray,
    radius_pixels: int = 3,
    min_lognhi: float = 20.3,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.asarray(indices, dtype=int)
    n_spec = len(indices)
    n_wave = len(wavelength)
    pixel = np.arange(n_wave, dtype=np.float32)
    offset = np.zeros((n_spec, n_wave), dtype=np.float32)
    weight = np.zeros((n_spec, n_wave), dtype=np.float32)

    for row, idx in enumerate(indices):
        for slot in (1, 2):
            if labels["N_DLA"][idx] < slot:
                continue
            lognhi = float(labels[f"LOGNHI{slot}"][idx])
            if lognhi < min_lognhi:
                continue
            lambda_dla = LYA * (1.0 + float(labels[f"Z_DLA{slot}"][idx]))
            center_float = float(np.interp(lambda_dla, wavelength, pixel))
            center_nearest = int(round(center_float))
            lo = max(0, center_nearest - radius_pixels)
            hi = min(n_wave, center_nearest + radius_pixels + 1)
            local_pixel = pixel[lo:hi]
            local_weight = np.exp(-0.5 * ((local_pixel - center_float) / 1.5) ** 2).astype(np.float32)
            offset_view = offset[row, lo:hi]
            weight_view = weight[row, lo:hi]
            better = local_weight > weight_view
            offset_view[better] = (center_float - local_pixel[better]).astype(np.float32)
            weight_view[better] = local_weight[better]

    return offset, weight


def make_scored_count_targets(
    labels: dict[str, np.ndarray],
    indices: np.ndarray,
    min_lognhi: float = 20.3,
) -> np.ndarray:
    counts = np.zeros(len(indices), dtype=np.int64)
    for row, idx in enumerate(np.asarray(indices, dtype=int)):
        count = 0
        for slot in (1, 2):
            if labels["N_DLA"][idx] >= slot and float(labels[f"LOGNHI{slot}"][idx]) >= min_lognhi:
                count += 1
        counts[row] = min(count, 2)
    return counts


def build_channels(
    flux: np.ndarray,
    clean: np.ndarray | None,
    wavelength: np.ndarray,
    wave_norm: np.ndarray,
    z_qso: float,
    snr: float,
    input_mode: str,
) -> np.ndarray:
    zq_channel = np.full_like(flux, z_qso, dtype=np.float32)
    snr_channel = np.full_like(flux, snr, dtype=np.float32)
    blue_mask = (wavelength < LYA * (1.0 + z_qso)).astype(np.float32)
    smooth = smooth_flux(flux, 15)
    if input_mode == "flux":
        channels = [flux, smooth, zq_channel, snr_channel, wave_norm, blue_mask]
    elif input_mode == "residual":
        if clean is None:
            raise ValueError("input_mode=residual requires FLUX_CLEAN")
        residual = clean - flux
        channels = [flux, clean, residual, zq_channel, snr_channel, wave_norm, blue_mask]
    elif input_mode == "all":
        if clean is None:
            raise ValueError("input_mode=all requires FLUX_CLEAN")
        residual = clean - flux
        channels = [flux, clean, residual, smooth, zq_channel, snr_channel, wave_norm, blue_mask]
    else:
        raise ValueError(f"unknown input_mode: {input_mode}")
    return np.stack(channels, axis=0).astype(np.float32)


class HybridTrainDataset(Dataset):
    def __init__(
        self,
        targets_npz: str | Path,
        train_fits: str | Path,
        split: str,
        input_mode: str = "all",
        max_samples: int | None = None,
    ):
        self.data = np.load(targets_npz)
        self.labels = read_labels(train_fits)
        self.input_mode = input_mode
        self.wavelength = self.data["wavelength"].astype(np.float32)
        self.wave_norm = wavelength_norm(self.wavelength)
        self.indices = self.data[f"{split}_idx"]
        self.center = self.data[f"{split}_center"]
        self.region = self.data[f"{split}_region"] if f"{split}_region" in self.data.files else self.center
        self.lognhi = self.data[f"{split}_lognhi"]
        self.mask = self.data[f"{split}_mask"]
        self.offset, self.offset_weight = make_offset_targets(self.wavelength, self.labels, self.indices)
        if max_samples is not None:
            self.indices = self.indices[:max_samples]
            self.center = self.center[:max_samples]
            self.region = self.region[:max_samples]
            self.lognhi = self.lognhi[:max_samples]
            self.mask = self.mask[:max_samples]
            self.offset = self.offset[:max_samples]
            self.offset_weight = self.offset_weight[:max_samples]
        self.x_all = self.data["x"]
        self.clean_all = self.data["x_clean"] if "x_clean" in self.data.files else None
        self.snr = self.data["snr_proxy"][self.indices].astype(np.float32)
        self.zq = self.labels["Z_QSO"][self.indices].astype(np.float32)
        min_lognhi = float(self.data["target_min_lognhi"]) if "target_min_lognhi" in self.data.files else 20.3
        self.count = make_scored_count_targets(self.labels, self.indices, min_lognhi=min_lognhi)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = int(self.indices[i])
        clean = None if self.clean_all is None else self.clean_all[idx].astype(np.float32)
        x = build_channels(
            self.x_all[idx].astype(np.float32),
            clean,
            self.wavelength,
            self.wave_norm,
            float(self.zq[i]),
            float(self.snr[i]),
            self.input_mode,
        )
        return (
            torch.from_numpy(x),
            torch.from_numpy(self.center[i].astype(np.float32)),
            torch.from_numpy(self.region[i].astype(np.float32)),
            torch.from_numpy(self.lognhi[i].astype(np.float32)),
            torch.from_numpy(self.mask[i].astype(np.float32)),
            torch.from_numpy(self.offset[i].astype(np.float32)),
            torch.from_numpy(self.offset_weight[i].astype(np.float32)),
            torch.tensor(int(self.count[i]), dtype=torch.long),
            int(idx),
        )


class HybridTestDataset(Dataset):
    def __init__(self, test_fits: str | Path, input_mode: str = "all"):
        self.input_mode = input_mode
        self.wavelength = read_image(test_fits, "WAVELENGTH").astype(np.float32)
        flux = read_image(test_fits, "FLUX").astype(np.float32)
        scale = np.percentile(flux, 75, axis=1, keepdims=True).astype(np.float32)
        self.x_all = normalize_with_scale(flux, scale)
        self.clean_all = None
        if input_mode in {"residual", "all"}:
            clean = read_image(test_fits, "FLUX_CLEAN").astype(np.float32)
            self.clean_all = normalize_with_scale(clean, scale)
        self.wave_norm = wavelength_norm(self.wavelength)
        self.snr = estimate_snr_gu_proxy(self.wavelength, flux).astype(np.float32)
        self.meta = read_meta(test_fits)
        self.targetid = self.meta["TARGETID"].astype(np.int64)
        self.zq = self.meta["Z_QSO"].astype(np.float32)

    def __len__(self) -> int:
        return len(self.targetid)

    def __getitem__(self, i: int):
        clean = None if self.clean_all is None else self.clean_all[i].astype(np.float32)
        x = build_channels(
            self.x_all[i].astype(np.float32),
            clean,
            self.wavelength,
            self.wave_norm,
            float(self.zq[i]),
            float(self.snr[i]),
            self.input_mode,
        )
        return torch.from_numpy(x), int(i)
