from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import maximum_filter1d, median_filter, uniform_filter1d
from torch.utils.data import Dataset

from csst_dla.fits_utils import read_image, read_labels, read_meta
from csst_dla.snr import estimate_snr_gu_proxy
from csst_dla.targets import LYA


def normalize_with_scale(flux: np.ndarray, scale: np.ndarray, eps: float = 1e-30) -> np.ndarray:
    x = flux / np.maximum(scale, eps)
    x = np.nan_to_num(x, nan=1.0, posinf=1.0, neginf=1.0)
    return np.clip(x, 0.0, 3.0).astype(np.float32)


def normalize_wzx_style(flux: np.ndarray, eps: float = 1e-30) -> np.ndarray:
    """Per-spectrum median/std normalization used by the WZX input branch.

    The existing hybrid channels retain their percentile scaling.  Adding this
    view gives an early-fusion model both normalizations without changing the
    original eight-channel layout.
    """
    values = np.asarray(flux, dtype=np.float32)
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros_like(values, dtype=np.float32)
    median = np.float32(np.median(values[finite]))
    filled = np.where(finite, values, median).astype(np.float32, copy=False)
    std = np.float32(np.std(filled))
    if not np.isfinite(std) or std < eps:
        std = np.float32(1.0)
    return np.nan_to_num((filled - median) / std, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def build_wzx_style_channels(flux: np.ndarray, clean: np.ndarray) -> np.ndarray:
    """The exact three spectral channels consumed by WZX in ``feature_mode=all``."""
    residual = np.asarray(clean, dtype=np.float32) - np.asarray(flux, dtype=np.float32)
    return np.stack(
        [normalize_wzx_style(flux), normalize_wzx_style(clean), normalize_wzx_style(residual)],
        axis=0,
    ).astype(np.float32, copy=False)


def _normalize_wzx_style_rows(values: np.ndarray, eps: float = 1e-30, block_size: int = 4096) -> np.ndarray:
    """Vectorized row-wise form of :func:`normalize_wzx_style`.

    It keeps the exact per-spectrum statistic but avoids a Python call for all
    500,000 spectra when materializing fusion caches.
    """
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"expected [rows, wavelength], got {values.shape}")
    output = np.empty_like(values, dtype=np.float32)
    for start in range(0, len(values), block_size):
        stop = min(len(values), start + block_size)
        block = values[start:stop]
        finite = np.isfinite(block)
        has_finite = finite.any(axis=1)
        with np.errstate(invalid="ignore"):
            median = np.nanmedian(np.where(finite, block, np.nan), axis=1).astype(np.float32)
        median = np.where(has_finite, median, 0.0).astype(np.float32)
        filled = np.where(finite, block, median[:, None]).astype(np.float32, copy=False)
        std = np.std(filled, axis=1).astype(np.float32)
        std = np.where(has_finite & np.isfinite(std) & (std >= eps), std, 1.0).astype(np.float32)
        output[start:stop] = np.nan_to_num(
            (filled - median[:, None]) / std[:, None], nan=0.0, posinf=0.0, neginf=0.0
        ).astype(np.float32, copy=False)
    return output


def build_wzx_style_channels_rows(flux: np.ndarray, clean: np.ndarray) -> np.ndarray:
    """Vectorized exact WZX views for a batch of raw spectra."""
    flux = np.asarray(flux, dtype=np.float32)
    clean = np.asarray(clean, dtype=np.float32)
    if flux.shape != clean.shape or flux.ndim != 2:
        raise ValueError("flux and clean must be same-shape [rows, wavelength] arrays")
    residual = clean - flux
    return np.stack(
        [
            _normalize_wzx_style_rows(flux),
            _normalize_wzx_style_rows(clean),
            _normalize_wzx_style_rows(residual),
        ],
        axis=1,
    ).astype(np.float32, copy=False)


WZX_FEATURE_MODES = frozenset({"flux", "flux_clean", "residual", "flux_residual", "all"})


def wzx_feature_requires_clean(feature_mode: str) -> bool:
    """Return whether the WZX spectral view needs FLUX_CLEAN."""
    if feature_mode not in WZX_FEATURE_MODES:
        raise ValueError(f"unknown WZX feature_mode={feature_mode!r}")
    return feature_mode != "flux"


def build_wzx_feature_channels(
    flux: np.ndarray,
    clean: np.ndarray | None,
    feature_mode: str,
) -> np.ndarray:
    """Build one WZX-compatible spectral view without inventing clean inputs.

    The channel order follows csst_dla_wzx_pkg.data. This lets feature fusion
    use a flux-only WZX checkpoint without materializing the clean FITS input.
    """
    if feature_mode not in WZX_FEATURE_MODES:
        raise ValueError(f"unknown WZX feature_mode={feature_mode!r}")
    flux = np.asarray(flux, dtype=np.float32)
    if flux.ndim != 1:
        raise ValueError(f"expected one spectrum [wavelength], got {flux.shape}")
    if wzx_feature_requires_clean(feature_mode):
        if clean is None:
            raise ValueError(f"WZX feature_mode={feature_mode!r} requires clean")
        clean = np.asarray(clean, dtype=np.float32)
        if clean.shape != flux.shape:
            raise ValueError("flux and clean must have matching shapes")
        residual = clean - flux
    else:
        residual = None
    channels: list[np.ndarray] = []
    if feature_mode in {"flux", "flux_residual", "all"}:
        channels.append(normalize_wzx_style(flux))
    if feature_mode in {"flux_clean", "all"}:
        channels.append(normalize_wzx_style(clean))
    if feature_mode in {"residual", "flux_residual", "all"}:
        channels.append(normalize_wzx_style(residual))
    return np.stack(channels, axis=0).astype(np.float32, copy=False)


def build_wzx_feature_channels_rows(
    flux: np.ndarray,
    clean: np.ndarray | None,
    feature_mode: str,
) -> np.ndarray:
    """Vectorized build_wzx_feature_channels for many spectra."""
    if feature_mode not in WZX_FEATURE_MODES:
        raise ValueError(f"unknown WZX feature_mode={feature_mode!r}")
    flux = np.asarray(flux, dtype=np.float32)
    if flux.ndim != 2:
        raise ValueError(f"expected [rows, wavelength], got {flux.shape}")
    if wzx_feature_requires_clean(feature_mode):
        if clean is None:
            raise ValueError(f"WZX feature_mode={feature_mode!r} requires clean")
        clean = np.asarray(clean, dtype=np.float32)
        if clean.shape != flux.shape:
            raise ValueError("flux and clean must have matching shapes")
        residual = clean - flux
    else:
        residual = None
    channels: list[np.ndarray] = []
    if feature_mode in {"flux", "flux_residual", "all"}:
        channels.append(_normalize_wzx_style_rows(flux))
    if feature_mode in {"flux_clean", "all"}:
        channels.append(_normalize_wzx_style_rows(clean))
    if feature_mode in {"residual", "flux_residual", "all"}:
        channels.append(_normalize_wzx_style_rows(residual))
    return np.stack(channels, axis=1).astype(np.float32, copy=False)


def smooth_flux(flux: np.ndarray, width: int = 15) -> np.ndarray:
    kernel = np.ones(width, dtype=np.float32) / float(width)
    return np.convolve(flux, kernel, mode="same").astype(np.float32)


# --- engineered channels (flux-only, challenge compliant) -------------------
# 8 A/pixel sampling makes a DLA core sub-pixel: the discriminative structure is
# a one-or-two-pixel dip riding on damping wings a few pixels wide, buried in a
# noise level that varies along the spectrum.  A convolutional stem reproduces
# any *linear* filter of the flux for free, so the only useful additions are
# non-linear statistics that a conv/attention stack cannot synthesise itself:
# a robust *median*-based local noise scale, a *min*-pooled trough depth, and a
# *max*-over-scales matched filter.
SIGMA_WINDOW = 31  # ~250 A: wide enough for a stable local noise estimate
TROUGH_WINDOW = 11  # ~88 A: +-5 px, the probe's search window
MF_WIDTHS = (1, 3, 7, 15)
SIG_CLIP = 12.0
MF_CLIP = 20.0


def local_noise_sigma(high_pass: np.ndarray, width: int = SIGMA_WINDOW) -> np.ndarray:
    """Per-pixel robust noise scale of a high-pass spectrum, via a sliding MAD.

    ``median_filter`` is a non-linear operator: no stack of convolutions can
    reproduce it, so this is genuine added accessibility rather than a
    re-parameterisation of the flux channel.
    """
    width = int(width) | 1
    median = median_filter(high_pass, size=width, mode="reflect")
    mad = median_filter(np.abs(high_pass - median), size=width, mode="reflect")
    return (1.4826 * mad).astype(np.float32)


def sliding_min(values: np.ndarray, width: int = TROUGH_WINDOW) -> np.ndarray:
    """Deepest value inside a centred window (min-pooling, non-linear)."""
    return (-maximum_filter1d(-values, size=int(width) | 1, mode="reflect")).astype(np.float32)


def multiscale_matched_filter(significance: np.ndarray, widths=MF_WIDTHS) -> np.ndarray:
    """Matched-filter SNR for top-hat absorption templates of several widths.

    With unit-variance noise the optimal statistic for a boxcar of width ``w``
    is the windowed sum divided by ``sqrt(w)``.  Absorption is a negative
    excursion, so the response is sign-flipped; the strongest scale is kept.
    """
    response = None
    for width in widths:
        width = int(width) | 1
        smoothed = uniform_filter1d(significance, size=width, mode="reflect")
        current = -(smoothed * np.sqrt(float(width))).astype(np.float32)
        response = current if response is None else np.maximum(response, current)
    return response.astype(np.float32)


def build_engineered_views(
    flux: np.ndarray, smooth: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (significance, trough, matched_filter) built from flux alone."""
    residual = (flux - smooth).astype(np.float32)
    sigma = local_noise_sigma(residual)
    significance = np.clip(residual / np.maximum(sigma, 1e-6), -SIG_CLIP, SIG_CLIP).astype(np.float32)
    trough = np.clip(sliding_min(significance), -SIG_CLIP, SIG_CLIP).astype(np.float32)
    matched = np.clip(multiscale_matched_filter(significance), 0.0, MF_CLIP).astype(np.float32)
    return significance, trough, matched



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
    if input_mode == "raw":
        # Local-architecture entry point: one median-normalized raw-flux channel
        # is built by the upstream FITS reader; keep the channel layout intact.
        channels = [flux]
    elif input_mode == "flux":
        channels = [flux, smooth, zq_channel, snr_channel, wave_norm, blue_mask]
    elif input_mode == "flux_aug":
        # flux (6ch) + derived high-frequency views from raw FLUX only.
        # resid = flux - smooth15 (continuum-removed absorption signature)
        # grad  = np.gradient(flux) (edge strength, localizes DLA boundary)
        #
        # DEPRECATED for transformer towers.  resid = (I - B15) flux and
        # grad = D flux are *exact linear* functions of the flux channel, so a
        # single convolution reproduces them at zero cost and they add
        # I(Y; resid, grad | flux) = 0.  Measured: +4.7 pp on the weak CNN but
        # -1.5 pp (v3c) / -3.5 pp (SOTA) on transformers.  Use ``flux``,
        # ``flux_sig`` or ``flux_feat`` instead.
        resid = (flux - smooth).astype(np.float32)
        grad = np.gradient(flux).astype(np.float32)
        channels = [flux, smooth, zq_channel, snr_channel, wave_norm, blue_mask, resid, grad]
    elif input_mode == "flux_sig":
        # flux (6ch) + local noise-normalised significance.  Unlike resid this
        # is non-linear (sliding MAD), so the conv stem cannot synthesise it.
        significance, _, _ = build_engineered_views(flux, smooth)
        channels = [
            flux,
            smooth,
            zq_channel,
            snr_channel,
            wave_norm,
            blue_mask,
            significance,
        ]
    elif input_mode == "flux_feat":
        # flux_sig (7ch) + two classic non-linear absorption detectors:
        # trough = deepest significance in a window (min-pooled)
        # mf     = strongest multi-scale matched-filter response
        significance, trough, matched = build_engineered_views(flux, smooth)
        channels = [
            flux,
            smooth,
            zq_channel,
            snr_channel,
            wave_norm,
            blue_mask,
            significance,
            trough,
            matched,
        ]
    elif input_mode == "residual":
        if clean is None:
            raise ValueError("input_mode=residual requires FLUX_CLEAN")
        residual = clean - flux
        channels = [flux, clean, residual, zq_channel, snr_channel, wave_norm, blue_mask]
    elif input_mode in {"all", "all_wzx"}:
        if clean is None:
            raise ValueError(f"input_mode={input_mode} requires FLUX_CLEAN")
        residual = clean - flux
        channels = [flux, clean, residual, smooth, zq_channel, snr_channel, wave_norm, blue_mask]
        if input_mode == "all_wzx":
            channels.extend(
                [
                    normalize_wzx_style(flux),
                    normalize_wzx_style(clean),
                    normalize_wzx_style(residual),
                ]
            )
    else:
        raise ValueError(f"unknown input_mode: {input_mode}")
    return np.stack(channels, axis=0).astype(np.float32)


def channel_count(input_mode: str) -> int:
    if input_mode == "raw":
        return 1
    if input_mode == "flux":
        return 6
    if input_mode == "flux_aug":
        return 8
    if input_mode == "flux_sig":
        return 7
    if input_mode == "flux_feat":
        return 9
    if input_mode == "residual":
        return 7
    if input_mode == "all":
        return 8
    if input_mode == "all_wzx":
        return 11
    raise ValueError(f"unknown input_mode: {input_mode}")


class HybridTrainDataset(Dataset):
    def __init__(
        self,
        targets_npz: str | Path,
        train_fits: str | Path,
        split: str,
        input_mode: str = "all",
        max_samples: int | None = None,
        cache_channels: bool = False,
    ):
        self.labels = read_labels(train_fits)
        self.input_mode = input_mode
        with np.load(targets_npz) as source:
            self.target_min_lognhi = (
                float(source["target_min_lognhi"])
                if "target_min_lognhi" in source.files
                else 20.3
            )
            self.wavelength = source["wavelength"].astype(np.float32)
            self.indices = source[f"{split}_idx"]
            self.center = source[f"{split}_center"]
            if f"{split}_region" in source.files:
                self.region = source[f"{split}_region"]
            else:
                self.region = self.center
            self.lognhi = source[f"{split}_lognhi"]
            self.mask = source[f"{split}_mask"]
            self.x_all = source["x"]
            self.clean_all = None
            if self.input_mode in {"residual", "all", "all_wzx"}:
                if "x_clean" not in source.files:
                    raise ValueError(f"input_mode={self.input_mode} requires x_clean")
                self.clean_all = source["x_clean"]
            self.snr = source["snr_proxy"][self.indices].astype(np.float32)
        self.wzx_raw_flux_all = None
        self.wzx_raw_clean_all = None
        if self.input_mode == "all_wzx":
            # The WZX views must be built before hybrid's percentile scaling
            # and clipping; otherwise their pretrained input distribution is
            # not faithfully reproduced.
            self.wzx_raw_flux_all = read_image(train_fits, "FLUX").astype(np.float32)
            self.wzx_raw_clean_all = read_image(train_fits, "FLUX_CLEAN").astype(np.float32)
        self.wave_norm = wavelength_norm(self.wavelength)
        self.offset, self.offset_weight = make_offset_targets(self.wavelength, self.labels, self.indices)
        if max_samples is not None:
            self.indices = self.indices[:max_samples]
            self.center = self.center[:max_samples]
            self.region = self.region[:max_samples]
            self.lognhi = self.lognhi[:max_samples]
            self.mask = self.mask[:max_samples]
            self.offset = self.offset[:max_samples]
            self.offset_weight = self.offset_weight[:max_samples]
        self.zq = self.labels["Z_QSO"][self.indices].astype(np.float32)
        min_lognhi = self.target_min_lognhi
        self.count = make_scored_count_targets(self.labels, self.indices, min_lognhi=min_lognhi)
        self.channels = self._build_channel_cache() if cache_channels else None

    def __len__(self) -> int:
        return len(self.indices)

    def _build_channel_cache(self) -> np.ndarray:
        cached = np.empty(
            (len(self), channel_count(self.input_mode), len(self.wavelength)),
            dtype=np.float32,
        )
        for i in range(len(self)):
            clean = None if self.clean_all is None else self.clean_all[self.indices[i]].astype(np.float32)
            base_mode = "all" if self.input_mode == "all_wzx" else self.input_mode
            base = build_channels(
                self.x_all[self.indices[i]].astype(np.float32),
                clean,
                self.wavelength,
                self.wave_norm,
                float(self.zq[i]),
                float(self.snr[i]),
                base_mode,
            )
            if self.input_mode == "all_wzx":
                cached[i, :8] = base
            else:
                cached[i] = base
        if self.input_mode == "all_wzx":
            cached[:, -3:] = build_wzx_style_channels_rows(
                self.wzx_raw_flux_all[self.indices],
                self.wzx_raw_clean_all[self.indices],
            )
        return cached

    def __getitem__(self, i: int):
        idx = int(self.indices[i])
        if self.channels is not None:
            x = self.channels[i].copy()
        else:
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
            if self.input_mode == "all_wzx":
                x[-3:] = build_wzx_style_channels(self.wzx_raw_flux_all[idx], self.wzx_raw_clean_all[idx])
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
        if input_mode in {"residual", "all", "all_wzx"}:
            clean = read_image(test_fits, "FLUX_CLEAN").astype(np.float32)
            self.clean_all = normalize_with_scale(clean, scale)
        self.wzx_raw_flux_all = flux if input_mode == "all_wzx" else None
        self.wzx_raw_clean_all = clean if input_mode == "all_wzx" else None
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
        if self.input_mode == "all_wzx":
            x[-3:] = build_wzx_style_channels(self.wzx_raw_flux_all[i], self.wzx_raw_clean_all[i])
        return torch.from_numpy(x), int(i)
