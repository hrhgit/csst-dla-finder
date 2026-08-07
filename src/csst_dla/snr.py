from __future__ import annotations

import numpy as np


def robust_noise_from_diff(flux: np.ndarray) -> np.ndarray:
    """Estimate per-spectrum pixel noise from adjacent-pixel differences."""
    diff = np.diff(flux, axis=1)
    med = np.nanmedian(diff, axis=1, keepdims=True)
    mad = np.nanmedian(np.abs(diff - med), axis=1)
    return (1.4826 * mad / np.sqrt(2.0)).astype(np.float32)


def estimate_snr_gu_proxy(
    wavelength: np.ndarray,
    flux: np.ndarray,
    wave_min: float = 2500.0,
    wave_max: float = 4000.0,
) -> np.ndarray:
    """Estimate the GU-band S/N-like quantity from spectra alone.

    The training labels include SNR_GU but test metadata does not. This proxy is
    designed to be available for both train and test and is strongly correlated
    with SNR_GU on the training set.
    """
    mask = (wavelength >= wave_min) & (wavelength < wave_max)
    if int(mask.sum()) < 10:
        raise ValueError(f"Not enough wavelength pixels in {wave_min}-{wave_max} A")
    region = flux[:, mask]
    signal = np.nanmedian(region, axis=1)
    noise = robust_noise_from_diff(region)
    proxy = signal / np.maximum(noise, 1e-30)
    return np.nan_to_num(proxy, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def snr_bin_index(snr: np.ndarray) -> np.ndarray:
    """Map S/N values to the challenge bins: [0,1), [1,2), ..., [7,inf)."""
    return np.digitize(snr, [1, 2, 3, 4, 5, 6, 7]).astype(np.int16)

