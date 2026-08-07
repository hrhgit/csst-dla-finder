from __future__ import annotations

import numpy as np

LYA = 1215.67


def make_center_targets(
    wavelength: np.ndarray,
    labels: dict[str, np.ndarray],
    indices: np.ndarray,
    sigma_pixels: float = 3.0,
    min_lognhi: float = 20.3,
    mid_lognhi: float = 21.0,
    high_lognhi: float = 21.5,
    very_high_lognhi: float = 22.0,
    low_lognhi_radius_pixels: int = 10,
    mid_lognhi_radius_pixels: int = 20,
    high_lognhi_radius_pixels: int = 50,
    very_high_lognhi_radius_pixels: int = 80,
    return_region: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create dense CNN targets on the observed wavelength grid.

    center_target is a Gaussian peak centered on each true DLA.
    lognhi_target stores LOGNHI near the DLA center and zero elsewhere.
    mask marks pixels where lognhi_target should contribute to a loss.
    Higher-column systems get wider LOGNHI masks so damping wings contribute.
    """
    indices = np.asarray(indices, dtype=int)
    n_spec = len(indices)
    n_wave = len(wavelength)
    center_target = np.zeros((n_spec, n_wave), dtype=np.float32)
    region_target = np.zeros((n_spec, n_wave), dtype=np.float32)
    lognhi_target = np.zeros((n_spec, n_wave), dtype=np.float32)
    mask = np.zeros((n_spec, n_wave), dtype=np.float32)
    pixel = np.arange(n_wave, dtype=np.float32)

    for row, idx in enumerate(indices):
        for slot in (1, 2):
            if labels["N_DLA"][idx] < slot:
                continue
            lognhi = float(labels[f"LOGNHI{slot}"][idx])
            if lognhi < min_lognhi:
                continue
            z_dla = float(labels[f"Z_DLA{slot}"][idx])
            lambda_dla = LYA * (1.0 + z_dla)
            center = int(np.argmin(np.abs(wavelength - lambda_dla)))
            peak = np.exp(-0.5 * ((pixel - center) / sigma_pixels) ** 2).astype(np.float32)
            center_target[row] = np.maximum(center_target[row], peak)
            if lognhi >= very_high_lognhi:
                radius = very_high_lognhi_radius_pixels
            elif lognhi >= high_lognhi:
                radius = high_lognhi_radius_pixels
            elif lognhi >= mid_lognhi:
                radius = mid_lognhi_radius_pixels
            else:
                radius = low_lognhi_radius_pixels
            local = np.abs(pixel - center) <= radius
            lognhi_target[row, local] = lognhi
            mask[row, local] = 1.0
            region_target[row, local] = 1.0

    if return_region:
        return center_target, region_target, lognhi_target, mask
    return center_target, lognhi_target, mask


def normalize_flux(flux: np.ndarray, eps: float = 1e-30) -> np.ndarray:
    """A simple robust normalization for fixed-grid 1D spectra."""
    scale = np.percentile(flux, 75, axis=1, keepdims=True)
    scale = np.maximum(scale, eps)
    x = flux / scale
    x = np.nan_to_num(x, nan=1.0, posinf=1.0, neginf=1.0)
    return np.clip(x, 0.0, 3.0).astype(np.float32)
