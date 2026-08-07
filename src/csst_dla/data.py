from __future__ import annotations

from pathlib import Path

import numpy as np

from .fits_utils import read_image, read_labels, read_meta


def load_train(path: str | Path = "train.fits") -> dict[str, np.ndarray]:
    return {
        "wavelength": read_image(path, "WAVELENGTH").astype(np.float32),
        "flux": read_image(path, "FLUX").astype(np.float32),
        "flux_clean": read_image(path, "FLUX_CLEAN").astype(np.float32),
        "labels": read_labels(path),
    }


def load_test(path: str | Path = "test.fits") -> dict[str, np.ndarray]:
    return {
        "wavelength": read_image(path, "WAVELENGTH").astype(np.float32),
        "flux": read_image(path, "FLUX").astype(np.float32),
        "flux_clean": read_image(path, "FLUX_CLEAN").astype(np.float32),
        "meta": read_meta(path),
    }


def make_stratified_split(
    labels: dict[str, np.ndarray], val_fraction: float = 0.2, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_dla = labels["N_DLA"].astype(int)
    z_qso = labels["Z_QSO"]
    z_bins = np.digitize(z_qso, [2.0, 2.5, 3.0, 3.5])
    strata = n_dla * 10 + z_bins

    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    for value in np.unique(strata):
        idx = np.flatnonzero(strata == value)
        rng.shuffle(idx)
        n_val = max(1, int(round(len(idx) * val_fraction)))
        val_parts.append(idx[:n_val])
        train_parts.append(idx[n_val:])

    train_idx = np.concatenate(train_parts)
    val_idx = np.concatenate(val_parts)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx.astype(np.int64), val_idx.astype(np.int64)

