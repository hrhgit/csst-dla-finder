#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse

import numpy as np

from csst_dla.fits_utils import read_image, read_labels
from csst_dla.snr import estimate_snr_gu_proxy
from csst_dla.targets import make_center_targets, normalize_flux


def normalize_with_scale(flux: np.ndarray, scale: np.ndarray, eps: float = 1e-30) -> np.ndarray:
    scale = np.maximum(scale, eps)
    x = flux / scale
    x = np.nan_to_num(x, nan=1.0, posinf=1.0, neginf=1.0)
    return np.clip(x, 0.0, 3.0).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create dense targets for a CNN detector.")
    parser.add_argument("--train-fits", default="train.fits")
    parser.add_argument("--split", default="splits/split_seed42.npz")
    parser.add_argument("--out", default="outputs/cnn_targets_seed42.npz")
    parser.add_argument("--sigma-pixels", type=float, default=3.0)
    parser.add_argument("--min-lognhi", type=float, default=20.3)
    parser.add_argument("--mid-lognhi", type=float, default=21.0)
    parser.add_argument("--high-lognhi", type=float, default=21.5)
    parser.add_argument("--very-high-lognhi", type=float, default=22.0)
    parser.add_argument("--low-lognhi-radius-pixels", type=int, default=10)
    parser.add_argument("--mid-lognhi-radius-pixels", type=int, default=20)
    parser.add_argument("--high-lognhi-radius-pixels", type=int, default=50)
    parser.add_argument("--very-high-lognhi-radius-pixels", type=int, default=80)
    args = parser.parse_args()

    split = np.load(args.split)
    labels = read_labels(args.train_fits)
    wavelength = read_image(args.train_fits, "WAVELENGTH").astype(np.float32)
    flux = read_image(args.train_fits, "FLUX").astype(np.float32)
    flux_clean = read_image(args.train_fits, "FLUX_CLEAN").astype(np.float32)
    scale = np.percentile(flux, 75, axis=1, keepdims=True).astype(np.float32)
    x = normalize_with_scale(flux, scale)
    x_clean = normalize_with_scale(flux_clean, scale)
    snr_proxy = estimate_snr_gu_proxy(wavelength, flux)

    train_center, train_region, train_lognhi, train_mask = make_center_targets(
        wavelength,
        labels,
        split["train_idx"],
        sigma_pixels=args.sigma_pixels,
        min_lognhi=args.min_lognhi,
        mid_lognhi=args.mid_lognhi,
        high_lognhi=args.high_lognhi,
        very_high_lognhi=args.very_high_lognhi,
        low_lognhi_radius_pixels=args.low_lognhi_radius_pixels,
        mid_lognhi_radius_pixels=args.mid_lognhi_radius_pixels,
        high_lognhi_radius_pixels=args.high_lognhi_radius_pixels,
        very_high_lognhi_radius_pixels=args.very_high_lognhi_radius_pixels,
        return_region=True,
    )
    val_center, val_region, val_lognhi, val_mask = make_center_targets(
        wavelength,
        labels,
        split["val_idx"],
        sigma_pixels=args.sigma_pixels,
        min_lognhi=args.min_lognhi,
        mid_lognhi=args.mid_lognhi,
        high_lognhi=args.high_lognhi,
        very_high_lognhi=args.very_high_lognhi,
        low_lognhi_radius_pixels=args.low_lognhi_radius_pixels,
        mid_lognhi_radius_pixels=args.mid_lognhi_radius_pixels,
        high_lognhi_radius_pixels=args.high_lognhi_radius_pixels,
        very_high_lognhi_radius_pixels=args.very_high_lognhi_radius_pixels,
        return_region=True,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        wavelength=wavelength,
        x=x,
        x_clean=x_clean,
        snr_proxy=snr_proxy,
        snr_gu=labels["SNR_GU"].astype(np.float32),
        train_idx=split["train_idx"],
        val_idx=split["val_idx"],
        train_center=train_center,
        train_region=train_region,
        train_lognhi=train_lognhi,
        train_mask=train_mask,
        val_center=val_center,
        val_region=val_region,
        val_lognhi=val_lognhi,
        val_mask=val_mask,
        target_min_lognhi=np.float32(args.min_lognhi),
        target_mid_lognhi=np.float32(args.mid_lognhi),
        target_high_lognhi=np.float32(args.high_lognhi),
        target_very_high_lognhi=np.float32(args.very_high_lognhi),
        target_low_lognhi_radius_pixels=np.int32(args.low_lognhi_radius_pixels),
        target_mid_lognhi_radius_pixels=np.int32(args.mid_lognhi_radius_pixels),
        target_high_lognhi_radius_pixels=np.int32(args.high_lognhi_radius_pixels),
        target_very_high_lognhi_radius_pixels=np.int32(args.very_high_lognhi_radius_pixels),
    )
    print(f"wrote {out}")
    print(f"x shape: {x.shape}")
    print("snr_proxy p10/p50/p90:", [round(float(v), 4) for v in np.percentile(snr_proxy, [10, 50, 90])])
    print(f"train targets: {train_center.shape}  val targets: {val_center.shape}")
    print(f"train positive pixels: {int((train_center > 0.1).sum())}")
    print(f"val positive pixels: {int((val_center > 0.1).sum())}")
    print(f"train region pixels: {int(train_region.sum())}")
    print(f"val region pixels: {int(val_region.sum())}")
    print(f"train lognhi mask pixels: {int(train_mask.sum())}")
    print(f"val lognhi mask pixels: {int(val_mask.sum())}")


if __name__ == "__main__":
    main()
