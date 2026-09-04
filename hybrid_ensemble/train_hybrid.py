#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from csst_dla.scoring import labels_to_truth, score_catalog
from data import HybridTrainDataset
from model import HybridDlaNet, input_channels
from decode import average_predictions, decode_validation_catalog, predict_member


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device(name)


def focal_bce_with_logits(logits, target, alpha=0.85, gamma=2.0, weight=None):
    bce = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
    prob = torch.sigmoid(logits)
    p_t = prob * target + (1.0 - prob) * (1.0 - target)
    alpha_t = alpha * target + (1.0 - alpha) * (1.0 - target)
    loss = alpha_t * (1.0 - p_t).pow(gamma) * bce
    if weight is not None:
        return (loss * weight).sum() / weight.sum().clamp_min(1.0)
    return loss.mean()


def balanced_class_weights(counts: np.ndarray, power: float) -> list[float]:
    class_counts = np.bincount(counts, minlength=3).astype(np.float64)
    if np.any(class_counts == 0):
        raise ValueError("count head requires all three classes in the training split")
    weights = (len(counts) / (3.0 * class_counts)) ** power
    return (weights / weights.mean()).astype(np.float32).tolist()


def train_one_epoch(model, loader, opt, device, args, count_class_weights=None, epoch=0):
    model.train()
    total = 0.0
    progress_every = max(1, len(loader) // 5)
    for batch_idx, (x, center, region, lognhi, mask, offset, offset_weight, count, _) in enumerate(loader):
        x = x.to(device)
        center = center.to(device)
        region = region.to(device)
        lognhi = lognhi.to(device)
        mask = mask.to(device)
        offset = offset.to(device)
        offset_weight = offset_weight.to(device)
        count = count.to(device)

        out = model(x)
        high_mask = ((lognhi >= args.high_lognhi_threshold) & (mask > 0)).float()
        center_weight = 1.0 + (args.high_lognhi_center_weight - 1.0) * high_mask
        center_loss = focal_bce_with_logits(out["center_logits"], center, weight=center_weight)
        region_loss = focal_bce_with_logits(out["region_logits"], region, alpha=0.75)
        if mask.sum() > 0:
            log_weight = mask * (1.0 + (args.high_lognhi_log_weight - 1.0) * high_mask)
            log_loss = (((20.3 + out["lognhi_raw"] - lognhi) ** 2) * log_weight).sum()
            log_loss = log_loss / log_weight.sum().clamp_min(1.0)
        else:
            log_loss = torch.tensor(0.0, device=device)
        if "offset_raw" in out and offset_weight.sum() > 0:
            offset_loss = (((out["offset_raw"] - offset) ** 2) * offset_weight).sum()
            offset_loss = offset_loss / offset_weight.sum().clamp_min(1.0)
        else:
            offset_loss = torch.tensor(0.0, device=device)
        count_loss = nn.functional.cross_entropy(
            out["count_logits"], count, weight=count_class_weights
        )
        loss = (
            center_loss
            + args.region_loss_weight * region_loss
            + args.lognhi_loss_weight * log_loss
            + args.offset_loss_weight * offset_loss
            + args.count_loss_weight * count_loss
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += float(loss.detach().cpu()) * len(x)
        if batch_idx == 0 or (batch_idx + 1) % progress_every == 0:
            print(
                json.dumps(
                    {
                        "progress": "train",
                        "epoch": epoch,
                        "batch": batch_idx + 1,
                        "total_batches": len(loader),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    return total / len(loader.dataset)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one Hybrid Ensemble member.")
    parser.add_argument("--targets", default="outputs/cnn_targets_seed42.npz")
    parser.add_argument("--train-fits", default="train.fits")
    parser.add_argument("--out-dir", default="hybrid_ensemble/runs/member_all_seed42")
    parser.add_argument("--input-mode", choices=["raw", "flux", "residual", "all", "all_wzx"], default="all")
    parser.add_argument("--hidden", type=int, default=96, help="DilatedResNet base channel width.")
    parser.add_argument(
        "--num-blocks",
        type=int,
        choices=[2, 3, 4, 5],
        default=4,
        help="Number of wavelength-resolution-aware DilatedResNet stages.",
    )
    parser.add_argument(
        "--norm-type",
        choices=["batch", "layer"],
        default="layer",
        help="Normalization used inside the local five-head architecture.",
    )
    parser.add_argument(
        "--head-layers",
        type=int,
        choices=[1, 2, 3],
        default=1,
        help="Number of layers in each local five-head output head.",
    )
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--min-z-dla", type=float, default=1.55)
    parser.add_argument("--region-loss-weight", type=float, default=0.2)
    parser.add_argument("--lognhi-loss-weight", type=float, default=0.05)
    parser.add_argument("--offset-loss-weight", type=float, default=0.1)
    parser.add_argument("--count-loss-weight", type=float, default=0.25)
    parser.add_argument(
        "--count-class-weight-power",
        type=float,
        default=0.0,
        help="0 keeps unweighted cross entropy; 1 uses inverse-frequency weights.",
    )
    parser.add_argument(
        "--truth-min-lognhi",
        type=float,
        default=20.3,
        help="Minimum LOGNHI used when constructing validation truth for model selection.",
    )
    parser.add_argument("--high-lognhi-threshold", type=float, default=22.0)
    parser.add_argument("--high-lognhi-center-weight", type=float, default=4.0)
    parser.add_argument("--high-lognhi-log-weight", type=float, default=4.0)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu", "mps"], default="auto")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_ds = HybridTrainDataset(
        args.targets,
        args.train_fits,
        "train",
        args.input_mode,
        max_samples=args.max_train_samples,
        cache_channels=True,
    )
    val_ds = HybridTrainDataset(
        args.targets,
        args.train_fits,
        "val",
        args.input_mode,
        max_samples=args.max_val_samples,
        cache_channels=True,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=args.norm_type == "batch",
        num_workers=args.num_workers,
        pin_memory=False,
        persistent_workers=args.num_workers > 0,
    )
    # BatchNorm cannot update statistics from a singleton training batch.
    # Drop only that final partial batch; validation remains in eval mode.
    # Validation data is already cached in the parent process. Keeping it out
    # of worker IPC avoids sharing one large backing array per batch and is
    # faster for this small validation split.
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    device = resolve_device(args.device)

    train_counts = np.asarray(train_ds.count, dtype=np.int64)
    class_counts = np.bincount(train_counts, minlength=3).tolist()
    class_weights = balanced_class_weights(train_counts, args.count_class_weight_power)
    count_class_weights = (
        torch.tensor(class_weights, dtype=torch.float32, device=device)
        if args.count_class_weight_power > 0.0
        else None
    )
    print(
        json.dumps(
            {
                "train_count_distribution": dict(enumerate(class_counts)),
                "count_class_weights": class_weights,
                "truth_min_lognhi": args.truth_min_lognhi,
                "target_min_lognhi": val_ds.target_min_lognhi,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    model = HybridDlaNet(
        input_channels(args.input_mode),
        hidden=args.hidden,
        num_blocks=args.num_blocks,
        with_offset=True,
        norm_type=args.norm_type,
        head_layers=args.head_layers,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    history = []
    best_score = -1.0
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, opt, device, args, count_class_weights, epoch=epoch)
        pred = predict_member(model, val_loader, device)
        avg = average_predictions([pred])
        catalog = decode_validation_catalog(
            avg,
            val_ds.indices,
            val_ds.labels,
            val_ds.wavelength,
            threshold=args.threshold,
            min_z_dla=args.min_z_dla,
            lognhi_min=args.truth_min_lognhi,
        )
        truth = labels_to_truth(val_ds.labels, val_ds.indices, min_lognhi=args.truth_min_lognhi)
        score = score_catalog(truth, catalog)
        row = {
            "epoch": epoch,
            "loss": loss,
            "final_score": score.final_score,
            "detection_score": score.detection_score,
            "parameter_score": score.parameter_score,
            "completeness": score.completeness,
            "purity": score.purity,
            "n_pred": score.n_pred,
            "n_match": score.n_match,
            "std_dv": score.std_dv,
            "std_dlognhi": score.std_dlognhi,
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if score.final_score > best_score:
            best_score = score.final_score
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": {
                        "input_mode": args.input_mode,
                        "hidden": args.hidden,
                        "num_blocks": args.num_blocks,
                        "with_offset": True,
                        "norm_type": args.norm_type,
                        "head_layers": args.head_layers,
                    },
                    "threshold": args.threshold,
                    "score": row,
                    "training_config": vars(args),
                },
                out_dir / "best_model.pt",
            )

    torch.save(
        {
            "model_state": model.state_dict(),
            "config": {
                "input_mode": args.input_mode,
                "hidden": args.hidden,
                "num_blocks": args.num_blocks,
                "with_offset": True,
                "norm_type": args.norm_type,
                "head_layers": args.head_layers,
            },
            "threshold": args.threshold,
            "score": history[-1],
            "training_config": vars(args),
        },
        out_dir / "model.pt",
    )
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (out_dir / "training_args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    print(f"wrote {out_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
